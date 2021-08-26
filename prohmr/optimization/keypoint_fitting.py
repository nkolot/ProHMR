import torch
import torch.nn as nn
from typing import Dict

from .optimization_task import OptimizationTask, rel_change
from .losses import keypoint_fitting_loss

class KeypointFitting(OptimizationTask):

    def __call__(self,
                 flow_net: nn.Module,
                 regression_output: Dict,
                 data: Dict,
                 full_frame: bool = True,
                 use_hips: bool = False) -> Dict:
        """
        Fit SMPL to 2D keypoint data.
        Args:
            flow_net (nn.Module): Pretrained Conditional Normalizing Flows network.
            regression_output (Dict): Output of ProHMR for the given input images.
            data (Dict): Dictionary containing images and their corresponding annotations.
            full_frame (bool): If True, perform fitting in the original image. Otherwise fit in the cropped box.
            use_hips (bool): If True, use hip keypoints for fitting. Hips are usually problematic.
        Returns:
            Dict: Optimization output containing SMPL parameters, camera, vertices and model joints.
        """

        pred_cam = regression_output['pred_cam'][:, 0]
        batch_size = pred_cam.shape[0]

        # Differentiating between fitting on the cropped box or the original image coordinates
        if full_frame:
            # Compute initial camera translation
            box_center = data['box_center']
            box_size = data['box_size']
            img_size = data['img_size']
            camera_center = 0.5 * img_size
            depth = 2 * self.cfg.EXTRA.FOCAL_LENGTH / (box_size.reshape(batch_size, 1) * pred_cam[:,0].reshape(batch_size, 1) + 1e-9)
            init_cam_t = torch.zeros_like(pred_cam)
            init_cam_t[:, :2] = pred_cam[:, 1:] + (box_center - camera_center) * depth / self.cfg.EXTRA.FOCAL_LENGTH
            init_cam_t[:, -1] = depth.reshape(batch_size)
            keypoints_2d = data['orig_keypoints_2d']
        else:
            # Translation has been already computed in the forward pass
            init_cam_t = regression_output['pred_cam_t'][:, 0]
            keypoints_2d = data['keypoints_2d']
            keypoints_2d[:, :, :-1] = self.cfg.MODEL.IMAGE_SIZE * (keypoints_2d[:, :, :-1] + 0.5)
            img_size = torch.tensor([self.cfg.MODEL.IMAGE_SIZE, self.cfg.MODEL.IMAGE_SIZE], device=pred_cam.device, dtype=pred_cam.dtype).reshape(1, 2).repeat(batch_size, 1)
            camera_center = 0.5 * img_size

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.detach().clone()

        # Get detected joints and their confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, [-1]]
        if not use_hips:
            joints_conf[:, [8, 9, 12, 25+2, 25+3, 25+14]] *= 0.0


        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones_like(camera_center)

        # Get predicted betas
        betas = regression_output['pred_smpl_params']['betas'][:,0].detach().clone()
        # Initialize latent to 0 (mode of the regressed distribution)
        z = torch.zeros(batch_size, 144, requires_grad=True, device=pred_cam.device)

        # Make z, betas and camera_translation optimizable
        z.requires_grad=True
        betas.requires_grad=True
        camera_translation.requires_grad = True

        # Setup optimizer
        opt_params = [z, betas, camera_translation]
        optimizer = torch.optim.LBFGS(opt_params, lr=1.0, line_search_fn='strong_wolfe')

        # As explained in Section 3.6 of the paper the pose prior reduces to ||z||_2^2
        def pose_prior():
            return (z ** 2).sum(dim=1)

        # Define fitting closure
        def closure():
            optimizer.zero_grad()
            smpl_params, _, _, _, _ = flow_net(z.unsqueeze(1))
            smpl_params = {k: v.squeeze(dim=1) for k,v in smpl_params.items()}
            # Override regression betas with the optimizable variable
            smpl_params['betas'] = betas
            smpl_output = self.smpl(**smpl_params, pose2rot=False)
            model_joints = smpl_output.joints
            loss = keypoint_fitting_loss(smpl_params, model_joints,
                                        camera_translation, camera_center, img_size,
                                        joints_2d, joints_conf, pose_prior,
                                        focal_length)
            loss.backward()
            return loss

        # Run fitting until convergence
        prev_loss = None
        for i in range(self.max_iters):
            loss = optimizer.step(closure)
            if i > 0:
                loss_rel_change = rel_change(prev_loss, loss.item())
                if loss_rel_change < self.ftol:
                    break
            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in opt_params if var.grad is not None]):
                break
            prev_loss = loss.item()

        # Get and save final parameter values
        opt_output = {}
        with torch.no_grad():
            smpl_params, _, _, _, _ = flow_net(z.unsqueeze(1))
            smpl_params = {k: v.squeeze(dim=1) for k,v in smpl_params.items()}
            smpl_params['betas'] = betas
            smpl_output = self.smpl(**smpl_params, pose2rot=False)
            model_joints = smpl_output.joints
            vertices = smpl_output.vertices
        opt_output['smpl_params'] = smpl_params
        opt_output['model_joints'] = model_joints
        opt_output['vertices'] = vertices
        opt_output['camera_translation'] = camera_translation.detach()

        return opt_output

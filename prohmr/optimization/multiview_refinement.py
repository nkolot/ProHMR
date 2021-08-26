import torch
import torch.nn as nn
from typing import Dict

from .optimization_task import OptimizationTask, rel_change
from .losses import multiview_loss

class MultiviewRefinement(OptimizationTask):

    def __call__(self,
                 flow_net: nn.Module,
                 regression_output: Dict,
                 data: Dict):
        """
        Multiple view refinement via optimization.
        Args:
            flow_net (nn.Module): Pretrained Conditional Normalizing Flows network.
            regression_output (Dict): Output of ProHMR for the given input images.
            data (Dict): Dictionary containing images and their corresponding annotations.
        Returns:
            Dict: Optimization output containing SMPL parameters, camera, vertices and model joints.
        """

        # Average predicted betas
        betas = regression_output['pred_smpl_params']['betas'][:,0].detach().clone()
        batch_size = betas.shape[0]
        betas = betas.mean(dim=0).unsqueeze(dim=0).repeat(batch_size, 1)

        # Initialize latent to 0 (mode of the regressed distribution)
        z = torch.zeros(batch_size, 144, requires_grad=True, device=betas.device)

        # Make z optimizable
        z.requires_grad=True

        # Setup optimizer
        opt_params = [z]
        optimizer = torch.optim.LBFGS(opt_params, lr=1.0, line_search_fn='strong_wolfe')

        # As explained in Section 3.6 of the paper the pose prior reduces to ||z||_2^2
        def pose_prior():
            return (z ** 2).sum(dim=1)

        # Define fitting closure
        def closure():
            optimizer.zero_grad()
            smpl_params, _, _, _, _ = flow_net(z.unsqueeze(1))
            smpl_params = {k: v.squeeze(dim=1) for k,v in smpl_params.items()}
            # Override regression betas with the average betas
            smpl_params['betas'] = betas
            loss = multiview_loss(smpl_params, pose_prior)
            loss.backward()
            return loss

        # Run fitting until convergence
        prev_loss = None
        images = []
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
        opt_output = {}

        # Get and save final parameter values
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

        return opt_output

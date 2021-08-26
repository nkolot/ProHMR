import torch
import numpy as np
import pytorch_lightning as pl
from typing import Any, Dict, Tuple

from prohmr.models import SMPL
from yacs.config import CfgNode

from prohmr.utils import SkeletonRenderer
from prohmr.utils.geometry import aa_to_rotmat, perspective_projection
from prohmr.optimization import OptimizationTask
from .backbones import create_backbone
from .heads import SMPLFlow
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss


class ProHMR(pl.LightningModule):

    def __init__(self, cfg: CfgNode):
        """
        Setup ProHMR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        # Create Normalizing Flow head
        self.flow = SMPLFlow(cfg)
        # Create discriminator
        self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smpl_parameter_loss = ParameterLoss()

        # Instantiate SMPL model
        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)                   

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        self.renderer = SkeletonRenderer(self.cfg)
        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        optimizer = torch.optim.AdamW(params=list(self.backbone.parameters()) + list(self.flow.parameters()),
                                     lr=self.cfg.TRAIN.LR,
                                     weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                           lr=self.cfg.TRAIN.LR,
                                           weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        return optimizer, optimizer_disc

    def initialize(self, batch: Dict, conditioning_feats: torch.Tensor):
        """
        Initialize ActNorm buffers by running a dummy forward step
        Args:
            batch (Dict): Dictionary containing batch data
            conditioning_feats (torch.Tensor): Tensor of shape (N, C) containing the conditioning features extracted using thee backbonee
        """
        # Get ground truth SMPL params, convert them to 6D and pass them to the flow module together with the conditioning feats.
        # Necessary to initialize ActNorm layers.
        smpl_params = {k: v.clone() for k,v in batch['smpl_params'].items()}
        batch_size = smpl_params['body_pose'].shape[0]
        has_smpl_params = batch['has_smpl_params']['body_pose'] > 0
        smpl_params['body_pose'] = aa_to_rotmat(smpl_params['body_pose'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)[has_smpl_params]
        smpl_params['global_orient'] = aa_to_rotmat(smpl_params['global_orient'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)[has_smpl_params]
        smpl_params['betas'] = smpl_params['betas'].unsqueeze(1)[has_smpl_params]
        conditioning_feats = conditioning_feats[has_smpl_params]
        with torch.no_grad():
            _, _ = self.flow.log_prob(smpl_params, conditioning_feats)
            self.initialized |= True

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        if train:
            num_samples = self.cfg.TRAIN.NUM_TRAIN_SAMPLES
        else:
            num_samples = self.cfg.TRAIN.NUM_TEST_SAMPLES


        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]

        # Compute keypoint features using the backbone
        conditioning_feats = self.backbone(x)

        # If ActNorm layers are not initialized, initialize them
        if not self.initialized.item():
            self.initialize(batch, conditioning_feats)

        # If validation draw num_samples - 1 random samples and the zero vector
        if num_samples > 1:
            pred_smpl_params, pred_cam, log_prob, _, pred_pose_6d = self.flow(conditioning_feats, num_samples=num_samples-1)
            z_0 = torch.zeros(batch_size, 1, self.cfg.MODEL.FLOW.DIM, device=x.device)
            pred_smpl_params_mode, pred_cam_mode, log_prob_mode, _,  pred_pose_6d_mode = self.flow(conditioning_feats, z=z_0)
            pred_smpl_params = {k: torch.cat((pred_smpl_params_mode[k], v), dim=1) for k,v in pred_smpl_params.items()}
            pred_cam = torch.cat((pred_cam_mode, pred_cam), dim=1)
            log_prob = torch.cat((log_prob_mode, log_prob), dim=1)
            pred_pose_6d = torch.cat((pred_pose_6d_mode, pred_pose_6d), dim=1)
        else:
            z_0 = torch.zeros(batch_size, 1, self.cfg.MODEL.FLOW.DIM, device=x.device)
            pred_smpl_params, pred_cam, log_prob, _,  pred_pose_6d = self.flow(conditioning_feats, z=z_0)

        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}
        output['log_prob'] = log_prob.detach()
        output['conditioning_feats'] = conditioning_feats
        output['pred_pose_6d'] = pred_pose_6d

        # Compute camera translation
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, num_samples, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, :, 1],
                                  pred_cam[:, :, 2],
                                  2*focal_length[:, :, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, :, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size * num_samples, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size * num_samples, -1)
        smpl_output = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, num_samples, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, num_samples, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, num_samples, -1, 2)
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_smpl_params = output['pred_smpl_params']
        pred_pose_6d = output['pred_pose_6d']
        conditioning_feats = output['conditioning_feats']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']


        batch_size = pred_smpl_params['body_pose'].shape[0]
        num_samples = pred_smpl_params['body_pose'].shape[1]
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smpl_params = batch['smpl_params']
        has_smpl_params = batch['has_smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d.unsqueeze(1).repeat(1, num_samples, 1, 1))
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d.unsqueeze(1).repeat(1, num_samples, 1, 1), pelvis_id=25+14)

        # Compute loss on SMPL parameters
        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            gt = gt_smpl_params[k].unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size * num_samples, -1, 3, 3)
            has_gt = has_smpl_params[k].unsqueeze(1).repeat(1, num_samples)
            loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(batch_size, num_samples, -1), gt.reshape(batch_size, num_samples, -1), has_gt)

        # Compute mode and expectation losses for 3D and 2D keypoints
        # The first item of the second dimension always corresponds to the mode
        loss_keypoints_2d_mode = loss_keypoints_2d[:, [0]].sum() / batch_size
        if loss_keypoints_2d.shape[1] > 1:
            loss_keypoints_2d_exp = loss_keypoints_2d[:, 1:].sum() / (batch_size * (num_samples - 1))
        else:
            loss_keypoints_2d_exp = torch.tensor(0., device=device, dtype=dtype)

        loss_keypoints_3d_mode = loss_keypoints_3d[:, [0]].sum() / batch_size
        if loss_keypoints_3d.shape[1] > 1:
            loss_keypoints_3d_exp = loss_keypoints_3d[:, 1:].sum() / (batch_size * (num_samples - 1))
        else:
            loss_keypoints_3d_exp = torch.tensor(0., device=device, dtype=dtype)
        loss_smpl_params_mode = {k: v[:, [0]].sum() / batch_size for k,v in loss_smpl_params.items()}
        if loss_smpl_params['body_pose'].shape[1] > 1:
            loss_smpl_params_exp = {k: v[:, 1:].sum() / (batch_size * (num_samples - 1)) for k,v in loss_smpl_params.items()}
        else:
            loss_smpl_params_exp = {k: torch.tensor(0., device=device, dtype=dtype) for k,v in loss_smpl_params.items()}


        # Filter out images with corresponding SMPL parameter annotations
        smpl_params = {k: v.clone() for k,v in gt_smpl_params.items()}
        smpl_params['body_pose'] = aa_to_rotmat(smpl_params['body_pose'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)
        smpl_params['global_orient'] = aa_to_rotmat(smpl_params['global_orient'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)
        smpl_params['betas'] = smpl_params['betas'].unsqueeze(1)
        has_smpl_params = (batch['has_smpl_params']['body_pose'] > 0)
        smpl_params = {k: v[has_smpl_params] for k, v in smpl_params.items()}
        # Compute NLL loss
        # Add some noise to annotations at training time to prevent overfitting
        if train:
            smpl_params = {k: v + self.cfg.TRAIN.SMPL_PARAM_NOISE_RATIO * torch.randn_like(v) for k, v in smpl_params.items()}
        if smpl_params['body_pose'].shape[0] > 0:
            log_prob, _ = self.flow.log_prob(smpl_params, conditioning_feats[has_smpl_params])
        else:
            log_prob = torch.zeros(1, device=device, dtype=dtype)
        loss_nll = -log_prob.mean()

        # Compute orthonormal loss on 6D representations
        pred_pose_6d = pred_pose_6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_pose_6d = ((torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - torch.eye(2, device=pred_pose_6d.device, dtype=pred_pose_6d.dtype).unsqueeze(0)) ** 2)
        loss_pose_6d = loss_pose_6d.reshape(batch_size, num_samples, -1)
        loss_pose_6d_mode = loss_pose_6d[:, 0].mean()
        loss_pose_6d_exp = loss_pose_6d[:, 1:].mean()

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_EXP'] * loss_keypoints_3d_exp+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_EXP'] * loss_keypoints_2d_exp+\
               self.cfg.LOSS_WEIGHTS['NLL'] * loss_nll+\
               self.cfg.LOSS_WEIGHTS['ORTHOGONAL'] * (loss_pose_6d_exp+loss_pose_6d_mode)+\
               sum([loss_smpl_params_exp[k] * self.cfg.LOSS_WEIGHTS[(k+'_EXP').upper()] for k in loss_smpl_params_exp])+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_MODE'] * loss_keypoints_3d_mode+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_MODE'] * loss_keypoints_2d_mode+\
               sum([loss_smpl_params_mode[k] * self.cfg.LOSS_WEIGHTS[(k+'_MODE').upper()] for k in loss_smpl_params_mode])

        losses = dict(loss=loss.detach(),
                      loss_nll=loss_nll.detach(),
                      loss_pose_6d_exp=loss_pose_6d_exp,
                      loss_pose_6d_mode=loss_pose_6d_mode,
                      loss_keypoints_2d_exp=loss_keypoints_2d_exp.detach(),
                      loss_keypoints_3d_exp=loss_keypoints_3d_exp.detach(),
                      loss_keypoints_2d_mode=loss_keypoints_2d_mode.detach(),
                      loss_keypoints_3d_mode=loss_keypoints_3d_mode.detach())

        for k, v in loss_smpl_params_exp.items():
            losses['loss_' + k + '_exp'] = v.detach()
        for k, v in loss_smpl_params_mode.items():
            losses['loss_' + k + '_mode'] = v.detach()

        output['losses'] = losses

        return loss

    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        summary_writer = self.logger.experiment
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        images = 255*images.permute(0, 2, 3, 1).cpu().numpy()
        num_samples = self.cfg.TRAIN.NUM_TRAIN_SAMPLES if mode == 'train' else self.cfg.TRAIN.NUM_TEST_SAMPLES

        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, num_samples, -1, 3)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, num_samples, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, num_samples, -1, 2)

        for loss_name, val in losses.items():
            summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)
        num_samples_per_image = min(num_samples, self.cfg.EXTRA.NUM_LOG_SAMPLES_PER_IMAGE)

        gt_keypoints_3d = batch['keypoints_3d']
        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, num_samples, -1, 3)

        # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
        predictions = self.renderer(pred_keypoints_3d[:num_images, :num_samples_per_image],
                                    gt_keypoints_3d[:num_images],
                                    2 * gt_keypoints_2d[:num_images],
                                    images=images[:num_images],
                                    camera_translation=pred_cam_t[:num_images, :num_samples_per_image])
        summary_writer.add_image('%s/predictions' % mode, predictions.transpose((2, 0, 1)), step_count)


    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    body_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            body_pose (torch.Tensor): Regressed body pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = body_pose.shape[0]
        gt_body_pose = batch['body_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_body_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(body_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, joint_batch: Dict, batch_idx: int, optimizer_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']
        optimizer, optimizer_disc = self.optimizers(use_pl_optimizer=True)
        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_smpl_params = output['pred_smpl_params']
        num_samples = pred_smpl_params['body_pose'].shape[1]
        pred_smpl_params = output['pred_smpl_params']
        loss = self.compute_loss(batch, output, train=True)
        disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1), pred_smpl_params['betas'].reshape(batch_size * num_samples, -1))
        loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
        loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        loss_disc = self.training_step_discriminator(mocap_batch, pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1), pred_smpl_params['betas'].reshape(batch_size * num_samples, -1), optimizer_disc)
        output['losses']['loss_gen'] = loss_adv
        output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        return output

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        pred_smpl_params = output['pred_smpl_params']
        num_samples = pred_smpl_params['body_pose'].shape[1]
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output

    def downstream_optimization(self, regression_output: Dict, batch: Dict, opt_task: OptimizationTask, **kwargs: Any) -> Dict:
        """
        Run downstream optimization using current regression output
        Args:
            regression_output (Dict): Dictionary containing batch data
            batch (Dict): Dictionary containing batch data
            opt_task (OptimizationTask): Class object for desired optimization task. Must implement __call__ method.
        Returns:
            Dict: Dictionary containing regression output.
        """
        conditioning_feats = regression_output['conditioning_feats']
        flow_net = lambda x: self.flow(conditioning_feats, z=x)
        return opt_task(flow_net=flow_net,
                        regression_output=regression_output,
                        data=batch,
                        **kwargs)

import torch
import pytorch_lightning as pl
from typing import Dict

from prohmr.models import SMPL
from yacs.config import CfgNode

from prohmr.utils import SkeletonRenderer
from .backbones import create_backbone
from .heads import SkeletonFlow
from .losses import Keypoint3DLoss


class ProSkeleton(pl.LightningModule):

    def __init__(self, cfg: CfgNode):
        """
        Setup ProbSkeleton model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        # Create Normalizing Flow head
        self.flow = SkeletonFlow(cfg)

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        self.renderer = SkeletonRenderer(self.cfg)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        optimizer = torch.optim.AdamW(params=list(self.parameters()),
                                     lr=self.cfg.TRAIN.LR,
                                     weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        return optimizer
    
    def initialize(self, batch: Dict, conditioning_feats: torch.Tensor):
        """
        Initialize ActNorm buffers by running a dummy forward step
        Args:
            batch (Dict): Dictionary containing batch data
            conditioning_feats (torch.Tensor): Tensor of shape (N, C) containing the conditioning features extracted using thee backbonee
        """
        gt_keypoints_3d = batch['keypoints_3d'][:, :, :-1].clone()
        batch_size = gt_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.reshape(batch_size, 1, -1)
        with torch.no_grad():
            _, _ = self.flow.log_prob(gt_keypoints_3d.unsqueeze(1).contiguous(), conditioning_feats)
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
        x = batch['keypoints_2d'][:, :, :-1]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # Compute keypoint features using the backbone
        conditioning_feats = self.backbone(x)

        # If ActNorm layers are not initialized, initialize them
        if not self.initialized.item():
            self.initialize(batch, conditioning_feats)

        # If validation draw num_samples - 1 random samples and the zero vector
        if num_samples > 1:
            pred_keypoints_3d, log_prob, _ = self.flow(conditioning_feats, num_samples=num_samples-1)
            z_0 = torch.zeros(batch_size, 1, self.cfg.MODEL.FLOW.DIM, device=x.device)
            pred_keypoints_3d_mode, log_prob_mode, _ = self.flow(conditioning_feats, z=z_0)
            pred_keypoints_3d = torch.cat((pred_keypoints_3d_mode, pred_keypoints_3d), dim=1)
            log_prob = torch.cat((log_prob_mode, log_prob), dim=1)
        else:
            z_0 = torch.zeros(batch_size, 1, self.cfg.MODEL.FLOW.DIM, device=x.device)
            pred_keypoints_3d, log_prob = self.flow(conditioning_feats, z=z_0)

        # Store useful regression outputs to the output dict
        output = {}
        output['log_prob'] = log_prob.detach()
        output['conditioning_feats'] = conditioning_feats
        output['pred_keypoints_3d'] = pred_keypoints_3d
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

        pred_keypoints_3d = output['pred_keypoints_3d']
        conditioning_feats = output['conditioning_feats']

        batch_size = pred_keypoints_3d.shape[0]
        num_samples = pred_keypoints_3d.shape[1]
        device = pred_keypoints_3d.device
        dtype = pred_keypoints_3d.dtype

        # Get annotations
        gt_keypoints_3d = batch['keypoints_3d'][:, :, :-1]

        pred_keypoints_3d = pred_keypoints_3d.reshape(batch_size, num_samples, -1, 3)

        # Compute 3D keypoint loss
        loss_keypoints_3d = ((gt_keypoints_3d.unsqueeze(1).repeat(1, num_samples, 1, 1) - pred_keypoints_3d).abs())

        # Compute mode and expectation losses for the 3D keypoints
        # The first item of the second dimension always corresponds to the mode

        loss_keypoints_3d_mode = loss_keypoints_3d[:, [0]].sum() / batch_size
        if loss_keypoints_3d.shape[1] > 1:
            loss_keypoints_3d_exp = loss_keypoints_3d[:, 1:].sum() / (batch_size * (num_samples - 1))
        else:
            loss_keypoints_3d_exp = torch.tensor(0., device=device, dtype=dtype)

        # Compute NLL loss
        # Add some noise to annotations at training time to prevent overfitting
        if train:
            gt_keypoints_3d = gt_keypoints_3d.clone() + self.cfg.TRAIN.POSE_3D_NOISE_RATIO * torch.randn_like(gt_keypoints_3d)
        log_prob, _ = self.flow.log_prob(gt_keypoints_3d.reshape(batch_size, 1, -1), conditioning_feats)
        loss_nll = -log_prob.mean()

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_EXP'] * loss_keypoints_3d_exp+\
               self.cfg.LOSS_WEIGHTS['NLL'] * loss_nll+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_MODE'] * loss_keypoints_3d_mode

        losses = dict(loss=loss.detach(),
                      loss_nll=loss_nll.detach(),
                      loss_keypoints_3d_exp=loss_keypoints_3d_exp.detach(),
                      loss_keypoints_3d_mode=loss_keypoints_3d_mode.detach())

        output['losses'] = losses
        output['loss'] = loss

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
        num_samples = self.cfg.TRAIN.NUM_TRAIN_SAMPLES if mode == 'train' else self.cfg.TRAIN.NUM_TEST_SAMPLES

        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, num_samples, -1, 3)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']

        for loss_name, val in losses.items():
            summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            batch (Dict): Dictionary containing input and annotations.
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        output = self.forward_step(batch, train=True)
        loss = self.compute_loss(batch, output, train=True)

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
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output

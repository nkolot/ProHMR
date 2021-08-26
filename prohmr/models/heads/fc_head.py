import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from yacs.config import CfgNode

class FCHead(nn.Module):

    def __init__(self, cfg: CfgNode):
        """
        Fully connected head for camera and betas regression.
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(FCHead, self).__init__()
        self.cfg = cfg
        self.npose = 6 * (cfg.SMPL.NUM_BODY_JOINTS + 1)
        self.layers = nn.Sequential(nn.Linear(cfg.MODEL.FLOW.CONTEXT_FEATURES,
                                              cfg.MODEL.FC_HEAD.NUM_FEATURES),
                                              nn.ReLU(inplace=False),
                                              nn.Linear(cfg.MODEL.FC_HEAD.NUM_FEATURES, 13))
        nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)

        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32))[None, None]
        init_betas = torch.from_numpy(mean_params['shape'].astype(np.float32))[None, None]

        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_betas', init_betas)

    def forward(self, smpl_params: Dict, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run forward pass.
        Args:
            smpl_params (Dict): Dictionary containing predicted SMPL parameters.
            feats (torch.Tensor): Tensor of shape (N, C) containing the features computed by the backbone.
        Returns:
            pred_betas (torch.Tensor): Predicted SMPL betas.
            pred_cam (torch.Tensor): Predicted camera parameters.
        """

        batch_size = feats.shape[0]
        num_samples = smpl_params['body_pose'].shape[1]

        offset = self.layers(feats).reshape(batch_size, 1, 13).repeat(1, num_samples, 1)
        betas_offset = offset[:, :, :10]
        cam_offset = offset[:, :, 10:]
        pred_cam = cam_offset + self.init_cam
        pred_betas = betas_offset + self.init_betas

        return pred_betas, pred_cam

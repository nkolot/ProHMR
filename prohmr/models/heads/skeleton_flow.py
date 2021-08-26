import torch
import torch.nn as nn
from nflows.flows import ConditionalGlow
from yacs.config import CfgNode
from typing import Optional, Tuple

class SkeletonFlow(nn.Module):

    def __init__(self, cfg: CfgNode):
        """
        Probabilistic 3D pose prediction head using Normalizing Flows.
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(SkeletonFlow, self).__init__()
        self.cfg = cfg
        self.flow = ConditionalGlow(cfg.MODEL.FLOW.DIM, cfg.MODEL.FLOW.LAYER_HIDDEN_FEATURES,
                                    cfg.MODEL.FLOW.NUM_LAYERS, cfg.MODEL.FLOW.LAYER_DEPTH,
                                    context_features=cfg.MODEL.FLOW.CONTEXT_FEATURES)

    def log_prob(self, samples: torch.Tensor, feats: torch.Tensor) -> Tuple:
        """
        Compute the log-probability of a set of 3D pose given a batch of images.
        Args:
            samples (Dict): Dictionary containing a set oßf 3D pose samples.
            feats (torch.Tensor): Conditioning features of shape (N, C).
        Returns:
            log_prob (torch.Tensor): Log-probability of the samples with shape (B,ß N).
            z (torch.Tensor): The Gaussian latent corresponding to each sample with shape (B, N, 3*J).
        """
        batch_size = feats.shape[0]
        num_samples = samples.shape[1]
        feats = feats.reshape(batch_size, 1, -1).repeat(1, num_samples, 1)
        log_prob, z = self.flow.log_prob(samples.reshape(batch_size*num_samples, -1).to(feats.dtype), feats.reshape(batch_size*num_samples, -1))
        log_prob = log_prob.reshape(batch_size, num_samples)
        z = z.reshape(batch_size, num_samples, -1)
        return log_prob, z

    def forward(self, feats: torch.Tensor, num_samples: Optional[int] = None, z: Optional[torch.Tensor] = None) -> Tuple:
        """
        Run a forward pass of the model.
        If z is not specified, then the model randomly draws num_samples samples for each image in the batch.
        Otherwise the batch of latent vectors z is transformed using the Conditional Normalizing Flows model.
        Args:
            feats (torch.Tensor): Conditioning features of shape (N, C).
            num_samples (int): Number of samples to draw per image.
            z (torch.Tensor): A batch of latent vectors of shape (B, N, 3*J).
        Returns:
            pred_pose (torch.Tensor): Predicted 3D pose samples of shape (B, N, J, 3).
            log_prob (torch.Tensor): Log-probability of the samples with shape (B, N).
            z (torch.Tensor): Either the input z or the randomly drawn batch of latent Gaussian vectors.
        """

        batch_size = feats.shape[0]

        if z is None:
            samples, log_prob, z = self.flow.sample_and_log_prob(num_samples, context=feats)
            z = z.reshape(batch_size, num_samples, -1)
            pred_pose = samples.reshape(batch_size, num_samples, -1 , 3)
        else:
            num_samples = z.shape[1]
            samples, log_prob, z = self.flow.sample_and_log_prob(num_samples, context=feats, noise=z)
            pred_pose = samples.reshape(batch_size, num_samples, -1 , 3)

        return pred_pose, log_prob, z
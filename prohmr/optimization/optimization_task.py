import torch
import numpy as np
from yacs.config import CfgNode
from prohmr.models import SMPL

def rel_change(prev_val: float, curr_val: float) -> float:
    """
    Compute relative change. Code from https://github.com/vchoutas/smplify-x
    Args:
        prev_val (float): Previous value
        curr_val (float): Current value
    Returns:
        float: Relative change
    """
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])

class OptimizationTask:

    def __init__(self,
                 cfg: CfgNode,
                 max_iters: int = 10,
                 ftol: float = 1e-9,
                 gtol: float = 1e-9,
                 device: torch.device = torch.device('cuda')):
        """
        Base downstream optimization class.
        Args:
            cfg (CfgNode): Model config file.
            max_iters (int): Maximum number of iterations to run fitting for.
            ftol (float): Relative loss change tolerance.
            gtol (float): Absolute gradient value tolerance.
            device (torch.device): Device to run fitting on.
        """

        # Store options
        self.cfg = cfg
        self.max_iters = max_iters
        self.ftol = ftol
        self.gtol = gtol
        self.device = device

        # Load SMPL model
        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg).to(device)

import torch
import torch.nn as nn
from typing import Dict, Callable

from prohmr.utils.geometry import perspective_projection

def gmof(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Geman-McClure error function.
    Args:
        x (torch.Tensor): Raw error signal.
        sigma (float): Robustness hyperparameter
    Returns:
        torch.Tensor: Robust error signal
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

def keypoint_fitting_loss(smpl_params: Dict,
                          model_joints: torch.Tensor,
                          camera_translation: torch.Tensor,
                          camera_center: torch.Tensor,
                          img_size: torch.Tensor,
                          joints_2d: torch.Tensor,
                          joints_conf: torch.Tensor,
                          pose_prior: Callable,
                          focal_length: torch.Tensor,
                          sigma: float = 100.0,
                          pose_prior_weight: float = 4.0,
                          shape_prior_weight: float = 6.0) -> torch.Tensor:
    """
    Loss function for fitting the SMPL model on 2D keypoints.
    Args:
        model_joints (torch.Tensor): Tensor of shape (B, N, 3) containing the SMPL 3D joint locations.
        camera_translation (torch.Tensor): Tensor of shape (B, 3) containing the camera translation.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        img_size (torch.Tensor): Tensor of shape (B, 2) containing the image size in pixels (height, width).
        joints_2d (torch.Tensor): Tensor of shape (B, N, 2) containing the target 2D joint locations.
        joints_conf (torch.Tensor): Tensor of shape (B, N, 1) containing the target 2D joint confidences.
        pose_prior (Callable): Returns the pose prior value.
        focal_length (float): Focal length value in pixels.
        pose_prior_weight (float): Pose prior loss weight.
        shape_prior_weight (float): Shape prior loss weight.
    Returns:
        torch.Tensor: Total loss value.
    """
    betas = smpl_params['betas']
    batch_size = betas.shape[0]
    img_size = img_size.max(dim=-1)[0]

    # Heuristic for scaling data_weight with resolution used in SMPLify-X
    data_weight = (1000. / img_size).reshape(-1, 1, 1).repeat(1, 1, 2)

    # Project 3D model joints
    projected_joints = perspective_projection(model_joints, camera_translation, focal_length, camera_center=camera_center)

    # Compute robust reprojection loss
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = ((data_weight ** 2) * (joints_conf ** 2) * reprojection_error).sum()

    # Compute pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior().sum()

    # Compute shape prior loss
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum()

    # Add up all losses
    total_loss = reprojection_loss + pose_prior_loss + shape_prior_loss

    return total_loss.sum()

def multiview_loss(smpl_params: Dict,
                   pose_prior: Callable,
                   pose_prior_weight: float = 1.0,
                   consistency_weight: float = 300.0):
    """
    Loss function for multiple view refinement (Eq. 12)
    Args:
        smpl_params (Dict): Dictionary containing the SMPL model parameters.
        pose_prior (Callable): Returns the pose prior value.
        pose_prior_weight (float): Pose prior loss weight.
        shape_prior_weight (float): Shape prior loss weight.
    Returns:
        torch.Tensor: Total loss value.
    """
    body_pose = smpl_params['body_pose']

    # Compute pose consistency loss.
    mean_pose = body_pose.mean(dim=0).unsqueeze(dim=0)
    pose_diff = ((body_pose - mean_pose) ** 2).sum(dim=-1)
    consistency_loss = consistency_weight ** 2 * pose_diff.sum()

    # Compute pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior().sum()

    # Add up all losses
    total_loss = consistency_loss + pose_prior_loss

    return total_loss

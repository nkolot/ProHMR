import numpy as np
from typing import Dict
from yacs.config import CfgNode

from .dataset import Dataset


class SkeletonDataset(Dataset):

    def __init__(self, cfg: CfgNode, dataset_file: str, mean_params: str, train: bool = True, **kwargs):
        """
        Dataset class used for loading 2D keypoints and annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            mean_params (str): Path to npz file with entries 'mean' and 'std' that contain the mean and variance of the 2D keypoints in the dataset.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        self.train = train
        self.cfg = cfg
        self.data = np.load(dataset_file)
        mean_params = np.load(mean_params)

        keypoint_list_2d = list(range(13)) + [14, 16, 17, 18]
        keypoint_list_3d = list(range(13)) + [16, 17, 18]

        self.keypoints_2d = self.data['extra_keypoints_2d'][:, keypoint_list_2d].astype(np.float32)
        self.keypoints_2d_mean = mean_params['mean'].astype(np.float32)
        self.keypoints_2d_std = mean_params['std'].astype(np.float32)
        self.keypoints_2d[:, :, :-1] = (self.keypoints_2d[:, :, :-1] - self.keypoints_2d_mean[np.newaxis]) / self.keypoints_2d_std[np.newaxis]
        
        self.keypoints_3d = self.data['extra_keypoints_3d'][:, keypoint_list_3d].astype(np.float32)

    def __len__(self):
        return len(self.keypoints_2d)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns an example from the dataset.
        """
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = self.keypoints_3d[idx].copy()
        item = {}
        item['keypoints_2d'] = keypoints_2d
        item['keypoints_3d'] = keypoints_3d
        return item

from typing import Dict, Optional

import torch
import numpy as np
import pytorch_lightning as pl
from yacs.config import CfgNode

from prohmr.configs import to_lower
from .dataset import Dataset
from .image_dataset import ImageDataset
from .mocap_dataset import MoCapDataset
from .openpose_dataset import OpenPoseDataset
from .batched_image_dataset import BatchedImageDataset
from .skeleton_dataset import SkeletonDataset

def create_dataset(cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True) -> Dataset:
    """
    Instantiate a dataset from a config file.
    Args:
        cfg (CfgNode): Model configuration file.
        dataset_cfg (CfgNode): Dataset configuration info.
        train (bool): Variable to select between train and val datasets.
    """

    dataset_type = Dataset.registry[dataset_cfg.TYPE]
    return dataset_type(cfg, **to_lower(dataset_cfg), train=train)

class MixedDataset:

    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True) -> None:
        """
        Setup Mixed dataset containing different dataset mixed together.
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            train (bool): Variable to select between train and val datasets.
        """

        dataset_list = cfg.DATASETS.TRAIN if train else cfg.DATASETS.VAL
        self.datasets = [create_dataset(cfg, dataset_cfg[dataset], train=train) for dataset, v in dataset_list.items()]
        self.weights = np.array([v.WEIGHT for dataset, v in dataset_list.items()]).cumsum()

    def __len__(self) -> int:
        """
        Returns:
            int: Sum of the lengths of each dataset
        """
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, i: int) -> Dict:
        """
        Index an element from the dataset.
        This is done by randomly choosing a dataset using the mixing percentages
        and then randomly choosing from the selected dataset.
        Returns:
            Dict: Dictionary containing data and labels for the selected example
        """
        p = torch.rand(1).item()
        for i in range(len(self.datasets)):
            if p <= self.weights[i]:
                p = torch.randint(0, len(self.datasets[i]), (1,)).item()
                return self.datasets[i][p]

class ProHMRDataModule(pl.LightningDataModule):

    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode) -> None:
        """
        Initialize LightningDataModule for ProHMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load datasets necessary for training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
        """
        self.train_dataset = MixedDataset(self.cfg, self.dataset_cfg, train=True)
        self.val_dataset = MixedDataset(self.cfg, self.dataset_cfg, train=False)
        self.mocap_dataset = MoCapDataset(**to_lower(self.dataset_cfg[self.cfg.DATASETS.MOCAP]))

    def train_dataloader(self) -> Dict:
        """
        Setup training data loader.
        Returns:
            Dict: Dictionary containing image and mocap data dataloaders
        """
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, self.cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS)
        mocap_dataloader = torch.utils.data.DataLoader(self.mocap_dataset, self.cfg.TRAIN.NUM_TRAIN_SAMPLES * self.cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=1)
        return {'img': train_dataloader, 'mocap': mocap_dataloader}

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Setup val data loader.
        Returns:
            torch.utils.data.DataLoader: Validation dataloader  
        """
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, self.cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS)
        return val_dataloader

class ProSkeletonDataModule(pl.LightningDataModule):

    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode) -> None:
        """
        Initialize LightningDataModule for ProbSkeleton training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load datasets necessary for training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
        """
        mean_stats = self.dataset_cfg[self.cfg.DATASETS.TRAIN].MEAN_STATS
        self.train_dataset = SkeletonDataset(self.cfg, self.dataset_cfg[self.cfg.DATASETS.TRAIN].DATASET_FILE, mean_stats, train=True)
        self.val_dataset = SkeletonDataset(self.cfg, self.dataset_cfg[self.cfg.DATASETS.VAL].DATASET_FILE, mean_stats, train=False)

    def train_dataloader(self) -> Dict:
        """
        Setup training data loader.
        Returns:
            Dict: Dictionary containing image and mocap data dataloaders
        """
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, self.cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS)
        return train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Setup val data loader.
        Returns:
            torch.utils.data.DataLoader: Validation dataloader  
        """
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, self.cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS)
        return val_dataloader

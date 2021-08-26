"""
Script used to train ProHMR.
Example usage:
python train_prohmr.py --root_dir=/path/to/experiment/folder

Running the above will use the default config file to train ProHMR as in the paper.
The code uses PyTorch Lightning for training.
"""
import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from prohmr.configs import get_config, prohmr_config, dataset_config
from prohmr.datasets import ProHMRDataModule
from prohmr.models import ProHMR

parser = argparse.ArgumentParser(description='ProHMR training code')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file')
parser.add_argument('--root_dir', type=str, required=True, help='Directory to save logs and checkpoints')


args = parser.parse_args()

# Load model config
if args.model_cfg is None:
    model_cfg = prohmr_config()
else:
    model_cfg = get_config(args.model_cfg)

# Load dataset config
dataset_cfg = dataset_config()

# Setup training and validation datasets
data_module = ProHMRDataModule(model_cfg, dataset_cfg)

# Setup model
model = ProHMR(model_cfg)

# Setup Tensorboard logger
logger = TensorBoardLogger(os.path.join(args.root_dir, 'tensorboard'), name='', version='', default_hp_metric=False)

# Setup checkpoint saving
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(args.root_dir, 'checkpoints'), every_n_train_steps=model_cfg.GENERAL.CHECKPOINT_STEPS)

# Setup PyTorch Lightning Trainer
trainer = pl.Trainer(default_root_dir=args.root_dir,
                     logger=logger,
                     gpus=1,
                     limit_val_batches=1,
                     num_sanity_val_steps=0,
                     log_every_n_steps=model_cfg.GENERAL.LOG_STEPS,
                     flush_logs_every_n_steps=model_cfg.GENERAL.LOG_STEPS,
                     val_check_interval=model_cfg.GENERAL.VAL_STEPS,
                     progress_bar_refresh_rate=1,
                     precision=16,
                     max_steps=model_cfg.GENERAL.TOTAL_STEPS,
                     move_metrics_to_cpu=True,
                     callbacks=[checkpoint_callback])

# Train the model
trainer.fit(model, datamodule=data_module)

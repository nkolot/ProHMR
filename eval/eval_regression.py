"""
Script used for evaluating the 3D pose errors of ProHMR (mode + minimum).

Example usage:
python eval_regression.py --checkpoint=/path/to/checkpoint --dataset=3DPW-TEST

Running the above will compute the Reconstruction Error for the mode as well as the minimum error for the test set of 3DPW.
"""
import torch
import argparse
from tqdm import tqdm
from prohmr.configs import get_config, prohmr_config, dataset_config
from prohmr.models import ProHMR
from prohmr.utils import Evaluator, recursive_to
from prohmr.datasets import create_dataset

parser = argparse.ArgumentParser(description='Evaluate trained models')
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
parser.add_argument('--dataset', type=str, default='H36M-VAL-P2', choices=['H36M-VAL-P2', '3DPW-TEST', 'MPI-INF-TEST'], help='Dataset to evaluate')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
parser.add_argument('--num_samples', type=int, default=4096, help='Number of test samples to draw')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
parser.add_argument('--log_freq', type=int, default=10, help='How often to log results')
parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')


args = parser.parse_args()

# Use the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model config
if args.model_cfg is None:
    model_cfg = prohmr_config()
else:
    model_cfg = get_config(args.model_cfg)

# Load dataset config
dataset_cfg = dataset_config()[args.dataset]

# Update number of test samples drawn to the desired value
model_cfg.defrost()
model_cfg.TRAIN.NUM_TEST_SAMPLES = args.num_samples
model_cfg.freeze()

# Setup model
model = ProHMR.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
model.eval()

# Create dataset and data loader
dataset = create_dataset(model_cfg, dataset_cfg, train=False)
dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

# List of metrics to log
metrics = ['mode_re', 'min_re']

# Setup evaluator object
evaluator = Evaluator(dataset_length=len(dataset), keypoint_list=dataset_cfg.KEYPOINT_LIST, pelvis_ind=model_cfg.EXTRA.PELVIS_IND, metrics=metrics)

# Go over the images in the dataset.
for i, batch in enumerate(tqdm(dataloader)):
    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)
    evaluator(out, batch)
    if i % args.log_freq == args.log_freq - 1:
        evaluator.log()

"""
Script used for evaluating the 3D pose errors for multiview refinement.

Example usage:
python eval_multiview.py --checkpoint=/path/to/checkpoint --dataset=H36M-VAL-MULTIVIEW

Running the above will compute the MPJPE and Reconstruction Error before and after fitting for the test set of Human3.6M.
"""
import torch
import argparse
from tqdm import tqdm
from prohmr.configs import get_config, prohmr_config, dataset_config
from prohmr.models import ProHMR
from prohmr.optimization import MultiviewRefinement
from prohmr.utils import Evaluator, recursive_to
from prohmr.datasets import create_dataset

parser = argparse.ArgumentParser(description='Evaluate trained models')
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
parser.add_argument('--dataset', type=str, choices=['H36M-VAL-MULTIVIEW', 'MANNEQUIN'], help='Dataset to evaluate')
parser.add_argument('--log_freq', type=int, default=10, help='How often to log results')

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

# Setup model
model = ProHMR.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
model.eval()

# Setup fitting
multiview_refinement = MultiviewRefinement(model_cfg, max_iters=30)

# Create dataset
dataset = create_dataset(model_cfg, dataset_cfg, train=False)

# List of metrics to log
metrics = ['mode_mpjpe', 'mode_re', 'opt_mpjpe', 'opt_re']

# Setup evaluator object
evaluator = Evaluator(dataset_length=dataset.total_length(), keypoint_list=dataset_cfg.KEYPOINT_LIST, pelvis_ind=model_cfg.EXTRA.PELVIS_IND, metrics=metrics)

# Go over the examples in the dataset. Each batch contains all views of the same pose
for i, batch in enumerate(tqdm(dataset)):
    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)
    opt_out = model.downstream_optimization(regression_output=out,
                                            batch=batch,
                                            opt_task=multiview_refinement)
    evaluator(out, batch, opt_output=opt_out)
    if i % args.log_freq == args.log_freq - 1:
        evaluator.log()

import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + "/../.."))
import torch
from basicts import launch_training

torch.set_num_threads(2)  


def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    parser.add_argument(
        "-c", "--cfg",
        default="doignn/Autoencoder_CMADS_AVE_TEM.py",
        help="training config"
    )
    parser.add_argument("--gpus", default="4,5", help="visible gpus, e.g. '0' or '0,1'")
    
    parser.add_argument("--seed", type=int, default=None, help="random seed (optional)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    launch_training(args.cfg, args.gpus)

import os
import cv2
from A import cfgs as cfgsA
from A import launch as launchA
import argparse
import torch
import numpy as np
import random

def load_args():
    """
    Load the configuration parameters of this task
    :return: The corresponding configuration parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='A',
                        help='which task to perform. possible choices: A, B')
    args = parser.parse_args()

    return args


def load_config(args):
    """
    Construct the configuration object used for running

    :param args: The input parameters
    :return: The total configuration parameters
    """
    path = os.getcwd()
    # Setup cfg.
    assert args.task == 'A' or args.task == 'B'
    base_path = os.path.abspath('.')
    if args.task == 'A':
        cfg = cfgsA.get_cfg()
    else:
        cfg = None

    cfg.LR_TRAIN_PATH = os.path.join(path, "data", "DIV2K_train_LR_bicubic", "X4")
    cfg.HR_TRAIN_PATH = os.path.join(path, "data", "DIV2K_train_HR")
    cfg.LR_VAL_PATH = os.path.join(path, "data", "DIV2K_valid_LR_bicubic", "X4")
    cfg.HR_VAL_PATH = os.path.join(path, "data", "DIV2K_valid_HR")

    return cfg


def main():

    # Load the input arguments
    args = load_args()

    # Merge the input arguments in the configuration.
    cfg = load_config(args)

    # Fix random seed
    random.seed(cfg.RAND_SEED)
    np.random.seed(cfg.RAND_SEED)
    torch.manual_seed(cfg.RAND_SEED)
    torch.cuda.manual_seed(cfg.RAND_SEED)
    torch.cuda.manual_seed_all(cfg.RAND_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

    if args.task == "A":
       launchA.train(cfg)
       #launchA.test(cfg, dataset="B100", trial_time=48000)

main()
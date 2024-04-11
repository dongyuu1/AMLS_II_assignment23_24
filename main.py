import os
import random
import argparse
import torch
import numpy as np
from A import cfgs as cfgsA
from A import launch as launchA


def load_args():
    """
    Load the configuration parameters of this task
    :return: The corresponding configuration parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="A",
        help="which task to perform. possible choices: A, B",
    )
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument(
        "--cmid", type=int, default=32, help="number of channels of the latent features"
    )
    parser.add_argument(
        "--cup",
        type=int,
        default=24,
        help="number of channels of the latent features in the reconstruction module",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-3, help="the initial learning rate"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="B100",
        help="the test dataset selected, not used for training",
    )
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
    assert args.task == "A" or args.task == "B"
    if args.task == "A":
        cfg = cfgsA.get_cfg()
    else:
        cfg = None

    cfg.LR_TRAIN_PATH = os.path.join(path, "Datasets", "DIV2K_train_LR_bicubic", "X4")
    cfg.HR_TRAIN_PATH = os.path.join(path, "Datasets", "DIV2K_train_HR")
    cfg.LR_VAL_PATH = os.path.join(path, "Datasets", "DIV2K_valid_LR_bicubic", "X4")
    cfg.HR_VAL_PATH = os.path.join(path, "Datasets", "DIV2K_valid_HR")
    cfg.MODEL.CMID = args.cmid
    cfg.MODEL.CUP = args.cup
    cfg.TRAIN.LR = args.lr
    return cfg


def main():
    """
    The main function switching the mode
    :return: None
    """
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
        if args.mode == "train":
            launchA.train(cfg)
            launchA.test(cfg, dataset="DIV2K", trial_time=48000)
        elif args.mode == "test":
            launchA.test(cfg, dataset=args.test_data, trial_time=48000)
        else:
            launchA.visualization(cfg, dataset=args.test_data, photo_num=6)


if __name__ == "__main__":
    main()

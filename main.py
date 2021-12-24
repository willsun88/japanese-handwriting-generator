import numpy as np
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import argparse



if __name__ == "__main__":
    # Check the arguments for training or generating
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--gen", action="store_true")
    args = parser.parse_args()

    if args.train:
        pass
    elif args.gen:
        pass
    else:
        print("Must either include the --train or --gen flag!")
        exit()


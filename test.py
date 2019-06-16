"""Test model"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TrainDataset
from model import Model
from utils import get_predictions



def test(ckpt, gpu):
    """Run model testing"""

    model_info = torch.load(ckpt, map_location='cpu')
    model = Model(ckpt=ckpt, gpu=gpu).eval()
    if gpu:
        model = model.cuda()

    # dataset
    dataset = TrainDataset(
        phase='test',
        shape=model_info['input_size']
    )
    dataloader = DataLoader(dataset, batch_size=1)

    total = 0
    correct = 0

    for X, y in dataloader:
        total += 1
        if gpu:
            X = X.cuda()

        with torch.no_grad():
            logit, _, _ = model(X)
            logit = logit.squeeze()
            logit = torch.softmax(logit, dim=0)
            logit = logit.cpu().data.numpy()
            pred = np.argmax(logit)
            if pred == y[0].item():
                correct += 1
            print(correct, total, correct /total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', 
                        help='Path to checkpoint file')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    test(args.ckpt, args.gpu)


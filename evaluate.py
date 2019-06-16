"""Test model"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TestDataset
from model import Model



def evaluate(ckpt, image_folder, gpu):
    """Run model evaluation
    
    Running this evaluation script will output a csv file, `result.csv` at the
    working directory.
    """
    model_info = torch.load(ckpt, map_location='cpu')
    model = Model(ckpt=ckpt, gpu=gpu).eval()
    if gpu:
        model = model.cuda()

    # dataset
    dataset = TestDataset(
        image_folder=image_folder,
        shape=model_info['input_size']
    )
    dataloader = DataLoader(dataset, batch_size=1)

    filenames = []
    predictions = []
    for filename, each_image in dataloader:
        print(filename[0])
        if gpu:
            each_image = each_image.cuda()

        with torch.no_grad():
            logit, _, _ = model(each_image)
            logit = logit.squeeze()
            logit = torch.softmax(logit, dim=0)
            predictions.append(logit.cpu().data.numpy())
            filenames.append(filename)

    # write out the result
    filenames = pd.DataFrame(np.stack(filenames, axis=0))
    predictions = pd.DataFrame(np.stack(predictions, axis=0))
    result = pd.concat([filenames, predictions], axis=1)
    result.to_csv('result.csv', index=None, header=False)

    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_images',
                        help='Path to single image or folder of image')
    parser.add_argument('--ckpt', default='ckpt/model.ckpt',
                        help='Path to checkpoint file')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    result = evaluate(args.ckpt, args.input_images, args.gpu)


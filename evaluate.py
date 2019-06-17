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
from utils import get_predictions



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
    logits = []
    predicted_indices, predicted_strings = [], []
    for filename, each_image in dataloader:
        print(filename[0])
        if gpu:
            each_image = each_image.cuda()

        with torch.no_grad():
            logit, _, _ = model(each_image)
            logit = logit.squeeze()
            logit = torch.softmax(logit, dim=0)
            logit = logit.cpu().data.numpy()
            predicted_idx, predicted_str = get_predictions(logit)

            logits.append(logit)
            filenames.append(filename)
            predicted_indices.append(predicted_idx)
            predicted_strings.append(predicted_str)

    # write out the result
    filenames = pd.DataFrame(np.stack(filenames, axis=0))
    logits = pd.DataFrame(np.stack(logits, axis=0))
    predicted_indices = pd.DataFrame(np.stack(predicted_indices, axis=0))
    predicted_strings = pd.DataFrame(np.stack(predicted_strings, axis=0))

    # save the confidence and result
    confidence = pd.concat([filenames, logits], axis=1)
    confidence.to_csv('./data/confidence.csv', index=None, header=False)

    result = pd.concat([filenames, predicted_indices, predicted_strings],
                       axis=1)
    result.to_csv('./data/result.csv', index=None, header=False)

    return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_images',
                        help='Path to single image or folder of image')
    parser.add_argument('--ckpt', default='ckpt/010.ckpt',
                        help='Path to checkpoint file')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    logit = evaluate(args.ckpt, args.input_images, args.gpu)


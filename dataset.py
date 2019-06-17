"""
Set up dataset to be consumed by the model
"""
import os
import glob

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import imgaug as ia
import imgaug.augmenters as iaa


__all__ = ['TrainDataset']


augmenter = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Sometimes(0.5, iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode='median'
        )),
        iaa.SomeOf((0, 5),
            [
                iaa.Sometimes(0.5, iaa.Superpixels(p_replace=(0, 1.0),
                              n_segments=(20, 200))),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True),
                iaa.Add((-10, 10), per_channel=0.5),
                iaa.AddToHueAndSaturation((-20, 20)),
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0))
                    )
                ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
            random_order=True
        )
    ],
    random_order=True
)


class TrainDataset(Dataset):
    """This dataset assume there exists
        - Folders: `data/train`, `data/val` and `data/test`, each contains
        the corresponding image.
        - Files: `data/train.csv`, `data/val.csv`, and `data/test.csv`, each
        contains the file name and corresponding label (no-header csv file)
    """

    def __init__(self, phase='train', shape=(256, 256)):
        """Initialize the dataset"""
        self.phase = phase
        self.shape = shape
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.image_folder = 'data'
        self.label = pd.read_csv('data/{}.csv'.format(phase), header=None)

    def __getitem__(self, index):
        """Get the corresponding image and label

        # Arguments
            index [int]: the index of the file in the dataset

        # Returns
            [FloatTensor]: image of shape CxHxW
            [int]: the object class
        """
        filename = self.label[0][index]
        filepath = os.path.join(self.image_folder, filename)

        # load and transform image
        image = cv2.imread(filepath)
        image = augmenter.augment_image(image)
        image = ia.imresize_single_image(image, self.shape)
        image = image / 255
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)
        image = torch.FloatTensor(image)

        # load label
        object_class = self.label[1][index].item() - 1

        return image, object_class

    def __len__(self):
        """Return the size of the dataset"""
        return self.label.shape[0]


class TestDataset(Dataset):
    """This dataset takes in a list of images"""

    def __init__(self, image_folder, shape=(256, 256)):
        """Initialize the dataset"""
        self.shape = shape
        self.image_folder = image_folder
        self.image_list = []
        
        # retrieve the image files
        if os.path.isfile(image_folder):
            self.image_list = [image_folder]
        else:
            self.image_list = glob.glob(
                os.path.join(image_folder, '**', '*'),
                recursive=True
            )
            self.image_list.sort()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __getitem__(self, index):
        """Get the corresponding image and filename

        # Arguments
            index [int]: the index of the file in the dataset

        # Returns
            [str]: the filename
            [FloatTensor]: image of shape (C x H x W)
        """
        filepath = self.image_list[index]

        # load test image the same way with train image
        image = cv2.imread(filepath)
        image = augmenter.augment_image(image)
        image = ia.imresize_single_image(image, self.shape)
        image = image / 255
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)
        image = torch.FloatTensor(image)

        return filepath, image

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.image_list)

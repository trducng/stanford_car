import glob
import os

import numpy as np
import numpy.random as random
import scipy.io

import imgaug as ia
import imgaug.augmenters as iaa


def split_train_test(original_folder):
    """Split the dataset into train/val/test"""
    label_file = scipy.io.loadmat(
        os.path.join(original_folder, 'devkit', 'cars_train_annos.mat'))
    train_annot = label_file['annotations'][0]
    train_annot = random.permutation(train_annot)

    n_files = train_annot.shape[0]
    n_train = int(0.9 * n_files)
    n_val = int(0.05 * n_files)

    train_files = list(train_annot[:n_train])
    val_files = list(train_annot[n_train:n_train + n_val])
    test_files = list(train_annot[n_train + n_val:])

    np.save(os.path.join(original_folder, 'train.npy'), train_files)
    np.save(os.path.join(original_folder, 'val.npy'), val_files)
    np.save(os.path.join(original_folder, 'test.npy'), test_files)


if __name__ == '__main__':
    split_train_test('data')

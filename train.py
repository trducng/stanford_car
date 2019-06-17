"""Perform training"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TrainDataset
from model import Model
from utils import Tracker, get_predictions, SuperConvergence



def main(output_dir, n_attentions, image_shape, batch_size,
         learning_rate, gpu):
    """Perform model training"""
    
    # initialize the dataset
    train_set = TrainDataset(phase='train', shape=image_shape)
    val_set = TrainDataset(phase='val', shape=image_shape)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                            num_workers=8, pin_memory=True)

    # initialize the model
    model = Model(n_classes=196, input_size=image_shape,
                  n_attentions=n_attentions, gpu=gpu)
    if gpu:
        model = model.cuda()

    # initialize related optimization methods
    criterion = nn.CrossEntropyLoss()
    criterion_attention = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    feature_center = torch.zeros(196, n_attentions * 2208)
    scheduler = SuperConvergence(optimizer, max_lr=learning_rate, stepsize=5000,
                                 better_as_larger=False, last_epoch=-1)
    if gpu:
        feature_center = feature_center.cuda()

    # initialize other hyperparameters
    crop_threshold = 0.5
    drop_threshold = 0.5
    focal_weight = 0.4

    # perform the training
    epoch = 0
    while True:
        print('Starting epoch {:03d}'.format(epoch))

        # statistic tracking
        train_loss_tracker = Tracker()
        train_accuracy_tracker = Tracker()

        model = model.train()
        for idx, (X, y) in enumerate(train_loader):
            if gpu:
                X = X.cuda()
                y = y.cuda()

            mini_batch = X.size(0)
            logits, feature_matrix, sampled_attentions = model(X)

            loss = (criterion(logits, y)
                    + criterion_attention(feature_matrix, feature_center[y]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            feature_center[y] = feature_center[y] + (focal_weight
                * (feature_matrix.detach() - feature_center[y]))

            preds, _ = get_predictions(logits.squeeze().cpu().data.numpy())
            preds = np.array(preds) == y.cpu().squeeze().data.numpy()
            accuracy = np.mean(preds)

            train_loss_tracker.step(loss.item() * mini_batch, mini_batch)
            train_accuracy_tracker.step(
                accuracy * mini_batch, mini_batch)
            
            # perform data cropping
            with torch.no_grad():
                crop_attentions = F.interpolate(
                    sampled_attentions.unsqueeze(1),
                    size=image_shape,
                    mode='bilinear',
                    align_corners=False)
                crop_attentions = crop_attentions > crop_threshold
                cropped_images = []
                for _idx in range(crop_attentions.size(0)):
                    positive_indices = torch.nonzero(crop_attentions[_idx])
                    x_min = torch.min(positive_indices[:, 2])
                    y_min = torch.min(positive_indices[:, 1])
                    x_max = torch.max(positive_indices[:, 2])
                    y_max = torch.max(positive_indices[:, 1])
                    cropped_image = F.interpolate(
                        crop_attentions[_idx, :, y_min:y_max+1, x_min:x_max+1]
                            .float().unsqueeze(0)
                        * X[_idx, :, y_min:y_max+1, x_min:x_max+1].unsqueeze(0),
                        size=image_shape,
                        mode='bilinear',
                        align_corners=False)
                    cropped_images.append(cropped_image)
                cropped_images = torch.cat(cropped_images, dim=0)

            logits, _, _ = model(cropped_images)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # perform attention dropping
            with torch.no_grad():
                drop_attentions = F.interpolate(
                    sampled_attentions.unsqueeze(1),
                    size=image_shape,
                    mode='bilinear',
                    align_corners=False
                )
                drop_attentions = (drop_attentions < drop_threshold).float()
                dropped_images = drop_attentions * X

            logits, _, _ = model(dropped_images)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stop = (epoch == 10)
            scheduler.step(epoch=None, metrics=train_loss_tracker.get_average(),
                           stop=stop)

            if idx % 100 == 0:
                _temp_lr = optimizer.param_groups[0]['lr']
                print('Batch {}, average loss {} - average accuracy {}, lr {}'
                    .format(idx, train_loss_tracker.get_average(),
                            train_accuracy_tracker.get_average(),
                            _temp_lr))

        # do validation pass
        val_loss_tracker = Tracker()
        val_accuracy_tracker = Tracker()

        model = model.eval()
        for X_val, y_val in val_loader:
            if gpu:
                X_val = X_val.cuda()
                y_val = y_val.cuda()

            mini_batch = X_val.size(0)

            with torch.no_grad():
                logits, _, _ = model(X_val)
                val_loss = criterion(logits, y_val)

                preds, _ = get_predictions(logits.squeeze().cpu().data.numpy())
                preds = np.array(preds) == y_val.cpu().squeeze().data.numpy()
                accuracy = np.mean(preds)

                val_loss_tracker.step(val_loss.item() * mini_batch, mini_batch)
                val_accuracy_tracker.step(
                    accuracy * mini_batch, mini_batch)

        state_dict = {
            'n_classes': 196,
            'input_size': image_shape,
            'n_attentions': n_attentions,
            'state_dict': model.state_dict()
        }
        torch.save(state_dict, os.path.join(output_dir, '{:03d}.ckpt'.format(epoch)))
        print('Validation - loss {}, accuracy {}'.format(
            val_loss_tracker.get_average(),
            val_accuracy_tracker.get_average()
        ))
        epoch += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir',
                        help='The output directory to store checkpoint')
    parser.add_argument('--n_attentions', default=32, type=int,
                        help='Number of attention channels')
    parser.add_argument('--image_size', nargs='+', default=(256, 256),
                        help='The image size, default "256 256"')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='The batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='Learing rate')
    parser.add_argument('--gpu', action='store_true',
                        help='Set to use GPU')

    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        n_attentions=args.n_attentions,
        image_shape=args.image_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gpu=args.gpu)
 

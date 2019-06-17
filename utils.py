"""Some utility functions"""
import json
import warnings

import numpy as np
import pandas as pd
import torch.optim as optim


with open('data/meta.json', 'r') as f_in:
    META = json.load(f_in)


def get_predictions(logits):
    """Get predictions from logits

    # Arguments
        logit [1D or 2D np array]: the logit, 1D is single batch

    # Returns
        [list of ints]: the predicted class index
        [list of strs]: the predicted class string
    """
    if logits.ndim == 1:
        logits = np.expand_dims(logits, axis=0)

    predicted_idx = list(np.argmax(logits, axis=1))
    predicted_str = [META.get(str(each), '0') for each in predicted_idx]

    return predicted_idx, predicted_str


class Tracker():
    """Track average value"""

    def __init__(self):
        """Initialize the object"""
        self.total = 0.0
        self.instances = 0

    def step(self, value, n_items=1):
        """Take a step

        # Arguments
            value [float}: the value at each step
            n_items [int]: the number of instances for that step
        """
        self.total += value
        self.instances += n_items

    def get_average(self):
        """Get the average

        # Returns
            [float]: the average
        """
        return self.total / self.instances


class SuperConvergence(optim.lr_scheduler._LRScheduler):
    r"""Super-convergence learning rate scheduler

        Implement based on
            Super-Convergence: Very Fast Training of Neural Networks Using
            Large Learning Rates (https://arxiv.org/abs/1708.07120)

    ```
                                                   max_lr
            /\          /\          /\
           /  \        /  \        /  \
          /    \      /    \      /    \
         /      \    /      \    /      \
        /        \  /        \  /        \
       /          \/          \/          \        base_lr
      |<---->| stepsize                    \
                                            \
                                             \
                                              \
                                               \
                                                \
                                                 \
                                                  \
                                           |<---->| stepsize
    ```

    According to the paper, the rule of thumb is to set `max_rl` two times the
    converging learning rate or to find using the `lr_finder`, and divide the
    `max_rl` by 3 or 4 to get `base_lr`.

    # Arguments
        optimizer [optim object]: the torch optimizer
        max_lr [float]: the maximum learning rate
        base_lr [float]: the base learning rate (if None == max_lr / 4)
        stepsize [int]: the number of training iterations to do half cycle
        omega [float]: after cyclical, reduce the learning rate by `omega` for
            every stepsize // 5 iterations
        patience [int]: if not None, automatically activate RL on plateau
        better_as_larger [bool]: the comparision scheme for RL on plateau
        save_model [func]: a function to save model (shouldn't take any
            argument)
        last_epoch [int]: the begin epoch (@NOTE: the epoch here looks more
            like iteration than epoch)
    """

    def __init__(self, optimizer, max_lr, base_lr=None, stepsize=10000,
                 omega=0.1, patience=None, better_as_larger=True,
                 save_model=None, last_epoch=-1):
        """Initialize the object"""

        # cyclical learning rate
        self.max_lr = max_lr
        self.base_lr = max_lr / 4 if base_lr is None else base_lr
        self.base_lrs = [self.base_lr] * len(optimizer.param_groups)
        for param_group in optimizer.param_groups:
            # we would want to start training when lr is smallest
            param_group['lr'] = self.base_lr
        self.lr_range = self.max_lr - self.base_lr
        if self.lr_range < 0:
            raise AttributeError('`base_lr` should be smaller than `max_lr`')
        self.stepsize = stepsize

        # after cyclical learning rate
        self.stop_intend = False
        self.stop = None
        self.omega = omega
        if omega >= 1 or omega <= 0:
            raise AttributeError('`omega` should be in range between 0 and 1')

        self.patience = patience
        if patience is not None and patience <= 10:
            warnings.warn('the `patience` number is not the number of epochs '
                          'but the number of training iterations, so patience '
                          '{} might not be what you intend'.format(patience))
        self.n_bad_periods = 0
        self.cooldown_counter = 0
        self.metrics = None
        self.best = None
        self.better = max if better_as_larger else min

        self.save_model = save_model

        super(SuperConvergence, self).__init__(optimizer, last_epoch)

    def add_save_model(self, save_model):
        """Add save model method

        # Arguments
            save_model [func]: a function to save model (shouldn't take any
                argument)
        """
        self.save_model = save_model

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if
            key not in ['optimizer', 'save_model']}

    def step(self, epoch=None, metrics=None, stop=False):
        """Increment the training iteration

        # Arguments
            metrics [float/int]: a metric number to determine if training has
                progress
            stop [bool]: indicate whether cyclical phase has done. This flag
                will be effective when it is set the first time
        """
        self.stop_intend = stop
        if (self.stop_intend and
            self.stop is None and
            self.last_epoch % self.stepsize == 0 and
            self.last_epoch // self.stepsize % 2 == 0):
            # mark the time begin stopping cyclical learning rate when the
            # the learning is the lowest
            self.stop = self.last_epoch
            self.base_lrs = [self.base_lr] * len(self.optimizer.param_groups)

        # only update the metrics when we stop CLR
        if self.stop:
            self.metrics = metrics

        super(SuperConvergence, self).step(epoch)

    def get_lr(self):
        """Calculate and return the learning rate

        # Returns
            [float]: the learning rate
        """
        # handle after CLR case
        if self.stop is not None:
            return self.after_clr()

        return self.clr()

    def clr(self):
        """Handle cyclical learning rate operation

        # Returns
            [list of floats]: list of learning rate to be consumed by optimizer
        """

        delta = (self.last_epoch % self.stepsize) / self.stepsize
        if self.last_epoch // self.stepsize % 2 == 1:
            delta = 1 - delta
        delta *= self.lr_range

        if (delta == 0 and self.last_epoch // self.stepsize % 2 == 0 and
            self.save_model is not None and self.last_epoch > 0):
            # delta == 0 means the learning rate at lowest point
            self.save_model()

        return [base_lr + delta for base_lr in self.base_lrs]

    def after_clr(self):
        """Handle learning rate operation after CLR"""

        # patience == None -> normal step-wise learning rate
        if self.patience is None:
            omega = (self.omega **
                    ((self.last_epoch - self.stop) // (self.stepsize / 10)))
            return [base_lr * omega for base_lr in self.base_lrs]

        # perform decay when plateau
        if self.is_better(self.metrics):
            self.best = self.metrics
            self.cooldown_counter = 0
            return [base_lr * self.omega ** self.n_bad_periods
                    for base_lr in self.base_lrs]

        self.cooldown_counter += 1
        if self.cooldown_counter >= self.patience:
            self.n_bad_periods += 1
            self.cooldown_counter = 0

            if self.n_bad_periods >= 10:
                if self.save_model is not None:
                    self.save_model()

        return [base_lr * self.omega ** self.n_bad_periods
                for base_lr in self.base_lrs]

    def is_better(self, metrics):
        """Determine whether the metrics is better than the best

        # Arguments
            metrics [float/int]: a metric number to determine if training has
                progress

        # Returns
            [bool]: whether the `metrics` is better than the best
        """
        if self.best is None:
            return True

        if abs(self.better(metrics, self.best) - metrics) < 1e-7:
            # additional condition to check if metrics == self.best
            if abs(self.better(metrics, self.best) - self.best) > 1e-7:
                return True

        return False


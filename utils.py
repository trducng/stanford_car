"""Some utility functions"""
import json

import numpy as np
import pandas as pd


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
    if logits.size == 1:
        logits = np.expand_dims(logits, axis=0)

    predicted_idx = list(np.argmax(logits, axis=1))
    predicted_str = [META[str(each)] for each in predicted_idx]

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


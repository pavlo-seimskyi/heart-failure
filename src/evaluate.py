import numpy as np
import pandas as pd
from copy import deepcopy


def train_valid_split(x, y, valid_start, valid_end):
    """Split into train/test sets, based on index location."""
    train_idx, valid_idx = _get_train_valid_idx(valid_start, valid_end, x.shape[0])
    x_train = x.copy().iloc[train_idx]
    y_train = y.copy().iloc[train_idx]
    x_valid = x.copy().iloc[valid_idx]
    y_valid = y.copy().iloc[valid_idx]
    return x_train, y_train, x_valid, y_valid


def generate_cv_splits(total_rows, cv_folds=10):
    """Generate a list of start & end idx for cross validation."""
    step = total_rows // cv_folds
    splits = []
    for split in range(0, cv_folds):
        start = step * split
        end = start + step
        splits.append([start, end])
    rows_left = total_rows // cv_folds != total_rows / cv_folds
    if rows_left:
        # extend the last validation set to last row
        splits[-1][-1] = total_rows
    return splits


def _get_train_valid_idx(valid_start_idx, valid_end_idx, total_rows):
    """Transform validation start and end indexes into a
    list of training and validation indexes."""
    valid_idx = np.arange(valid_start_idx, valid_end_idx)
    all_idx = np.arange(total_rows)
    train_mask = np.isin(all_idx, valid_idx, invert=True)
    train_idx = all_idx[train_mask]
    return train_idx, valid_idx

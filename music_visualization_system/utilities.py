""" Assortment of utility functions"""
import numpy as np


def average_filterbank(
    left_channel: np.ndarray, right_channel: np.ndarray
) -> np.ndarray:
    return ((left_channel / 2) + (right_channel / 2)).astype(np.uint32)

# -*- coding: utf-8 -*-

"""
Utility methods for training wavenet.
"""

# Basic import(s)
import numpy as np


def transform_params (params):
    """
    Transform initial parameter configuration to conform with as many
    constraints as possible, so as to avoid exploding gradients.

    Arguments:
        params: Initial, random parameter configuration, as array of size (N,).

    Returns
        Transformed `params`.
    """

    # Definitions
    N = len(params)

    # Utility functions
    # -----------------
    # 2-norm of vector `x`.
    norm_ = lambda x: np.sqrt(np.sum(np.square(x)))

    # Get high-pass filter coefficients `b` from low-pass filter coefficiets `a`.
    b_    = lambda a: np.flip(a,0) * np.power(-1., np.arange(N))

    # Constraint functions
    # --------------------
    # Add constant to low-pass filter coefficiets `a` so as to satisfy (C1).
    c1_ = lambda a: a + (1. - np.sum(a)) / float(len(a))

    # Divide low-pass filter coefficiets `a` by constant so as to satisfy (C2, m=0).
    c2_ = lambda a: a / norm_(a)

    # Add constant to high-pass filter coefficiets `b` so as to satisfy (C4).
    c4_ = lambda a: a - b_(np.repeat(- np.sum(b_(a)) / float(N), N))

    # Iteratively enforce each of the above conditions
    for _ in range(10):
        params = c1_(params)
        params = c2_(params)
        params = c4_(params)
        pass

    return params

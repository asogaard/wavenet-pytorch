# -*- coding: utf-8 -*-

"""
Loss function(s).
"""

# Basic import(s)
import numpy as np

# Pytorch import(s)
import torch
from torch.autograd import Variable

# General definitions
dtype = torch.FloatTensor


def gini (c):
    """
    Comput the Gini coefficient -- a measure of sparsity -- for a set of wavelet
    coefficients `c.`

    Arguments:
        c: Wavelet coefficients, as torch.autograd.Variable of shape (Nc,).

    Returns:
        Gini coefficient, as torch.autograd.Variable of shape (1,).
    """
    Nc = len(c)
    k = np.arange(Nc)
    k = Variable(torch.FloatTensor(2 * k - Nc - 1).type(dtype))

    _, idx = c.abs().sort()
    cs = c.abs().index_select(0,idx)
    return (k * cs).sum() / (Nc * cs.sum())

# -*- coding: utf-8 -*-

"""
Create distribution tensors for wavelet filter operations and -regularisation.
"""

# Basic import(s)
import numpy as np
from scipy.ndimage.interpolation import shift

# Pytorch import(s)
import torch
from torch.autograd import Variable


def dist_low_ (N, m):
    """
    Create distribution tensor `D`, such that matrix product of filter
    coefficients `a` and `D` yield the low-pass filter matrix, `a * D = L`.

    Arguments:
        N: Number of filter coefficients.
        m: Dimension (radix-2) matrix, column-wise.

    Returns:
        Low-pass filter distribution tensor, as np.array.
    """
    L = np.zeros((N, 2**(m-1), 2**m))
    for i in range(N):
        for j in range(L.shape[1]):
            L[i,j,(2*j - i + N/2) % L.shape[2]] = 1
            pass
        pass
    return np.transpose(L, [1,0,2])


def dist_high_ (N, m):
    """
    Create distribution tensor `D`, such that matrix product of filter
    coefficients `a` and `D` yield the high-pass filter matrix, `a * D = H`.

    Arguments:
        N: Number of filter coefficients.
        m: Dimension (radix-2) matrix, column-wise.

    Returns:
        High-pass filter distribution tensor, as np.array.
    """
    H = np.zeros((N, 2**(m-1), 2**m))
    for i in range(N):
        for j in range(H.shape[1]):
            H[i,j,(2*j + i - N/2 + 1) % H.shape[2]] = (-1) ** i
            pass
        pass
    return np.transpose(H, [1,0,2])


def dist_reg_ (N):
    """
    Create distribution tensor `D` for regularisation terms (R3-5), such that
    each term is a row in the resulting matrix `a * D * a`, e.g.

    Arguments:
        N: Number of filter coefficients.

    Returns:
        Regularisation distribution tensor, as torch.autograd.Variable.
    """

    D = np.zeros((N, N - 1, N))
    for idx in range(N):
        D[idx,:,idx] = 1
        for row, m in enumerate(range(-N//2 + 1, N//2)):
            D[idx,row,:] = shift(D[idx,row,:], 2*m)
            pass
        pass
    D = np.transpose(np.round(D).astype(np.int), [1,0,2])
    return Variable(torch.FloatTensor(D), requires_grad=False)


def delta_ (N):
    """
    Vector Kronecker-delta, with `1` on central entry.

    Arguments:
        N: Number of vector entries, required to be odd.

    Return:
        Vector Kronecker-delta, as torch.autograd.Variable.
    """
    # Check(s)
    assert N % 2 == 1, "delta_: Number of entries ({}) has to be odd.".format(N)

    # Kronecker delta
    delta = Variable(torch.zeros(N), requires_grad=False)
    delta[N//2] = 1
    return delta

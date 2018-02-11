# -*- coding: utf-8 -*-

"""
Pytorch modules for constructing wavenet.
"""

# Basic import(s)
from collections import namedtuple
import numpy as np

# Pytorch import(s)
import torch
from torch.autograd import Variable

# Project import(s)
from wavenet.ops import *

# General definitions
dtype = torch.FloatTensor


class Filter (torch.nn.Module):
    def __init__ (self, params, input_dim):
        """General wavelet filter from scale `input_dim` to `input_dim - 1`."""
        super(Filter, self).__init__()
        self.params = params
        self.dist   = None  # Distribution tensor; to be set by dervied class.
        return

    def W (self):
        return self.params.matmul(self.dist)

    def forward (self, x):
        return self.W().matmul(x)

    def backward (self, c):
        """
        Just using numpy arrays -- we don't care about gradients here anyway.
        """
        return np.matmul(c, self.W().data.numpy())
    pass


class Lowpass (Filter):
    def __init__ (self, params, input_dim):
        """Low-pass filter from scale `input_dim` to `input_dim - 1`."""
        super(Lowpass, self).__init__(params, input_dim)
        self.dist = Variable(torch.FloatTensor( dist_low_(len(params), input_dim) ).type(dtype), requires_grad=False)
        return
    pass


class Highpass (Filter):
    def __init__ (self, params, input_dim):
        """High-pass filter from scale `input_dim` to `input_dim - 1`."""
        super(Highpass, self).__init__(params, input_dim)
        self.dist = Variable(torch.FloatTensor( dist_high_(len(params), input_dim) ).type(dtype), requires_grad=False)
        return
    pass


class Regularisation (torch.nn.Module):
    def __init__ (self, params):
        """Wavelet-regularisation for filter coefficient `params`."""
        super(Regularisation, self).__init__()
        self.a = params
        self.b = None
        self.N = len(self.a.data)
        self.D = dist_reg_(self.N)       # Distribution tensor
        self.delta = delta_(self.N - 1)  # Kronecker-delta
        return

    def forward (self):
        # Compute high-pass filter coefficients
        idx = Variable(torch.LongTensor(list(reversed(range(self.N)))))
        self.b = self.a.index_select(0, idx)  # Reverse `a`
        sign = Variable(torch.FloatTensor(np.power(-1, np.arange(self.N))), requires_grad=True)
        self.b = self.b * sign  # Flip sign of every other element in `b`

        # (R1)
        R1 = torch.pow(self.a.sum() - np.sqrt(2.), 2.)

        # (R2)
        R2 = torch.pow(self.a.matmul(self.D).matmul(self.a) - self.delta, 2.).sum()

        # (R3)
        R3 = torch.pow(self.b.matmul(self.D).matmul(self.b) - self.delta, 2.).sum()

        # (R4)
        R4 = torch.pow(self.b.sum(), 2.)

        # (R5)
        R5 = torch.pow(self.a.matmul(self.D).matmul(self.b), 2.).sum()

        return R1 + R2 + R3 + R4 + R5
    pass


# Struct containing one high- and one low-pass filter for each scale `m`.
Layer = namedtuple("Layer", "high low")

class Wavenet1D_ (torch.nn.Module):
    def __init__ (self, params, input_size):
        """
        ...
        """
        super(Wavenet1D_, self).__init__()
        print "Constructing 1D Wavenet."

        # Check(s)
        assert np.log2(input_size) % 1 == 0, \
        "Wavenet: `input_size` must be radix 2. Recieved {}.".format(input_size)

        # Member variable(s)
        self.params      = params
        self.input_shape = (input_size,)
        self.layers      = list()

        # Create layers
        for m in reversed(range(int(np.log2(input_size)))):
            self.layers.append(Layer(high=Highpass(self.params, m + 1),
                                     low =Lowpass (self.params, m + 1)))
            pass
        return

    def forward (self, x):
        output = list()
        for ilayer, layer in enumerate(self.layers):
            output.insert(0, layer.high.forward(x))  # High-pass
            x = layer.low.forward(x)                 # Low-pass
            pass
        return torch.cat([x] + output)

    def backward (self, c):
        """
        Just using numpy arrays -- we don't care about gradients here anyway.
        """
        try:
            c = c.data.numpy()
        except:
            # Already numpy array
            pass
        output = c[:1]
        N = len(c)
        for m, layer in enumerate(reversed(self.layers)):
            output = layer.high.backward(c[2**m:2**(m+1)]) + layer.low.backward(output)
            pass
        return output
    pass


class Wavenet2D_ (torch.nn.Module):
    def __init__ (self, params, input_shape):
        """
        ...
        """
        super(Wavenet2D_, self).__init__()
        print "Constructing 2D Wavenet."

        # Check(s)
        assert len(input_shape) == 2

        # Member variable(s)
        self.params      = params
        self.input_shape = input_shape
        self.wavenets = list()

        # Create 1D-wavenets for each row
        for input_size in input_shape:
            self.wavenets.append(Wavenet1D_(params, input_size))
            pass
        return

    def forward (self, x):
        c = x
        for i in range(c.shape[0]):
            c[i,:] = self.wavenets[0].forward(c[i,:])
            pass
        for j in range(c.shape[1]):
            c[:,j] = self.wavenets[1].forward(c[:,j])
            pass
        return c

    def backward (self, c):
        """
        Just using numpy arrays -- we don't care about gradients here anyway.
        """
        x = np.array(c)
        for j in range(x.shape[1]):
            x[:,j] = self.wavenets[1].backward(x[:,j])
            pass
        for i in range(x.shape[0]):
            x[i,:] = self.wavenets[0].backward(x[i,:])
            pass
        return x
    pass


def Wavenet (params, input_shape):
    """
    General factory method for getting the correct Wavenet class.
    """
    if isinstance(input_shape, int):
        return Wavenet1D_(params, input_shape)
    elif len(input_shape) == 1:
        return Wavenet1D_(params, input_shape[0])
    elif len(input_shape) == 2:
        return Wavenet2D_(params, input_shape)
    else:
        assert False, "Wavenet: input_shape {} not supported.".format(input_shape)
    return

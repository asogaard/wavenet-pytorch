#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for testing wavelet bases learned using a neural network approach.
"""

# Basic import(s)
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Pytorch import(s)
import torch
from torch.autograd import Variable

# Project import(s)
from wavenet.generators import *

# Type definitions
dtype = torch.FloatTensor

# General definitions
seed = 22


# Main function definition.
def main ():

    # Get paths for wavenets to test
    paths = sys.argv[1:]
    if not paths:
        print "Please specify at least one wavenet to test, as:"
        print "  $ python {} <path> [...]".format(sys.argv[0])
        return

    # Select only paths with correct suffix
    paths_filtered = filter(lambda path: path.endswith('.pt'), paths)
    if len(paths_filtered) != len(paths):
        print "Only selecting paths ending in `.pt`."
        pass

    paths = paths_filtered

    if not paths:
        print "No paths ending in `.pt` was found."
        return

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load wavenet model(s) from file
    wavenets = map(torch.load, paths)

    # Study wavenet(s)
    for path, wavenet in zip(paths, wavenets):

        print "== {}:".format(path)
        tag = '__'.join(path.split('/')[-1].split('.')[0].split('__')[1:])
        params      = wavenet.params
        input_shape = wavenet.input_shape



        for idx in range(input_shape[0]):
            # Compute basis function
            c = np.zeros(input_shape)
            c[idx] = 1
            basis = wavenet.backward(c)

            # Plot basis functions
            plt.clf()
            plt.plot(basis, drawstyle='steps-mid')
            plt.ylim(-1,1)
            plt.savefig('figures/basis__{}__idx{}.pdf'.format(tag, idx))
            pass
        # ...
        pass


    """
    # Loss
    # @TODO: Move to `test.py` script.
    plt.clf()
    for key in losses[0].keys():
        if key == 'compactness': continue
        loss = [loss_[key] for loss_ in losses]
        plt.semilogy(loss, label=key)
        pass
    plt.legend()
    plt.savefig('tmp_losses.pdf')

    # Study wavelet basis
    # @TODO: Move to `test.py` script.
    for i in range(input_shape[0]):  # @TEMP: For 1D only
        c = Variable(torch.zeros(input_shape).type(dtype), requires_grad=False)
        c.data[i] = 1
        basis = w.backward(c)

        plt.clf()
        plt.plot(np.zeros(len(basis)), color='gray', linewidth=1)
        plt.plot(basis, drawstyle='steps-mid')
        plt.ylim(-1,1)

        plt.savefig('tmp_basis{}.pdf'.format(i))
        pass

        """

    # ...

    return


# Main function call.
if __name__ == '__main__':
    main()
    pass

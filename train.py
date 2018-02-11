#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for learning opitimal wavelet bases using a neural network approach.
"""

# Basic import(s)
import time
import json
import numpy as np

# Pytorch import(s)
import torch
from torch.autograd import Variable

# Project import(s)
from wavenet.loss import *
from wavenet.utils import *
from wavenet.modules import *
from wavenet.generators import *

# Type definitions
dtype = torch.FloatTensor

# General definitions
seed = 22
num_params = 16
batch_size = 1
input_shape = (64,)
generator_name = 'spikes'
gen_opts = dict(input_shape=input_shape)

# Main function definition.
def main ():

    # Generator
    if   generator_name == 'sine':
        generator = generate_sine
    elif generator_name == 'spikes':
        generator = generate_spikes
    else:
        raise "Generator {} not supported.".format(generator_name)

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Learnable, universal filter coefficients
    params = np.random.randn(num_params)
    params = transform_params(params)

    # Pytorch variables
    params  = Variable(torch.FloatTensor(params), requires_grad=True)
    indices = Variable(torch.arange(num_params).type(dtype) / np.float(num_params - 1), requires_grad=False)
    x       = Variable(torch.randn(input_shape).type(dtype), requires_grad=False)

    # Wavenet instance
    w = Wavenet(params, input_shape)

    # Optimiser
    optimiser  = torch.optim.Adam([params], lr=1e-02)
    lambda_reg = 1.0E+02
    num_steps  = 5000

    # Regularisation
    reg = Regularisation(params)

    # Training loop
    print "Initial parameters:", params.data.numpy()
    loss_dict = lambda : {'sparsity': 0, 'regularisation': 0, 'combined': 0, 'compactness': 0}

    losses = {'sparsity': [0], 'regularisation': [0], 'combined': [0], 'compactness': [0]}

    print "=" * 80
    print "START RUNNING"
    print "-" * 80
    start = time.time()
    for step, x_ in enumerate(generator(**gen_opts)):

        # Stop condition
        if step >= num_steps:
            break

        # Set input
        x.data = torch.from_numpy(x_).type(dtype)

        # Get wavelet coefficients
        c = w.forward(x)

        # Sparsity loss
        sparsity = 1. - gini(c)

        # Regularisation loss
        regularisation = reg.forward()

        # Compactness  loss
        compactness = torch.sum(torch.dot(indices - 0.5, params.abs() / params.abs().sum()))

        # Combined loss
        combined = sparsity + lambda_reg * (regularisation) + compactness

        # Perform backpropagation
        combined.backward()

        # Parameter update
        if step % batch_size == 0:
            optimiser.step()
            optimiser.zero_grad()
            pass


        # Non-essential stuff below
        # -------------------------------------------------------------------------

        # Log
        if step % 1000 == 0:
            print "Step {}/{}".format(step, num_steps)
            pass

        # Logging loss history
        losses['sparsity'][-1]       += np.float(sparsity)
        losses['regularisation'][-1] += np.float(regularisation)
        losses['compactness'][-1]    += np.float(compactness)
        losses['combined'][-1]       += np.float(combined)

        if step % batch_size ==  0:
            for key in losses:
                losses[key][-1] /= float(batch_size)
                losses[key].append(0.)
                pass
            pass

        # Draw model diagram
        if step == 0:
            from torchviz import make_dot
            dot = make_dot(sparsity, params={'params': params, 'input': x})
            dot.format = 'pdf'
            dot.render('output/model')
            pass

        pass

    end = time.time()
    print "-" * 80
    print "Took {:.1f} sec.".format(end - start)
    print "=" * 80

    # Clean-up
    for key in losses:
        losses[key].pop(-1)
        pass
    print "Final parameters:", params.data.numpy()

    # Save to file
    tag = '{}__N{}__{}'.format('x'.join(map(str, input_shape)), num_params, generator_name)

    # -- Model
    torch.save(w, 'output/model__{}.pt'.format(tag))

    # -- Loss
    with open('output/loss__{}.json'.format(tag), 'w') as f:
        json.dump(losses, f)
        pass

    return


# Main function call.
if __name__ == '__main__':
    main()
    pass

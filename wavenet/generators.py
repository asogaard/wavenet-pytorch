# -*- coding: utf-8 -*-

"""
Data generators for training wavenet.
"""

# Basic import(s)
import numpy as np
rng = np.random.RandomState(21)  # For reproducibility


def validate_shape_ (shape):
    """
    Ensure that the `shape` argument is, or can be converted to, a tuple of ints.

    Arguments:
        shape: int, or iterable of ints, specifying the shape of inputs.

    Returns:
        Validated shape, as tuple of ints.

    Raises:
        AssertionError: In case `shape` argument is not valid.
    """

    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, list):
        shape = tuple(shape)
        pass
    assert isinstance(shape, tuple)
    assert all(map(lambda d: isinstance(d, int), shape))
    return shape


def generate_spikes (input_shape, threshold=0.8):
    """
    Generate toy spike-type samples.

    Arguments:
        shape: int, or iterable of ints, specifying shape of samples to be
            generated.
        threshold: Fraction of bins containing spikes.

    Yields:
        Generated samples, as numpy arrays of shape `shape`.

    Raises:
        AssertionError: In case `shape` argument is not valid.
    """

    # Check(s)
    shape = validate_shape_(input_shape)

    # Loop
    while True:
        y = np.zeros(*shape)
        while np.sum(y) == 0:
            y = (rng.rand(*shape) > threshold).astype(np.float)
            pass
        yield y


def generate_sine (input_shape):
    """
    Generate toy sine-type samples.

    Arguments:
        shape: int, or iterable of ints, specifying shape of samples to be
            generated.

    Yields:
        Generated samples, as numpy arrays of shape `shape`.

    Raises:
        AssertionError: In case `shape` argument is not valid.
    """

    # Check(s)
    shape = validate_shape_(input_shape)

    # Loop
    mesh = np.meshgrid(*map(np.arange, shape))

    while True:
        y = np.ones(*shape).astype(np.float)
        for dim, x in enumerate(mesh):
            x = x.astype(np.float) / np.float(len(x) - 1) * 2. * np.pi  # x in [0,2π]
            y *= rng.rand() * np.sin(x  * (4. + rng.randn() * 2) + rng.rand())
            pass
        y /= np.sum(np.abs(y))
        yield y

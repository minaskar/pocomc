import numpy as np


def assert_array_2d(x: np.ndarray):
    if len(x.shape) != 2:
        raise ValueError(f"Input should have 2 dimensions, but got {len(x.shape)}")


def assert_array_1d(x: np.ndarray):
    if len(x.shape) != 1:
        raise ValueError(f"Input should have 1 dimension, but got {len(x.shape)}")


def assert_equal_type(x, y):
    if type(x) != type(y):
        raise ValueError(f"Expected inputs to have equal types, but got {type(x)} and {type(y)}")


def assert_arrays_equal_shape(x: np.ndarray,
                              y: np.ndarray):
    if x.shape != y.shape:
        raise ValueError(f"Inputs should have equal shape, but got {x.shape} and {y.shape}")


def assert_array_within_interval(x: np.ndarray,
                                 left: np.ndarray,
                                 right: np.ndarray,
                                 left_open: bool = False,
                                 right_open: bool = False):
    left = left.copy()
    left[np.isnan(left)] = -np.inf

    right = right.copy()
    right[np.isnan(right)] = np.inf

    if left_open and right_open:
        condition = (left < x) & (x < right)
        interval_string = f'({left}, {right})'
    elif left_open and not right_open:
        condition = (left < x) & (x <= right)
        interval_string = f'({left}, {right}]'
    elif not left_open and right_open:
        condition = (left <= x) & (x < right)
        interval_string = f'[{left}, {right})'
    else:
        condition = (left <= x) & (x <= right)
        interval_string = f'[{left}, {right}]'

    if not np.all(condition):
        x_min = np.min(x)
        x_max = np.max(x)
        raise ValueError(f"Expected input to be within interval {interval_string}, "
                         f"but got minimum = {x_min} and maximum = {x_max}")


def assert_array_float(x: np.ndarray):
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError(f"Expected input to have dtype float, but got {x.dtype}")

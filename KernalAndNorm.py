"""
Different functions are provided to calculate the difference between two patterns.
Such as, L2, L1 distance.

Methods to normalize the pattern is also included.
"""

import numpy as np
from numba import jit


@jit
def squared_l2_norm(pattern):
    """
    Calculate the squared L2 norm of the pattern.

    :param pattern: a numpy array, arbitrary shape and dimension
    :return: np.linalg.norm(pattern)
    """
    return np.linalg.norm(pattern)


@jit
def squared_l2_norm_batch(pattern_stack):
    """
    Calculate the l2 norm of a stack of patterns.

    :param pattern_stack: a stack of patterns of the shape (pattern_num, d1,d2,..,,dn)
    :return: a numpy array of shape (pattern_num). The value of each pattern is the squared l2 norm of that pattern.
    """

    return np.linalg.norm(pattern_stack, axis=0)


@jit
def inner_product(pattern_one, pattern_two):
    """
    Calculate the inner product of the two patterns as a vecter.

    :param pattern_one: a numpy array
    :param pattern_two: a numpy array with the same shape as pattern_one
    :return: np.sum(np.multiply(pattern_one, pattern_two))
    """

    return np.sum(np.multiply(pattern_one, pattern_two))


@jit
def inner_product_batch(pattern_stack_one, pattern_num_one, pattern_stack_two, pattern_num_two):
    """
    Calculate the inner product pair of each pattern in batch one and batch two.
    Notice that the pattern_stack_one variable represent the pattern along the zero dimension while the
    pattern_stack_two variable represent patterns along dimension one in the final distance matrix.

    :param pattern_stack_one: numpy array, (number of patterns, shape of each pattern)
    :param pattern_num_one: number of patterns in the first stack.
    :param pattern_stack_two: numpy array, (number of patterns, shape of each pattern)
    :param pattern_num_two: number of patterns in the second stack.
    :return: a numpy array in the shape of (pattern_num_one, pattern_num_two)
    """

    """
    Notice that the two stacks can be different. So we can not deduce the lower triangular pattern from the 
    other half.
    """
    holder = np.zeros((pattern_num_one, pattern_num_two))
    for l in range(pattern_num_one):
        for m in range(pattern_num_two):
            holder[l, m] = np.sum(np.multiply(pattern_stack_one[l], pattern_stack_two[m]))

    return holder

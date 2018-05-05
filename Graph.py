"""
Different functions are provided to calculate the difference between two patterns.
Such as, L2, L1 distance.

Methods to normalize the pattern is also included.
"""

import numpy as np
from numba import jit, int64, float64


@jit(nopython=True, parallel=True)
def l2_norm(pattern):
    """
    Calculate the squared L2 norm of the pattern.

    :param pattern: a numpy array, arbitrary shape and dimension
    :return: np.linalg.norm(pattern)
    """
    return np.linalg.norm(pattern)


@jit(nopython=True, parallel=True)
def l2_norm_batch(pattern_stack):
    """
    Calculate the l2 norm of a stack of patterns.

    :param pattern_stack: a stack of patterns of the shape (pattern_num, d1,d2,..,,dn)
    :return: a numpy array of shape (pattern_num). The value of each pattern is the squared l2 norm of that pattern.
    """

    return np.linalg.norm(pattern_stack, axis=0)


@jit(nopython=True, parallel=True)
def inner_product(pattern_one, pattern_two):
    """
    Calculate the inner product of the two patterns as a vecter.

    :param pattern_one: a numpy array
    :param pattern_two: a numpy array with the same shape as pattern_one
    :return: np.sum(np.multiply(pattern_one, pattern_two))
    """

    return np.sum(np.multiply(pattern_one, pattern_two))


@jit(nopython=True, parallel=True)
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


@jit(nopython=True, parallel=True)
def l2_square_from_inner_product(matrix):
    """
    This function extract the squared l2 norm from the inner product matrix.

    :param matrix: inner product matrix
    :return: np.diagnal(matrix)
    """
    return np.diag(matrix)


@jit(float64[:, :](float64[:, :], int64), nopython=True, parallel=True)
def inner_product_to_L2_square(matrix, length):
    """
    Turns the inner product matrix into the |pattern1 - pattern2|_{2}^2 matrix. The matrix has to be a square matrix

    :param matrix: inner product matrix. This has to be a square matrix
    :param length: The length of the matrix along each edge.
    :return: |pattern1 - pattern2|_{2}^2 matrix
    """
    squared_norm = l2_square_from_inner_product(matrix)

    return np.reshape(squared_norm, [length, 1]) + np.reshape(squared_norm, [1, length]) - matrix


@jit(float64[:, :](float64[:, :], int64), nopython=True, parallel=True)
def inner_product_to_normalized_L2_square(matrix, length):
    """
    Turns the inner product matrix into the |pattern1 - pattern2|_{2}^2 matrix. Here the pattern is normalized.
    Therefore, it can be reformulated into

    |pattern1 - pattern2|_{2}^2 = 2 - 2 <pattern1, pattern2>/ |pattern1||pattern2|

    :param matrix:inner product matrix. This has to be a square matrix.
    :param length: The length of the matrix along each edge.
    :return: The normalized squared distance matrix
    """
    norm = np.divide(1, np.sqrt(l2_square_from_inner_product(matrix)))

    normalized_inner_product = np.multiply(np.multiply(np.reshape(norm, [length, 1]), matrix),
                                           np.reshape(norm, [1, length]))
    return 2 - 2 * normalized_inner_product


@jit(float64[:, :](float64[:, :], float64), nopython=True, parallel=True)
def gaussian_dense(matrix, two_sigma_square):
    """
    Apply np.exp( - matrix/two_sigma_square) element-wise.

    :param matrix: Distance matrix.
    :param two_sigma_square: 2\sigma^2
    :return: np.exp( - matrix/two_sigma_square)
    """

    return np.exp(- matrix / two_sigma_square)


@jit(float64[:, :](float64[:, :], int64, int64, int64), nopython=True, parallel=True)
def nn_strip_dense_num(matrix, num_line, length, num_to_keep):
    """
    Keep the hightnest several values along each line in this matrix strip.

    :param matrix: The matrix strip to inspect.
    :param num_line: The number of lines in this strip
    :param length: The length of each line
    :param num_to_keep: The number of values to keep along each line
    :return: a dense matrix strip, each line of which only contains several non-zero values.
    """

    pass

@jit(float64[:, :](float64[:, :], int64, int64, int64), nopython=True, parallel=True)
def nn_strip_dense_threshold(matrix, num_line, length, threshold):
    """
    Keep the hightnest several values along each line in this matrix strip.

    :param matrix: The matrix strip to inspect.
    :param num_line: The number of lines in this strip
    :param length: The length of each line
    :param threshold: The threshold below which the values will be set to zeros.
    :return: a dense matrix strip, each line of which only contains several non-zero values.
    """

    pass

@jit(float64[:, :](float64[:, :], int64, int64, int64), nopython=True, parallel=True)
def nn_strip_sparse_num(matrix, num_line, length, num_to_keep):
    """
    Keep the hightnest several values along each line in this matrix strip.

    :param matrix: The matrix strip to inspect.
    :param num_line: The number of lines in this strip
    :param length: The length of each line
    :param num_to_keep: The number of values to keep along each line
    :return: a sparse matrix strip, each line of which only contains several non-zero values.
    """

    pass

@jit(float64[:, :](float64[:, :], int64, int64, int64), nopython=True, parallel=True)
def nn_strip_sparse_threshold(matrix, num_line, length, threshold):
    """
    Keep the hightnest several values along each line in this matrix strip.

    :param matrix: The matrix strip to inspect.
    :param num_line: The number of lines in this strip
    :param length: The length of each line
    :param threshold: The threshold below which the values will be set to zeros.
    :return: a sparse matrix strip, each line of which only contains several non-zero values.
    """

    pass

def laplacian():

    pass

def normalized_laplacian():

    pass

def symmetrized_laplacian():

    pass

def symmetrized_normalized_laplacian():

    pass





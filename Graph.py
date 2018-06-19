"""
Different functions are provided to calculate the difference between two patterns.
Such as, L2, L1 distance.

Methods to normalize the pattern is also included.
"""

import numpy as np
from numba import jit, int64, float64
from scipy import sparse
from scipy.sparse import linalg


@jit
def l2_norm(pattern):
    """
    Calculate the squared L2 norm of the pattern.

    :param pattern: a numpy array, arbitrary shape and dimension
    :return: np.linalg.norm(pattern)
    """
    return np.linalg.norm(pattern)


@jit
def l2_norm_batch(pattern_stack):
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


@jit(nopython=True, parallel=True)
def l2_square_from_inner_product(matrix):
    """
    This function extract the squared l2 norm from the inner product matrix.

    :param matrix: inner product matrix
    :return: np.diagnal(matrix)
    """
    return np.diag(matrix)


"""
I would like to write this in a jit and nophthon way, but it turns out to be difficult. Therefore, I feel that since 
this function will possibly only run once, and just use the regular python function is okey.
"""


def inner_product_to_L2_square(matrix):
    """
    Turns the inner product matrix into the |pattern1 - pattern2|_{2}^2 matrix. The matrix has to be a square matrix

    :param matrix: inner product matrix. This has to be a square matrix
    :return: |pattern1 - pattern2|_{2}^2 matrix
    """

    length = matrix.shape[0]
    squared_norm = np.reshape(np.diag(matrix), (length, 1))

    return squared_norm + np.transpose(squared_norm) - 2 * matrix


def inner_product_to_normalized_L2_square(matrix):
    """
    Turns the inner product matrix into the |pattern1 - pattern2|_{2}^2 matrix. Here the pattern is normalized.
    Therefore, it can be reformulated into

    |pattern1 - pattern2|_{2}^2 = 2 - 2 <pattern1, pattern2>/ |pattern1||pattern2|

    :param matrix:inner product matrix. This has to be a square matrix.
    :param length: The length of the matrix along each edge.
    :return: The normalized squared distance matrix
    """

    length = matrix.shape[0]
    norm = np.divide(1, np.sqrt(l2_square_from_inner_product(matrix)))

    normalized_inner_product = np.multiply(np.multiply(np.reshape(norm, [length, 1]), matrix),
                                           np.reshape(norm, [1, length]))
    return 2 - 2 * normalized_inner_product


@jit(float64[:, :](float64[:, :], float64))
def gaussian_dense(matrix, two_sigma_square):
    """
    Apply np.exp( - matrix/two_sigma_square) element-wise.

    :param matrix: Distance matrix.
    :param two_sigma_square: 2\sigma^2
    :return: np.exp( - matrix/two_sigma_square)
    """

    return np.exp(- matrix / two_sigma_square)


@jit(float64[:, :](float64[:, :], int64))
def nearest_points_indexes_without_self(matrix, num_to_keep):
    """
    Each row of the matrix, let's say the jth row, represents the distance between the other
    data point from the jth point. This function returns the indexes for the points with the
    smallest distances with respect to each point represented by that specified row.

    By row, I mean the 0th dimension. Also notice that this function does not include the
    target particle, i.e. the diagonal element along the matrix is set to zero.

    :param matrix: The matrix strip to inspect.
    :param num_to_keep: The number of values to keep along each line
    :return: a dense matrix strip, each line of which only contains several non-zero values.
    """

    # Set the diagonal to 0
    np.fill_diagonal(matrix, 0)
    # Get the position for the resulted values
    sort_arg = np.argsort(matrix, axis=1)

    return sort_arg[:, : num_to_keep]


@jit(float64[:, :](float64[:, :], int64))
def nearest_points_indexes_with_self(matrix, num_to_keep):
    """
    Each row of the matrix, let's say the jth row, represents the distance between the other
    data point from the jth point. This function returns the indexes for the points with the
    smallest distances with respect to each point represented by that specified row.

    By row, I mean the 0th dimension. Also notice that this function includes the
    target particle, i.e. the diagonal element along the matrix is set to 1.

    :param matrix: The matrix strip to inspect.
    :param num_to_keep: The number of values to keep along each line
    :return: a dense matrix strip, each line of which only contains several non-zero values.
    """

    # Set the diagonal to 1
    np.fill_diagonal(matrix, 1)
    # Get the position for the resulted values
    sort_arg = np.argsort(matrix, axis=1)

    return sort_arg[:, : num_to_keep]


@jit(float64[:, :](float64[:, :], int64))
def nearest_points_values_without_self(matrix, num_to_keep):
    """
    Each row of the matrix, let's say the jth row, represents the distance between the other
    data point from the jth point. This function returns the indexes for the points with the
    smallest distances with respect to each point represented by that specified row.

    By row, I mean the 0th dimension. Also notice that this function does not include the
    target particle, i.e. the diagonal element along the matrix is set to zero.

    :param matrix: The matrix strip to inspect.
    :param num_to_keep: The number of values to keep along each line
    :return: a dense matrix strip, each line of which only contains several non-zero values.
    """

    # Set the diagonal to 0
    np.fill_diagonal(matrix, 0)
    # Get the position for the resulted values
    sort = np.sort(matrix, axis=1)

    return sort[:, : num_to_keep]


@jit(float64[:, :](float64[:, :], int64))
def nearest_points_values_with_self(matrix, num_to_keep):
    """
    Each row of the matrix, let's say the jth row, represents the distance between the other
    data point from the jth point. This function returns the indexes for the points with the
    smallest distances with respect to each point represented by that specified row.

    By row, I mean the 0th dimension. Also notice that this function includes the
    target particle, i.e. the diagonal element along the matrix is set to 1.

    :param matrix: The matrix strip to inspect.
    :param num_to_keep: The number of values to keep along each line
    :return: a dense matrix strip, each line of which only contains several non-zero values.
    """

    # Set the diagonal to 1
    np.fill_diagonal(matrix, 1)
    # Get the position for the resulted values
    sort = np.sort(matrix, axis=1)

    return sort[:, : num_to_keep]


@jit(float64[:](float64[:, :]))
def degree_batch(matrix):
    """
    :param matrix: a numpy array to sum up along the 1st axis
    :return: np.sum(matrix, axis=1)
    """
    return np.sum(matrix, axis=1)


@jit(float64[:, :](float64[:]))
def degree_vector_to_matrix(vector):
    """
    Convert the degree vector to degree function.
    """
    return np.diag(vector)


@jit(float64[:, :](float64[:], float64[:, :]))
def laplacian(degree_vector, weight_matrix):
    """
    Construct the fundamental Laplacian matrix.

    :param degree_vector: The degree of each data point. It's a vector.
    :param weight_matrix: The weight matrix of data points.
    :return: np.diag(degree_vector) - weight_matrix
    """
    return np.diag(degree_vector) - weight_matrix


@jit(float64[:, :](float64[:], float64[:, :], int64))
def normalized_laplacian(degree_vector, weight_matrix, length):
    """
    Construct the normalized laplacian matrix.

    :param degree_vector: The vector contains the degree of each data point.
    :param weight_matrix: The weight matrix
    :param length: The number of points in this data set.
    :return: np.eye(length) - vector^(-1) * weight_matrix
    """
    holders = np.zeros((length, 1))
    holders[:, 0] = 1 / degree_vector

    return np.eye(length) - holders * weight_matrix


@jit(float64[:, :](float64[:], float64[:, :], int64))
def symmetrized_normalized_laplacian(degree_vector, weight_matrix, length):
    """
    Construct the normalized laplacian matrix.

    :param degree_vector: The vector contains the degree of each data point.
    :param weight_matrix: The weight matrix
    :param length: The number of points in this data set.
    :return: np.eye(length) - vector^(-1/2) * weight_matrix *vector^(-1/2).T
    """
    holders = np.zeros((length, 1))
    holders[:, 0] = np.sqrt(1 / degree_vector)

    return np.eye(length) - holders * weight_matrix * holders.T


def solve_for_eigenvectors(matrix, num, mode="general"):
    """
    Solve for several eigenvectors.

    :param matrix: This should be a sparse square matrix object.
    :param num: The number of eigenvectors to solve
    :param mode: If the matrix is symmetric, then set mode = symmetric. Otherwise, set mode = general.
    :return: eigenvlues, eigenvectors
    """

    # Construct a sparse matrix
    if mode == "general":
        return linalg.eigs(matrix, num)

    if mode == "symmetric":
        return linalg.eigsh(matrix, num)


##################################################################
#
#       Normalization
#
##################################################################
@jit(["void(float64[:, :], float64[:], float64[:], int64[2])"], nopython=True, parallel=True)
def normalization(matrix, scaling_dim0, scaling_dim1, matrix_shape):
    """
    Scale each row in the matrix by a corresponding value in scaling_dim0.
    Scale each col in the matrix by a corresponding value in scaling_dim1.

    :param matrix: The matrix to scale.
    :param scaling_dim0: Scaling factors for each row.
    :param scaling_dim1: Scaling factors for each column.
    :param matrix_shape: The shape of the matrix
    """

    for l in range(matrix_shape[0]):
        matrix[l, :] *= scaling_dim0[l]
    for m in range(matrix_shape[1]): 
        matrix[:, m] *= scaling_dim1[m]

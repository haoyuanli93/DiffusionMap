"""
Different functions are provided to calculate the difference between two patterns.
Such as, L2, L1 distance.

Methods to normalize the pattern is also included.

Notice that in this script, the matrix needs to be sparse.
"""

import numpy as np
import scipy.sparse
from numba import jit, int64, float64


##################################################################
#
#       Construct Laplacian matrixes
#
##################################################################
def degree_mat(weight_matrix):
    """
    Obtain the degree matrix to construct the Laplacian matrix.
    :param weight_matrix:The weight matrix of data points. This has to be sparse
    :return: The csr diagonal degree matrix
    """

    # Calculate the degree. Because the weight matrix is symmetric, it does not matter
    # Along which axis to sum.
    length = weight_matrix.shape[0]
    _degree = np.reshape(weight_matrix.sum(axis=0), (1, length))

    _degree_mat = scipy.sparse.dia_matrix((_degree, np.array([0, ])), shape=(length, length))
    _degree_mat.tocsr()

    return _degree_mat


def inverse_degree_mat(weight_matrix):
    """
    Obtain the degree matrix to construct the normalized Laplacian matrix.
    :param weight_matrix:The weight matrix of data points. This has to be sparse
    :return: The csr diagonal inverse degree matrix
    """

    # Calculate the degree. Because the weight matrix is symmetric, it does not matter
    # Along which axis to sum.
    length = weight_matrix.shape[0]
    _degree = np.reshape(1. / (weight_matrix.sum(axis=0)), (1, length))

    _degree_mat = scipy.sparse.dia_matrix((_degree, np.array([0, ])), shape=(length, length))
    _degree_mat.tocsr()

    return _degree_mat


def inverse_sqrt_degree_mat(weight_matrix):
    """
    Obtain the degree matrix to construct the normalized symmetric Laplacian matrix.
    :param weight_matrix:The weight matrix of data points. This has to be sparse
    :return: The csr diagonal inverse sqrt degree matrix
    """

    # Calculate the degree. Because the weight matrix is symmetric, it does not matter
    # Along which axis to sum.
    length = weight_matrix.shape[0]
    _degree = np.reshape(1. / np.sqrt(weight_matrix.sum(axis=0)), (1, length))

    _degree_mat = scipy.sparse.dia_matrix((_degree, np.array([0, ])), shape=(length, length))
    _degree_mat.tocsr()

    return _degree_mat


def laplacian(degree_matrix, weight_matrix):
    """
    Construct the fundamental Laplacian matrix.

    :param degree_matrix: The csr diagonal matrix of the degree matrix.
    :param weight_matrix: The weight matrix of data points. This has to be sparse
    :return: degree_matrix - weight_matrix
    """
    return degree_matrix - weight_matrix


def normalized_laplacian(degree_matrix, weight_matrix):
    """
    Construct the normalized laplacian matrix.

    :param degree_matrix: The csr diagonal matrix of the degree matrix.
    :param weight_matrix: The weight matrix
    :return: scipy.sparse.eye - degree_matrix * weight_matrix
    """
    length = degree_matrix.shape[0]

    return scipy.sparse.eye(m=length, n=length,
                            format="csr") - degree_matrix * weight_matrix


def get_symmetric_normalized_laplacian(degree_matrix, weight_matrix):
    """
    Construct the normalized symmetric laplacian matrix.

    :param degree_matrix: The csr diagonal matrix of the degree matrix.
    :param weight_matrix: The weight matrix
    :return: scipy.sparse.eye - degree_matrix * weight_matrix * degree_matrix
    """
    length = degree_matrix.shape[0]

    return scipy.sparse.eye(m=length, n=length,
                            format="csr") - degree_matrix * weight_matrix * degree_matrix


##################################################################
#
#       Normalization
#
##################################################################
@jit(["void(float64[:, :], float64[:], float64[:], int64[2])"], nopython=True, parallel=True)
def normalization(matrix, std_dim0, std_dim1, matrix_shape):
    """
    Convert the inner product matrix to Pearson correlation coefficient matrix.
    i.e.
    E[XY] ->  (E[XY] - E[X]E[Y])/Var(X)Var(Y)

    :param matrix: The matrix to scale.
    :param std_dim0: standard deviation for each element along dimension 0.
    :param std_dim1: standard deviation for each element along dimension 1.
    :param matrix_shape: The shape of the matrix
    :return: The normalized matrix
    """

    for l in range(matrix_shape[0]):
        matrix[l, :] /= std_dim0[l]

    for m in range(matrix_shape[1]):
        matrix[:, m] /= std_dim1[m]


@jit(["void(float64[:, :], float64[:], float64[:], int64[2])"], nopython=True, parallel=True)
def shift(matrix, mean_dim0, mean_dim1, matrix_shape):
    """
    Convert the inner product matrix to Pearson correlation coefficient matrix.
    i.e.
    E[XY] ->  (E[XY] - E[X]E[Y])/Var(X)Var(Y)

    :param matrix: The matrix to scale.
    :param mean_dim0: mean value for each element along dimension 0.
    :param mean_dim1: mean value for each element along dimension 1.
    :param matrix_shape: The shape of the matrix
    :return: The normalized matrix
    """

    for l in range(matrix_shape[0]):
        matrix[l, :] -= mean_dim0[l] * mean_dim1


@jit(["void(float64[:, :], float64[:], float64[:],  float64[:], float64[:], int64[2])"],
     nopython=True, parallel=True)
def shift_and_normalization(matrix, std_dim0, std_dim1, mean_dim0, mean_dim1, matrix_shape):
    """
    Convert the inner product matrix to Pearson correlation coefficient matrix.
    i.e.
    E[XY] ->  (E[XY] - E[X]E[Y])/Var(X)Var(Y)

    :param matrix: The matrix to scale.
    :param std_dim0: standard deviation for each element along dimension 0.
    :param std_dim1: standard deviation for each element along dimension 1.
    :param mean_dim0: mean value for each element along dimension 0.
    :param mean_dim1: mean value for each element along dimension 1.
    :param matrix_shape: The shape of the matrix
    :return: The normalized matrix
    """

    for l in range(matrix_shape[0]):
        matrix[l, :] -= mean_dim0[l] * mean_dim1
        matrix[l, :] /= std_dim0[l]

    for m in range(matrix_shape[1]):
        matrix[:, m] /= std_dim1[m]


##################################################################
#
#       Value Extraction
#
##################################################################

@jit(["void(int64[:, :], int64[:, :], int64[:, :], int64[2])"], nopython=True, parallel=True)
def get_values_int(source, indexes, holder, holder_size):
    """
    Use this function to update the indexes along dimension 1.

    :param source: The constructed index holder: aux_dim1_index
    :param indexes: The local index find by da.argtopk
    :param holder: The holder variable: row_idx_to_keep
    :param holder_size: The shape of row_idx_to_keep
    """
    for l in range(holder_size[0]):
        for m in range(holder_size[1]):
            holder[l, m] = source[l, indexes[l, m]]


@jit(["void(float64[:, :], int64[:, :], float64[:, :], int64[2])"], nopython=True, parallel=True)
def get_values_float(source, indexes, holder, holder_size):
    """
    Use this function to update the indexes along dimension 1.

    :param source: The constructed value matrix.
    :param indexes: The local index find by da.argtopk
    :param holder: The holder variable: row_val_to_keep
    :param holder_size: The shape of row_val_to_keep
    """
    for l in range(holder_size[0]):
        for m in range(holder_size[1]):
            holder[l, m] = source[l, indexes[l, m]]

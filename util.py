"""
This module contains functions that I don't know where to put.
"""


def generate_patch_index_list(batch_number, mode="upper_triangle"):
    """
    Generate a list contains the patch indexes.

    :param batch_number: The number of batches to consider.
    :param mode: upper_triangle, only create indexes for upper_triangle. lower_triangle. only create indexes for
    the lower triangle.
    :return: a list contains the indexes. For example
                [ [0,0],[0,1],[0,2], ... , [0, batch_number = n],
                        [1,1],[1,2], ... , [1, n]
                                     ...
                                     ...
                                           [n,n]]
    """

    patch_indexes = []

    if mode == "upper_triangle":
        for l in range(batch_number):
            for m in range(l, batch_number):
                patch_indexes.append([l, m])
    elif mode == "lower_triangle":
        for l in range(batch_number):
            for m in range(0, l):
                patch_indexes.append([l, m])
    else:
        raise Exception("At present, this program can only calculate the upper or lower triangular part"
                        "of the distance matrix.")
    return patch_indexes

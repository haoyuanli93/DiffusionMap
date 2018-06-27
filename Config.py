"""
This file contains the configuration information for the calculation.

The reason that I choose a python script is multiple.
1. It's simple.
2. Python is powerful for this purpose.
"""
import numpy

CONFIGURATIONS = {

    # The following six parameters are used to calculate the correlation matrix
    # or the Laplacian matrix. If you only want to calculate the correlation matrix, then
    # you only need to modify this part.

    "batch_num_dim0": int(7),  # Batch number along dimension 0
    "batch_num_dim1": int(2),  # Batch number along dimension 1
    "input_file_list": str("../input/file_list.txt"),  # The txt file containing the h5 files to process
    "output_folder": str("../output/"),  # The output folder to store the output
    "neighbor_number": int(37),  # The number of nearest neighbors to keep

    # The following two parameters are for the eigenvector calculation. During that calculation, one need
    # to specify the output folder and the neighborhood number. Those two parameters are specified in the
    # preceding six parameters.

    "sparse_matrix_npz": str("../output/laplacian_matrix.npz"),  # The npz file containing the sparse matrix
    "eig_num": int(10)  # The number of (eigenvector, eigenvalue) pairs to compute.

}


##################################################################
#
#       Check
#
##################################################################
def check(comm_size):
    """
    This function check if the parameter type is correct and if the batch_num_dim0 is comm_size - 1
    and check if comm_size is an even number.

    :return: Void
    """

    global CONFIGURATIONS

    config = CONFIGURATIONS

    # The batch_num_dim0 has to be an odd number. and has to be comm_size - 1
    if not (type(config["batch_num_dim0"]) is int):
        raise Exception("batch_num_dim0 has to be an integer.")

    if comm_size - 1 != config["batch_num_dim0"]:
        raise Exception("The number of nodes, i.e. the comm_size has to be config[\"batch_num_dim0\"] + 1.")

    if not (type(config["batch_num_dim1"]) is int):
        raise Exception("batch_num_dim1 has to be an integer.")

    if not (type(config["input_file_list"]) is str):
        raise Exception("input_file_list has to be a python string.")

    if not (type(config["output_folder"]) is str):
        raise Exception("output_folder has to be a python string.")

    if not (type(config["neighbor_number"]) is int):
        raise Exception("neighbor_number has to be an integer.")

    if not (type(config["sparse_matrix_npz"]) is str):
        raise Exception("sparse_matrix_npz has to be a python string.")

    if not (type(config["eig_num"]) is int):
        raise Exception("eig_num has to be an integer.")

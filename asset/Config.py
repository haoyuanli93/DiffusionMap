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

    "batch_num_dim1": int(40),  # Batch number along dimension 1
    "input_file_list": str("../input/file_list.txt"),  # The txt file containing the h5 files to process
    "mask_file": str("../input/mask.npy"),  # The txt file containing the h5 files to process
    "output_folder": str("../output/"),  # The output folder to store the output
    "neighbor_number": int(50),  # The number of nearest neighbors to keep

    # Construct the Laplacian matrix term
    "keep_diagonal": bool(False),  # Whether to keep the diagonal terms or not.for
    "Laplacian_matrix": str("symmetric normalized laplacian"),

    # The following two parameters are for the eigenvector calculation. During that calculation, one need
    # to specify the output folder and the neighborhood number. Those two parameters are specified in the
    # preceding six parameters.

    "sparse_matrix_npz": str("../output/partial_weight_matrix.npz"),  # The npz file containing the weight matrix
                                                                      # From the weight matrix, the laplacian matrix
                                                                      # will be constructed.
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

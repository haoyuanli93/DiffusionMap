"""
This file contains the configuration information for the calculation.

The reason that I choose a python script is multiple.
1. It's simple.
2. Python is powerful for this purpose.
"""

CONFIGURATIONS = {

    ###############################################################################################
    # Specify parameters to calculate the similarity matrix
    ###############################################################################################

    "batch_num_dim1": int(1),  # Batch number along dimension 1
    "input_file_list": str("../input/file_list.txt"),  # The txt file containing the h5 files to process
    "mask_file": str("../input/mask.npy"),  # The txt file containing the h5 files to process
    "output_folder": str("../output/"),  # The output folder to store the output
    "keep_diagonal": bool(False),  # Whether to keep the diagonal terms or not.
    # The total number of nearest neighbors to calculate when one calculate the similarity matrix.
    "neighbor_number_similarity_matrix": int(1000),
    "zeros_mean_shift": True,  # shift the pattern so that the mean is 0.
    "normalize_by_std": True,  # normalize the pattern so that the standard deviation is 1


    ###############################################################################################
    # Specify parameters to construct and solve the Laplacian matrix
    ###############################################################################################
    # Specify the Laplacian matrix type
    "Laplacian_matrix": str("symmetric normalized laplacian"),
    # The neighbor to use when one convert the similarity matrix into the Laplacian matrix
    "neighbor_number_Laplacian_matrix": int(1000),
    "eig_num": int(10),  # The number of (eigenvector, eigenvalue) pairs to compute.

    # tau: The similarity matrix in the program is essentially the Pearson correlation coefficient
    #        This value can be negative which is not valid for a similarity matrix. Therefore one
    #        needs to cast negative values into positive ones. Currently, this is accomplished by applying
    #        np.exp(x/tau) where x represents the array containing the correlation coefficients.
    "tau": float(0.5)

}


##################################################################
#
#       Check
#
##################################################################
def check():
    """
    This function check if the parameter type is correct and if the batch_num_dim0 is comm_size - 1
    and check if comm_size is an even number.
    """

    global CONFIGURATIONS

    config = CONFIGURATIONS

    #####################################################################
    # Check parameter data type
    #####################################################################

    if not (type(config["batch_num_dim1"]) is int):
        raise Exception("batch_num_dim1 has to be an integer.")

    if not (type(config["input_file_list"]) is str):
        raise Exception("input_file_list has to be a python string.")

    if not (type(config["mask_file"]) is str):
        raise Exception("mask_file has to be a python string.")

    if not (type(config["output_folder"]) is str):
        raise Exception("output_folder has to be a python string.")

    if not (type(config["keep_diagonal"]) is bool):
        raise Exception("keep_diagonal has to be a boolean value.")

    if not (type(config["neighbor_number_similarity_matrix"]) is int):
        raise Exception("neighbor_number_similarity_matrix has to be an integer.")

    if not (type(config["Laplacian_matrix"]) is str):
        raise Exception("Laplacian_matrix has to be a python string.")

    if not (type(config["neighbor_number_Laplacian_matrix"]) is int):
        raise Exception("neighbor_number_Laplacian_matrix has to be an integer.")

    if not (type(config["eig_num"]) is int):
        raise Exception("eig_num has to be an integer.")

    if not (type(config["tau"]) is float):
        raise Exception("tau has to be a float value.")

    #####################################################################
    # Check parameter relation
    #####################################################################
    if config["neighbor_number_Laplacian_matrix"] > config["neighbor_number_similarity_matrix"]:
        raise Exception("neighbor_number_Laplacian_matrix can not be " +
                        "larger than neighbor_number_similarity_matrix.")

"""
This file contains the configuration information for the calculation.

The reason that I choose a python script is multiple.
1. It's simple.
2. Python is powerful for this purpose.

Things to notice:
1. You can either specify the parameters in this file or specify the corresponding parameter with commandline.
2. If you specify the parameter in the command line, then the program will use the parameter you have specified
    with commandline, but it will live this file along, i.e. the only way you can modify the value in this file
    is to edit this file directly yourself with editor or other IO methods.
3. It's recommended to use this configuration file.
"""

CONFIGURATIONS = {

    """
    The following six parameters are used to calculate the correlation matrix
    or the Laplacian matrix. If you only want to calculate the correlation matrix, then
    you only need to modify this part.
    """

    "batch_num_dim0": 8,  # Batch number along dimension 0
    "batch_num_dim1": 2,  # Batch number along dimension 1
    "input_file_list": "../input/file_list_2.txt",  # The txt file containing the h5 files to process
    "output_folder": "../output/",  # The output folder to store the output
    "neighbor_number": 37,  # The number of nearest neighbors to keep
    "keep_diagonal": False,  # When counting the nearest neighbors, whether to include the point itself.

    """
    The following two parameters are for the eigenvector calculation. During that calculation, one need 
    to specify the output folder and the neighborhood number. Those two parameters are specified in the 
    preceding six parameters.  
    """

    "sparse_matrix_npz": "../output/laplacian_matrix.npz",  # The npz file containing the sparse matrix
    "eig_num": 10  # The number of (eigenvector, eigenvalue) pairs to compute.

}

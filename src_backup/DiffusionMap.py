"""
ToDo: Use data source module to reduce the number of parameters required for operations.
ToDo: Allow the program to read the pattern number information from data source.
ToDo: Use h5py to organize different output.
"""

import numpy as np
import argparse
from scipy import sparse

import IOfun
import Graph

# Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('distance_matrix', type=str, help="The numpy array of the similarity matrix.")
parser.add_argument('Kernel', type=str, help="The kernel function used to calculate the Laplacian matirx.")
parser.add_argument("Output_Address", type=str, help="The output address.")
parser.add_argument("Laplacian_Type", type=str, help="The Laplacian matrix type")
parser.add_argument("Neighbor_num", type=int, help="The number of connections between points")

# Parse
args = parser.parse_args()
address_distance_matrix = args.distance_matrix
kernel_type = args.Kernel
address_output = args.Output_Address
laplacian_type = args.Laplacian_Type
neighbor_num = args.Neighbor_num

"""
Step one. Load similarity matrix and construct the Lapacian matrix.
"""
# Load similarity matrix
distance_matrix = np.load(address_distance_matrix)

# Tmp
data_num_total = distance_matrix.shape[0]
entry_num_total = data_num_total * neighbor_num

# Apply kernel
distance_matrix = Graph.gaussian_dense(distance_matrix, 1)

# Get degree vector
values_to_keep = Graph.nearest_points_values_without_self(distance_matrix, neighbor_num)
degree_vector = Graph.degree_batch(values_to_keep)

# Get Laplacian with extra entries
holders = np.zeros((data_num_total, 1))
holders[:, 0] = 1 / degree_vector
laplacian_pre = - holders * values_to_keep

# Extract info to build the sparse matrix
indexes_to_keep = Graph.nearest_points_indexes_without_self(distance_matrix, neighbor_num)
indexes_to_keep = indexes_to_keep.reshape(entry_num_total)

# This is the index of each point along the 0th dimension.
aux_index = np.dot(np.reshape(np.arange(data_num_total), (data_num_total, 1)), np.ones((1, neighbor_num)))
aux_index = aux_index.reshape(entry_num_total)

# Values to keep
values_to_keep = values_to_keep.reshape(entry_num_total)

# Construct the sparse matrix
laplacian_sparse = sparse.csr_matrix((values_to_keep, (aux_index, indexes_to_keep)),
                                     shape=(data_num_total, data_num_total)) + sparse.identity(data_num_total)
"""
Step two. Solve for the eigenvalues and eigenvectors.
"""
eigenvalues, eigenvectors = Graph.solve_for_eigenvectors(laplacian_sparse, 5, mode="general")

np.save(address_output + '/eigenvalues.npy', eigenvalues)
np.save(address_output + "/eigenvectors.npy", eigenvectors)

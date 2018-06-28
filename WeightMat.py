# Standard modules
import time

import dask.array as da
import h5py
import numpy as np
import scipy.sparse
from mpi4py import MPI

import Config
# project modules
import DataSource
import Graph

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Check if the configuration information is valid and compatible with the MPI setup
Config.check(comm_size=comm_size)

# Parse
batch_num_dim0 = Config.CONFIGURATIONS["batch_num_dim0"]
batch_num_dim1 = Config.CONFIGURATIONS["batch_num_dim1"]
input_file_list = Config.CONFIGURATIONS["input_file_list"]
output_folder = Config.CONFIGURATIONS["output_folder"]

if Config.CONFIGURATIONS["keep_diagonal"]:
    neighbor_number = Config.CONFIGURATIONS["neighbor_number"]
else:
    # If one does not want to keep the diagonal value, then just calculate for one more value and then
    # remove the diagonal value.
    neighbor_number = Config.CONFIGURATIONS["neighbor_number"] + 1

"""
Step One: Initialization
"""
if comm_rank == 0:
    data_source = DataSource.DataSourceFromH5pyList(source_list_file=input_file_list)

    # Time for making batches
    tic_local = time.time()
    # Build the batches
    data_source.make_batches(batch_num_dim0=batch_num_dim0, batch_num_dim1=batch_num_dim1)
    toc_local = time.time()
    print("It takes {} seconds to construct the batches.".format(toc_local - tic_local))

else:
    data_source = None

print("Sharing the datasource and job list info.")
data_source = comm.bcast(obj=data_source, root=0)
print("Process {} receives the datasource."
      "There are totally {} jobs for this process.".format(comm_rank, len(data_source.batch_ends_local_dim1)))

comm.Barrier()  # Synchronize

"""
Step Two: Calculate mean and std for each diffraction pattern
"""
tic = time.time()
if comm_rank != 0:

    # Get the correct chunk size
    data_shape = data_source.source_dict["shape"]
    chunk_size = tuple([100, ] + list(data_shape))
    data_num = data_source.batch_num_list_dim0[comm_rank - 1]

    # Construct the data for diagonal patch
    info_holder_dim0 = data_source.batch_ends_local_dim0[comm_rank - 1]

    ####################################################################################################################
    #
    #   Begin the calculation of the diagonal term.
    #
    ####################################################################################################################

    # Open the files to do calculation. Remember to close them in the end
    h5file_holder_dim0 = {}
    dataset_holder_dim0 = []
    for file_name in info_holder_dim0["files"]:
        h5file_holder_dim0.update({file_name: h5py.File(file_name, 'r')})

        # Get the dataset names and the range in that dataset
        data_name_list = info_holder_dim0[file_name]["Datasets"]
        data_ends_list = info_holder_dim0[file_name]["Ends"]

        for data_idx in range(len(data_name_list)):
            data_name = data_name_list[data_idx]

            # Load the datasets for the specified range.
            tmp_data_holder = h5file_holder_dim0[file_name][data_name]
            tmp_dask_data_holder = da.from_array(tmp_data_holder[data_ends_list[data_idx][0]:
                                                                 data_ends_list[data_idx][1]], chunks=chunk_size)
            dataset_holder_dim0.append(tmp_dask_data_holder)

    # Create dask arrays based on these h5 files
    dataset_dim0 = da.reshape(da.concatenate(dataset_holder_dim0, axis=0), (data_num, np.prod(data_shape)))
    print("Finishes loading data.")

    # Calculate the mean value of each pattern of the vector
    data_mean_dim0 = da.mean(a=dataset_dim0, axis=-1)
    # Calculate the standard deviation of each pattern of the vector
    data_std_dim0 = da.std(a=dataset_dim0, axis=-1)

    # Calculate the concrete values
    data_mean_dim0, data_std_dim0 = [np.array(data_mean_dim0), np.array(data_std_dim0)]

    print("Finishes calculating the mean, the standard variation and the inner product matrix.")

    ####################################################################################################################
    #
    #   Finish the calculation of the diagonal term. Now Clean things up
    #
    ####################################################################################################################

    # Create a holder for all standard variations and means
    std_all = np.empty(data_source.data_num_total, dtype=np.float64)
    mean_all = np.empty(data_source.data_num_total, dtype=np.float64)
    print("Process {} finishes the first stage.".format(comm_rank))

else:
    # Create several holders in the master node. These values have no meaning.
    # They only keep the pycharm quiet.
    chunk_size = None
    data_shape = data_source.source_dict["shape"]
    data_std_dim0 = None
    data_mean_dim0 = None
    data_num = None
    holder_size = None
    info_holder_dim0 = None
    h5file_holder_dim0 = None
    dataset_dim0 = None
    std_all = np.empty(data_source.data_num_total, dtype=np.float64)
    mean_all = np.empty(data_source.data_num_total, dtype=np.float64)

# Let the master node to gather and assemble all the norms.
std_data = comm.gather(data_std_dim0, root=0)
mean_data = comm.gather(data_mean_dim0, root=0)
comm.Barrier()  # Synchronize

"""
Step Three: The master node receive and organize all the norms
"""
if comm_rank == 0:
    std_all = np.concatenate(std_data[1:], axis=0)
    mean_all = np.concatenate(mean_data[1:], axis=0)
    print("This is process {}, the shape of mean_all is {}".format(comm_rank, mean_all.shape))
    np.save(output_folder + "/mean_all.npy", mean_all)
    np.save(output_folder + "/std_all.npy", std_all)

# Share this information to all worker nodes.
comm.Bcast(std_all, root=0)
comm.Bcast(mean_all, root=0)
comm.Barrier()  # Synchronize

"""
Step Four: Calculate the sparse correlation matrix
"""
if comm_rank != 0:

    # Create holders to store the largest values and the corresponding indexes of the correlation matrix
    holder_size = np.array([data_num, neighbor_number], dtype=np.int64)  # Auxiliary variable
    idx_to_keep_dim1 = np.zeros((data_num, neighbor_number), dtype=np.int64)
    val_to_keep = -10. * np.ones((data_num, neighbor_number), dtype=np.float64)

    ####################################################################################################################
    #
    #   Begin the calculation of a non-diagonal term.
    #
    ####################################################################################################################

    for batch_idx_dim1 in range(batch_num_dim1):

        # Data number for this patch along dimension 1
        data_num_dim1 = data_source.batch_num_list_dim1[batch_idx_dim1]
        global_idx_start = data_source.batch_global_idx_range_dim1[batch_idx_dim1, 0]
        global_idx_end = data_source.batch_global_idx_range_dim1[batch_idx_dim1, 1]

        # Construct the data along dimension 1
        info_holder_dim1 = data_source.batch_ends_local_dim1[batch_idx_dim1]

        # Open the files to do calculation. Remember to close them in the end
        h5file_holder_dim1 = {}
        dataset_holder_dim1 = []
        for file_name in info_holder_dim1["files"]:
            h5file_holder_dim1.update({file_name: h5py.File(file_name, 'r')})

            # Get the dataset names and the range in that dataset
            dataset_name_list_dim1 = info_holder_dim1[file_name]["Datasets"]
            dataset_ends_list_dim1 = info_holder_dim1[file_name]["Ends"]

            for data_idx in range(len(dataset_name_list_dim1)):
                data_name = dataset_name_list_dim1[data_idx]

                # Load the datasets for the specified range.
                tmp_data_holder = h5file_holder_dim1[file_name][data_name]
                tmp_dask_data_holder = da.from_array(tmp_data_holder[dataset_ends_list_dim1[data_idx][0]:
                                                                     dataset_ends_list_dim1[data_idx][1]],
                                                     chunks=chunk_size)
                dataset_holder_dim1.append(tmp_dask_data_holder)

        # Create dask arrays based on these h5 files
        dataset_dim1 = da.reshape(da.concatenate(dataset_holder_dim1, axis=0), (data_num_dim1, np.prod(data_shape)))
        print("Finishes loading data.")

        # Calculate the correlation matrix.
        inner_prod_matrix = da.dot(dataset_dim0, da.transpose(dataset_dim1)) / float(np.prod(data_shape))
        inner_prod_matrix = np.array(inner_prod_matrix)

        ################################################################################################################
        #
        #   Finish the calculation of a non diagonal term. Now Clean things up
        #
        ################################################################################################################

        # prepare some auxiliary variables for later process
        data_std_dim1 = std_all[global_idx_start:global_idx_end]
        data_mean_dim1 = mean_all[global_idx_start:global_idx_end]

        # Construct the global index for each entry along dimension 1
        aux_dim1_index = np.outer(np.ones(data_num, dtype=np.int64), np.arange(global_idx_start - neighbor_number,
                                                                               global_idx_end, dtype=np.int64))
        # Store the index for the entry from the last iteration
        aux_dim1_index[:, :neighbor_number] = idx_to_keep_dim1

        # Normalize the inner product matrix
        Graph.normalization(matrix=inner_prod_matrix,
                            std_dim0=data_std_dim0,
                            std_dim1=data_std_dim1,
                            mean_dim0=data_mean_dim0,
                            mean_dim1=data_mean_dim1,
                            matrix_shape=np.array([data_num, data_num_dim1]))

        # Put previously selected values together with the new value and do the sort
        inner_prod_matrix = np.concatenate((val_to_keep, inner_prod_matrix), axis=1)

        # Find the local index of the largest values
        idx_pre_dim1 = np.argsort(a=inner_prod_matrix, axis=1)[:, :-(neighbor_number + 1):-1]

        # Turn the local index into global index
        Graph.get_values_int(source=aux_dim1_index,
                             indexes=idx_pre_dim1,
                             holder=idx_to_keep_dim1,
                             holder_size=holder_size)

        # Calculate the largest values
        Graph.get_values_float(source=inner_prod_matrix,
                               indexes=idx_pre_dim1,
                               holder=val_to_keep,
                               holder_size=holder_size)

        # Close all h5 file opened for the column dataset
        for file_name in h5file_holder_dim1.keys():
            h5file_holder_dim1[file_name].close()

    # Close all h5 files opened for the row dataset
    for file_name in h5file_holder_dim0.keys():
        h5file_holder_dim0[file_name].close()

else:
    # Auxiliary variables.
    idx_to_keep_dim1 = None
    val_to_keep = None
    idx_pre_dim1 = None

# Let the master node to gather and assemble the matrix.
index_to_keep_dim1_data = comm.gather(idx_to_keep_dim1, root=0)
value_to_keep_data = comm.gather(val_to_keep, root=0)
comm.Barrier()  # Synchronize

"""
Step Five: Collect all the patches and assemble them.
"""
if comm_rank == 0:
    values_all = np.concatenate(value_to_keep_data[1:], axis=0)
    idx_dim1_all = np.concatenate(index_to_keep_dim1_data[1:], axis=0)

    # Constuct the holder for index for each point along dimension 0
    holder_size = (data_source.data_num_total, neighbor_number)
    idx_dim0_all = np.outer(np.arange(data_source.data_num_total, dtype=np.int),
                            np.ones(neighbor_number, dtype=np.int))

    # Convert 1-D array to construct the sparse matrix
    size_num = data_source.data_num_total * neighbor_number
    values_all = values_all.reshape(size_num)
    idx_dim0_all = idx_dim0_all.reshape(size_num)
    idx_dim1_all = idx_dim1_all.reshape(size_num)

    # Calculate the time to construct the correlation matrix
    toc_0 = time.time()

    # Construct a sparse matrix
    matrix = scipy.sparse.coo_matrix((values_all, (idx_dim0_all, idx_dim1_all)),
                                     shape=(data_source.data_num_total, data_source.data_num_total))

    # Save the matrix
    scipy.sparse.save_npz(file=output_folder + "/correlation_matrix.npz", matrix=matrix, compressed=True)

    # Finishes the calculation.
    toc_1 = time.time()
    print("Time to construct the correlation matrix is {}".format(toc_1 - toc_0))

    if not (Config.CONFIGURATIONS["Laplacian_matrix"] is None):
        if Config.CONFIGURATIONS["Laplacian_matrix"] == "symmetric normalized laplacian":
            # Convert negative values, if there is any, to positive values
            np.exp(matrix.data, out=matrix.data)
            # Cast the weight matrix to a symmetric format.
            matrix_trans = matrix.transpose(copy=True)
            matrix_sym = (matrix + matrix_trans) / 2.
            matrix_asym = (matrix - matrix_trans) / 2.
            np.absolute(matrix_asym.data, out=matrix_asym.data)
            matrix_sym += matrix_asym
            # Remove the diagonal term
            if not Config.CONFIGURATIONS["keep_diagonal"]:
                matrix_sym.setdiag(values=0, k=0)
            # Calculate the degree matrix for normalization
            degree = Graph.inverse_sqrt_degree_mat(weight_matrix=matrix)
            # Calculate the laplacian matrix
            laplacian = Graph.symmetrized_normalized_laplacian(degree_matrix=degree, weight_matrix=matrix_sym)

            laplacian.tocsr()
            scipy.sparse.save_npz(file=output_folder + "/laplacian_matrix.npz", matrix=laplacian, compressed=True)

            toc_2 = time.time()
            print("Time to construct the Laplacian matrix is {}".format(toc_2 - toc_1))
            print("This finishes all calculation. The total running time is {}".format(toc_2 - tic))

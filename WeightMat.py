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
neighbor_number = Config.CONFIGURATIONS["neighbor_number"]
keep_diagonal = Config.CONFIGURATIONS["keep_diagonal"]

"""
Step One: Initialization
"""
tic_0 = time.time()
if comm_rank == 0:
    data_source = DataSource.DataSourceFromH5pyList(source_list_file=input_file_list)

    # Time for making batches
    tic = time.time()
    # Build the batches
    data_source.make_batches(batch_num_dim0=batch_num_dim0, batch_num_dim1=batch_num_dim1)
    toc = time.time()
    print("It takes {} seconds to construct the batches.".format(toc - tic))

else:
    data_source = None

print("Sharing the datasource and job list info.")
data_source = comm.bcast(obj=data_source, root=0)
print("Process {} receives the datasource."
      "There are totally {} jobs for this process.".format(comm_rank,
                                                           len(data_source.batch_ends_local_dim1[comm_rank - 1])))
comm.Barrier()  # Synchronize

"""
Step Two: Calculate the diagonal patch
"""
tic = time.time()
if comm_rank != 0:

    # Get the correct chunk size
    data_shape = data_source.source_dict["shape"]
    chunk_size = tuple([100, ] + list(data_shape))
    data_num = data_source.batch_num_list_dim0[comm_rank - 1]

    # Construct the data for diagonal patch
    info_holder_dim0 = data_source.batch_ends_local_dim0[comm_rank - 1]

    print("Process {}".format(comm_rank), info_holder_dim0)

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
    dataset_dim0 = da.concatenate(dataset_holder_dim0, axis=0)
    print("Finishes loading data.")

    axes_range = list(range(1, len(data_shape) + 1))
    # Calculate the mean value of each pattern of the vector
    data_mean_dim0 = da.mean(a=dataset_dim0, axis=tuple(axes_range))
    # Calculate the standard deviation of each pattern of the vector
    data_std_dim0 = da.std(a=dataset_dim0, axis=tuple(axes_range))
    # Calculate the correlation matrix.
    inner_prod_matrix = da.tensordot(dataset_dim0, dataset_dim0, axes=(axes_range, axes_range))

    # Calculate the concrete values
    data_mean_dim0, data_std_dim0, inner_prod_matrix = [np.array(data_mean_dim0),
                                                        np.array(data_std_dim0),
                                                        np.array(inner_prod_matrix) / float(np.prod(data_shape))]

    print("Finishes calculating the mean, the standard variation and the inner product matrix.")

    ####################################################################################################################
    #
    #   Finish the calculation of the diagonal term. Now Clean things up
    #
    ####################################################################################################################

    # Normalize the inner product matrix
    Graph.normalization(matrix=inner_prod_matrix,
                        std_dim0=data_std_dim0,
                        std_dim1=data_std_dim0,
                        mean_dim0=data_mean_dim0,
                        mean_dim1=data_mean_dim0,
                        matrix_shape=np.array([data_num, data_num]))

    """
    Notice that, finally, when we calculate the eigenvectors for the whole matrix,
    one needs the global index rather than the local index. Therefore, one should 
    keep the global index.
    """
    # Remove the diagonal value
    inner_prod_matrix -= np.eye(data_num)
    
    batch_ends = data_source.batch_global_idx_range_dim0[comm_rank - 1, 0]
    idx_pre_dim1 = np.argsort(a=inner_prod_matrix, axis=1)[:, :-(neighbor_number + 1):-1]

    holder_size = np.array([data_num, neighbor_number], dtype=np.int64)

    val_to_keep = np.zeros_like(idx_pre_dim1, dtype=np.float64)
    Graph.get_values_float(source=inner_prod_matrix, indexes=idx_pre_dim1, holder=val_to_keep,
                           holder_size=holder_size)

    idx_to_keep_dim1 = idx_pre_dim1 + batch_ends

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
    num_dim = None
    holder_size = None
    info_holder_dim0 = None
    h5file_holder_dim0 = None
    dataset_dim0 = None
    axes_range = None
    std_all = np.empty(data_source.data_num_total, dtype=np.float64)
    mean_all = np.empty(data_source.data_num_total, dtype=np.float64)
    # For assembling
    idx_pre_dim1 = None
    idx_to_keep_dim1 = None
    val_to_keep = None

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
Step Four: Calculate the off-diagonal patch
"""
if comm_rank != 0:

    # Get the batch bin info along this line.
    bin_list_dim1 = data_source.batch_ends_local_dim1[comm_rank - 1]
    bin_number = len(bin_list_dim1)

    ####################################################################################################################
    #
    #   Begin the calculation of a non-diagonal term.
    #
    ####################################################################################################################

    # Loop through bins along this line
    """
    Notice that, during this process, there are two things to do.
    First, extract all the data contained in this bin and build a dask array for them.
    Second, construct a numpy array containing the corresponding index and mean and std to later process.
    """
    for bin_idx in range(bin_number):

        # Loop through batches in this bins.
        # Remember the first element is the batches, the second element is the corresponding index along dim0
        batches_in_bin = bin_list_dim1[bin_idx][0]
        original_idx_on_dim0 = bin_list_dim1[bin_idx][1]

        # Construct holder arrays for the indexes and norms.
        tmp_num_list = np.array([0, ] + [data_source.batch_num_list_dim0[x] for x in original_idx_on_dim0],
                                dtype=np.int)
        tmp_end_list = np.cumsum(tmp_num_list)
        data_num_dim1 = np.sum(tmp_num_list)

        idx_dim1 = np.zeros(data_num_dim1 + neighbor_number, dtype=np.int)
        data_mean_dim1 = np.ones(data_num_dim1, dtype=np.float)
        data_std_dim1 = np.ones(data_num_dim1, dtype=np.float)

        # Open the files to do calculation. Remember to close them in the end
        h5file_holder_dim1 = {}
        dataset_holder_dim1 = []

        for batch_local_idx in range(len(batches_in_bin)):

            # Define auxiliary variables
            batch_idx = original_idx_on_dim0[batch_local_idx]
            _tmp_start = data_source.batch_global_idx_range_dim0[batch_idx, 0]
            _tmp_end = data_source.batch_global_idx_range_dim0[batch_idx, 1]

            # Assign index value and norm values to corresponding variables.
            # The idx_dim1 is shifted according to the previously listed reason
            idx_dim1[tmp_end_list[batch_local_idx] + neighbor_number:
                     tmp_end_list[batch_local_idx + 1] + neighbor_number] = np.arange(_tmp_start, _tmp_end)

            data_std_dim1[tmp_end_list[batch_local_idx]:tmp_end_list[batch_local_idx + 1]] = std_all[
                                                                                             _tmp_start:_tmp_end]
            data_mean_dim1[tmp_end_list[batch_local_idx]:tmp_end_list[batch_local_idx + 1]] = mean_all[
                                                                                              _tmp_start:_tmp_end]

            # Extract the batch info
            info_holder_dim1 = batches_in_bin[batch_local_idx]  # For different horizontal patches

            for file_name in info_holder_dim1["files"]:

                # To prevent hanging opened files, check existing files.
                if not (file_name in h5file_holder_dim1):
                    h5file_holder_dim1.update({file_name: h5py.File(file_name, 'r')})

                data_name_list_dim1 = info_holder_dim1[file_name]["Datasets"]
                data_ends_list_dim1 = info_holder_dim1[file_name]["Ends"]
                for data_idx in range(len(data_name_list_dim1)):
                    col_data_name = data_name_list_dim1[data_idx]

                    tmp_data_holder = h5file_holder_dim1[file_name][col_data_name]
                    tmp_dask_data_holder = da.from_array(tmp_data_holder[data_ends_list_dim1[data_idx][0]:
                                                                         data_ends_list_dim1[data_idx][1]],
                                                         chunks=chunk_size)
                    dataset_holder_dim1.append(tmp_dask_data_holder)

        print("Finishes loading data.")

        # Create dask arrays based on these h5 files
        dataset_dim1 = da.concatenate(dataset_holder_dim1, axis=0)

        # Calculate the correlation matrix.
        inner_prod_matrix = da.tensordot(dataset_dim0, dataset_dim1, axes=(axes_range, axes_range))
        inner_prod_matrix = np.array(inner_prod_matrix) / float(np.prod(data_shape))

        ################################################################################################################
        #
        #   Finish the calculation of a non diagonal term. Now Clean things up
        #
        ################################################################################################################

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

        # Create auxiliary variable to update index along dimension 1
        aux_dim1_index = np.outer(np.ones(data_num, dtype=np.int), idx_dim1)
        # Assign corresponding values to the first neighbor_number elements along dimension 1
        aux_dim1_index[:, :neighbor_number] = idx_to_keep_dim1

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
    idx_dim0_all = np.zeros(holder_size, dtype=np.int)

    # Fill the idx_dim0_all with desired values
    for idx in range(batch_num_dim0):
        _tmp_start = data_source.batch_global_idx_range_dim0[idx, 0]
        _tmp_end = data_source.batch_global_idx_range_dim0[idx, 1]
        idx_dim0_all[_tmp_start:_tmp_end] = np.outer(np.arange(_tmp_start, _tmp_end, dtype=np.int),
                                                     np.ones(neighbor_number, dtype=np.int))

    # Convert 1-D array to construct the sparse matrix
    size_num = data_source.data_num_total * neighbor_number

    values_all = values_all.reshape(size_num)
    idx_dim0_all = idx_dim0_all.reshape(size_num)
    idx_dim1_all = idx_dim1_all.reshape(size_num)

    # Calculate the time to construct the correlation matrix
    toc_1 = time.time()

    # Construct a sparse matrix
    matrix = scipy.sparse.coo_matrix((values_all, (idx_dim0_all, idx_dim1_all)),
                                     shape=(data_source.data_num_total, data_source.data_num_total))
    matrix.tocsr()

    # Save the matrix
    scipy.sparse.save_npz(file=output_folder + "/correlation_matrix.npz", matrix=matrix, compressed=True)

    # Finishes the calculation.
    toc_0 = time.time()
    print("Finishes all calculation.")
    print("Total calculation time is {}".format(toc_0 - tic_0))
    print("Time to construct the correlation matrix is {}".format(toc_0 - toc_1))

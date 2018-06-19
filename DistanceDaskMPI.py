# Standard modules
from mpi4py import MPI
import numpy as np
import argparse
import time
import dask.array as da
import h5py
import scipy.sparse

# project modules
import DataSource
import Graph
import util

# Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--batch_number_dim0', type=int, help="batch number along dimension 0. This value should be "
                                                          "the same as the worker node number. The worker node"
                                                          "number is the total node number -1.")
parser.add_argument('--batch_number_dim1', type=int, help="batch number along dimension 1. This value "
                                                          "can be arbitrary.")
parser.add_argument('--output_folder', type=str, help="Specify the folder to put the calculated data.")
parser.add_argument("--input_file_list", type=str, help="Specify the text file for the input file list.")
parser.add_argument("--neighbor_number", type=int, help="Specify the number of neighbors.")
parser.add_argument("--keep_diagonal", type=bool, help="Specify the number of neighbors.")

# Parse
args = parser.parse_args()
batch_num_dim0 = args.batch_number_dim0
batch_num_dim1 = args.batch_number_dim1
input_file_list = args.input_file_list
output_folder = args.output_folder
neighbor_number = args.neighbor_number
keep_diagonal = args.keep_diagonal

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

"""
Step One: Initialization
"""
if comm_rank == 0:
    data_source = DataSource.DataSourceFromH5pyList(source_list_file=input_file_list)

    # Time for making batches
    tic = time.time()
    # Build the batches
    data_source.make_batches(batch_num_dim0=batch_num_dim0, batch_num_dim1=batch_num_dim1)
    toc = time.time()
    print("It takes {} seconds to construct the batches.".format(toc - tic))

    # Starting calculating the time
    tic_0 = MPI.Wtime()

else:
    data_source = None

print("Sharing the datasource and job list info.")
# The worker receives the job instruction
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
    row_info_holder = data_source.batch_ends_local_dim0[comm_rank - 1]
    
    print("Process {}".format(comm_rank), row_info_holder)
    
    # Open the files to do calculation
    # Remember to close them in the end
    row_h5file_holder = {}
    row_dataset_holder = []
    for file_name in row_info_holder.keys():
        row_h5file_holder.update({file_name: h5py.File(file_name, 'r')})

        # Get the dataset names and the range in that dataset
        data_name_list = row_info_holder[file_name]["Datasets"]
        data_ends_list = row_info_holder[file_name]["Ends"]

        for data_idx in range(len(data_name_list)):
            data_name = data_name_list[data_idx]

            # Load the datasets for the specified range.
            tmp_data_holder = row_h5file_holder[file_name][data_name]
            tmp_dask_data_holder = da.from_array(tmp_data_holder[data_ends_list[data_idx][0]:
                                                                 data_ends_list[data_idx][1]], chunks=chunk_size)
            row_dataset_holder.append(tmp_dask_data_holder)

    # Create dask arrays based on these h5 files
    dataset = da.concatenate(row_dataset_holder, axis=0)

    print("Finishes loading data.")

    # Calculate the correlation matrix.
    num_dim = len(data_shape) + 1
    inner_prod_matrix = da.tensordot(dataset, dataset, axes=(list(range(1, num_dim)), list(range(1, num_dim))))
    inner_prod_matrix = np.array(inner_prod_matrix)

    # Debug
    #dataset = np.array(dataset)
    #dataset = dataset.reshape(data_num, np.prod(data_shape))
    #inner_prod_matrix = np.matmul(dataset, dataset.T)
    
    
    print("Finishes calculating the matrix.")
    # Get the diagonal values
    inv_norm = 1. / (np.sqrt(np.diag(inner_prod_matrix)))

    # Save the result to disk
    np.save(output_folder + "/aux/inv_norm_batch_{}.npy".format(comm_rank - 1), inv_norm)
        
    # Normalize the inner product matrix
    Graph.normalization(matrix=inner_prod_matrix,
                                            scaling_dim0=inv_norm,
                                            scaling_dim1=inv_norm,
                                            matrix_shape=np.array([data_num, data_num]))
    
    inner_prod_matrix -= np.eye(N=data_num,dtype=np.float)

    # sort get the index of the largest value
    """
    Notice that, finally, when we calculate the eigenvectors for the whole matrix,
    one needs the global index rather than the local index. Therefore, one should 
    keep the global index.
    """
    batch_ends = data_source.batch_global_idx_range_dim0[comm_rank - 1, 0]
    row_idx_pre = np.argsort(a=inner_prod_matrix, axis=1)[:, :-(neighbor_number + 1):-1]

    holder_size = np.array([data_num, neighbor_number], dtype=np.int64)

    row_val_to_keep = np.zeros_like(row_idx_pre, dtype=np.float64)
    util.get_values_float(source=inner_prod_matrix, indexes=row_idx_pre, holder=row_val_to_keep,
                                            holder_size=holder_size)

    row_idx_to_keep = row_idx_pre + batch_ends

    # Create a holder for all norms
    inv_norm_all = np.empty(data_source.data_num_total, dtype=np.float64)

    print("Finishes the first stage.")

else:
    # Create several holders in the master node. These values have no meaning.
    # They only keep the communication robust.
    chunk_size = None
    dataset = None
    data_num = None
    num_dim = None
    row_val_to_keep = None
    row_idx_to_keep = None
    holder_size = None
    row_info_holder = None
    row_h5file_holder = None

comm.Barrier()  # Synchronize

"""
Step Three: The master node receive and organize all the norms
"""
if comm_rank == 0:
    
    # Load all the norms and assemble them
    holder = []
    for l in range(comm_size - 1):
        holder.append(np.load(output_folder + "/aux/inv_norm_batch_{}.npy".format(l)))
        
    inv_norm_all = np.concatenate(holder, axis=0)
    np.save(output_folder + "/inverse_norms.npy", inv_norm_all)

"""
One needs to synchronize here because I try to use the following strategy.

During the first stage, one calculate the diagonal patches and therefore get the norm for each pattern.
If we only have 10**6 patterns, then this is only of a size of a 1024*1024 diffraction pattern.

During the first stage, each worker will sort his patterns and save the largest 50 ones along each rows 
in the memory and remember the global index for each saved value. 

When the worker moves to the next patch, it will first load the norms for patterns in that patch and 
calculate the inner product and normalize and compare with the previous 50 values. In the end, the worker 
will save the largest 50 values along dimension 1.

During this process, each worker need the norm of all the patterns, therefore I have to synchronize here.
"""
comm.Bcast(inv_norm_all, root=0)
comm.Barrier()  # Synchronize

"""
Step Four: Calculate the off-diagonal patch
"""
if comm_rank != 0:
    # Get the batch bin info along this line.
    bin_list_dim1 = data_source.batch_ends_local_dim1[comm_rank - 1]
    bin_number = len(bin_list_dim1)

    # Loop through bins along this line
    """
    Notice that, during this process, there are two things to do.
    First, extract all the data contained in this bin and build a dask array for them.
    Second, construct a numpy array containing the corresponding index and norm to later process.
    """
    for bin_idx in range(bin_number):

        # Loop through batches in this bins.
        # Remember the first element is the batches, the second element is the corresponding index along dim0
        batches_in_bin = bin_list_dim1[bin_idx][0]
        batch_idx_on_dim0 = bin_list_dim1[bin_idx][1]

        # Construct holder arrays for the indexes and norms.
        tmp_num_list = np.array([0, ] + [data_source.batch_num_list_dim0[x] for x in batch_idx_on_dim0], dtype=np.int)
        tmp_end_list = np.cumsum(tmp_num_list)

        data_num_row = np.sum(tmp_num_list)
        """
        The reason that col_idx is longer than the right_inv_norm is that when I use da.argtopk, the returned 
        value is the index in the synthetic array rather than the global index. Therefore, I need an array to 
        keep track of the global index. 
        
        Further more, when I do the sorting, I also need to include the previously selected nearest neighbors,
        therefore, the container should be large enough to include these data. 
        """
        col_idx = np.zeros(data_num_row + neighbor_number, dtype=np.int)
        right_inv_norm = np.zeros(data_num_row, dtype=np.float)

        # Open the files to do calculation. Remember to close them in the end
        col_h5file_holder = {}
        col_dataset_holder = []

        for batch_local_idx in range(len(batches_in_bin)):

            # Define auxiliary variables
            batch_idx = batch_idx_on_dim0[batch_local_idx]
            _tmp_start = data_source.batch_global_idx_range_dim0[batch_idx, 0]
            _tmp_end = data_source.batch_global_idx_range_dim0[batch_idx, 1]

            # Assign index value and norm values to corresponding variables.
            # The col_idx is shifted according to the previously listed reason
            col_idx[tmp_end_list[batch_local_idx] + neighbor_number:
                    tmp_end_list[batch_local_idx + 1] + neighbor_number] = np.arange(_tmp_start, _tmp_end)

            right_inv_norm[tmp_end_list[batch_local_idx]:tmp_end_list[batch_local_idx + 1]] = inv_norm_all[
                                                                                              _tmp_start:_tmp_end]

            # Extract the batch info
            col_info_holder = batches_in_bin[batch_local_idx]  # For different horizontal patches

            for file_name in col_info_holder.keys():
                col_h5file_holder.update({file_name: h5py.File(file_name, 'r')})

                col_data_name_list = col_info_holder[file_name]["Datasets"]
                col_data_ends_list = col_info_holder[file_name]["Ends"]
                for data_idx in range(len(col_data_name_list)):
                    col_data_name = col_data_name_list[data_idx]

                    tmp_data_holder = col_h5file_holder[file_name][col_data_name]
                    tmp_dask_data_holder = da.from_array(tmp_data_holder[col_data_ends_list[data_idx][0]:
                                                                         col_data_ends_list[data_idx][1]],
                                                         chunks=chunk_size)
                    col_dataset_holder.append(tmp_dask_data_holder)

        print("Finishes loading data.")

        # Create auxiliary variable to update index along dimension 1
        aux_dim1_index = np.outer(np.ones(data_source.batch_num_list_dim0[comm_rank - 1], dtype=np.int), col_idx)
        # Assign corresponding values to the first neighbor_number elements along dimension 1
        aux_dim1_index[:, :neighbor_number] = row_idx_to_keep

        # Create dask arrays based on these h5 files
        col_dataset = da.concatenate(col_dataset_holder, axis=0)

        # Calculate the correlation matrix.
        inner_prod_matrix = da.tensordot(dataset, col_dataset,
                                         axes=(list(range(1, num_dim)), list(range(1, num_dim))))
        inner_prod_matrix = np.array(inner_prod_matrix)

        # Normalize the inner product matrix
        Graph.normalization(matrix=inner_prod_matrix,
                            scaling_dim0=inv_norm,
                            scaling_dim1=right_inv_norm,
                            matrix_shape=np.array([data_num, data_num_row]))
        
        # Put previously selected values together with the new value and do the sort
        inner_prod_matrix = np.concatenate((row_val_to_keep, inner_prod_matrix), axis=1)

        # Find the local index of the largest values
        row_idx_pre = np.argsort(a=inner_prod_matrix, axis=1)[:, :-(neighbor_number + 1):-1]

        # Turn the local index into global index
        util.get_values_int(source=aux_dim1_index,
                                              indexes=row_idx_pre,
                                              holder=row_idx_to_keep,
                                              holder_size=holder_size)

        # Calculate the largest values
        util.get_values_float(source=inner_prod_matrix,
                                                indexes=row_idx_pre,
                                                holder=row_val_to_keep,
                                                holder_size=holder_size)

        # Close all h5 file opened for the column dataset
        for file_name in col_h5file_holder.keys():
            col_h5file_holder[file_name].close()

    # Close all h5 files opened for the row dataset
    for file_name in row_info_holder.keys():
        row_h5file_holder[file_name].close()

    # Save the distance patch
    name_to_save = output_folder + "/distances/distance_batch_{}.h5".format(comm_rank - 1)
    with h5py.File(name_to_save, 'w') as _tmp_h5file:
        _tmp_h5file.create_dataset("/row_index", data=row_idx_to_keep)
        _tmp_h5file.create_dataset("/row_values", data=row_val_to_keep)

comm.Barrier()  # Synchronize

"""
Step Five: Collect all the patches and assemble them.
"""
if comm_rank == 0:
    holder_size = (data_source.data_num_total, neighbor_number)
    # Load all the patches and assemble them as a sparse matrix
    idx_dim1 = np.zeros(holder_size, dtype=np.int)
    idx_dim0 = np.zeros(holder_size, dtype=np.int)
    values = np.zeros(holder_size, dtype=np.float)

    # Fill the variables with desired values
    for idx in range(batch_num_dim0):
        with h5py.File(output_folder + "/distances/distance_batch_{}.h5".format(idx)) as h5file:
            _tmp_start = data_source.batch_global_idx_range_dim0[idx, 0]
            _tmp_end = data_source.batch_global_idx_range_dim0[idx, 1]

            idx_dim0[_tmp_start:_tmp_end] = np.outer(np.arange(_tmp_start, _tmp_end, dtype=np.int),
                                                     np.ones(neighbor_number, dtype=np.int))

            idx_dim1[_tmp_start:_tmp_end] = np.array(h5file['row_index'])
            values[_tmp_start:_tmp_end] = np.array(h5file['row_values'])

    size_num = data_source.data_num_total * neighbor_number

    values = values.reshape(size_num)
    idx_dim0 = idx_dim0.reshape(size_num)
    idx_dim1 = idx_dim1.reshape(size_num)

    # Construct a sparse matrix
    matrix = scipy.sparse.coo_matrix((values,(idx_dim0, idx_dim1)),
                                             shape=(data_source.data_num_total, data_source.data_num_total))

    # Symmetrize this matrix
    matrix += np.transpose(matrix)

    # Convert to compressed sparse row matrix
    matrix.tocsr(copy=True)

    # Save the matrix
    scipy.sparse.save_npz(file=output_folder + "/correlation_matrix.npz", matrix=matrix, compressed=True)

    # Finishes the calculation.
    toc_0 = MPI.Wtime()
    print("Finishes all calculation.")
    print("Total calculation time is {}".format(toc_0 - tic_0))

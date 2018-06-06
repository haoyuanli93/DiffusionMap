# Standard modules
from mpi4py import MPI
import numpy as np
import argparse
import time
import dask.array as da
import h5py

# project modules
import DataSourceBK
import OutPut
import Graph
import util

# Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--batch_number', type=int, help="batch number")
parser.add_argument('--calculation_mode', type=str, help="Specify how to partition the matrix.")
parser.add_argument('--address_output', type=str, help="Specify the folder to put the calculated data.")
parser.add_argument("--address_input", type=str, help="Specify the text file for the input file list.")
parser.add_argument("--input_mode", type=str, help="Specify the input file type.")
parser.add_argument("--neighbor_number", type=str, help="Specify the number of neighbors.")

# Parse
args = parser.parse_args()
batch_number = args.batch_number
calculation_mode = args.calculation_mode
address_input = args.address_input
address_output = args.address_output
input_mode = args.input_mode
"""
Because in diffusion map, one would not include the the diagonal terms,
when one sort the distances, ideally, one should remove the diagonal terms.
However, somehow, I don't know how to achieve that efficiently. Therefore
instead, I keep one more terms when sorting the sequence and then remove the 
diagonal line in total. 
"""
neighbor_number = args.neighbor_number + 1

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

"""
Step One: Initialization
"""
if comm_rank == 0:
    data_source = DataSourceBK.create_data_source(source_type="DataSourceV2",
                                                  param={"source_list_file": address_input,
                                                       "file_type": input_mode})
    # Build the batches
    data_source.make_batches(batch_number=batch_number)
    batch_ends = np.cumsum(np.array([0, ] + data_source.batch_number_list))  # Starting global index for each batch.
    # Jobs for each slave
    patch_index = util.generate_patch_index_list(batch_number=batch_number, mode=calculation_mode)
    job_list = util.generate_job_list(param={"batch_number": batch_number,
                                             "patch_index": patch_index}, mode=calculation_mode)
    # Starting calculating the time
    tic_0 = time.time()

else:
    data_source = None
    patch_index = None
    batch_ends = None
    job_list = None

print("Sharing the datasource and job list info.")
# The worker receives the job instruction
data_source = comm.bcast(obj=data_source, root=0)
patch_index = comm.bcast(obj=patch_index, root=0)
batch_ends = comm.bcast(obj=batch_ends, root=0)
job_list = comm.bcast(obj=job_list, root=0)
print("Process {} receives the job instruction."
      "There are totally {} jobs for this process.".format(comm_rank, len(job_list[comm_rank - 1])))
print("The job list is:")
print(job_list[comm_rank - 1])
comm.Barrier()  # Synchronize

"""
Step Two: Calculate the diagonal patch
"""
tic = time.time()
if comm_rank != 0:
    # Construct the data for diagonal patch
    row_info_holder = data_source.batch_ends_local[comm_rank - 1]

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
                                                                 data_ends_list[data_idx][1]]
                                                 , chunks=(10, 128, 128))
            row_dataset_holder.append(tmp_dask_data_holder)

    # Create dask arrays based on these h5 files
    dataset = da.concatenate(row_dataset_holder, axis=0)

    # Calculate the correlation matrix.
    num_dim = len(dataset.shape)
    inner_prod_matrix = da.tensordot(dataset, dataset, axes=(list(range(1, num_dim)), list(range(1, num_dim))))
    inner_prod_matrix.compute(scheduler='threads')

    # Get the diagonal values
    inv_norm = 1. / (da.sqrt(da.diag(inner_prod_matrix)))
    inv_norm.compute(scheduler='threads')

    # Save the diagonal values
    name_to_save = address_output + "/distances/inv_norm_{}.h5".format(comm_rank - 1)
    da.to_hdf5(name_to_save, '/inv_norm', inv_norm)

    # Normalize the inner product matrix
    inner_prod_matrix = da.tensordot(lhs=da.tensordot(lhs=inv_norm,
                                                      rhs=inner_prod_matrix,
                                                      axes=([0, ], [0, ])),
                                     rhs=inv_norm,
                                     axes=([0, ], [1, ]))

    # sort and create holders for the largest values along each row
    """
    Notice that, finally, when we calculate the eigenvectors for the whole matrix,
    one needs the global index rather than the local index. Therefore, one should 
    keep the global index.
    """
    row_idx_to_keep = da.argtopk(a=inner_prod_matrix, k=neighbor_number, axis=1) + batch_ends[comm_rank - 1]
    row_val_to_keep = da.topk(a=inner_prod_matrix, k=neighbor_number, axis=1)

    # There is no need to save these values at present.
    """
    The reason to treat these two variables differently is that I don't need to do complicated 
    manipulations on the values but I did not have a better way to find the correct global index
    other than going back to a numpy array and perform some complicated index manipulations.
    """
    row_val_to_keep.compute()
    row_idx_pre = np.array(row_idx_to_keep)

    # Construct an auxiliary numpy array to extract useful information
    aux_dim0_index = np.outer(np.arange(row_idx_pre.shape[0], dtype=np.int),
                              np.ones(row_idx_pre.shape[1], dtype=np.int))
    aux_dim1_index = np.outer(np.ones(row_idx_pre.shape[0], dtype=np.int),
                              np.arange(row_idx_pre.shape[1], dtype=np.int))

"""
One needs to synchronize here because to avoid excessive io, I would try to use the following strategy.

During the first stage, one calculate the diagonal patches and therefore get the norm for each pattern.
If we only have 10**6 patterns, then this is only of a size of a 1024*1024 diffraction pattern.

During the first stage, each worker will sort his patterns and save the largest 50 ones along each rows 
in the memory and remember the global index for each saved value. 

When the worker moves to the next patch, it will first load the norms for patterns in that patch and 
calculate the inner product and normalize and compare with the previous 50 values. In the end, the worker 
will save the largest 50 values along dimension 1.

During this process, one does not need sparse matrix since I feel it is easier to manipulate the index 
manually.  
"""
comm.Barrier()  # Synchronize

"""
Step Four: Calculate the off-diagonal patch
"""
if comm_rank != 0:
    # Construct the data for off-diagonal patch
    patch_number = len(job_list[comm_rank - 1]) - 1

    for _local_idx in range(1, patch_number):  # The first patch calculated for each row is the diagonal patch.

        # Get to know which patch is to process
        job_idx = job_list[comm_rank - 1][_local_idx]
        col_info_holder = data_source.batch_ends_local[job_idx[1]]  # For different horizontal patches

        # Open the files to do calculation
        # Remember to close them in the end
        col_h5file_holder = {}
        col_dataset_holder = []
        for file_name in col_info_holder.keys():
            col_h5file_holder.update({file_name: h5py.File(file_name, 'r')})

            col_data_name_list = col_info_holder[file_name]["Datasets"]
            col_data_ends_list = col_info_holder[file_name]["Ends"]
            for data_idx in range(len(col_data_name_list)):
                col_data_name = col_data_ends_list[data_idx]

                tmp_data_holder = col_h5file_holder[file_name][col_data_name]
                tmp_dask_data_holder = da.from_array(tmp_data_holder[col_data_ends_list[data_idx][0]:
                                                                     col_data_ends_list[data_idx][1]]
                                                     , chunks=(10, 128, 128))
                col_dataset_holder.append(tmp_dask_data_holder)

        # Create dask arrays based on these h5 files
        col_dataset = da.concatenate(col_dataset_holder, axis=0)

        # Calculate the correlation matrix.
        inner_prod_matrix = da.tensordot(dataset, col_dataset, axes=(list(range(1, num_dim)), list(range(1, num_dim))))

        # Load the norm for the off-diagonal patch
        """Open the file, remember to close it after this iteration."""
        right_inv_norm_holder = h5py.File(address_output + "/distances/inv_norm_{}.h5".format(job_idx[1]))
        right_inv_norm = da.from_array(right_inv_norm_holder['inv_norm']
                                       , chunks=(data_source.batch_number_list[job_idx[1]],))

        # Normalize the inner product matrix
        inner_prod_matrix = da.tensordot(lhs=da.tensordot(lhs=inv_norm,
                                                          rhs=inner_prod_matrix,
                                                          axes=([0, ], [0, ])),
                                         rhs=right_inv_norm,
                                         axes=([0, ], [1, ]))

        # Put previously selected values together with the new value and do the sort
        inner_prod_matrix = da.concatenate([row_val_to_keep, inner_prod_matrix], axis=1)

        # Calculate the largest values
        row_val_to_keep = da.topk(a=inner_prod_matrix, k=neighbor_number, axis=1)
        row_val_to_keep.compute()

        # Notice that this is a new variable.
        row_idx_to_keep = np.array(da.argtopk(a=inner_prod_matrix, k=neighbor_number, axis=1))

        """
        Notice that here, the row_val_to_keep contains values from different patches. Therefore, the 
        index should reflect this fact.
        
        Index larger than the neighborhood number belong to newer patch. Index smaller than the neighborhood
        number belong to the previous patches. The value should be that in the row_idx_pre with the same
        row index and column index.
        """
        row_idx_to_keep[row_idx_to_keep >= neighbor_number] += (batch_ends[job_idx[1]] - neighbor_number)

        # Row index for values from row_idx_pre
        tmp_row_idx_for_pre = row_idx_to_keep[row_idx_to_keep < neighbor_number]
        # Indexes to replace in row_idx_to_keep
        tmp_row_idx = aux_dim1_index[row_idx_to_keep < neighbor_number]
        tmp_col_idx = aux_dim0_index[row_idx_to_keep < neighbor_number]
        row_idx_to_keep[tmp_col_idx, tmp_row_idx] = row_idx_pre[tmp_col_idx, tmp_row_idx_for_pre]

        # Close all h5 file opened for the column dataset
        for file_name in col_info_holder.keys():
            col_h5file_holder[file_name].close()

    # Close all h5 files opened for the row dataset
    for file_name in row_info_holder.keys():
        row_h5file_holder[file_name].close()

    # Save the distance patch
    name_to_save = address_output + "/distances/distance_batch_{}.h5".format(comm_rank - 1)
    da.to_hdf5(name_to_save, {'/row_index': row_idx_to_keep, '/row_values': row_val_to_keep})

comm.Barrier()  # Synchronize

"""
Step Five: Collect all the patches and assemble them.
"""
if comm_rank == 0:
    print("Finishes all calculation.")
    pass

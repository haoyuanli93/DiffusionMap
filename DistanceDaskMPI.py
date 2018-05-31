# Standard modules
from mpi4py import MPI
import numpy as np
import argparse
import time
from dask.distributed import Client, LocalCluster
import dask.array as da
import h5py

# project modules
import DataSource
import OutPut
import Graph
import util

# Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('param', type=int, help="batch number")
parser.add_argument('mode', type=str, help="Specify whether param refers to batch_num or batch_size.")
parser.add_argument('address_output', type=str, help="Specify the folder to put the calculated data.")
parser.add_argument("address_input", type=str, help="Specify the input h5 file.")
parser.add_argument("input_mode", type=str, help="Specify the input mode.")
parser.add_argument("source_type", type=str, help="Specify the datasource type.")

# Parse
args = parser.parse_args()
param = args.param
mode = args.mode
address_input = args.address_input
address_output = args.address_output
input_mode = args.input_mode

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

"""
TODO: Need to replace the direct usage of comm_size with batch_number
"""
batch_number = comm_size - 1

"""
Step One: Create the DataSource class.
"""
if comm_rank == 0:
    data_source = DataSource.create_data_source(source_type="DataSourceV2",
                                                param={"source_list_file": address_input,
                                                       "file_type": input_mode})
    """
    TODO: Need to replace the direct usage of comm_size with batch_number
    """
    data_source.make_batches(batch_number=batch_number)
else:
    data_source = None

data_source = comm.bcast(obj=data_source, root=0)
comm.Barrier()  # Synchronize

"""
Step Two: Setup dask client
"""
# The coordinator node generates the patch list for each worker node
if comm_rank == 0:

    # Starting calculating the time
    tic_0 = time.time()

    # Jobs for each slave
    patch_index = util.generate_patch_index_list(batch_number=batch_number, mode=mode)
    job_list = util.generate_job_list(param={"batch_number": batch_number,
                                             "patch_index": patch_index}, mode=mode)
else:
    cluster = LocalCluster()
    client = Client(cluster, processes=False)

    patch_index = None
    job_list = None

# The worker receives the job instruction
patch_index = comm.bcast(obj=patch_index, root=0)
job_list = comm.bcast(obj=job_list, root=0)
comm.Barrier()  # Synchronize

print("Process {} receives the job instruction."
      "There are totally {} jobs for this process.".format(comm_rank, len(job_list[comm_rank - 1])))
print("The job list is:")
print(job_list[comm_rank - 1])

"""
Step Three: Calculate the diagonal patch
"""
tic = time.time()
if comm_rank != 0:
    # Construct the data for diagonal patch
    row_info_holder = data_source.batch_ends_local[comm_rank - 1]

    # Open the files to do calculation
    # Remember to close them in the end
    h5file_holder = {}
    dataset_holder = []
    for file_name in row_info_holder.keys():
        h5file_holder.update({file_name: h5py.File(file_name, 'r')})

        data_name_list = row_info_holder[file_name]["Datasets"]
        data_ends_list = row_info_holder[file_name]["Ends"]
        for data_idx in range(len(data_name_list)):
            data_name = data_name_list[data_idx]

            tmp_data_holder = h5file_holder[file_name][data_name]
            tmp_dask_data_holder = da.from_array(tmp_data_holder[data_ends_list[data_idx][0]:
                                                                 data_ends_list[data_idx][1]]
                                                 , chunks='auto')
            dataset_holder.append(tmp_dask_data_holder)

    # Create dask arrays based on these h5 files
    dataset = da.concatenate(dataset_holder, axis=0)

    # Calculate the correlation matrix.
    num_dim = len(dataset.shape)
    inner_prod_matrix = da.tensordot(dataset, dataset, axes=(list(range(1, num_dim)), list(range(1, num_dim))))

    # Save the distance patch
    name_to_save = address_output + "/distances/patch_{}_{}.npy".format(comm_rank - 1, comm_rank - 1)
    da.to_npy_stack(name_to_save, inner_prod_matrix)

    # comm.Barrier()  # There is no need to synchronize here

    """
    Step Four: Calculate the off-diagonal patch
    """
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
                                                     , chunks='auto')
                col_dataset_holder.append(tmp_dask_data_holder)

        # Create dask arrays based on these h5 files
        col_dataset = da.concatenate(col_dataset_holder, axis=0)

        # Calculate the correlation matrix.
        inner_prod_matrix = da.tensordot(dataset, col_dataset, axes=(list(range(1, num_dim)), list(range(1, num_dim))))

        # Save the distance patch
        name_to_save = address_output + "/distances/patch_{}_{}.npy".format(job_idx[0], job_idx[1])
        da.to_npy_stack(name_to_save, inner_prod_matrix)

        # Close all h5 file opened for the column dataset
        for file_name in col_info_holder.keys():
            col_h5file_holder[file_name].close()

    # Close all h5 files opened for the row dataset
    for file_name in row_info_holder.keys():
        h5file_holder[file_name].close()

comm.Barrier()  # Synchronize

"""
Step Five: Collect all the patches and assemble them.
"""
if comm_rank == 0:
    # Starting calculating the time
    toc_0 = time.time()
    print("Total time for this calculation is {} seconds".format(toc_0 - tic_0))

    # Assemble the distance matrix
    tot_matrix = OutPut.assemble(data_source)
    # Symmetrize matrix
    sym = np.transpose(np.triu(tot_matrix)) + np.triu(tot_matrix) - np.diag(np.diag(tot_matrix))
    print("Finish assembling the inner product matrix")

    # Calculate the distance matrix
    distance_matrix = Graph.inner_product_to_normalized_L2_square(sym)

    # Save the matrix
    np.save(data_source.output_path + "/distance_matrix.npy", distance_matrix)
    print("Save the complete distance matrix")

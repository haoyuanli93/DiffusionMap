from mpi4py import MPI
import numpy as np
import argparse

# Load other modules
import DataSourceBK
import OutPut
import Graph
import time

# Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('param', type=int, help="batch number")
parser.add_argument('mode', type=str, help="Specify whether param refers to batch_num or batch_size.")
parser.add_argument('address_output', type=str, help="Specify the folder to put the calculated data.")
parser.add_argument("address_input", type=str, help="Specify the input h5 file.")
parser.add_argument("input_mode", type=str, help="Specify the input mode.")

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
Step One: Create the DataSource class.
"""
data_source = DataSourceBK.DataSource(source=address_input, output_path=address_output, mode=input_mode)
data_source.make_indexes(param=param, mode=mode)
comm.Barrier()  # Synchronize

"""
Step Two: Calculate the distance
"""
# The master node generates the patch list for each slave node
if comm_rank == 0:

    # Starting calculating the time
    tic_0 = time.time()

    # Jobs for each slave
    job_list = []
    job_num_tot = data_source.batch_number * (data_source.batch_number + 1) // 2

    # Generate a total list of different patches.
    patch_indexes = []
    for l in range(data_source.batch_number):
        for m in range(l, data_source.batch_number):
            patch_indexes.append([l, m])

    # Calculate the job number for each slave
    # When the number of slaves is a factor of the batch number, the situation is clean
    if np.mod(job_num_tot, comm_size - 1) == 0:
        job_num_each = job_num_tot // (comm_size - 1)
        for l in range(comm_size - 1):
            job_list.append(patch_indexes[l * job_num_each: (l + 1) * job_num_each])
    else:
        # Extra jobs that should be shared among nodes
        extra = np.mod(job_num_tot, comm_size - 1)
        job_num_each = job_num_tot // (comm_size - 1)

        # The first several slaves share the extra batches
        for l in range(extra):
            job_list.append(patch_indexes[l * (job_num_each + 1): (l + 1) * (job_num_each + 1)])

        for l in range(comm_size - 1 - extra):
            start = extra * (job_num_each + 1) + l * job_num_each
            end = start + job_num_each
            job_list.append(patch_indexes[start: end])

    # Send jobs to each slave
    for l in range(1, comm_size):
        destination_process = l
        comm.send(job_list[l - 1], dest=destination_process, tag=1)
        print("sending job instruction to process {}".format(destination_process))

# The slaves receive the job instruction
else:
    job = comm.recv(source=0, tag=1)
    print("Process {} receives the job instruction."
          "There are totally {} jobs for this process.".format(comm_rank, len(job)))
    """
    Begin calculation
    """
    tic = time.time()
    for patch_index in job:
        # Load data, pattern_batch_l refers to pattern indexes along vertical direction.
        # pattern_batch_r refers to pattern indexes along horizontal direction
        pattern_batch_l = data_source.load_data_batch_from_stacks(patch_index[0])
        pattern_num_l = pattern_batch_l.shape[0]
        pattern_batch_r = data_source.load_data_batch_from_stacks(patch_index[1])
        pattern_num_r = pattern_batch_r.shape[0]
        # Calculate variance
        distance = Graph.inner_product_batch(pattern_batch_l, pattern_num_l, pattern_batch_r, pattern_num_r)
        # Save variance
        OutPut.save_distances(data_source, distance, patch_index)
    toc = time.time()
    print("It takes {} seconds for process {} to calculate distance patches".format(toc - tic, comm_rank))

comm.Barrier()  # Synchronize

"""
Step Three: Finish the calculation of the distance
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

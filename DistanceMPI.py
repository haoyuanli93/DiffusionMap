from mpi4py import MPI
import argparse

# Load other modules
import DataSource as DS
import OutPut.py as OP
import KernalAndNorm as KN
import time

# Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('param', type=int, help="batch number")
parser.add_argument('mode', type=str, help="Specify whether param refers to batch number or batch size.")

# Parse
args = parser.parse_args()
param = args.param
mode = args.mode

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

"""
Step One: Create the DataSource class.
"""
data_source = DS.DataSource(source='./raw.h5', output_path='.', mode="test")
data_source.make_indexes(param=param, mode=mode)
comm.Barrier()  # Synchronize

"""
Step Two: Calculate the variance
"""
# The master node generates the batch list for each slave node
if comm_rank == 0:
    # Jobs for each slave
    job_list = []
    job_num_tot = data_source.batch_number
    job_num_each = job_num_tot // (comm_size - 1) + 1

    # The first comm_size-2 slaves will do more than or equal to the last slave
    for l in range(comm_size - 2):
        job_list.append(range(l * job_num_each, (l + 1) * job_num_each))
    # Deal with the last slave
    job_list.append(range((comm_size - 2) * job_num_each, job_num_tot))

    # Send jobs to each slave
    for l in range(1, comm_size):
        destination_process = l
        comm.send(job_list[l - 1], dest=destination_process, tag="normalization")
        print("sending job instruction to process {}".format(destination_process))

# The slaves receive the job instruction
else:
    job = comm.recv(source=0, tag="normalization")
    print("Process {} receives the job instruction."
          "There are totally {} jobs for this process.".format(comm_rank, len(job)))
    """
    Begin calculation
    """
    tic = time.time()
    for batch_index in job:
        # Load data
        pattern_batch = data_source.load_data_batch_from_stacks(batch_index)
        # Calculate variance
        variance = KN.squared_l2_norm_batch(pattern_batch)
        # Release memory
        del pattern_batch
        # Save variance
        OP.save_variances(data_source, variance, batch_index)
    toc = time.time()
    print("It takes {} seconds for process {} to finish the normalization".format(toc - tic, comm_rank))

comm.Barrier()  # Synchronize
"""
Step Three: Calculate the distance
"""
# The master node generates the patch list for each slave node
if comm_rank == 0:
    # Jobs for each slave
    job_list = []
    job_num_tot = data_source.batch_number * (data_source.batch_number - 1) // 2
    job_num_each = job_num_tot // (comm_size - 1) + 1

    # Generate a total list of different patches.
    patch_indexes = []
    for l in range(data_source.batch_number):
        for m in range(l, data_source.batch_number):
            patch_indexes.append([l, m])

    # The first comm_size-2 slaves will do more than or equal to the last slave
    for l in range(comm_size - 2):
        job_list.append(patch_indexes[l * job_num_each: (l + 1) * job_num_each])
    # Deal with the last slave
    job_list.append(patch_indexes[(comm_size - 2) * job_num_each:job_num_tot])

    # Send jobs to each slave
    for l in range(1, comm_size):
        destination_process = l
        comm.send(job_list[l - 1], dest=destination_process, tag="distance")
        print("sending job instruction to process {}".format(destination_process))

# The slaves receive the job instruction
else:
    job = comm.recv(source=0, tag="distance")
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
        distance = KN.inner_product_batch(pattern_batch_l, pattern_num_l, pattern_batch_r, pattern_num_r)
        # Save variance
        OP.save_distances(data_source, distance, patch_index)
    toc = time.time()
    print("It takes {} seconds for process {} to finish the normalization".format(toc - tic, comm_rank))

comm.Barrier()  # Synchronize

import time
import numpy as np
from pDiffusionMap import util, abbr, DataSource
from mpi4py import MPI

try:
    import Config
except ImportError:
    raise Exception("This package use Config.py file to set parameters. Please use the start_a_new_project.py "
                    "script to get a folder \'proj_***\'. Move this folder to a desirable address and modify"
                    "the Config.py file in the folder \'proj_***/pDiffusionMap\' and execute DiffusionMap calculation"
                    "in this folder.")
# Check if the configuration information is valid and compatible with the MPI setup
Config.check()

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
batch_num_dim0 = comm_size - 1

# Parse
batch_num_dim1 = Config.CONFIGURATIONS["batch_num_dim1"]
input_file_list = Config.CONFIGURATIONS["input_file_list"]
output_folder = Config.CONFIGURATIONS["output_folder"]
mask_file = Config.CONFIGURATIONS["mask_file"]
zeros_mean_shift = Config.CONFIGURATIONS["zeros_mean_shift"]
normalize_by_std = Config.CONFIGURATIONS["normalize_by_std"]

if Config.CONFIGURATIONS["keep_diagonal"]:
    neighbor_number = Config.CONFIGURATIONS["neighbor_number_similarity_matrix"]
else:
    # If one does not want to keep the diagonal value, then just calculate for one more value and then
    # remove the diagonal value.
    neighbor_number = Config.CONFIGURATIONS["neighbor_number_similarity_matrix"] + 1

"""
Step One: Initialization
"""
if comm_rank == 0:
    data_source = DataSource.DataSourceFromH5pyList(source_list_file=input_file_list)

    # Build the batches
    tic_local = time.time()
    data_source.make_batches(batch_num_dim0=batch_num_dim0, batch_num_dim1=batch_num_dim1)
    toc_local = time.time()
    print("It takes {} seconds to construct the batches.".format(toc_local - tic_local))

else:
    data_source = None

comm.Barrier()  # Synchronize
data_source = comm.bcast(obj=data_source, root=0)
print("Process {} receives the datasource."
      "There are totally {} jobs for this process.".format(comm_rank, len(data_source.batch_ends_local_dim1)))
comm.Barrier()  # Synchronize

"""
Step Two: Calculate mean and std
"""
# Global timer
tic = time.time()
if comm_rank != 0:

    # Get the correct chunk size
    data_shape = data_source.source_dict["shape"]
    chunk_size = tuple([100, ] + list(data_shape))
    data_num = data_source.batch_num_list_dim0[comm_rank - 1]

    # Construct the data for diagonal patch
    info_holder_dim0 = data_source.batch_ends_local_dim0[comm_rank - 1]

    # Load data and calculate the mean and std
    [dataset_dim0, data_mean_dim0,
     data_std_dim0, bool_mask_1d, mask] = abbr.get_data_and_stat(batch_info=info_holder_dim0,
                                                                 maskfile=mask_file,
                                                                 data_num=data_num,
                                                                 data_shape=data_shape)

    # Create a holder for all standard variations and means
    std_all = np.empty(data_source.data_num_total, dtype=np.float64)
    mean_all = np.empty(data_source.data_num_total, dtype=np.float64)
    print("Process {} finishes the first stage.".format(comm_rank))

else:
    # Auxiliary variables.
    data_std_dim0 = None
    data_mean_dim0 = None

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

# Share this information to all worker nodes.
comm.Bcast(std_all, root=0)
comm.Bcast(mean_all, root=0)
comm.Barrier()  # Synchronize

"""
Step Four: Calculate the sparse weight matrix
"""
if comm_rank != 0:

    # Create holders to store the largest values and the corresponding indexes of the correlation matrix
    holder_size = np.array([data_num, neighbor_number], dtype=np.int64)  # Auxiliary variable
    idx_to_keep_dim1 = np.zeros((data_num, neighbor_number), dtype=np.int64)
    val_to_keep = (-2e+100) * np.ones((data_num, neighbor_number), dtype=np.float64)

    #  Loop through each rows.
    for batch_idx_dim1 in range(batch_num_dim1):
        print("Node {} begins to process batch {}. There are {} more batches to process.".format(comm_rank,
                                                                                                 batch_idx_dim1,
                                                                                                 batch_num_dim1 -
                                                                                                 batch_idx_dim1 - 1))
        abbr.update_nearest_neighbors(data_source=data_source, dataset_dim0=dataset_dim0,
                                      data_num=data_num, std_all=std_all, mean_all=mean_all,
                                      neighbor_number=neighbor_number, data_shape=data_shape,
                                      batch_idx_dim1=batch_idx_dim1, bool_mask_1d=bool_mask_1d,
                                      data_std_dim0=data_std_dim0, data_mean_dim0=data_mean_dim0,
                                      holder_size=holder_size, idx_to_keep_dim1=idx_to_keep_dim1,
                                      val_to_keep=val_to_keep, normalize_by_std=normalize_by_std,
                                      zeros_mean_shift=zeros_mean_shift)

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
    # Load the mask
    mask = np.load(mask_file)
    util.save_correlation_values_and_positions(values=values_all,
                                               index_dim0=idx_dim0_all,
                                               index_dim1=idx_dim1_all,
                                               output_address=output_folder,
                                               mask=mask,
                                               means=mean_all,
                                               std=std_all)
    # Finishes the calculation.
    toc = time.time()
    print("The total calculation time is {} seconds".format(toc - tic))

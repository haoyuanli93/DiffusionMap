import numpy as np
from pDiffusionMap import Graph, util


def update_nearest_neighbors(data_source, dataset_dim0, data_num,
                             std_all, mean_all, neighbor_number, data_shape,
                             batch_idx_dim1, bool_mask_1d, data_std_dim0,
                             data_mean_dim0, holder_size,
                             idx_to_keep_dim1, val_to_keep):
    """
    This is an abbreviation of the original flow for to find the nearest neighbors.

    :param data_source: The data_source object.
    :param dataset_dim0: The dataset_dim0 the dataset along dimension 0
    :param data_num: The data number along dimension 0
    :param std_all: All standard deviation
    :param mean_all: All mean values
    :param neighbor_number: The number of neighbors to keep.
    :param data_shape: The shape of each pattern.
    :param batch_idx_dim1: The batch index along dimension 1
    :param bool_mask_1d: The 1D boolean mask
    :param data_std_dim0: The standard deviation of the dimension 0 batch.
    :param data_mean_dim0: The mean values of the dimension 0 batch.
    :param holder_size: The size of the following two holders.
    :param idx_to_keep_dim1: The holder for the indexes
    :param val_to_keep: The holder for the values.
    :return: None
    """
    # Data number for this patch along dimension 1
    data_num_dim1 = data_source.batch_num_list_dim1[batch_idx_dim1]
    global_idx_start = data_source.batch_global_idx_range_dim1[batch_idx_dim1, 0]
    global_idx_end = data_source.batch_global_idx_range_dim1[batch_idx_dim1, 1]

    # Construct the data along dimension 1
    info_holder_dim1 = data_source.batch_ends_local_dim1[batch_idx_dim1]

    # Create dask arrays based on these h5 files
    dataset_dim1 = np.reshape(util.h5_dataloader(batch_dict=info_holder_dim1,
                                                 pattern_number=data_num,
                                                 pattern_shape=data_shape),
                              (data_num_dim1, np.prod(data_shape)))
    # Apply the mask
    dataset_dim1 = dataset_dim1[:, bool_mask_1d]

    # Calculate the correlation matrix.
    inner_prod_matrix = np.dot(dataset_dim0, np.transpose(dataset_dim1)) / float(np.prod(data_shape))

    ################################################################################################################
    #   Finish the calculation of a non diagonal term. Clean things up
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


def get_data_and_stat(batch_info, maskfile, data_num, data_shape):
    """
    Use the batch_info to load the data along dimension 0 and calculate the mean value and standard deviation
    of each pattern.
    :param batch_info: The dictionary containing the info to extract pattern in this batch.
    :param maskfile: A string containing the address of the numpy array
    :param data_num: Number of patterns in this batch.
    :param data_shape: Shape of each pattern
    :return: reshaped_data_of_this_batch, data_mean, data_std, bool_mask_1d
    """
    # Load data
    dataset = util.h5_dataloader(batch_dict=batch_info,
                                 batch_number=data_num,
                                 pattern_shape=data_shape)
    dataset = dataset.reshape((data_num, np.prod(data_shape)))

    # Load the mask
    mask = np.load(maskfile)
    bool_mask_1d = util.get_bool_mask_1d(mask=mask)
    # Apply the mask to the dataset_dim0
    dataset = dataset[:, bool_mask_1d]

    # Calculate the mean value of each pattern of the vector
    data_mean = np.mean(dataset, axis=-1)
    # Calculate the standard deviation of each pattern of the vector
    data_std = np.std(dataset, axis=-1)

    return dataset, data_mean, data_std, bool_mask_1d, mask

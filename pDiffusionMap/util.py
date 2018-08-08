"""
This module contains functions that I don't know where to put.
"""
import time
import datetime

import h5py
import numpy as np
import scipy
import scipy.sparse
from pDiffusionMap import Graph


##################################################################
#
#       Parse the list of data files
#
##################################################################

def parse_data_list(txt_file, file_type="h5"):
    """
    Provide an interface to parse the data list.

    :param txt_file: The txt file containing the list of all data files.
    :param file_type: The type of data files
    :return: A dictionary in the following format.

            {"Files":[A list of all names in the txt file] ,
             Name_1:{Datasets:[a list of datsets to process],
                     data_num:[a list of data number in each data set]}
             ...

             }

    """

    available_file_type = ["h5", ]

    if file_type == "h5":
        return _parse_h5_data_list(txt_file)
    else:
        raise Exception("Invalid value for file_type. Currently, available file types are: ",
                        available_file_type)


def _parse_h5_data_list(txt_file):
    """
    Exam each h5 files listed in the txt_file. It returns a dictionary contains the addresses to each
    h5 files and the datasets to process and the pattern number of each data sets.
    It also checks the shape of each data set.

    Notice that, the order of the file is strictly the order given by the user. If the user
    has specified the order the dataset, then the order of the dataset is strictly that specified by the user.
    If the user has not specified the order of the dataset. The it's ordered according to lexicographical order.
    i.e.

    keys = list(file.keys())
    keys = keys.sort(key = str.lower)

    This also means, your dataset name can only use ASCII characters.

    :param txt_file: The txt file containing the list to parse.
    :return: a dictionary containing the necessary information.
    """

    dict_holder = {"Files": []}

    # Read the lines. Do not change the order of the files in the txt file.
    with open(txt_file, 'r') as txtFile:
        lines_holder = txtFile.readlines()

    # Remove redundant "/n" symbol and blank spaces
    lines_holder = [x.strip('\n') for x in lines_holder if x.strip('\n')]
    lines_holder = [x.strip() for x in lines_holder if x.strip()]

    """
    The initial step is to remove all lines starting with #
    """
    lines = []
    for x in lines_holder:
        if x[0] == "#":
            continue
        lines.append(x)

    """
    This parser exams Three times through the file.
    The first loop, create entries for each h5 file. 
    The second loop, exams whether the user has specified data sets to process.
    The third loop, exam each data set to check for pattern number and pattern shape.
    """

    # Record the line number for each line beginning with "File:"
    file_pos = []

    """
    First loop, check for existence
    """
    for num in range(len(lines)):
        line = lines[num]

        # If the first 5 characters are "File:", then append
        # the address to the corresponding file to dict_holder["Files"]
        if line[:5] == "File:":

            address = line[5:]
            # Check if the file exists and is intact
            try:
                with h5py.File(address, 'r'):
                    pass

                # Because the file exist, append the file address to dict_holder["Files"]
                dict_holder["Files"].append(address)

                # Create entries for this h5 file.
                dict_holder.update({address: {"Datasets": [],
                                              "data_num": []}})

                # Record the line number of this file
                file_pos.append(num)

            except IOError:
                raise Exception("The file {} does not exit or is damaged.".format(address) +
                                "Please make sure all the data source h5 files are intact" +
                                "before launching this program.")

    file_list = dict_holder["Files"]
    # Check if all the files are different. Because I am using the absolute path, this can be done easily
    if len(set(file_list)) != len(file_list):
        raise Exception("There are duplicated h5 files specified in the list." +
                        "Please make sure that all the h5 files listed in {} ".format(txt_file) +
                        "are unique.")

    """
    Second loop, check for user specified data sets
    """
    file_num_total = len(file_list)
    file_pos.append(len(lines))  # To specify the range of lines to search, for the last file in the list.

    # First check all the other files except the last one.
    for file_idx in range(file_num_total):
        # Check lines between this file and the next file
        """
        The default behavior is to process all data sets in the h5 file. 
        If the user specify a data set, then set this flag to 0. In this case,
        only process those data sets specified by the user.
        """
        default_flag = 1
        for line_idx in range(file_pos[file_idx], file_pos[file_idx + 1]):

            line = lines[line_idx]
            # Check if this line begins with "Dataset:"
            if line[:8] == "Dataset:":
                # If user specifies data sets, set this flag to zero to disable the default behavior.
                default_flag = 0
                dict_holder[file_list[file_idx]]["Datasets"].append(line[8:])

        # If it's to use default behavior.
        if default_flag == 1:
            with h5py.File(file_list[file_idx], 'r') as h5file:
                keys = list(h5file.keys())
                # Make sure the keys are in lexicographical order
                dict_holder[file_list[file_idx]]["Datasets"] = keys

    """
    Third loop, check for data number and data shape
    """

    # Get a shape
    with h5py.File(file_list[0], 'r') as h5file:
        key = dict_holder[file_list[0]]["Datasets"][0]
        data_set = h5file[key]
        dict_holder.update({"shape": data_set.shape[1:]})

    for file_address in file_list:

        with h5py.File(file_address, 'r') as h5file:
            for key in dict_holder[file_address]["Datasets"]:
                data_set = h5file[key]
                dict_holder[file_address]["data_num"].append(data_set.shape[0])
                # Check if the data size is correct
                if dict_holder["shape"] != data_set.shape[1:]:
                    raise Exception("The shape of the dataset {}".format(key) +
                                    "in file {}".format(file_address) +
                                    "is different from the intended shape." +
                                    "Please check if the shape of all samples are the same.")

    # Return the result
    return dict_holder


##################################################################
#
#       Get batch number list
#
##################################################################

def get_batch_num_list(total_num, batch_num):
    """
    Generate a list containing the data number per batch.
    The idea is that the difference between each batches is at most one pattern.

    :param total_num: The total number of patterns.
    :param batch_num: The number of batches to build.
    :return: A list containing the data number in each batch.
    """

    redundant_num = np.mod(total_num, batch_num)
    if redundant_num != 0:
        number_per_batch = total_num // batch_num
        batch_num_list = [number_per_batch + 1, ] * redundant_num
        batch_num_list += [number_per_batch, ] * (batch_num - redundant_num)
    else:
        number_per_batch = total_num // batch_num
        batch_num_list = [number_per_batch, ] * batch_num

    return batch_num_list


##################################################################
#
#       Get global index map
#
##################################################################

def get_global_index_map(data_num_total,
                         file_num,
                         data_num_per_file,
                         dataset_num_per_file,
                         data_num_per_dataset):
    """
    Return an array containing the map from the global index to file index, dataset index and the local
    index for the specific pattern.

    :param data_num_total: The total number of data points.
    :param file_num: The number of files.
    :param data_num_per_file: The data point number in each file
    :param dataset_num_per_file: The dataset number in each file
    :param data_num_per_dataset: The data point number in each dataset.
    :return: A numpy array containing the map
                           [
     global index -->       [file index, dataset index, local index]],
                            [file index, dataset index, local index]],
                            [file index, dataset index, local index]],
                                    ....
                           ]

    """
    holder = np.zeros((3, data_num_total), dtype=np.int64)
    # Starting point of the global index for different files
    global_idx_file_start = 0
    for file_idx in range(file_num):
        # End point of the global index for different files
        global_idx_file_end = global_idx_file_start + data_num_per_file[file_idx]
        # Assign file index
        holder[0, global_idx_file_start: global_idx_file_end] = file_idx
        """
        Postpone the update of the starting point until the end of the loop.
        """

        # Process the dataset index
        # Starting point of the global index for different dataset
        global_idx_dataset_start = global_idx_file_start
        for dataset_idx in range(dataset_num_per_file[file_idx]):
            # End point of the global index for different dataset
            global_idx_dataset_end = global_idx_dataset_start + data_num_per_dataset[file_idx][dataset_idx]
            # Assign the dataset index
            holder[1, global_idx_dataset_start: global_idx_dataset_end] = dataset_idx
            # Assign the local index within each dataset
            holder[2, global_idx_dataset_start:global_idx_dataset_end] = np.arange(
                data_num_per_dataset[file_idx][dataset_idx])

            # Update the starting global index of the dataset
            global_idx_dataset_start = global_idx_dataset_end

        # update the start point for the global index of the file
        global_idx_file_start = global_idx_file_end

    return holder


##################################################################
#
#       Get batch ends
#
##################################################################

def get_batch_ends(index_map, global_index_range_list, file_list, source_dict):
    """
    Generate the batch ends for each batch given all information.

    :param index_map: The map between global index and (file index, dataset index, local index)
    :param global_index_range_list: A numpy array containing the starting and ending global
                                    index of the corresponding batch
                [
       batch 0 -->  [starting global index, ending global index],
       batch 1 -->  [starting global index, ending global index],
       batch 2 -->  [starting global index, ending global index],
                        ...
                ]
    :param file_list: The list containing the file names.
    :param source_dict: The information of the source.
    :return: A list containing information for dask to retrieve the data in this batch.
             The structure of this variable is

        The first layer is a list ----->   [
                                        " This is for the first batch"
        The second layer is a dic ----->     {
                                              files :[ A list containing the addresses for files in this
                                                       folder. Notice that this list has the same order
                                                       as that listed in the input file list.]

                                              file name 1:
        The third layer is a dic  ----->                    {Dataset name:
        The forth layer is a list ----->                     [A list of the dataset names],

                                                             Ends:
                                                             [A list of the ends in the dataset. Each is a
                                                              small list: [start,end]]}
                                                             ,
                                              file name 2:
        The third layer is a dic  ----->                    {Dataset name:
        The forth layer is a list ----->                     [A list of the dataset names],

                                                             Ends:
                                                             [A list of the ends in the dataset. Each is a
                                                              small list: [start,end]]}
                                                             , ... }

                                         " This is for the second batch"
                                         ...
                                            ]
    """

    batch_number = global_index_range_list.shape[0]
    # Create a variable to hold batch ends.
    batch_ends_local = []

    for batch_idx in range(batch_number):

        global_idx_batch_start = global_index_range_list[batch_idx, 0]
        global_idx_batch_end = global_index_range_list[batch_idx, 1]

        # Create an element for this batch
        batch_ends_local.append({})
        # This entry contains the list of files contained in this batch
        batch_ends_local[-1].update({"files": []})

        # Find out how many h5 files are covered by this range
        """
        Because the way the map variable is created guarantees that the file index is 
        increasing, the returned result need not be sorted. Similar reason applies for 
        the other layers.
        """
        file_pos_holder = index_map[0, global_idx_batch_start: global_idx_batch_end]
        dataset_pos_holder = index_map[1, global_idx_batch_start: global_idx_batch_end]
        data_pos_holder = index_map[2, global_idx_batch_start: global_idx_batch_end]

        file_range = np.unique(file_pos_holder)

        # Create the entry for the batch
        for file_idx in file_range:

            # The file address of this file
            batch_ends_local[-1]["files"].append(file_list[file_idx])

            # Holder for dataset information for this file in this batch
            batch_ends_local[-1].update({file_list[file_idx]: {"Datasets": [],
                                                               "Ends": []}})
            # Find out which datasets are covered within this file for this batch
            dataset_range = np.unique(dataset_pos_holder[file_pos_holder == file_idx])
            for dataset_idx in dataset_range:
                # Attach this dataset name
                batch_ends_local[-1][file_list[file_idx]]["Datasets"].append(
                    source_dict[file_list[file_idx]]["Datasets"][dataset_idx])
                # Find out the ends for this dataset
                """
                Notice that, because later, I will use [start:end] to retrieve the data
                from the h5 file. Therefore, the end should be the true end of the python-style
                index plus 1.
                """
                tmp_start = np.min(
                    data_pos_holder[(file_pos_holder == file_idx) & (dataset_pos_holder == dataset_idx)])
                tmp_end = np.max(
                    data_pos_holder[(file_pos_holder == file_idx) & (dataset_pos_holder == dataset_idx)]) + 1
                # Attach this dataset range
                batch_ends_local[-1][file_list[file_idx]]["Ends"].append([tmp_start, tmp_end])

    return batch_ends_local


##################################################################
#
#       Provide batch index list to merge
#
##################################################################

def get_batch_idx_per_list(batch_num):
    """
    The batch number is calculated in this way.

                --------------------------
                | 00 | 11 | 11 | 11 | 11 |
                --------------------------
                | 11 | 00 | 11 | 11 | 11 |
                --------------------------
                | 11 | 11 | 00 | 11 | 11 |
                --------------------------
                | 11 | 11 | 11 | 00 | 11 |
                --------------------------
                | 11 | 11 | 11 | 11 | 00 |
                --------------------------

    I want the index along each line for each 11 element.

    :param batch_num: The number of batches along each line.
    :return: A numpy array containing the dim1 idx of the 11 element in this matrix.
    """
    batch_num_per_line = batch_num - 1
    holder = np.zeros((batch_num, batch_num_per_line), dtype=np.int)

    # Deal with the first line and the last line
    holder[0, :] = np.arange(1, batch_num, dtype=np.int)
    holder[batch_num - 1, :] = np.arange(batch_num_per_line, dtype=np.int)
    for l in range(1, batch_num - 1):
        holder[l, :l] = np.arange(l, dtype=np.int)
        holder[l, l:] = np.arange(l + 1, batch_num, dtype=np.int)

    return holder


##################################################################
#
#       Sampling
#
##################################################################

def get_sampled_pattern(global_index, global_index_map, data_dict):
    """
    This function return the data with the corresponding global index

    :param global_index: The global index of the corresponding pattern.
    :param global_index_map: The global_index_map which is defined in  get_global_index_map
    :param data_dict: The source_dict in data_source object
    :return: The corresponding pattern.
    """
    # Decipher the global index
    file_index = global_index_map[global_index, 0]
    dataset_index = global_index_map[global_index, 1]
    local_index = global_index_map[global_index, 2]

    # Get file name and dataset name
    file_name = data_dict["Files"][file_index]
    dataset_name = data_dict[file_name]["Datasets"][dataset_index]

    with h5py.File(file_name, 'r') as h5file:
        return np.array(h5file[dataset_name][local_index])


def get_sampled_pattern_batch(global_index_array, global_index_map, data_dict):
    """
    This function return the data with the corresponding global index

    :param global_index_array: The array containing all global indexes of interests to us.
    :param global_index_map: The global_index_map which is defined in  get_global_index_map
    :param data_dict: The source_dict in data_source object
    :return: A numpy array containing all patterns of interests to us .
    """

    holder_shape = tuple([global_index_array.shape[0], ] + list(data_dict['shape']))
    pattern_holder = np.zeros(holder_shape)

    for l in range(global_index_array.shape[0]):
        global_index = global_index_array[l]

        # Decipher the global index
        file_index = global_index_map[0, global_index]
        dataset_index = global_index_map[1, global_index]
        local_index = global_index_map[2, global_index]

        # Get file name and dataset name
        file_name = data_dict["Files"][file_index]
        dataset_name = data_dict[file_name]["Datasets"][dataset_index]

        # Get the pattern
        with h5py.File(file_name, 'r') as h5file:
            pattern_holder[l] = np.array(h5file[dataset_name][local_index])

    return pattern_holder


def get_sampled_pattern_batch_efficient(global_index_array, global_index_map, data_dict):
    """
    This function return the data with the corresponding global index

    :param global_index_array: The array containing all global indexes of interests to us.
    :param global_index_map: The global_index_map which is defined in  get_global_index_map
    :param data_dict: The source_dict in data_source object
    :return: A numpy array containing all patterns of interests to us .
    """

    holder_shape = tuple([global_index_array.shape[0], ] + list(data_dict['shape']))
    pattern_holder = np.zeros(holder_shape)

    counter = 0

    """
    Adopt the method in data source here.
    """
    file_pos_holder = global_index_map[0, global_index_array]
    dataset_pos_holder = global_index_map[1, global_index_array]
    data_pos_holder = global_index_map[2, global_index_array]

    # Get all the files
    file_range = np.unique(file_pos_holder)

    # Create the entry for the batch
    for file_idx in file_range:

        # Directly open this h5file
        file_name = data_dict["Files"][file_idx]
        with h5py.File(file_name, 'r') as h5file:

            # Get all the dataset within this file
            dataset_range = np.unique(dataset_pos_holder[file_pos_holder == file_idx])
            # Load each dataset
            for dataset_idx in dataset_range:
                dataset_name = data_dict[file_name]["Datasets"][dataset_idx]
                # Get all the pattern local index
                local_index = data_pos_holder[(file_pos_holder == file_idx) & (dataset_pos_holder == dataset_idx)]

                for l in local_index:
                    pattern_holder[counter] = np.array(h5file[dataset_name][l])
                    counter += 1

    return pattern_holder


##################################################################
#
#       Assemble
#
##################################################################

def save_correlation_values_and_positions(values, index_dim0, index_dim1,
                                          means, std, mask, output_address):
    """
    Save the arrays that can be converted into the Laplacian matrix into a hdf5 file.
    As I imagine, no one would want to calculate the correlation matrix a lot of times,
    therefore I don't need a timestamp to automatically distinguish different calculations.

    :param values: The values to save. Notice that this is a 2D numpy array. Dimension 0 represent
                    the index of the sample. Dimension 1 represent the nearest neighbors. The values
                    along each row decrease. i.e. values[i,j] >= values[i,j+1] holds for any i and j.
    :param index_dim0: The index along dimension 0 for each value.
    :param index_dim1: The index along dimension 1 for each value.
    :param means: The mean value of each data pattern.
    :param std: The standard deviation for each data pattern
    :param mask: The mask utilized here.
    :param output_address: The output folder to save the result.
    :return: None
    """
    # Create a time stamp
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

    with h5py.File(output_address + "/partial_correlation_matrix.h5", 'w') as h5file:
        h5file.create_dataset('values', data=values, dtype=np.float64)
        h5file.create_dataset('index_dim0', data=index_dim0, dtype=np.int64)
        h5file.create_dataset('index_dim1', data=index_dim1, dtype=np.int64)
        h5file.create_dataset('matrix_shape', data=np.array([values.shape[0], values.shape[0]], dtype=np.int64),
                              dtype=np.int64)
        h5file.create_dataset('means', data=means, dtype=np.float64)
        h5file.create_dataset('mask', data=mask, dtype=np.int64)
        h5file.create_dataset('std', data=std, dtype=np.float64)
        h5file.create_dataset('time_stamp', data=stamp)


def assemble_laplacian_matrix(laplacian_type, correlation_matrix_file, neighbor_number, tau, keep_diagonal=False):
    """
    Assemble the Laplacian matrix from the weight matrix.

    :param laplacian_type: The type of Laplacian matrix to construct.
    :param correlation_matrix_file: The hdf5 file containing the information of the weight matrix.
    :param neighbor_number: The number of neighbors to keep in the Laplacian matrix
    :param tau: The casting parameter: correlation np.exp(correlation/tau)
    :param keep_diagonal: Whether to keep the diagonal term.
    :return: The csr sparse Laplacian matrix, and the shape of this matrix.
    """

    """
    This is a dirty trick. In previous calculation, when you have set the keep_diagonal to be false,
    I did not through away the diagonal terms directly for some reasons. Instead, I kept it and simply
    increase the the neighbor_number_similarity_matrix by one in the actual calculation. Therefore,
    you would find the shape of values to be 
        [total data number, neighbor_number_similarity_matrix + 1]
    
    Then when I construct the Laplacian matrix, I would simply set the diagonal values to be 0 after the casting.
    """

    with h5py.File(correlation_matrix_file, 'r') as h5file:
        values = np.array(h5file['values'])[:, :neighbor_number]
        idx_dim0 = np.array(h5file['index_dim0'])[:, :neighbor_number]
        idx_dim1 = np.array(h5file['index_dim1'])[:, :neighbor_number]
        matrix_shape = np.array(h5file['matrix_shape'])

    site_number = np.prod(values.shape)
    values = values.reshape(site_number)
    idx_dim0 = idx_dim0.reshape(site_number)
    idx_dim1 = idx_dim1.reshape(site_number)

    # Cast the values to positive
    np.exp(values / tau, out=values)

    # Construct a sparse weight matrix
    matrix = scipy.sparse.coo_matrix((values, (idx_dim0, idx_dim1)),
                                     shape=tuple(matrix_shape))

    # Convert the weight matrix in to a Laplacian matrix
    if laplacian_type == "symmetric normalized laplacian":
        # Cast the weight matrix to a symmetric format.
        matrix_trans = matrix.transpose(copy=True)
        matrix_sym = (matrix + matrix_trans) / 2.
        matrix_asym = (matrix - matrix_trans) / 2.
        np.absolute(matrix_asym.data, out=matrix_asym.data)
        matrix_sym += matrix_asym
        # Remove the diagonal term
        if not keep_diagonal:
            matrix_sym.setdiag(values=0, k=0)
        # Calculate the degree matrix for normalization
        degree = Graph.inverse_sqrt_degree_mat(weight_matrix=matrix_sym)
        # Calculate the laplacian matrix
        csr_matrix = Graph.symmetrized_normalized_laplacian(degree_matrix=degree, weight_matrix=matrix_sym)
        csr_matrix.tocsr()
    else:
        raise Exception("Currently, the only available Laplacian matrix type is \"symmetric normalized laplacian\".")

    return csr_matrix, matrix_shape


def save_eigensystem_and_calculation_parameters(eigenvectors, eigenvalues, config):
    """
    Save the eigensystem and the parameters used to obtain this result. Use a timestamp to distinguish
    different calculations.

    :param eigenvectors: The obtained eigenvectors.
    :param eigenvalues: The eigenvalue for each eigenvector.
    :param config: The configuration dictionary.
    :return: None
    """
    # Create a time stamp
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

    with h5py.File(config["output_folder"] + "/eigensystem_{}.h5".format(stamp), 'w') as h5file:
        h5file.create_dataset("eigenvalues", data=eigenvalues, dtype=np.float64)
        h5file.create_dataset("eigenvectors", data=eigenvectors, dtype=np.float64)
        h5file.create_dataset("neighbor_number", data=config["neighbor_number_Laplacian_matrix"],
                              dtype=np.int64)
        h5file.create_dataset("tau", data=config["tau"], dtype=np.float64)
        h5file.create_dataset("keep_diagonal", data=config["keep_diagonal"], dtype=np.float64)


##################################################################
#
#       Data Loader
#
##################################################################
def h5_dataloader(batch_dict, batch_number, pattern_shape):
    """
    Use this function to load the data
    :param batch_dict: The dictionary specifying which dataset to read and how many patterns to read from each dataset.
    :param batch_number: The number of patterns in this batch
    :param pattern_shape: The shape of each pattern.
    :return: A numpy array containing the corresponding patterns.
    """
    # First, create a holder for the data
    holder = np.empty((batch_number,) + tuple(pattern_shape), dtype=np.float64)
    # Get a counter to remember where we are when loading the patterns
    counter = 0

    # Second, iteratively open each file and load the dataset
    for file_name in batch_dict["files"]:
        with h5py.File(file_name, 'r') as h5file:
            # Get the dataset names and the range in that dataset
            data_name_list = batch_dict[file_name]["Datasets"]
            data_ends_list = batch_dict[file_name]["Ends"]

            for data_idx in range(len(data_name_list)):
                data_name = data_name_list[data_idx]

                # Obtain a holder for this dataset
                tmp_data_holder = h5file[data_name]
                # Calculate the patter number
                p_num = data_ends_list[data_idx][1] - data_ends_list[data_idx][1]

                # Load the range of patterns into memory
                holder[counter:counter + p_num] = np.array(tmp_data_holder[data_ends_list[data_idx][0]:
                                                                           data_ends_list[data_idx][1]])

                # update the counter
                counter += p_num

    return holder


##################################################################
#
#       Get Bool mask
#
##################################################################

def get_bool_mask_1d(mask):
    """
    Turn the numpy mask into a boolean mask.

    In the numpy array, 1 represents good while 0 represents bad.

    :param mask: numpy array mask.
    :return: The 1D boolean mask
    """

    bool_mask_holder = np.ones(mask.shape, dtype=np.bool)
    bool_mask_holder[mask > 0.5] = True
    bool_mask_holder[mask <= 0.5] = False

    return bool_mask_holder.reshape(np.prod(mask.shape))

"""
This module contains functions that I don't know where to put.
"""
import h5py
# from numba import jit, int64
import numpy as np
import copy


##################################################################
#
#       Generate job list
#
##################################################################

def generate_job_list(param={}, mode="IO_optimized"):
    """
    Generate the job list for each worker

    :param param: This contains information necessary for generate the job list. Currently, this argument
                    receives a dictionary contains

                    {"batch_number":  ,
                     "patch_index": }

                    The patch index is a list which contains the patch index for each patch.
    :param mode: The mode for calculation. Different mode may requires different param value. Currently
                 the only supported mode is "IO_optimized".
    :return: A list containing the jobs for each workder
            [[job1 ,job2, job3,...],
             [job1 ,job2, job3,...],
             [job1 ,job2, job3,...],
              ...                  ]
    """
    if mode == "IO_optimized":
        batch_number = param["batch_number"]
        jobs_list = copy.deepcopy(param["patch_index"])

        # When there is only one batch, do not change anything. Sine everything will be done at once.
        if batch_number == 1:
            return jobs_list

        # Move the diagonal patch to the first position.
        for worker_idx in range(1, batch_number):
            jobs_list[worker_idx][0], jobs_list[worker_idx][worker_idx] = jobs_list[worker_idx][worker_idx], \
                                                                          jobs_list[worker_idx][0]
        return jobs_list


##################################################################
#
#       Generate patch index
#
##################################################################

def generate_patch_index_list(batch_number, mode="IO_optimized"):
    """
    Generate a list contains the patch indexes.

    :param batch_number: The number of batches to consider.
    :param mode: upper_triangle, only create indexes for upper_triangle. lower_triangle. only create indexes for
    the lower triangle. For "IO_optimized", the pattern looks like
                    --------------------------
                    | 11 | 00 | 11 | 00 | 11 |
                    --------------------------
                    | 11 | 11 | 00 | 11 | 00 |
                    --------------------------
                    | 00 | 11 | 11 | 00 | 11 |
                    --------------------------
                    | 11 | 00 | 11 | 11 | 00 |
                    --------------------------
                    | 00 | 11 | 00 | 11 | 11 |
                    --------------------------

    :return: a list contains the indexes. For upper_triangule
                [ [0,0],[0,1],[0,2], ... , [0, batch_number = n],
                        [1,1],[1,2], ... , [1, n]
                                     ...
                                     ...
                                           [n,n]]
    """

    # Check if the batch_number is a integer
    if not isinstance(batch_number, int):
        raise Exception("The batch number has to be an integer.")

    patch_indexes = []

    if mode == "upper_triangle":
        for l in range(batch_number):
            for m in range(l, batch_number):
                patch_indexes.append([l, m])
    elif mode == "lower_triangle":
        for l in range(batch_number):
            for m in range(0, l):
                patch_indexes.append([l, m])
    elif mode == "IO_optimized":
        # Check if the batch_number is odd
        if np.mod(batch_number, 2) == 0:
            raise Exception(
                "At present, this mode only works when the batch number is odd. Please set a new batch number")

        for row in range(batch_number):
            patch_indexes.append([])

            for col in range(row, batch_number):
                if np.mod(col - row, 2) == 0:
                    patch_indexes[-1].append([row, col])

            for col in range(0, row):
                if np.mod(row - col, 2) == 1:
                    patch_indexes[-1].append([row, col])

    else:
        raise Exception("At present, this program can only calculate the upper or lower triangular part"
                        "of the distance matrix.")
    return patch_indexes


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
        lines = txtFile.readlines()

    # Remove redundant "/n" symbol and blank spaces
    lines = [x.strip('\n') for x in lines]
    lines = [x.strip() for x in lines]

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
            default_flag = 0  # If user specifies data sets, set this flag to zero to disable the default behavior.

            line = lines[line_idx]
            # Check if this line begins with "Dataset:"
            if line[:8] == "Dataset:":
                dict_holder[file_list[file_idx]]["Datasets"].append(line[8:])

        # If it's to use default behavior.
        if default_flag == 1:
            with h5py.File(file_list[file_idx], 'r') as h5file:
                keys = list(h5file.keys())
                # Make sure the keys are in lexicographical order
                keys = keys.sort(key=str.lower)
                dict_holder[file_list[file_idx]]["Datasets"].append(keys)

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
                dict_holder[file_address]["Datasets"].append(data_set.shape[0])
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
#       Find out the available memory for each node
#
##################################################################

##################################################################
#
#       MPI configuration parser
#
##################################################################

def mpi_config_parser(txt_file, mode="MPI+Dask"):
    pass

##################################################################
#
#       Get indexes info
#
##################################################################
# @jit(int64[:](int64, int64[:], int64, int64[:], int64[:]), nopython=True)
# def global_idx_to_file_dataset_local_idx(global_idx, data_number_file_accumulate, file_num,
#                                          data_number_dataset_accumulate, dataset_number_file_accumulate):
#     """
#     Get the file index, the dataset index and the local index in that dataset for a global index.
#
#     :param global_idx: the global index
#     :param data_number_file_accumulate: The accumulated sum of the number of data point in all files.
#                                         The first value is 0. The length is file_num + 1.
#     :param file_num: The number of h5 files.
#     :param data_number_dataset_accumulate: The accumulated sum of the number of data point in datasets. It does not
#                                             start from zero.
#     :param dataset_number_file_accumulate: The accumulated sum of the number of datasets in all files. Start from 0.
#     :return: a numpy array. [The file index, the dataset index, the local index in that dataset]
#     """
#
#     holder = np.zeros(3)
#
#     for file_idx in range(file_num):
#
#         # Find the file where the global index lives
#         """
#         The file_idx here is the python-style index of the file. Because
#         the data_number_file_accumulate begins with 0. One has to compare
#         with data_number_file_accumulate[file_idx + 1]. However, because
#         file_idx is the python-style index of the file, later on, one directly
#         use file_idx to find the correct value in the data_number_per_dataset and the
#         other variables.
#         """
#         if global_idx < data_number_file_accumulate[file_idx + 1]:
#             holder[0] = file_idx
#
#             # Find the dataset where the global index lives
#             """
#             The dataset_idx here is the python-style global dataset index of the corresponding dataset in a h5 file.
#             Because it is shifted to right by 1, (the initial value)
#             """
#             for dataset_idx in range(dataset_number_file_accumulate[file_idx],
#                                      dataset_number_file_accumulate[file_idx + 1]):
#                 if global_idx < (data_number_file_accumulate[file_idx] +
#                                  data_number_dataset_accumulate[file_idx, dataset_idx + 1]):
#                     holder[1] = dataset_idx
#                     holder[2] = global_idx - (data_number_file_accumulate[file_idx] +
#                                               data_number_dataset_accumulate[file_idx, dataset_idx])

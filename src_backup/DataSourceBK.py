"""

This module contains classes as interfaces to collect dataset.

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%                                        %%
       %%   ATTENTION! ATTENTION! ATTENTION!     %%
       %%                                        %%
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%                                        %%
       %%  There is not dependence between any   %%
       %%  two classes below. When you modify    %%
       %%  one of them, there is no need to      %%
       %%  worry the others.                     %%
       %%                                        %%
       %%  However, they all share several       %%
       %%  essential interfaces for the other    %%
       %%  modules. You should keep those        %%
       %%  interfaces to unchanged.              %%
       %%                                        %%
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Interfaces:
a. interface to select different data source. This is the function create_data_source.

b. interface of different kinds of data source
    Input: This is different for different data sources
    Output: This should be the same for all kind of data sources.

Tips:
      1. type checking is a good idea to make sure things are working properly.

"""

import h5py as h5
import numpy as np
import util


def create_data_source(source_type="DataSourceV1", param={}):
    """
    Because one can have very different data source, I have a difficult time creating a universal
    datasource class. Therefore, I use this function as a selector. One specifies the corresponding
    type of datasource to use with parameter source_type. Then one specifies the required
    parameters for that source with parameter param, which is dictionary.

    Later, I will modify the other classes to make this model work more smooth.

    :param source_type: "DataSourceV1", "DataSourceV2", ...
    :param param: The parameters required for the corresponding data source
    :return: a DataSource instance with the corresponding type.
    """

    if source_type == "DataSourceV1":
        return DataSourceV1()
    elif source_type == "DataSourceV2":
        return DataSourceV2(param)


class DataSourceV2:
    """

    A newer version of DataSource object. Several things are to be changed in this version.

    1. Add support to several h5 file. One use a txt file to specify which h5 file to read
       and which dataset in that file is to read.
    2. The interfaces for coordinator will be more standard. Compatible with readme.
    3. This class is compatible with dask, i.e. dask can use this program to retrieve the patterns for calculation
       (Hard, may take a while to realize.)
       Temporally, this class may return a list for and then dask will use that list to manage the calculation.

    """

    def __init__(self, param={}):
        """
        Initialize the instance with a text file containing files to process.

        :param source_list_file: The text file containing the list of files to process
        :param file_type: the file_type to process
        """

        if not param:
            print("No text file containing the source list has been specified." +
                  "Please initialize the this DataSource instance manually.")
        else:

            source_list_file = param['source_list_file']
            file_type = param['file_type']

            self.source_dict = util.parse_data_list(source_list_file, file_type=file_type)

            # Get essentials keys of the dict
            self.file_list = self.source_dict["Files"]
            # Get useful statistics
            self.file_num = len(self.file_list)
            self.dataset_num_total = 0
            self.dataset_num_per_file = []  # Use this to trace down the corresponding dataset index
            self.data_num_total = 0
            self.data_num_per_dataset = []
            self.data_num_per_file = []

            for file_address in self.file_list:
                tmp = self.source_dict[file_address]["data_num"]
                self.dataset_num_per_file.append(len(tmp))
                self.data_num_per_file.append(np.sum(tmp))
                self.data_num_per_dataset.append(tmp)

            self.data_num_total = np.sum(self.data_num_per_file)
            self.dataset_num_total = np.sum(self.dataset_num_per_file)

            self.batch_ends_local = []
            self.batch_number_list = []

    def initialize(self, param={}):
        """
        Initialize the instance with a text file containing files to process.

        :param source_list_file: The text file containing the list of files to process
        :param file_type: the file_type to process
        """

        source_list_file = param['source_list_file']
        file_type = param['file_type']

        self.source_dict = util.parse_data_list(source_list_file, file_type=file_type)

        # Get essentials keys of the dict
        self.file_list = self.source_dict["Files"]
        # Get useful statistics
        self.file_num = len(self.file_list)
        self.dataset_num_total = 0
        self.dataset_num_per_file = []  # Use this to trace down the corresponding dataset index
        self.data_num_total = 0
        self.data_num_per_dataset = []
        self.data_num_per_file = []

        for file_address in self.file_list:
            tmp = self.source_dict[file_address]["data_num"]
            self.dataset_num_per_file.append(len(tmp))
            self.data_num_per_file.append(np.sum(tmp))
            self.data_num_per_dataset.append(tmp)

        self.data_num_total = np.sum(self.data_num_per_file)
        self.dataset_num_total = np.sum(self.dataset_num_per_file)

        """
        The self.batch_ends_local is the most crucial variable for data retrieval. The structure of such a 
        variable is 
        
        The first layer is a list ----->   [                                                          
                                        " This is for the first batch"    
        The second layer is a dic ----->     {file name 1:
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
        self.batch_ends_local = []
        self.batch_number_list = []

    def make_batches(self, batch_number):
        """
        Get the info to extract the nth batch.The size of the batch is decided by the coordinator.
        This function only collect some essential info.

        :param batch_number: the number of batches we would like to have.
        :return: A list containing the necessary info.
        """

        ####################################################
        #
        #       Detailed explanation
        #
        ####################################################
        """
        To use this function, one has assumed that each node has almost the same memory available.
        It is not obvious, but the small patches does not necessarily have the same shape.
        The calculation model in my mind is the following:
            
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
        """

        redundant_num = np.mod(self.data_num_total, batch_number)
        if redundant_num != 0:
            number_per_batch = self.data_num_total // batch_number
            self.batch_number_list = [number_per_batch + 1, ] * redundant_num
            self.batch_number_list += [number_per_batch, ] * (batch_number - redundant_num)

        else:
            number_per_batch = self.data_num_total // batch_number
            self.batch_number_list = [number_per_batch + 1, ] * batch_number

        ######################################################################################
        #    Create a huge numpy array to calculate global index and local index
        ######################################################################################
        """
        Create a huge list containing every index.
        This is not an efficient and elegant way. However, I can not figure out a better way.
        """
        holder = np.zeros((3, self.data_num_total), dtype=np.int64)
        # Starting point of the global index for different files
        global_idx_file_start = 0
        for file_idx in range(self.file_num):
            # End point of the global index for different files
            global_idx_file_end = global_idx_file_start + self.data_num_per_file[file_idx]
            # Assign file index
            holder[0, global_idx_file_start: global_idx_file_end] = file_idx
            """
            Postpone the update of the starting point until the end of the loop.
            """

            # Process the dataset index
            # Starting point of the global index for different dataset
            global_idx_dataset_start = global_idx_file_start
            for dataset_idx in range(self.dataset_num_per_file[file_idx]):
                # End point of the global index for different dataset
                global_idx_dataset_end = global_idx_dataset_start + self.data_num_per_dataset[file_idx][dataset_idx]
                # Assign the dataset index
                holder[1, global_idx_dataset_start: global_idx_dataset_end] = dataset_idx
                # Assign the local index within each dataset
                holder[2, global_idx_dataset_start:
                          global_idx_dataset_end] = np.arange(self.data_num_per_dataset[file_idx][dataset_idx])

                # Update the starting global index of the dataset
                global_idx_dataset_start = global_idx_dataset_end

            # update the start point for the global index of the file
            global_idx_file_start = global_idx_file_end

        # The starting global index of this batch
        global_idx_batch_start = 0
        for batch_idx in range(batch_number):
            # The ending global index of this batch
            global_idx_batch_end = global_idx_batch_start + self.batch_number_list[batch_idx]

            #print("global_idx_batch_start", global_idx_batch_start)
            #print("global_idx_batch_end", global_idx_batch_end)

            # Create an element for this batch
            self.batch_ends_local.append({})

            # Find out how many h5 files are covered by this range
            """
            Because the way the holder variable is created guarantees that the file index is 
            increasing, the returned result need not be sorted. Similar reason applies for 
            the other layers.
            """
            file_pos_holder = holder[0, global_idx_batch_start: global_idx_batch_end]
            dataset_pos_holder = holder[1, global_idx_batch_start: global_idx_batch_end]
            data_pos_holder = holder[2, global_idx_batch_start: global_idx_batch_end]

            #print("file position holder", file_pos_holder[::1000])
            file_range = np.unique(file_pos_holder)
            #print("file_range",file_range)
            # Create the entry for the file
            for file_idx in file_range:
                # Create only in the element for this batch
                self.batch_ends_local[-1].update({self.file_list[file_idx]: {"Datasets": [],
                                                                             "Ends": []}})
                # Find out which datasets are covered within this file for this batch
                dataset_range = np.unique(dataset_pos_holder[file_pos_holder == file_idx])
                #print(dataset_range)
                for dataset_idx in dataset_range:
                    # Attach this dataset name
                    self.batch_ends_local[-1][self.file_list[file_idx]]["Datasets"].append(
                        self.source_dict[self.file_list[file_idx]]["Datasets"][dataset_idx])
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
                    self.batch_ends_local[-1][self.file_list[file_idx]]["Ends"].append([tmp_start, tmp_end])

            # Update the batch start and batch ends.
            global_idx_batch_start = global_idx_batch_end
        ######################################################################################
        #    End of this terrible method
        ######################################################################################


class DataSourceV1:
    """
    This is the first version, not compatible with the readme.

    Interface to extract data for calculation

    This class controls the data source for diffusion map. The only available source is a single h5py file.
    The target format is

    (In a single h5py file. There are several data sets.)
    "0"  (500,128,128)
    "1"  (500,128,128)
    ...

    Each data set is a stack of patterns. The last two dimensions are for spatial
    dimensions. The first dimensions record how many patterns are there in this
    file.

    The task completed by this file is
    1. Give a unique index for each pattern.
    2. Record the status of each pattern.
    3. Deliver the correct pattern to the correct thread.

    Attention!
    This solution is only a temporal solution for development and demo.
    At present, I assume that all the data are saved in a single h5 file.
    The original data source, the h5 file, only contains patterns. All the
    later on data are saved in a new h5 file, usually called, output.h5.
    """

    def __init__(self, source, output_path, mode="test"):
        """
        Initialize the DataSource class with source file

        :param source: String of the address of the source file. eg. /home/diffusion/test/source.h5
        :param output_path: String of the address of the output h5 file. eg. /home/diffusion/test/output.h5
        :param mode: Operation mode. Currently, the only available value is "test"
        """
        if mode != "test":
            print("Currently, the available type is the test type.\n"
                  "Please set type to \"test\")")

        self.source = source
        self.output_path = output_path

        # Obtain the structure of the data set.
        with h5.File(self.source, "r") as h5file:
            self.keys = list(h5file.keys())

            # Check if the list is empty
            if len(self.keys) == 0:
                raise Exception("Invalid h5 file. There is no valid data set in the source file.")

            stack_shape = h5file[self.keys[0]].shape
            self.pattern_shape = stack_shape[1:]

            # Obtain the pattern number information of each data set.
            self.pattern_number_total = 0
            self.pattern_number_array = []
            for l in range(len(self.keys)):
                stack_shape = h5file[self.keys[l]].shape
                self.pattern_number_total += stack_shape[0]
                self.pattern_number_array.append(stack_shape[0])

        # Set some other parameters.
        self.indexes = np.zeros((1, 1), dtype=np.int)  # Holder for global indexes for each pattern
        self.batch_number = 0  # The number of batches
        self.batch_size = 0  # The number of patterns in each batch
        self.stack_range = []  # The ends of each stack along the global index

        # Set some other parameters
        self.batch_size_list = []  # Later, this is used to check if the size of the saved patches are correct.

        # The holder for the positions of the ends of the batches
        """
        This is a nested structure. 

        [---------------------------------------------------------- This layer for different batches
            [------------------------------------------------------ This layer for different stacks for the same batch
                [ stack index, starting local, ending local index, starting global, ending global index]
            [
        [
        """
        self.batch_ends = []

    def make_indexes(self, param=1, mode="batch_size"):
        """
        :param param: value of the batch_size or batch_num depending on the choice mode parameter.
        :param mode: Use "batch_size" to specify the number of patterns with the param parameter.
                     Use "batch_num" to specify the number of batches with the param parameter.
        """

        if mode == "batch_size":
            self.batch_number = self.pattern_number_total // param
        elif mode == "batch_num":
            self.batch_number = param
        else:
            raise Exception("Invalid value for mode parameter. \n " +
                            "The mode parameter can only be set as \"batch_num\" or \"batch_size\".")

        # Make indexes
        self._make_indexes_for_batch_number()

    def _make_indexes_for_batch_number(self):
        """
        Evenly divide all the patterns into different batches.

        """

        # Create a global indexing system for all the patterns
        self.indexes = np.zeros((self.pattern_number_total, 3), dtype=np.int)

        # Generate batch_size_list. Some batches are larger than the others.
        extra = np.mod(self.pattern_number_total, self.batch_number)
        self.batch_size = self.pattern_number_total // self.batch_number
        # The first several batches contains one more patterns than the others
        self.batch_size_list = [self.batch_size + 1, ] * extra + [self.batch_size, ] * (self.batch_number - extra)

        # Get batch range
        self.batch_heads = np.zeros(self.batch_number, dtype=np.int)
        self.batch_tails = np.zeros(self.batch_number, dtype=np.int)
        tmp = np.insert(np.add.accumulate(np.array(self.batch_size_list)), 0, 0)
        self.batch_heads = tmp[:-1]
        self.batch_tails = self.batch_heads + np.array(self.batch_size_list)

        # Get stack range
        self.stack_heads = np.zeros(len(self.keys), dtype=np.int)
        self.stack_tails = np.zeros(len(self.keys), dtype=np.int)
        tmp = np.insert(np.add.accumulate(np.array(self.pattern_number_array)), 0, 0)
        self.stack_heads = tmp[:-1]
        self.stack_tails = self.stack_heads + self.pattern_number_array

        # Assign the correct index of the stack
        # Notice that the last number in a:b expression is not taken.
        for l in range(len(self.keys)):
            self.indexes[self.stack_heads[l]: self.stack_tails[l], 1] = l

        # Assign the global index
        self.indexes[:, 2] = np.arange(self.pattern_number_total)

        # Calculate the end points for each batch
        for l in range(self.batch_number):
            sublist_holder = self.get_batch_ends(l)
            self.batch_ends.append(sublist_holder)

    def get_batch_ends(self, batch_index):
        """
        Calculate the ends of each stacks that covers this batch.

        :param batch_index: The index of the batch that are under inspection.
        :return: a python list containing
        [
           [stack index, local starting index, local ending index]
           [stack index, local starting index, local ending index]
           [stack index, local starting index, local ending index]
                ...
        ]
        """

        sublist_holder = []

        # Starting stack index
        batch_start_global_index = self.batch_heads[batch_index]
        batch_end_global_index = self.batch_tails[batch_index]
        batch_start_stack_index = self.indexes[batch_start_global_index, 1]
        batch_end_stack_index = self.indexes[batch_end_global_index - 1, 1]

        """
        Notice that the first value in self.stack_range is zero. So if tmp_start_stack_index=1, then
        self.stack_range[tmp_start_stack_index] is refers to the pattern number in stack 0. 
        Therefore global_index_start - self.stack_range[tmp_start_stack_index] is the local starting index.
        """

        if batch_end_stack_index == batch_start_stack_index:
            sublist_holder.append([batch_start_stack_index,
                                   batch_start_global_index - self.stack_heads[batch_start_stack_index],
                                   batch_end_global_index - self.stack_heads[batch_start_stack_index]])
            return sublist_holder

        # If this batch is covered by two different stack:
        if batch_end_stack_index - batch_start_stack_index == 1:
            sublist_holder.append([batch_start_stack_index,
                                   batch_start_global_index - self.stack_heads[batch_start_stack_index],
                                   self.pattern_number_array[batch_start_stack_index]])

            sublist_holder.append([batch_end_stack_index,
                                   0,
                                   batch_end_global_index - self.stack_heads[batch_end_stack_index]])
            return sublist_holder

        # If this batch is covered by more than two different stacks
        # The first stack
        if batch_end_stack_index - batch_start_stack_index >= 2:
            sublist_holder.append([batch_start_stack_index,
                                   batch_start_global_index - self.stack_heads[batch_start_stack_index],
                                   self.pattern_number_array[batch_start_stack_index]])
            # The stacks in between
            for m in range(batch_start_stack_index + 1, batch_end_stack_index):
                sublist_holder.append([m, 0, self.pattern_number_array[m]])
            # The last stack
            sublist_holder.append([batch_end_stack_index,
                                   0,
                                   batch_end_global_index - self.stack_heads[batch_end_stack_index]])
            return sublist_holder

        raise Exception("Something is wrong.")

    def load_data_batch_from_stacks(self, batch_index):
        """
        Load data for one slave. Notice that this does not update the index.
        :returns: batch_of_pattern
        """

        # Create the holder for the pattern batch \
        holder = np.zeros((self.batch_size_list[batch_index], self.pattern_shape[0], self.pattern_shape[1]))
        tmp_position = 0  # The position in the holder

        # Load the patterns from the h5 file into the holder.
        stack_ends = self.batch_ends[batch_index]

        # Open the file and load data
        with h5.File(self.source, 'r') as h5file:
            # Loop through the indexes
            for l in range(len(stack_ends)):
                # For readability
                tmp_start = stack_ends[l][1]
                tmp_end = stack_ends[l][2]
                tmp_num = tmp_end - tmp_start
                tmp_key = self.keys[stack_ends[l][0]]

                holder[tmp_position: tmp_position + tmp_num] = np.copy(h5file[tmp_key][tmp_start:tmp_end])

                tmp_position += tmp_num

        return holder

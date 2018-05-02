"""
Use classes in this scripts to control the data source for diffusion map.
Currently the only available methods is h5py. The target format is

(In a single h5py file. There are several data sets.)
"0"  (500,128,128)
"1"  (500,128,128)
...

Each data set is a stack of patterns. The first two dimensions are for spatial
dimensions. The last dimensions record how many patterns are there in this
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

import h5py as h5
import numpy as np


class DataSource:
    """Interface to extract data for calculation"""

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

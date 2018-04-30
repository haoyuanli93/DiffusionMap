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
        with h5.File(self.source) as h5file:
            self.keys = h5file.keys()

            # Check if the list is empty
            if len(self.keys) == 0:
                raise Exception("Invalid h5 file. There is no valid data set in the source file.")

            stack_shape = h5file[self.keys[0]].shape
            self.pattern_shape = stack_shape[:-1]

            # Obtain the pattern number information of each data set.
            self.pattern_number_total = 0
            self.pattern_number_array = []
            for l in range(len(self.keys)):
                stack_shape = h5file[self.keys[0]].shape
                self.pattern_number_total += stack_shape[0]
                self.pattern_number_array.append(stack_shape[0])

        # Set some other parameters.
        self.indexes = np.zeros((1, 1))  # Holder for global indexes for each pattern
        self.batch_number = 0  # The number of batches
        self.batch_size = 0  # The number of patterns in each batch
        self.stack_range = []  # The ends of each stack along the global index

        # Set some other parameters
        self.batch_size_list = []  # Later, this is used to check if the size of the saved patches are correct.

        """
        Record for calculating different patches of distances. A 2D array. Upper-right triangle.
        """
        self.patch_status = np.zeros((1, 1))

        """
        Record for normalizing different batches.
        """
        self.batch_status = np.zeros(1)

        # Current processing batch for normalization.
        self.batch_index_to_process = 0

        # Current processing patch for distance matrix
        self.patch_index_to_process = np.zeros(2)

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
            self._make_indexes_for_batch_size(param)
        elif mode == "batch_num":
            self._make_indexes_for_batch_number(param)
        else:
            raise Exception("Invalid value for mode parameter. \n " +
                            "The mode parameter can only be set as \"batch_num\" or \"batch_size\".")

    def _make_indexes_for_batch_number(self, batch_num):
        """
        This function first calculate the corresponding batch_size and then use function
        self._make_indexes_for_batch_size(self, batch_size).

        The principle is to make sure the batch_num is the specified value.

        :param batch_num: Batch number to divide
        """

        self.batch_number = batch_num

        # When the batch number is a factor of the total pattern number:
        if np.mod(self.pattern_number_total, self.batch_size) == 0:
            self.batch_size = self.pattern_number_total // self.batch_number
            self._make_indexes_for_batch_size(self.batch_size)

        # When the batch number is not a factor of the total pattern number:
        else:
            self.batch_size = self.pattern_number_total // self.batch_number + 1
            self._make_indexes_for_batch_size(self.batch_size)

    def _make_indexes_for_batch_size(self, batch_size):
        """
        This function force the program to calculate the batch number according to the batch_size.
        It's basically the same as _make_indexes_for_batch_size_harsh. The only difference is that this function
        allow the pattern number in the last batch to float. If the batch_size is a factor of the total pattern
        number, then this function is the same as _make_indexes_for_batch_size_harsh. Otherwise, the last batch
        would be composed by a smaller number of patterns than that specified by the batch_size variable.

        :param batch_size: integer number. It is not necessarily to be a factor of the total pattern number.
        """

        self.batch_size = batch_size
        if isinstance(self.batch_size, int) and self.batch_size <= 0:
            raise Exception("Invalid batch_size. batch_size has to be an positive integer.")

        self.indexes = np.zeros((self.pattern_number_total, 3))

        # First deal with the batch index problem.
        # When the batch size is a factor of the total pattern number:
        if np.mod(self.pattern_number_total, self.batch_size) == 0:
            self.batch_number = self.pattern_number_total // self.batch_size

            # Assign the correct batch index
            for l in range(self.batch_number):
                self.indexes[l * self.batch_size:(l + 1) * self.batch_size, 0] = l

        # When the batch_size is not a factor of the total pattern number.
        else:
            self.batch_number = self.pattern_number_total // self.batch_size + 1

            # Assign the correct batch index first for those regular batches then to the last batch.
            for l in range(self.batch_number - 1):
                self.indexes[l * self.batch_size:(l + 1) * self.batch_size, 0] = l
            self.indexes[(self.batch_number - 1) * self.batch_size:, 0] = self.batch_number

        # Assign the correct index of the key
        # The positions of the end of each pattern stack along the global index
        self.stack_range = np.add.accumulate(self.pattern_number_array)
        self.stack_range = np.insert(self.stack_range, 0, 0)

        for l in range(len(self.keys)):
            self.indexes[self.stack_range[l]: self.stack_range[l + 1], 1] = l

        # Assign the global index
        self.indexes[:, 2] = np.arange(self.pattern_number_total)

        # Initialize the batch status
        self.batch_status = np.zeros((self.batch_number, 2))
        self.batch_status[:, 0] = np.arange(self.batch_number)

        # Calculate the end points for each batch and stack
        # First calculate the ends for the first batch_num - 1 batches. The last batch is different.
        for l in range(self.batch_number - 1):
            sublist_holder = self.get_batch_ends_harsh(l)
            self.batch_ends.append(sublist_holder)
        # Deal with the last batch. Notice that this formula is valid whether the batch_size is a factor or not.
        self.batch_ends.append(self.get_batch_ends((self.batch_number - 1) * batch_size, self.pattern_number_total))

    def _make_indexes_for_batch_size_harsh(self, batch_size):
        """
        This function force the program to calculate the batch number according to the batch_size.
        The basic design of this global index is as such:

        (batch index, index of the key, global index)

        This function also calculate the starting and ending point of each batch in each pattern stack.

        :param batch_size: integer number. It has to be a factor of the total number.
        :return: None
        """

        self.batch_size = batch_size

        # Check whether the batch_size variable is a positive integer or not.
        if isinstance(self.batch_size, int) and self.batch_size <= 0 and np.mod(self.pattern_number_total,
                                                                                batch_size) != 0:
            raise Exception("Invalid batch_size. batch_size has to be an positive integer.")

        self.indexes = np.zeros((self.pattern_number_total, 3))
        self.batch_number = self.pattern_number_total // self.batch_size

        # Assign the correct batch index
        for l in range(self.batch_number):
            self.indexes[l * self.batch_size:(l + 1) * self.batch_size, 0] = l

        # Assign the correct index of the key
        # The positions of the end of each pattern stack along the global index
        self.stack_range = np.add.accumulate(self.pattern_number_array)
        self.stack_range = np.insert(self.stack_range, 0, 0)

        for l in range(len(self.keys)):
            self.indexes[self.stack_range[l]: self.stack_range[l + 1], 1] = l

        # Assign the global index
        self.indexes[:, 2] = np.arange(self.pattern_number_total)

        # Initialize the batch status
        self.batch_status = np.zeros((self.batch_number, 2))
        self.batch_status[:, 0] = np.arange(self.batch_number)

        # Calculate the end points for each batch and stack
        for l in range(self.batch_number):
            sublist_holder = self.get_batch_ends_harsh(l)
            self.batch_ends.append(sublist_holder)

        return None

    def update_batch_status(self):
        """
        Increase the batch index to process by one and update the status record in the batch status variable.
        For normalizing the data.
        """
        self.batch_status[self.batch_index_to_process] = 1
        self.batch_index_to_process += 1

    def update_patch_status(self):
        """
        Move to the next patch and update the index.

        00 -> 01 -> 02 -> ~~~ -> 0n ->

           -> 11 -> 12 -> ~~~ -> 1n ->

                          ~~~

                              -> nn
        """
        self.patch_status[self.patch_index_to_process[0], self.patch_index_to_process[1]] = 1

        # update the index pair
        if self.patch_index_to_process[1] == self.batch_number:
            self.patch_index_to_process[0] += 1
            self.patch_index_to_process[1] = self.patch_index_to_process[0]
        else:
            self.patch_index_to_process[1] += 1

    def load_data_batch_from_stacks(self, batch_index):
        """
        Load data for one slave. Notice that this does not update the index.
        :returns: batch_of_pattern
        """

        # Create the holder for the pattern batch \
        holder = np.zeros((self.pattern_shape[0], self.pattern_shape[1], self.batch_size))
        tmp_position = 0  # The position in the holder

        # Load the patterns from the h5 file into the holder.
        stack_ends = self.batch_ends[batch_index]

        # Open the file and load data
        with h5.File(self.source) as h5file:
            # Loop through the indexes
            for l in range(len(stack_ends)):
                # For readability
                tmp_start = stack_ends[l][1]
                tmp_end = stack_ends[l][2]
                tmp_num = tmp_end - tmp_start
                tmp_key = self.keys[stack_ends[l][0]]

                holder[tmp_position: tmp_position + tmp_num] = np.copy(h5file[tmp_key][tmp_start:tmp_end])

        return holder

    def load_variances_batch(self, batch_index):
        """
        Return the variances for each batch.
        :param batch_index: The batch index for the batch to return
        :return: A numpy array for the variance.
        """
        with h5.File(self.output_path+'/variances/batch_{}.h5'.format(batch_index), 'r') as h5file:
            holder = h5file['{}'.format(batch_index)]

        return holder

    def get_batch_ends(self, global_index_start, global_index_end):
        """
        Calculate the ends of each stacks that covers this batch.

        :param global_index_start: The global index of the starting pattern in this batch
        :param global_index_end : The global index of the last pattern in this batch
        :return: a python list containing
        [
           [stack index, local starting index, local ending index, global starting index, global ending index],
           [stack index, local starting index, local ending index, global starting index, global ending index],
           [stack index, local starting index, local ending index, global starting index, global ending index],
                ...
        ]
        """

        sublist_holder = []

        # Starting stack index
        tmp_start_stack_index = self.indexes[global_index_start, 1]
        tmp_end_stack_index = self.indexes[global_index_end, 1]

        # If this batch is in the same stack
        """
        Notice that the first value in self.stack_range is zero. So if tmp_start_stack_index=1, then
        self.stack_range[tmp_start_stack_index] is refers to the pattern number in stack 0. 
        Therefore global_index_start - self.stack_range[tmp_start_stack_index] is the local starting index.
        """
        if tmp_start_stack_index == tmp_end_stack_index:
            sublist_holder.append([tmp_start_stack_index,
                                   global_index_start - self.stack_range[tmp_start_stack_index],
                                   global_index_end - self.stack_range[tmp_start_stack_index],
                                   global_index_start,
                                   global_index_end,
                                   ])
            return sublist_holder

        # If this batch is covered by two different stack:
        if tmp_end_stack_index - tmp_start_stack_index == 1:
            sublist_holder.append([tmp_start_stack_index,
                                   global_index_start - self.stack_range[tmp_start_stack_index],
                                   self.pattern_number_array[tmp_start_stack_index],
                                   global_index_start,
                                   self.stack_range[tmp_end_stack_index],
                                   ])

            sublist_holder.append([tmp_end_stack_index,
                                   0,
                                   global_index_end - self.stack_range[tmp_end_stack_index],
                                   self.stack_range[tmp_end_stack_index],
                                   global_index_end
                                   ])
            return sublist_holder

        # If this batch is covered by more than two different stacks
        # The first stack
        sublist_holder.append([tmp_start_stack_index,
                               global_index_start - self.stack_range[tmp_start_stack_index],
                               self.pattern_number_array[tmp_start_stack_index],
                               global_index_start,
                               self.stack_range[tmp_start_stack_index + 1],
                               ])
        # The stacks in between
        for m in range(tmp_start_stack_index + 1, tmp_end_stack_index - 1):
            sublist_holder.append([m, 0, self.pattern_number_array[m],
                                   self.stack_range[m],
                                   self.stack_range[m + 1]])
        # The last stack
        sublist_holder.append([tmp_end_stack_index,
                               0,
                               global_index_end - self.stack_range[tmp_start_stack_index],
                               self.stack_range[tmp_end_stack_index],
                               global_index_end
                               ])

    def get_batch_ends_harsh(self, batch_index):
        """
        Calculate the ends of each stacks that covers this batch.

        :param batch_index: The index of the batch to inspect
        :return: a python list containing

        [
            [stack index, local starting index, local ending index, global starting index, global ending index],
            [stack index, local starting index, local ending index, global starting index, global ending index],
            [stack index, local starting index, local ending index, global starting index, global ending index],
                ...
        ]
        """

        sublist_holder = []

        # Starting stack index
        tmp_start_global_index = self.batch_size * batch_index
        tmp_end_global_index = self.batch_size * (batch_index + 1)
        tmp_start_stack_index = self.indexes[tmp_start_global_index, 1]
        tmp_end_stack_index = self.indexes[tmp_end_global_index, 1]

        # If this batch is in the same stack
        """
        Notice that the first value in self.stack_range is zero. So if tmp_start_stack_index=1, then
        self.stack_range[tmp_start_stack_index] is refers to the pattern number in stack 0. 
        Therefore tmp_start_global_index - self.stack_range[tmp_start_stack_index] is the local starting index.
        """
        if tmp_start_stack_index == tmp_end_stack_index:
            sublist_holder.append([tmp_start_stack_index,
                                   tmp_start_global_index - self.stack_range[tmp_start_stack_index],
                                   tmp_end_global_index - self.stack_range[tmp_start_stack_index],
                                   tmp_start_global_index,
                                   tmp_end_global_index,
                                   ])
            return sublist_holder

        # If this batch is covered by two different stack:
        if tmp_end_stack_index - tmp_start_stack_index == 1:
            sublist_holder.append([tmp_start_stack_index,
                                   tmp_start_global_index - self.stack_range[tmp_start_stack_index],
                                   self.pattern_number_array[tmp_start_stack_index],
                                   tmp_start_global_index,
                                   self.stack_range[tmp_end_stack_index],
                                   ])

            sublist_holder.append([tmp_end_stack_index,
                                   0,
                                   tmp_end_global_index - self.stack_range[tmp_end_stack_index],
                                   self.stack_range[tmp_end_stack_index],
                                   tmp_end_global_index
                                   ])
            return sublist_holder

        # If this batch is covered by more than two different stacks
        # The first stack
        sublist_holder.append([tmp_start_stack_index,
                               tmp_start_global_index - self.stack_range[tmp_start_stack_index],
                               self.pattern_number_array[tmp_start_stack_index],
                               tmp_start_global_index,
                               self.stack_range[tmp_start_stack_index + 1],
                               ])
        # The stacks in between
        for m in range(tmp_start_stack_index + 1, tmp_end_stack_index - 1):
            sublist_holder.append([m, 0, self.pattern_number_array[m],
                                   self.stack_range[m],
                                   self.stack_range[m + 1]])
        # The last stack
        sublist_holder.append([tmp_end_stack_index,
                               0,
                               tmp_end_global_index - self.stack_range[tmp_start_stack_index],
                               self.stack_range[tmp_end_stack_index],
                               tmp_end_global_index
                               ])
        return sublist_holder

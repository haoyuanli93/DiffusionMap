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

import numpy as np

import util


class DataSourceFromH5pyList:
    """

    A newer version of DataSource object. Several things are to be changed in this version.

    1. Only support h5 file. One uses a txt file to specify which h5 file to read
       and which dataset in that file is to read.
    2. This class is only compatible with dask. This class may return a list for and then
       dask will use that list to manage the calculation.
    """

    def __init__(self, source_list_file=None):
        """
        Initialize the instance with a text file containing files to process.
        More information is contained in the comments of the function self.initialize.

        :param source_list_file: The text file containing the list of files to process
        """

        if not source_list_file:
            print("No text file containing the source list has been specified." +
                  "Please initialize the this DataSource instance manually.")
        else:

            self.source_dict = util.parse_data_list(source_list_file, file_type="h5")

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
            The self.batch_ends_local_dim0 is the crucial variable for data retrieval along dimension 0 
            variable is 

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

            """
            The self.batch_ends_local_dim1 is the crucial variable for data retrieval along dimension 1 
            variable is 

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

            self.batch_ends_local_dim0 = []
            self.batch_num_list_dim0 = []
            self.batch_global_idx_range_dim0 = None

            self.batch_ends_local_dim1 = []
            self.batch_num_list_dim1 = []
            self.batch_global_idx_range_dim1 = None

    def initialize(self, source_list_file=None):
        """
        Initialize the instance with a text file containing files to process.

        :param source_list_file: The text file containing the list of files to process
        """

        self.source_dict = util.parse_data_list(source_list_file, file_type="h5")

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

        self.batch_ends_local_dim0 = []
        self.batch_num_list_dim0 = []
        self.batch_global_idx_range_dim0 = None

        self.batch_ends_local_dim1 = []
        self.batch_num_list_dim1 = []
        self.batch_global_idx_range_dim1 = None

    def make_batches(self, batch_num_dim0, batch_num_dim1):
        """
        Get the info to extract the batches.

        :param batch_num_dim0: the number of batches along dimension 0.
        :param batch_num_dim1: the number of batches along dimension 1.
        :return: A list containing the necessary info.
        """

        """
                Create a huge list containing every index.
                   [
                    [file index, dataset index, local index]],   <-- global index
                    [file index, dataset index, local index]], 
                    [file index, dataset index, local index]],
                            ....
                   ]
        """
        global_index_map = util.get_global_index_map(data_num_total=self.data_num_total,
                                                     file_num=self.file_num,
                                                     data_num_per_file=self.data_num_per_file,
                                                     dataset_num_per_file=self.dataset_num_per_file,
                                                     data_num_per_dataset=self.data_num_per_dataset)

        #################################################################
        #  First process dimension 0
        #################################################################

        # Get batch number list along dimension 0
        self.batch_num_list_dim0 = util.get_batch_num_list(total_num=self.data_num_total, batch_num=batch_num_dim0)
        self.batch_global_idx_range_dim0 = np.zeros((batch_num_dim0, 2), dtype=np.int)
        tmp = np.cumsum([0, ] + self.batch_num_list_dim0)
        self.batch_global_idx_range_dim0[:, 0] = tmp[:-1]
        self.batch_global_idx_range_dim0[:, 1] = tmp[1:]

        # Get batch ends for dimension 0 and 1
        self.batch_ends_local_dim0 = util.get_batch_ends(index_map=global_index_map,
                                                         global_index_range_list=self.batch_global_idx_range_dim0,
                                                         file_list=self.file_list,
                                                         source_dict=self.source_dict)

        #################################################################
        #  Process dimension 1
        #################################################################

        # Get batch number list along dimension 1
        self.batch_num_list_dim1 = util.get_batch_num_list(total_num=self.data_num_total, batch_num=batch_num_dim1)
        self.batch_global_idx_range_dim1 = np.zeros((batch_num_dim1, 2), dtype=np.int)
        tmp = np.cumsum([0, ] + self.batch_num_list_dim1)
        self.batch_global_idx_range_dim1[:, 0] = tmp[:-1]
        self.batch_global_idx_range_dim1[:, 1] = tmp[1:]

        # Get batch ends for dimension 0 and 1
        self.batch_ends_local_dim1 = util.get_batch_ends(index_map=global_index_map,
                                                         global_index_range_list=self.batch_global_idx_range_dim1,
                                                         file_list=self.file_list,
                                                         source_dict=self.source_dict)

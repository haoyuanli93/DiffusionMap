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

            self.batch_ends_local_dim0 = []
            self.batch_num_list_dim0 = []
            self.batch_global_idx_range_dim0 = None

            self.batch_ends_local_dim1 = []
            self.batch_num_list_dim1 = []
            # self.batch_global_idx_range_dim1 = None

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
        The structure of the self.batch_ends_local_dim1 is different. Actually, this contains the job information
        for each worker. According to the explanation in the self.make_indexes function below, the calculation 
        scheme requires that each worker is in charge of one line. Therefore, when we have excessive workers, one might 
        want to divide the matrix finer along dimension 0 and less finer on dimension 1. Along each batch in dimension 
        0, we can bin the batch along dimension 0 a little bit. Detailed explanation is in the function 
        self.make_indexes. Therefore, each line will contains several different batches. 
        
        Then there comes another problem, I don't want to re find all the batches. So instead of finding new batches
        I group together the batches along dimension 0. This is the reason I use the following structure.
        
        The zeroth layer is a list ---->  [
                                        "This is for the first line"
                The first layer is a list ----->   [                                                          
                                                " This is for the first batch along this line"    
                The second layer is a list ----->    [ 
                                                        "The first element is the batches"
                The third layer is a list ----->     [
                                                        The 1st batch along dimension 0 that is in this batch,
                                                        The 2nd batch along dimension 0 that is in this batch,
                                                        ...
                                                      ],
                                                        " The second element is batches index of batches
                                                        along dimension 0 in this bin."
                                                      [ 1,2,3,4 ..]
            
                                                      ],
                                                
                                                 " This is for the second batch along this line"
                                                 ...         
                                                    ]
                                         "This is for the second line"
                                         []
                                         ...
                                         ]
        """
        self.batch_ends_local_dim0 = []
        self.batch_num_list_dim0 = []
        self.batch_global_idx_range_dim0 = None

        self.batch_ends_local_dim1 = []
        self.batch_num_list_dim1 = []
        # self.batch_global_idx_range_dim1 = None

    def make_batches(self, batch_num_dim0, batch_num_dim1):
        """
        Get the info to extract the batches.

                    ================
                    Detailed explain
                    ================

        To use this function, one has assumed that each node has almost the same memory available.
        It is not obvious, but the small patches does not necessarily have the same shape.
        The calculation model in my mind is the following:

                    --------------------------
                    | 11 | 11 | 11 | 00 | 00 |
                    --------------------------
                    | 00 | 11 | 11 | 11 | 00 |
                    --------------------------
                    | 00 | 00 | 11 | 11 | 11 |
                    --------------------------
                    | 11 | 00 | 00 | 11 | 11 |
                    --------------------------
                    | 11 | 11 | 00 | 00 | 11 |
                    --------------------------

        "11" represents that  I plan to calculate this patch. "00" means I will skip this patch.
        To avoid excessive io, each worker is in charge of patches along one row.

        Because, my calculation also requires scaling, so the calculation is done in such a way.

        1. All the workers will calculate the diagonal patches. After that, they will save the
            diagonal values to disk. The other jobs will read the value each time they need to do the scaling.
        2. The workers will load patches along each row. Calculate the inner product and do the
            scaling and sorting and etc. Notice that, because we could have a lot of workers, really
            a lot of workers, therefore, each row can be quite flat and if we keep the  partitions
            along the two dimensions the same, we might not use all the memory. Therefore, I'll let the user to
            choose the partition number along each dimension.
        3. Later, I would try to provide some more information and perhaps functions to help the user
            to set the optimal value for the partition number along dim1.

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
        """
        The batch number is calculated in this way.
        
                    --------------------------
                    | 00 | 11 | 11 | 00 | 00 |
                    --------------------------
                    | 00 | 00 | 11 | 11 | 00 |
                    --------------------------
                    | 00 | 00 | 00 | 11 | 11 |
                    --------------------------
                    | 11 | 00 | 00 | 00 | 11 |
                    --------------------------
                    | 11 | 11 | 00 | 00 | 00 |
                    --------------------------
                    
        Previously, I planned to do something more sophisticated such as partitioning each row again from scratch.
        Then I just realized that I did not want that for 
            1. The implementation would be complicated and inefficient unless I come up with some brighter idea.
                However, I love more profound math than these concrete ones, so I prefer to spend time on stochastic
                analysis and operator algebras.
            2. After I used the restroom for a while, I realized that, the finer implementation might not provide
                much better performance than the current one. Therefore, I did not want to spend time on this.
            3. I have left enough comments everywhere in this project. Therefore, in the future, some summer interns
                or younger and smarter students might finish the finer implementation.
        
        """
        # The number of vertical batches along each line.
        batch_num_per_line = (int(batch_num_dim0) - 1) // 2
        # Number of batches to bin together.
        batch_num_per_bin = util.get_batch_num_list(batch_num_per_line, batch_num_dim1)
        """
        This number is used to specify the range of batches inside a specific bin
        """
        bin_ends = np.cumsum([0, ] + batch_num_per_bin)
        # batch index along each line
        batch_idx_per_line = util.get_batch_idx_per_list(batch_num=batch_num_dim0)

        for line_idx in range(batch_num_dim0):

            # The second layer of list is for different batches in this line.
            # Create a holder for batch bins in this line
            self.batch_ends_local_dim1.append([])

            for bin_idx in range(batch_num_dim1):
                # Get the batch indexes to merge
                idx_list = batch_idx_per_line[line_idx, bin_ends[bin_idx]:bin_ends[bin_idx + 1]]

                # The third layer of list is for this batch in this line.
                tmp = [[self.batch_ends_local_dim0[x] for x in idx_list],
                       idx_list]

                self.batch_ends_local_dim1[-1].append(tmp)

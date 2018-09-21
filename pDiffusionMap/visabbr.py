import numpy as np
import time
import holoviews as hv
from holoviews import streams
from holoviews.operation.datashader import datashade
from pDiffusionMap import DataSource, util


def get_background_sample_and_streams(data_source, eigens, dim0, dim1, length, sample_number, sampled_index):
    """
    Abbreviation of the code to get the background pattern and the streams to generate the manifold.

    :param data_source: The datasource object.
    :param eigens:The eigenvectors numpy arrays of the shape [number of eigenvectors, dimension of the eigenvectors]
    :param dim0: The eigenvector used as the first dimension coordinate
    :param dim1: The eigenvector used as the second dimension coordinate
    :param length: The length of the embedded manifold shown on the screen
    :param sample_number: The number of samples to extract.
    :param sampled_index: The index of the selected samples.
    :return: points_all: the hv.Points object for all the data points
             background: the embedded manifold rendered with datashader.
             sampled_points: The hv.Points object for the samples.
             sampled_positions: The position numpy array for the sampled patterns.
             check: The check stream
             select: The polygon object
             path_stream: The select stream
    """

    #########################################################
    # [Auto] Create holoviews object for all the data points
    #########################################################
    data_all_coor = np.zeros((data_source.data_num_total, 2))
    data_all_coor[:, 0] = eigens[dim0, :]
    data_all_coor[:, 1] = eigens[dim1, :]

    points_all = hv.Scatter((eigens[dim0, :],
                             eigens[dim1, :])).options(height=length, width=length)

    # Datashade all the points.
    background = datashade(points_all, dynamic=True)

    # Get the coordinate of the sampled points
    sampled_positions = np.zeros((sample_number, 2))
    sampled_positions[:, 0] = data_all_coor[sampled_index, 0]
    sampled_positions[:, 1] = data_all_coor[sampled_index, 1]

    # Create the holoviews for the sampled points 
    sampled_points = hv.Points(sampled_positions).options(height=length,
                                                          width=length,
                                                          tools=['box_select', ],
                                                          size=4,
                                                          color='red',
                                                          nonselection_alpha=1,
                                                          nonselection_color='yellow',
                                                          selection_color='red')

    #########################################################
    # [Auto]Define stream for the sampled points
    #########################################################
    check = hv.streams.Selection1D(source=sampled_points)

    #########################################################
    # [Auto] Define streams related to all the data points
    #########################################################
    select = hv.Polygons([]).options(line_width=5, line_color='green', line_alpha=1, fill_alpha=0.6)
    path_stream = hv.streams.PolyDraw(source=select)

    return points_all, background, sampled_points, sampled_positions, check, select, path_stream


def load_data_and_get_samples(input_txtfile, sample_number):
    """
    Load the data and get some samples

    :param input_txtfile: The txt file containing the input h5 files
    :param sample_number: The number of samples to extract.
    :return: data_source, pattern_shape, global_index_map, sampled_index, sampled_patterns
    """

    #########################################################
    # Load Data
    #########################################################
    # Create a data_source object to access the raw data in hdf5 files
    data_source = DataSource.DataSourceFromH5pyList(source_list_file=input_txtfile)
    pattern_shape = data_source.source_dict['shape']

    # Create the global index map
    global_index_map = util.get_global_index_map(data_num_total=data_source.data_num_total,
                                                 file_num=data_source.file_num,
                                                 data_num_per_file=data_source.data_num_per_file,
                                                 dataset_num_per_file=data_source.dataset_num_per_file,
                                                 data_num_per_dataset=data_source.data_num_per_dataset)

    #########################################################
    # Extract samples
    #########################################################
    # Extract some global index
    sampled_index = np.sort(np.random.permutation(data_source.data_num_total)[:sample_number])

    tic = time.time()
    # Preload the sampled patterns
    sampled_patterns = util.get_sampled_pattern_batch_efficient(global_index_array=sampled_index,
                                                                global_index_map=global_index_map,
                                                                data_dict=data_source.source_dict)
    toc = time.time()
    print("It takes {} seconds to sample {} patterns.".format(toc - tic, sample_number))

    return data_source, pattern_shape, global_index_map, sampled_index, sampled_patterns

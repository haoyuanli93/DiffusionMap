import numpy as np
import holoviews as hv
import matplotlib.path as mpltPath
import pandas as pd
from holoviews.operation.datashader import datashade, rasterize
import datashader as ds
import bokeh.palettes as bp
import h5py


def assemble_patterns(data_holder, row_num, col_num, index, value_range, pattern_shape):
    """
    After the program has obtained the index of the patterns in the selected region,
    this function randomly choose several of the patterns to show in a grid-space.

    :param data_holder: The holder containing all the data shown in the diagram
    :param row_num: The row number of the grid space
    :param col_num: The column number of the grid space
    :param index: The index of all the data in the selected region
    :param range: The range of values to show a numpy array as RGB image.
    :param pattern_shape: The pattern shape
    :return: hv.GridSpace
    """
    index = np.array(index)
    index_num = index.shape[0]
    if index_num >= row_num * col_num:
        np.random.shuffle(index)
        sampled_index = index[:row_num * col_num]
        sampled_index = sampled_index.reshape((row_num, col_num))

        image_holder = {(x, y): hv.Image(data_holder[sampled_index[x, y]]).redim.range(z=(value_range[0],
                                                                                          value_range[1]))
                        for x in range(row_num) for y in range(col_num)}
    else:
        # When we do not have so many patterns, first layout
        # all the patterns available and then fill the other
        # positions with patterns of zeros.
        index_list = [(x, y) for x in range(row_num) for y in range(col_num)]
        image_holder = {index_list[l]: hv.Image(data_holder[index[l]]).redim.range(z=(value_range[0],
                                                                                      value_range[1]))
                        for l in range(index_num)}
        image_holder.update({index_list[l]: hv.Image(np.zeros(pattern_shape,
                                                              dtype=np.float64))
                             for l in range(index_num, row_num * col_num)})

    return hv.GridSpace(image_holder)


def save_selected_region(stream_holder, data_holder, output='./selected_index.npy', return_selected_region=False):
    """
    Use this function to parse the stream and find the index of the points contained in the specified region.
    This function also returns the index of the selected

    :param stream_holder: The holoviews stream containing the contour of the path.
    :param data_holder: The numpy array containing all the positions of each point
                        [[x, y],
                         [x, y],
                         ... ]     This is should be of the shape [number of points, 2]

    :param output: The file name to save the index of points of the selected region
    :param return_selected_region: Choose whether to return the selected region for inspection
    :return:
    """
    # Extract the x and y coordinates of different points along the path
    x_coor = stream_holder.data['xs'][0]
    y_coor = stream_holder.data['ys'][0]
    path_holder = [(x_coor[l], y_coor[l]) for l in range(len(x_coor))]

    # Construct the matplotlib path object
    poly_path = mpltPath.Path(path_holder)

    # Use the poly_path object to find the index of the particles that are going to be saved
    decision = poly_path.contains_points(data_holder)

    # Get the index of the points in the region
    index = np.arange(decision.shape[0])[decision]

    # Save the selected index
    np.save(output, index)

    if return_selected_region is True:
        # return the selected index and the selected points
        return index, data_holder[index]


def get_eigenvectors(eigenvectors):
    """
    Load the eigenvector npy file or convert the numpy array containing all the eigenvectors to a pandas DataFrame
    :param eigenvectors: A string containing the npy file position or a numpy array containing the eigenvectors
                        The shape of the eigenvector is
                                    [number of eigenvectors, dimension of eigenvectors]
    :return: A pandas DataFrame of the following format.
                {eigvec_0: The first eigenvector,
                 eigvec_1: The second eigenvector,
                 ...
                 number: The number of eigenvectors.
                 dimension: The dimension of the eigenvectors}
    """
    if type(eigenvectors) is str:
        data_holder = np.load(eigenvectors)
    else:
        data_holder = eigenvectors

    dict_holder = {"eigvec_{}".format(l): data_holder[l] for l in range(data_holder.shape[0])}

    return pd.DataFrame(data=dict_holder)


def get_random_samples(eigvec_holder, dim0, dim1, sample_num):
    """
    Get a randomly sampled subset of the total dataset containing the corresponding embedded dimensions
    and the indexes.

    :param eigvec_holder: The dataframe object containing all the embedded coordinates of all the data points.
    :param dim0: The name of the first dimension to show
    :param dim1: The name of the second dimension to show
    :param sample_num: The number of samples to extract.
    :return: A dataframe object containing
            {'index': index,
             'x': x coordinate,
             'y': y coordinate
    """

    index = np.random.permutation(eigvec_holder[dim0].shape[0])[:sample_num]

    dict_holder = {'idx': index}
    dict_holder.update({"x": np.array(eigvec_holder[dim0][index]),
                        "y": np.array(eigvec_holder[dim1][index])})

    return pd.DataFrame(dict_holder)


def construct_dataframe(dim0, dim1, eigensystem, variance, mean, attribute, category):
    """
    Load the eigensystem and some other useful stuff and then construct some
    dataframe for visualization
    :param dim0: The index of the eigenvector to be used as the x-coordinate
    :param dim1: The index of the eigenvector to be used as the y-coordinate
    :param eigensystem: The h5file containing the eigensystem to be visualized.
    :param variance: The address to the numpy file containing the variance for each pattern.
    :param mean: The address to the numpy file containing the mean for each pattern
    :param attribute: The address to the numpy file containing the interesting attribute
    :param category: The address to the numpy file containing the classification based on the attribute.
    :return: dataframe, dataframe sorted along dim0, dataframe sorted along dim1
    """
    data_dict = {}
    with h5py.File(eigensystem) as h5file:
        data_dict.update({"x": np.array(h5file['eigenvectors'][dim0]),
                          "y": np.array(h5file['eigenvectors'][dim1])})
    data_dict.update({"variance": np.load(variance),
                      "mean": np.load(mean),
                      "attribute": np.load(attribute),
                      "category": np.load(category)})

    # Construct a dataframe object
    dataframe = pd.DataFrame(data_dict)

    # Sort along x axis
    index = np.argsort(data_dict['x'])
    # Create a new datafrme
    dataframe_sort_along_x = pd.DataFrame({key: data_dict[key][index] for key in list(data_dict.keys())})

    # Sort along y axis
    index = np.argsort(data_dict['y'])
    dataframe_sort_along_y = pd.DataFrame({key: data_dict[key][index] for key in list(data_dict.keys())})

    return dataframe, dataframe_sort_along_x, dataframe_sort_along_y


def show_manifold_and_attribute(dataframe, dataframex, dataframey):
    """
    Show the manifold with some adjoint diagrams showing the distribution of the attribute
    :param dataframe: The raw dataframe.
    :param dataframex: The dataframe sorted along dimension x
    :param dataframey: The dataframe sorted along dimension y
    :return: The manifold and the distribution of the attribute along the two axis
    """
    density_x = hv.Curve(dataframex, kdims=['x', 'attribute'])
    density_y = hv.Curve(dataframey, kdims=['y', 'attribute'])

    density_curve_x = hv.operation.timeseries.rolling(density_x, rolling_window=50)
    density_curve_y = hv.operation.timeseries.rolling(density_y, rolling_window=50)

    density = hv.Points(data=dataframe,
                        kdims=["x", "y"],
                        vdims=['attribute', 'category']).options(color_index="attribute",
                                                                 width=600, height=400, colorbar=True,
                                                                 colorbar_position='left')
    return (density << density_curve_y.options(width=200,
                                               height=400) << density_curve_x.options(width=400, height=200))


def show_manifold_and_category(dataframe, dataframex, dataframey):
    """
    Show the manifold with some adjoint diagrams showing the distribution of the category
    :param dataframe: The raw dataframe.
    :param dataframex: The dataframe sorted along dimension x
    :param dataframey: The dataframe sorted along dimension y
    :return: The manifold and the distribution of the category along the two axis
    """
    density_x = hv.Curve(dataframex, kdims=['x', 'category'])
    density_y = hv.Curve(dataframey, kdims=['y', 'category'])

    density_curve_x = hv.operation.timeseries.rolling(density_x, rolling_window=50)
    density_curve_y = hv.operation.timeseries.rolling(density_y, rolling_window=50)

    density = hv.Points(data=dataframe,
                        kdims=["x", "y"],
                        vdims=['attribute', 'category']).options(color_index="category",
                                                                 width=600, height=400, colorbar=True,
                                                                 colorbar_position='left')
    return (density << density_curve_y.options(width=200,
                                               height=400) << density_curve_x.options(width=400, height=200))


def datashade_manifold_and_attribute(dataframe, dataframex, dataframey):
    """
    Show the manifold with some adjoint diagrams showing the distribution of the attribute
    :param dataframe: The raw dataframe.
    :param dataframex: The dataframe sorted along dimension x
    :param dataframey: The dataframe sorted along dimension y
    :return: The manifold and the distribution of the attribute along the two axis
    """
    density_x = hv.Curve(dataframex, kdims=['x', 'attribute'])
    density_y = hv.Curve(dataframey, kdims=['y', 'attribute'])

    density_curve_x = hv.operation.timeseries.rolling(density_x, rolling_window=50)
    density_curve_y = hv.operation.timeseries.rolling(density_y, rolling_window=50)

    density = hv.Points(data=dataframe,
                        kdims=["x", "y"],
                        vdims=['attribute', 'category'])

    shade = rasterize(density, aggregator=ds.mean('attribute')).options(width=600,
                                                                        height=400,
                                                                        colorbar=True,
                                                                        colorbar_position='left')

    return (shade << density_curve_y.options(width=200,
                                             height=400) << density_curve_x.options(width=400, height=200))


def datashade_manifold_and_category(dataframe, dataframex, dataframey):
    """
    Show the manifold with some adjoint diagrams showing the distribution of the category
    :param dataframe: The raw dataframe.
    :param dataframex: The dataframe sorted along dimension x
    :param dataframey: The dataframe sorted along dimension y
    :return: The manifold and the distribution of the category along the two axis
    """
    density_x = hv.Curve(dataframex, kdims=['x', 'category'])
    density_y = hv.Curve(dataframey, kdims=['y', 'attribute'])

    density_curve_x = hv.operation.timeseries.rolling(density_x, rolling_window=50)
    density_curve_y = hv.operation.timeseries.rolling(density_y, rolling_window=50)

    density = hv.Points(data=dataframe,
                        kdims=["x", "y"],
                        vdims=['attribute', 'category'])

    shade = rasterize(density, aggregator=ds.mean('category')).options(width=600,
                                                                       height=400,
                                                                       colorbar=True,
                                                                       colorbar_position='left')

    return (shade << density_curve_y.options(width=200,
                                             height=400) << density_curve_x.options(width=400, height=200))

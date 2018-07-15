import numpy as np
import holoviews as hv


def assemble_patterns(data_holder, row_num, col_num, index, pattern_shape):
    """
    After the program has obtained the index of the patterns in the selected region,
    this function randomly choose several of the patterns to show in a grid-space.

    :param data_holder: The holder containing all the data shown in the diagram
    :param row_num: The row number of the grid space
    :param col_num: The column number of the grid space
    :param index: The index of all the data in the selected region
    :param pattern_shape: The pattern shape
    :return: hv.GridSpace
    """
    index = np.array(index)
    index_num = index.shape[0]
    if index_num >= row_num * col_num:
        np.random.shuffle(index)
        sampled_index = index[:row_num * col_num]
        sampled_index = sampled_index.reshape((row_num, col_num))

        image_holder = {(x, y): hv.Image(data_holder[sampled_index[x, y]], label="Sample patterns")
                        for x in range(row_num) for y in range(col_num)}
    else:
        # When we do not have so many patterns, first layout
        # all the patterns available and then fill the other
        # positions with patterns of zeros.
        index_list = [(x, y) for x in range(row_num) for y in range(col_num)]
        image_holder = {index_list[l]: hv.Image(data_holder[index[l]], label="Sample patterns")
                        for l in range(index_num)}
        image_holder.update({index_list[l]: hv.Image(np.zeros(pattern_shape, dtype=np.float64), label="Sample patterns")
                             for l in range(index_num, row_num * col_num)})

    return hv.GridSpace(image_holder)

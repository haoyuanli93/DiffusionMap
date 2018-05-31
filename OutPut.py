"""
Save important intermediate results, on the one hand, it can be used for later on process,
on the other hand, it can be used for debugging.
"""

import numpy as np
import os
import h5py as h5


def save_distances(output_path, distance_patch, patch_position):
    """
    Save the distance patch to a numpy array.

    :param output_path: The folder to save the distance patch
    :param patch_position: the position of the patch in the large matrix. eg
                patch (0,0) | patch (0,1)
                --------------------------
                patch (1,0) | patch (1,1)

            The first number is the position along dimension zero. The second
            number is the position along dimension one.
    :param distance_patch: the distance patch to save
    """

    # load the data_source address
    address = output_path

    # check if the folder exist
    if not os.path.isdir(address + '/distances'):
        os.makedirs(address + '/distances')

    name = address + "/distances/patch_{}_{}.npy".format(patch_position[0], patch_position[1])
    np.save(name, distance_patch)


def assemble(data_source):
    """
    Assemble different patches of distance matrices into a single distance matrix
    :param data_source: An instance of data_source class.
    :return: The 2D distance matrix. Notice that this matrix is up-triangular.
    """

    folder_address = data_source.output_path + '/distances'

    batch_num = data_source.batch_number
    total_num = data_source.pattern_number_total

    # First check if the address is a folder
    if not os.path.isdir(folder_address):
        raise Exception("The target specified by the folder_address is not a folder.\n" +
                        "Please make sure the folder_address is the folder where you have \n" +
                        "saved the distance patches.")

    # Second check whether all the patches are present in the folder
    flag = True  # True if all the patches exist.
    for l in range(batch_num):
        for m in range(l, batch_num):
            patch_file = folder_address + "/patch_{}_{}.npy".format(l, m)
            if not os.path.isfile(patch_file):
                flag *= False

    if not flag:
        raise Exception("The patches are not complete. \n" +
                        "Please check if you have obtained " +
                        "all {} patterns required to get a complete distance matrix.".format(
                            batch_num * (batch_num + 1) // 2))

    """
    If we have all the files, we can assemble the distance matrix.
    start_dim_0 record the starting point of the patch along axis 0.
    start_dim_1 record the starting point of the patch along axis 1.
    """
    holder = np.zeros((total_num, total_num))

    start_dim_0 = 0
    for l in range(batch_num):
        # Notice that the first dimension increases slower.
        end_dim_0 = start_dim_0 + data_source.batch_size_list[l]

        start_dim_1 = start_dim_0
        for m in range(l, batch_num):

            patch_file = folder_address + "/patch_{}_{}.npy".format(l, m)

            # load patch
            patch_holder = np.load(patch_file)
            # Check if the dimension is correct
            if patch_holder.shape[0] != data_source.batch_size_list[l] or patch_holder.shape[1] != \
                    data_source.batch_size_list[m]:

                print(patch_holder.shape, data_source.batch_size_list[l], data_source.batch_size_list[m])
                raise Exception("The size of the ({},{}) patch does not match that in the record.\n"
                                "Please check if the distance patch is correct.".format(l, m))
            else:

                end_dim_1 = start_dim_1 + data_source.batch_size_list[m]
                holder[start_dim_0:end_dim_0, start_dim_1:end_dim_1] = patch_holder
                start_dim_1 = end_dim_1

        # update the starting position of the outer loop.
        start_dim_0 = end_dim_0

    return holder


def save_variances(data_source, variances, batch_index):
    """
    Save variances by batch.

    :param data_source: The data_source class that contains the raw patterns.
    :param batch_index: The batch index of to process.
    :param variances: The variances to save.
    """

    # load the data_source address
    address = data_source.output_path

    # check if the folder exist
    if not os.path.isdir(address + '/variances'):
        os.makedirs(address + '/variances')

    with h5.File(address + '/variances/batch_{}.h5'.format(batch_index), 'w') as h5file:
        h5file.create_dataset('{}'.format(batch_index), data=variances, chunks=True)

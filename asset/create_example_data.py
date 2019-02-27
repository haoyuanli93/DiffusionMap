# Load packages
import numpy as np
from scipy import ndimage
import h5py as h5
import os

file_num = 5  # Number of h5files to generate
batch_num = 2  # Number of datasets in each h5file. Each dataset will contain 100 images of shape 256*256
print("{} h5 files will be generated. Each will contain {} datasets".format(file_num, batch_num))
print("There will be 100 images in each dataset. Each dataset is of the shape (100,256,256)")


# Define a useful function
def shift_to_center(array):
    """
    Shift the image so that the center of mass of this image is the same as the geometric center
    of the image.
    :param array:
    :return: The shifted array
    """
    # Get array information
    shape = array.shape
    dim = len(shape)
    center = np.array([x / 2.0 for x in shape])

    # Get the position of the center of mass
    center_mass = ndimage.center_of_mass(array)

    # Shift the pattern
    shifted_array = ndimage.shift(array, shift=[center[l] - center_mass[l] for l in range(dim)])

    return shifted_array


# Load data
print("Load the example array in the file {}".format("../input/example_array.npy"))
sample = np.load("../input/example_array.npy")

# Padding the image
final_sample_holder = np.zeros((256, 256))
final_sample_holder[:sample.shape[0], :sample.shape[1]] += sample

# Shift the image to the center
final_sample = shift_to_center(final_sample_holder)

# Create the rotated image and save them to the corresponding h5 file

# Generate random angles
random_angles = np.linspace(start=0, stop=355, num=file_num * batch_num * 100, endpoint=True)

# Generate a category array
category_array = np.zeros_like(random_angles)
for l in range(5):
    category_array[l * 20 * file_num * batch_num:(l + 1) * 20 * file_num * batch_num] = l

np.save("../input/examples_angles.npy", arr=random_angles)
np.save("../input/examples_category.npy", arr=category_array)

print("Generate {} random angles".format(np.prod(random_angles.shape)))
print("These random values are saved to the file {}".format("../input/random_angles.npy"))

# Change the shape of the random angle array
random_angles = random_angles.reshape((file_num, batch_num, 100))

# Loop through all the angles
for l in range(5):

    with h5.File('../input/example_data_file_{}.h5'.format(l), 'w') as h5file:

        for m in range(2):

            # First create a dataset
            angle_holder = random_angles[l, m]
            dataset_holder = np.zeros((100, 256, 256))

            # Loop through all the angles
            for n in range(100):
                dataset_holder[n] = ndimage.rotate(input=final_sample,
                                                   angle=angle_holder[n],
                                                   reshape=False)

            h5file.create_dataset("Dataset_{}".format(m), data=dataset_holder)

        print("A new h5 file containing example datasets in created in {}".format(
            '../input/example_data_file_{}.h5'.format(l)))

###############################################
# Update the input_file_list.txt in the input folder
###############################################
print("Update the input_file_list.txt")
# Create useful files for the user
with open('../input/file_list.txt', 'w') as txtfile:
    txtfile.write("##################################################################\n")
    txtfile.write("# Dear users:\n")
    txtfile.write("# This is a file aims to help you to specify which dataset to process.\n")
    txtfile.write("# This txt file, or any txt file obeying the following roles can\n")
    txtfile.write("# be understood by the program.\n")
    txtfile.write(
        "# 1. The program ignore any lines that is empty, starts with # or blank space. \n")
    txtfile.write("# 2. The program can not handle addresses containing any blank spaces.\n")
    txtfile.write("#    Please pay attention to this.\n")
    txtfile.write("# 3. The h5py file address, preferably the absolute address, should \n")
    txtfile.write("#    follow \"File:the_address_no_blank_space\"\n")
    txtfile.write("# 4. The dataset in the hdf5 file to process is listed right below\n")
    txtfile.write("#    the corresponding hdf5 file line following \"Dataset:\". Each \n")
    txtfile.write("#    dataset should occupy a whole line.\n")
    txtfile.write("# 5. If one does not specify which dataset to process, the default\n")
    txtfile.write("#    action is to process all the datasets contained in this hdf5 file.\n")
    txtfile.write("# 6. The program detects if there are any duplicated files. Please\n")
    txtfile.write("#    make sure that all the h5py files are unique.\n")
    txtfile.write("# 7. The program does not detect if the datasets are unique, therefore\n")
    txtfile.write("#    please pay extra caution to this since this can incur errors\n")
    txtfile.write("#    that are extremely difficult to trace down.\n")
    txtfile.write("# \n")
    txtfile.write("# Best wishes.\n")
    txtfile.write("# Haoyuan Li\n")
    txtfile.write("# 7/17/2018 \n")
    txtfile.write("# \n")
    txtfile.write("#-----------------------------------------------------------------\n")
    txtfile.write("#             *This is an example*\n")
    txtfile.write("#-----------------------------------------------------------------\n")
    txtfile.write("#File:/reg/d/psdm/amo/amo86615/res/haoyuan/singles.h5\n")
    txtfile.write("#Dataset:patterns_one\n")
    txtfile.write("#Dataset:patterns_two\n")
    txtfile.write("#File:/reg/d/psdm/amo/amo86615/res/haoyuan/multiples.py\n")
    txtfile.write("##################################################################\n")
    for l in range(file_num):
        txtfile.write(
            "File:{}\n".format(os.path.abspath('../input/example_data_file_{}.h5'.format(l))))

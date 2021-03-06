""""
This script creates three sets of data. One is a disk, the other is a square.
Each set of data is composed of 500 patterns. They are  generated by add noises to
the original pattern.
"""

import numpy as np
import h5py as h5
import argparse

# Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('batch_num', type=int, help="batches.")
parser.add_argument('address', type=str, help="Specify where to save.")

"""
Generate a h5 file to hold the data
"""
args = parser.parse_args()
num = args.batch_num
address = args.address

with h5.File(address, 'w') as h5file:
    for r in range(num):
        """
        Generate patterns for square
        """
        original_square = np.zeros((128, 128))
        original_square[45:72, 45:72] = 40

        square_stack = np.zeros((51, 128, 128))
        for l in range(51):
            square_stack[l] = original_square + np.random.rand(128, 128) * 150

        """
        Generate patterns for circle
        """
        original_disk = np.zeros((128, 128))
        for l in range(128):
            for m in range(128):
                if (l - 63.5) ** 2 + (m - 63.5) ** 2 <= 400:
                    original_disk[l, m] = 68

        disk_stack = np.zeros((71, 128, 128))
        for l in range(71):
            disk_stack[l] = original_disk + np.random.rand(128, 128) * 150

        h5file.create_dataset(str(2 * r), data=square_stack)
        h5file.create_dataset(str(2 * r + 1), data=disk_stack)


import fileinput
import sys

# Get the absolute path of this file
path = sys.path[0]

# Use fileinput to initialize the path value in scripts.
for line in fileinput.input(('WeightMat.py', 'EigensSlepc.py', 'Visualization_Example.ipynb'), inplace="True"):
    line.rstrip().replace('This is a path_holder. Please use setup.py to initialize this value.', path)

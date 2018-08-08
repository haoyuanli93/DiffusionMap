import fileinput
import sys

# Get the absolute path of this file
path = sys.path[0]

# Define the holder for the path
holder = 'This is a path_holder. Please use setup.py to initialize this value.'

# Use fileinput to initialize the path value in scripts.
for line in fileinput.input(('WeightMat.py',
                             'EigensSlepc.py',
                             'Manifold_Browser1.ipynb',
                             'Manifold_Browser2.ipynb'), inplace="True"):
    if holder in line:
        line = line.replace(holder, path)
    sys.stdout.write(line)

# Standard modules
import argparse
import numpy

# Parse the parameters
parser = argparse.ArgumentParser()

parser.add_argument('--input_folder', type=str, help="Specify the folder containing the eigenvectors.")
parser.add_argument("--node_num", type=int, help="Specify how many pieces is related to a single eigenvector.")
parser.add_argument("--eig_num", type=int, help="Specify the number of eigenvectors to assemble.")

# Parse
args = parser.parse_args()
input_folder = args.input_folder
node_num = args.node_num
eig_num = args.eig_num

# The first is for different eigenvectors
for eig_idx in range(eig_num):
    # This is the holder for all the numpy array
    holder = []
    for node_idx in range(node_num):
        holder.append(numpy.load(input_folder + "/Eigenvec_{}_{}.npy".format(eig_idx, node_idx)))

    # Assemble all the pieces
    assembled_eigenvector = numpy.concatenate(tuple(holder))

    # For the first iteration, create a holder for all the eigenvectors
    if eig_idx == 0:
        total_holder = numpy.zeros((eig_num, assembled_eigenvector.shape[0]))

    total_holder[eig_idx, :] = assembled_eigenvector

# Save all the assembled eigenvectors
numpy.save(input_folder + "/Eigenvectors_all.npy", total_holder)


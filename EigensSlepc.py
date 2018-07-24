import sys, numpy as np
sys.path.append('/reg/neh/home5/haoyuan/Documents/my_repos/DiffusionMap')

from mpi4py import MPI
import time, scipy.sparse, numpy
from petsc4py import PETSc
from slepc4py import SLEPc

try:
    import Config
except ImportError:
    raise Exception("This package use Config.py file to set parameters. Please use the start_a_new_project.py "
                    "script to get a folder \'proj_****\'. Move this folder to a desirable address and modify"
                    "the Config.py file in the folder \'proj_****/src\' and execute DiffusionMap calculation"
                    "in this folder.")

# Parse
sparse_matrix_npz = Config.CONFIGURATIONS["sparse_matrix_npz"]
neighbor_number = Config.CONFIGURATIONS["neighbor_number"]
eig_num = Config.CONFIGURATIONS["eig_num"]
output_folder = Config.CONFIGURATIONS["output_folder"]

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

"""
Step One: Load the sparse matrix and get some information
"""
if comm_rank == 0:
    print("Begin loading the data", flush=True)
    tic = time.time()
comm.Barrier()  # Synchronize

# Load the matrix
csr_matrix = scipy.sparse.load_npz(sparse_matrix_npz)
mat_size = csr_matrix.shape

"""
Step Two: Initialize the petsc matrix
"""
petsc_mat = PETSc.Mat()
petsc_mat.create(PETSc.COMM_WORLD)

petsc_mat.setSizes(mat_size)
petsc_mat.setType('aij')  # sparse
petsc_mat.setPreallocationNNZ(neighbor_number)
petsc_mat.setUp()
rstart, rend = petsc_mat.getOwnershipRange()
print(rstart, rend)

p1 = csr_matrix.indptr
p2 = csr_matrix.indices
p3 = csr_matrix.data

petsc_mat.createAIJ(size=mat_size,
                    csr=(p1[rstart:rend + 1] - p1[rstart],
                         p2[p1[rstart]:p1[rend]],
                         p3[p1[rstart]:p1[rend]]))
petsc_mat.assemble()

"""
Step Three: Solve for the eigenvalues and eigenvectors
"""

Print = PETSc.Sys.Print
xr, xi = petsc_mat.createVecs()

# Setup the eigensolver
E = SLEPc.EPS().create()
E.setOperators(petsc_mat, None)
E.setDimensions(nev=eig_num, ncv=PETSc.DECIDE)
E.setProblemType(SLEPc.EPS.ProblemType.HEP)
E.setFromOptions()

# Solve the eigensystem
E.solve()

"""
Step Four: Inspect the result
"""

# Inspect the result and save the results
eigen_values = []
local_eigenvector_holder = numpy.zeros((eig_num, rend - rstart))

# Show some calculation information
Print("")
its = E.getIterationNumber()
Print("Number of iterations of the method: %i" % its)
sol_type = E.getType()
Print("Solution method: %s" % sol_type)
nev, ncv, mpd = E.getDimensions()
Print("Number of requested eigenvalues: %i" % nev)
tol, maxit = E.getTolerances()
Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
nconv = E.getConverged()
Print("Number of converged eigenpairs: %d" % nconv)
if not (nconv > 0):
    raise Exception(" The weight matrix is too singular, no converged eigen-pair is obtained.")

# Show the error and collect the eigen-pairs
Print("")
Print("        k          ||Ax-kx||/||kx|| ")
Print("----------------- ------------------")
for i in range(eig_num):
    k = E.getEigenpair(i, xr, xi)
    error = E.computeError(i)
    Print(" %12f       %12g" % (k.real, error))

    # Obtain the eigenvalue
    eigen_values.append(k.real)

    # Obtain the eigenvector
    local_eigenvector = xr.getArray()
    local_eigenvector_holder[i, :] = local_eigenvector

Print("")
comm.Barrier()  # Synchronize

# All the node send the eigenvector holder to the first node
eigenvector_pieces = comm.gather(local_eigenvector_holder, root=0)

if comm_rank == 0:
    # Save the eigenvalues
    vals = numpy.asarray(eigen_values)
    numpy.save(output_folder + "/Eigenvalues.npy", vals)

    # Assemble the eigenvectors and save them
    eigenvectors = numpy.concatenate(eigenvector_pieces, axis=1)
    numpy.save(output_folder + "/Eigenvectors.npy", eigenvectors)

    # Finishes everything.
    print("Finishes all calculation.", flush=True)
    toc = time.time()
    print("The total calculation time is {}".format(toc - tic), flush=True)

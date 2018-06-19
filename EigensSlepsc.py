# Standard modules
from mpi4py import MPI
import numpy
import argparse
import time
import scipy.sparse

from petsc4py import PETSc
from slepc4py import SLEPc

# Parse the parameters
parser = argparse.ArgumentParser()

parser.add_argument('--output_folder', type=str, help="Specify the folder to put the calculated data.")
parser.add_argument("--sparse_matrix_npz", type=str, help="Specify the npz file containing the sparse matrix.")
parser.add_argument("--neighbor_number", type=int, help="Specify the number of neighbors.")
parser.add_argument("--eig_num", type=int, help="Specify the number of eigenvectors to calculate.")

# Parse
args = parser.parse_args()
sparse_matrix_npz = args.sparse_matrix_npz
output_folder = args.output_folder
neighbor_number = args.neighbor_number
eig_num = args.eig_num

"""
Step One: Load the sparse matrix and get some information
"""
csr_matrix = scipy.sparse.load_npz(sparse_matrix_npz)
mat_size = csr_matrix.shape

"""
Step One: Initialize the petsc matrix
"""
petsc_mat = PETSc.Mat()
petsc_mat.create(PETSc.COMM_WORLD)

petsc_mat.setSizes(mat_size)
petsc_mat.setType('aij')  # sparse
petsc_mat.setPreallocationNNZ(neighbor_number)
petsc_mat.setUp()

if comm_rank == 0:
    # First extract the index and values for initialization
    p1 = csr_matrix.indptr
    p2 = csr_matrix.indices
    p3 = csr_matrix.data

    petsc_mat.createAIJ(size=mat_size,
                        csr=(p1, p2, p3), comm=PETSc.COMM_WORLD)
    print("Process {} finishes initialing the matrix.".format(comm_rank))
petsc_mat.assemble()

"""
Step Seven: Solve for the eigenvalues and eigenvectors
"""

Print = PETSc.Sys.Print
xr, xi = petsc_mat.createVecs()

# Setup the eigensolver
E = SLEPc.EPS().create()
E.setOperators(petsc_mat, None)
E.setDimensions(nev=eig_num, ncv=PETSc.DECIDE)
E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
E.setFromOptions()

# Solve the eigensystem
E.solve()

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
if nconv > 0:
    Print("")
    Print("        k          ||Ax-kx||/||kx|| ")
    Print("----------------- ------------------")
    for i in range(nconv):
        k = E.getEigenpair(i, xr, xi)
        error = E.computeError(i)
        if k.imag != 0.0:
            Print(" %9f%+9f j  %12g" % (k.real, k.imag, error))
        else:
            Print(" %12f       %12g" % (k.real, error))
    Print("")

if comm_rank == 0:
    toc_0 = MPI.Wtime()

    print("Finishes all the calculation.")
    print("The total calculation time is {}".format(toc_0 - tic_0))

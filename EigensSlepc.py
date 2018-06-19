# Standard modules
import argparse
import time
import scipy.sparse
import numpy

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
print("Begin loading the data", flush=True)
tic = time.time()

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
rstart, rend = petsc_mat.getOwnershipRange()

p1 = csr_matrix.indptr
p2 = csr_matrix.indices
p3 = csr_matrix.data

petsc_mat.createAIJ(size=csr_matrix.shape,
                    csr=(p1[rstart:rend + 1] - p1[rstart],
                         p2[p1[rstart]:p1[rend]],
                         p3[p1[rstart]:p1[rend]]), comm=PETSc.COMM_WORLD)
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
E.setProblemType(SLEPc.EPS.ProblemType.HEP)
E.setFromOptions()

# Solve the eigensystem
E.solve()

# Inspect the result and save the results
vals = []
vecs = []

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

        # Obtain the result
        vals.append(k)
        vecs = [complex(xr0, xi0) for xr0, xi0 in zip(xr.getArray(),
                                                      xi.getArray())]
        if k.imag != 0.0:
            Print(" %9f%+9f j  %12g" % (k.real, k.imag, error))
        else:
            Print(" %12f       %12g" % (k.real, error))
    Print("")
    vals = numpy.asarray(vals)
    vecs = numpy.asarray(vecs).T

    # Save the result
    numpy.save(output_folder + "/Eigenvec.npy", vecs)
    numpy.save(output_folder + "/Eigenval.npy", vals)

print("Finishes all calculation.", flush=True)
toc = time.time()
print("The total calculation time is {}".format(toc - tic), flush=True)

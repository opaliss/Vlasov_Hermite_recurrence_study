import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import sympy
import numpy as np
import pickle


# loop over velocity resolutions
for Nv in np.arange(4, 22, 2):
    # symbolic variables
    xi = sympy.symbols('xi')
    # must be an integer from definition
    k = sympy.symbols('k', integer=True)

    # advection matrix (off-diagonal)
    vec = sympy.zeros(Nv)
    for jj in range(1, Nv + 1):
        vec[jj - 1] = sympy.sqrt(jj) / (sympy.sqrt(2))

    # create an advection tri-diagonal matrix
    A = sympy.banded({1: tuple(vec[0, :-1]), -1: tuple(vec[0, :-1])})

    # identity matrix
    I = sympy.eye(Nv)

    # invert matrix
    M = sympy.SparseMatrix(I * xi - k / np.abs(k) * A)

    # get final response function
    R_approx = sympy.simplify(sympy.simplify(M.inv()[0, 1] / sympy.sqrt(2) * k / np.abs(k)))

    # save optimal R(nu*) (for k=1)
    with open("R_AW/R_" + str(Nv) + ".txt", "wb") as outf:
        pickle.dump(sympy.simplify(R_approx), outf)

    print("completed collisionless + truncation")
    print("Nv = ", Nv)

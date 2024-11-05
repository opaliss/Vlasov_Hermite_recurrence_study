import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import sympy
import numpy as np
import pickle

# setup the number of Hermite moments

for Nv in np.arange(20, 30, 2):
    # initialize the symbolic variables
    xi = sympy.symbols('xi')
    k = sympy.symbols('k', integer=True)
    c = sympy.symbols('c', complex=True)

    # create advection matrix A
    vec = sympy.zeros(Nv)
    for jj in range(1, Nv + 1):
        vec[jj - 1] = sympy.sqrt(jj) / (sympy.sqrt(2))

    A = sympy.banded({1: tuple(vec[0, :-1]), -1: tuple(vec[0, :-1])})

    A[-1, Nv - 1] += sympy.I * c * sympy.sqrt(Nv) / sympy.sqrt(2) * k / np.abs(k)

    # identity matrix
    I = sympy.eye(Nv, dtype=int)

    # eigenvalue matrix
    M = sympy.SparseMatrix(I * xi - k / np.abs(k) * A)

    # inversion
    R_approx = sympy.simplify(sympy.simplify(M.inv()[0, 1] / sympy.sqrt(2) * k / np.abs(k)))

    # adiabatic limit matching
    asymptotics_0 = R_approx.series(xi, 0, 4)
    sol_coeff = sympy.solve(asymptotics_0.coeff(xi, 1) + sympy.I*sympy.sqrt(sympy.pi), c)


    # save optimal (c)
    with open("optimal_q1_HP/coeff_" + str(Nv) + ".txt", "wb") as outf:
        pickle.dump(sol_coeff[0], outf)


    # save optimal R(c)
    with open("optimal_R_HP_q1/R_" + str(Nv) + ".txt", "wb") as outf:
        pickle.dump(sympy.simplify(R_approx.subs(c, sol_coeff[0])), outf)
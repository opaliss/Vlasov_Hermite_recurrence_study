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
    b = sympy.symbols('b', complex=True)
    a = sympy.symbols('a', complex=True)
    c = sympy.symbols('c', complex=True)

    # create advection matrix A
    vec = sympy.zeros(Nv)
    for jj in range(1, Nv + 1):
        vec[jj - 1] = sympy.sqrt(jj) / (sympy.sqrt(2))

    A = sympy.banded({1: tuple(vec[0, :-1]), -1: tuple(vec[0, :-1])})

    A[-1, Nv - 3] += sympy.I * a * sympy.sqrt(Nv) / sympy.sqrt(2) * k / np.abs(k)
    A[-1, Nv - 2] += sympy.I * b * sympy.sqrt(Nv) / sympy.sqrt(2) * k / np.abs(k)
    A[-1, Nv - 1] += sympy.I * c * sympy.sqrt(Nv) / sympy.sqrt(2) * k / np.abs(k)

    # identity matrix
    I = sympy.eye(Nv, dtype=int)

    # eigenvalue matrix
    M = sympy.SparseMatrix(I * xi - k / np.abs(k) * A)

    # inversion
    R_approx = sympy.simplify(sympy.simplify(M.inv(method="LDL")[0, 1] / sympy.sqrt(2) * k / np.abs(k)))
    print("I successfully inverted the matrix! ")

    # adiabatic limit matching
    asymptotics_0 = R_approx.series(xi, 0, 4)
    print("zeroth order is " + str(asymptotics_0.coeff(xi, 0)))

    sol_coeff = sympy.solve([asymptotics_0.coeff(xi, 0) + 1,
                             asymptotics_0.coeff(xi, 1) + sympy.I*sympy.sqrt(sympy.pi),
                             asymptotics_0.coeff(xi, 2) -2], [a, b, c])

    # save optimal (a, b, c)
    with open("optimal_q2_HP/coeff_" + str(Nv) + ".txt", "wb") as outf:
        pickle.dump(sol_coeff[0].evalf(), outf)

    # save optimal R(a, b, c)
    with open("optimal_R_HP_q2/R_" + str(Nv) + ".txt", "wb") as outf:
        pickle.dump(sympy.simplify(R_approx.subs([(a, sol_coeff[0][0]), (b, sol_coeff[0][1]), (c, sol_coeff[0][2])])),
                    outf)

    print("completed HP q2")
    print("Nv = ", Nv)
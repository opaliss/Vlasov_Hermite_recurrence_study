import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import sympy
import numpy as np
import pickle

# setup the number of Hermite moments

for Nv in np.arange(12, 14, 2):
    print("Nv = ", Nv)
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
    I = sympy.ImmutableSparseMatrix(sympy.eye(Nv))

    # eigenvalue matrix
    M = sympy.SparseMatrix(I * xi - k / np.abs(k) * A)

    # inversion
    cofactor = M.cofactor(0, 1)
    determinant = M.det(method="berkowitz")
    R_approx = sympy.simplify(cofactor / determinant / sympy.sqrt(2) * k / np.abs(k))
    #R_approx = sympy.simplify(sympy.simplify(M.inv(method="LDL")[0, 1] / sympy.sqrt(2) * k / np.abs(k)))
    print("I successfully inverted the matrix! ")

    # adiabatic limit matching
    asymptotics_0 = R_approx.series(xi, 0, 3)
    print("zeroth order is " + str(sympy.simplify(asymptotics_0.coeff(xi, 0))))
    print("first order is " + str(sympy.simplify(asymptotics_0.coeff(xi, 1))))
    print("second order is " + str(sympy.simplify(asymptotics_0.coeff(xi, 2))))

    sol_coeff = sympy.solve(asymptotics_0.coeff(xi, 1) + sympy.I*sympy.sqrt(sympy.pi), c)
    print("sol coeff = ", complex(sol_coeff[0]))

    # save optimal (c)
    with open("optimal_q1_HP/coeff_" + str(Nv) + ".txt", "wb") as outf:
        pickle.dump(complex(sol_coeff[0]), outf)


    # save optimal R(c)
    with open("optimal_R_HP_q1/R_" + str(Nv) + ".txt", "wb") as outf:
        pickle.dump(sympy.simplify(R_approx.subs(c, sol_coeff[0])), outf)

    print(sol_coeff)

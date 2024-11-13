import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import sympy
import numpy as np
import pickle
import scipy


# loop over velocity resolutions
for Nv in np.arange(4, 22, 2):
    # symbolic variables
    xi = sympy.symbols('xi')
    # must be an integer from definition
    k = sympy.symbols('k', integer=True)
    v0 = sympy.symbols('v0')

    # advection matrix (off-diagonal)
    vec = sympy.zeros(Nv)
    vec2 = sympy.zeros(Nv)
    for jj in range(1, Nv + 1):
        vec[jj - 1] = sympy.sqrt(jj)
        vec2[jj - 1] = (sympy.sqrt(1) - v0**2) * sympy.sqrt(jj)

    # create an advection tri-diagonal matrix
    A = sympy.banded({1: tuple(vec[0, :-1]), -1: tuple(vec2[0, :-1])})

    # identity matrix
    I = sympy.ImmutableSparseMatrix(sympy.eye(Nv))

    # invert matrix
    M = sympy.SparseMatrix(I * xi - k / np.abs(k) * A / sympy.sqrt(2))

    # save optimal R(nu*)
    R_approx = sympy.simplify(sympy.simplify(M.inv('LU')[0, 1] / sympy.sqrt(2) * k / np.abs(k)))
    print("I successfully inverted the matrix! ")

    asymptotics_0 = R_approx.series(xi, 0, 3, dir="+")
    print("zeroth order is " + str(sympy.simplify(asymptotics_0.coeff(xi, 0))))
    print("first order is " + str(sympy.simplify(asymptotics_0.coeff(xi, 1))))
    print("second order is " + str(sympy.simplify(asymptotics_0.coeff(xi, 2))))

    # objective function
    func = sympy.lambdify(v0, asymptotics_0.coeff(xi, 0) + 1, modules='numpy')
    # its derivative
    func_prime = sympy.lambdify(v0, sympy.diff(asymptotics_0.coeff(xi, 0), v0), modules="numpy")
    sol_coeff = scipy.optimize.newton(func=func, fprime=func_prime, x0=0, maxiter=10000, tol=1e-3, full_output=True)
    print("optimal coeff v0=", sol_coeff[0])
    # save optimal nu (for k=1)
    with open("optimal_nu_filter_klimas/nu_" + str(Nv) + ".txt", "wb") as outf:
        pickle.dump(sol_coeff[0], outf)

    # save optimal R(nu*) (for k=1)
    with open("optimal_R_filter_klimas/R_" + str(Nv) + ".txt", "wb") as outf:
        pickle.dump(sympy.simplify(R_approx.subs(v0, sol_coeff[0].real)), outf)

    print("completed collisionless + truncation")
    print("Nv = ", Nv)

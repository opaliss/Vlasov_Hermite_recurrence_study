import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import sympy
from sympy import symbols
import numpy as np
import scipy
import pickle


# loop over velocity resolutions
for Nv in np.arange(4, 14, 2):
    # symbolic variables
    xi = symbols('xi')
    # must be real and not complex
    nu = symbols('nu', real=True)
    # must be an integer from definition
    k = symbols('k', integer=True)

    # advection matrix (off-diagonal)
    vec = sympy.zeros(Nv)
    for jj in range(1, Nv + 1):
        vec[jj - 1] = sympy.sqrt(jj) / (sympy.sqrt(2))

    # advection matrix (main-diagonal)
    vec2 = sympy.zeros(Nv)
    for nn in range(0, Nv + 1):
        # hyper collisions coefficient
        vec2[nn] = np.power(nn/(Nv-1), 36) / np.sqrt(2)
        if vec2[nn] < 1e-16:
            vec2[nn] = 0

    # create an advection tri-diagonal matrix
    A = sympy.banded({1: tuple(vec[0, :-1]),
                     -1: tuple(vec[0, :-1]),
                      0: tuple(nu * vec2[0, :] / sympy.I)})

    # identity matrix
    I = sympy.eye(Nv)

    # invert matrix
    M = sympy.simplify(sympy.SparseMatrix(I * xi - A))

    # get final response function
    R_approx = sympy.simplify(M.inv()[0, 1] / sympy.sqrt(2))
    print("I successfully inverted the matrix! ")

    asymptotics_0 = R_approx.series(xi, 0, 2, dir="+")
    print("zeroth order is " + str(sympy.simplify(asymptotics_0.coeff(xi, 0))))

    # objective function
    func = sympy.lambdify(nu, abs(asymptotics_0.coeff(xi, 1) + sympy.I * sympy.sqrt(sympy.pi)), modules='numpy')
    # its derivative
    func_prime = sympy.lambdify(nu, sympy.diff(asymptotics_0.coeff(xi, 1) + sympy.I * sympy.sqrt(sympy.pi), nu), modules="numpy")
    sol_coeff = scipy.optimize.newton(func=func, fprime=func_prime, x0=10, maxiter=10000, tol=1e-3, full_output=True)

    # save optimal nu (for k=1)
    with open("optimal_nu_filter_Hou_Li/nu_" + str(Nv) + ".txt", "wb") as outf:
        pickle.dump(sol_coeff[0], outf)

    # save optimal R(nu*) (for k=1)
    with open("optimal_R_filter_Hou_Li/R_" + str(Nv) + ".txt", "wb") as outf:
        pickle.dump(sympy.simplify(R_approx.subs(nu, sol_coeff[0].real)), outf)

    print(sol_coeff)
    print("completed filter operator")
    print("Nv = ", Nv)
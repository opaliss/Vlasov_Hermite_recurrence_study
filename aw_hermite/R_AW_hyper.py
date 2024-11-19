import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import sympy
from sympy import banded, symbols, SparseMatrix
import numpy as np
import scipy
import pickle


def factorial_ratio(num1, denom1, num2, denom2):
    # return (num1!num2!) / (denom1!denom2!) with num1>denom1 and num2<denom2
    vector1 = range(denom1 + 1, num1 + 1)
    vector2 = range(num2 + 1, denom2 + 1)
    const = 1
    for ii in range(len(vector1)):
        const *= sympy.Rational(vector1[ii],  vector2[ii])
    if const < 0:
        return print("alert! negative diffusion")
    else:
        return sympy.simplify(const)


# loop over velocity resolutions
for Nv in np.arange(12, 14, 2):
    # hypercollisionality order ~n^{2alpha -1}
    # alpha = 1 (Lenard Bernstein 1958) ~n
    # alpha = 2 (Camporeale 2016) ~n^3
    for alpha in range(1, int(Nv/2) + 1):
        print("Nv = ", Nv)
        print("alpha = ", alpha)

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
            const = factorial_ratio(num1=nn, denom1=nn-2*alpha+1, num2=Nv-2*alpha, denom2=Nv-1)
            if const != 0:
                vec2[nn] = const

        # enforce k=1 for simplicity now
        k = 1

        # create an advection tri-diagonal matrix
        A = banded({1: tuple(vec[0, :-1]), -1: tuple(vec[0, :-1]), 0: tuple(nu * vec2[0, :] / (sympy.I * sympy.sqrt(2) * k))})

        # identity matrix
        I = sympy.ImmutableSparseMatrix(sympy.eye(Nv))

        # invert matrix
        # M = sympy.SparseMatrix(I * xi - k / np.abs(k) * A)
        M = sympy.SparseMatrix(I * xi - A)

        # get final response function
        # R_approx = sympy.simplify(sympy.simplify(M.inv()[0, 1] / sympy.sqrt(2) * k / np.abs(k))) # if k is not 1
        # R_approx = sympy.simplify(sympy.simplify(M.inv("QR")[0, 1] / sympy.sqrt(2)))
        cofactor = M.cofactor(0, 1)
        determinant = M.det(method="berkowitz")
        R_approx = sympy.simplify(cofactor / determinant / sympy.sqrt(2))

        asymptotics_0 = R_approx.series(xi, 0, 3)
        print("zeroth order is " + str(sympy.simplify(asymptotics_0.coeff(xi, 0))))
        print("first order is " + str(sympy.simplify(asymptotics_0.coeff(xi, 1))))
        print("second order is " + str(sympy.simplify(asymptotics_0.coeff(xi, 2))))

        func = sympy.lambdify(nu, asymptotics_0.coeff(xi, 1) + sympy.I * sympy.sqrt(sympy.pi), modules='numpy')
        func_prime = sympy.lambdify(nu, sympy.diff(asymptotics_0.coeff(xi, 1), nu), modules="numpy")
        func_prime2 = sympy.lambdify(nu, sympy.diff(sympy.diff(asymptotics_0.coeff(xi, 1), nu), nu), modules="numpy")
        sol_coeff = scipy.optimize.newton(func=func,
                                          fprime=func_prime,
                                          fprime2=func_prime2,
                                          x0=1,
                                          maxiter=20000,
                                          tol=1e-8,
                                          full_output=True)
        print("residual = ", np.abs(func(sol_coeff[0])))
        # sol_coeff = sympy.solve(asymptotics_0.coeff(xi, 1) + sympy.I * sympy.sqrt(sympy.pi), nu)


        # save optimal nu (for k=1)
        with open("optimal_nu_hyper_" + str(alpha) + "/nu_" + str(Nv) + ".txt", "wb") as outf:
            pickle.dump(sol_coeff[0], outf)

        # save optimal R(nu*) (for k=1)
        with open("optimal_R_hyper_" + str(alpha) + "/R_" + str(Nv) + ".txt", "wb") as outf:
            pickle.dump(sympy.simplify(R_approx.subs(nu, sol_coeff[0])), outf)

        print(sol_coeff)
        print("completed hypercollisional operator")

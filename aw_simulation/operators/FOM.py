"""Operators of the Spectral Plasma Solver Hermite-Fourier Expansion

Author: Opal Issan (oissan@ucsd.edu)
Date: Oct 31st, 2024
"""
import numpy as np
import pickle
import scipy.special


def psi_ln_aw(xi, n, alpha_s, u_s, v):
    """Hermite basis functions for plotting purposes

    :param alpha_s: float, velocity scaling parameter
    :param u_s, float, velocity shifting parameter
    :param v: float or array, the velocity coordinate sampled at version points
    :param xi: float or array, xi^{s} scaled velocity, i.e. xi = (v - u^{s})/alpha^{s}
    :param n: int, order of polynomial
    :return: float or array,  asymmetrically weighted (AW) hermite polynomial of degree n evaluated at xi
    """
    if n == 0:
        return np.exp(-xi ** 2) / np.sqrt(np.pi)
    if n == 1:
        return np.exp(-xi ** 2) * (2 * xi) / np.sqrt(2 * np.pi)
    else:
        psi = np.zeros((n + 1, len(xi)))
        psi[0, :] = np.exp(-xi ** 2) / np.sqrt(np.pi)
        psi[1, :] = np.exp(-xi ** 2) * (2 * xi) / np.sqrt(2 * np.pi)
        for jj in range(1, n):
            factor = - alpha_s * np.sqrt((jj + 1) / 2)
            psi[jj + 1, :] = (alpha_s * np.sqrt(jj / 2) * psi[jj - 1, :] + u_s * psi[jj, :] - v * psi[jj, :]) / factor
    return psi[n, :]


def fft_(coefficient, Nx, x, L):
    """evaluate the fourier expansion given the fourier coefficients.

    :param coefficient: array, vector of all fourier coefficients
    :param Nx: int, number of fourier modes (total 2Nx+1)
    :param x: array, spatial domain
    :param L: float, length of spatial domain
    :return: array, fourier expansion
    """
    sol = np.zeros(len(x), dtype="complex128")
    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        sol += coefficient[ii] * np.exp(1j * kk * x * 2 * np.pi / L)
    return sol.real


def D_matrix_full(Nx, L):
    """D matrix (spatial Fourier derivative operator)

    :param Nx: int, number of spatial spectral terms
    :param L: float, the length of the spatial domain
    :return: 2D matrix, D matrix (anti-symmetric + diagonal)
    """
    D = np.zeros((2 * Nx + 1, 2 * Nx + 1), dtype="complex128")
    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        D[ii, ii] = (2 * np.pi * 1j * kk) / L
    return scipy.sparse.dia_matrix(D)


def D_matrix_inv_full(Nx, L):
    """D matrix inverse (spatial Fourier derivative operator)

    :param Nx: int, number of spatial spectral terms
    :param L: float, the length of the spatial domain
    :return: 2D matrix, D matrix (anti-symmetric + diagonal)
    """
    D = np.zeros(((2 * Nx + 1), (2 * Nx + 1)), dtype="complex128")
    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        if kk != 0:
            D[ii, ii] = L / (2 * np.pi * 1j * kk)
    return scipy.sparse.dia_matrix(D)


def A_matrix_diag(Nv, D):
    """main diagonal component of the linear advection matrix

    :param D: matrix, diagonal matrix with Fourier derivative coefficients
    :param Nv: int, total number of Hermite spectral expansion coefficients
    :return: 2D matrix, A matrix in linear advection term
    """
    return -scipy.sparse.kron(scipy.sparse.identity(n=Nv), D, format="csr")


def factorial_ratio(num1, denom1, num2, denom2):
    """(num1!num2!) / (denom1!denom2!) with num1>denom1 and num2<denom2
        used for damping rate of hypercollisional operator

    :param num1: int, first numerator
    :param denom1: int, first denominator
    :param num2: int, second numerator
    :param denom2: int, second denominator
    :return: float (from zero to one), (num1!num2!) / (denom1!denom2!)
    """
    vector1 = range(denom1 + 1, num1 + 1)
    vector2 = range(num2 + 1, denom2 + 1)
    const = 1
    for ii in range(len(vector1)):
        const *= vector1[ii] / vector2[ii]
    if const >= 0:
        return const
    else:
        print("negative diffusion")


def A_matrix_col(Nx_total, Nv, M0, MF, col_type, hyper_rate):
    """collisional term of the advection matrix

    :param M0: matrix 0th index
    :param MF: matrix final index
    :param Nv: int, total number of Hermite spectral expansion coefficients
    :param Nx_total: int, total number of Fourier spectral expansion coefficients (Nx + 1)
    :param col_type: str, type of collisional operator, e.g. "hyper", "hou_li", or "collisionless"
    :param hyper_rate: int, e.g. 1 (LB), 2 (Camporeale et al. 2006), 3 (higher etc), ...
    :return: 2D matrix, A matrix in linear advection term
    """
    A = np.zeros((MF - M0, MF - M0), dtype="complex128")
    for ii, n in enumerate(range(M0, MF)):
        # main diagonal
        if col_type == "hyper":
            A[ii, ii] = factorial_ratio(num1=ii, denom1=ii-2*hyper_rate+1, num2=Nv - 2*hyper_rate, denom2=Nv-1)
        elif col_type == "hou_li":
            A[ii, ii] = (n / (Nv - 1))**36
        elif col_type != "collisionless":
            return print("we do not have the " + str(col_type) + " collisional operator implemented!")
    return -scipy.sparse.kron(A, scipy.sparse.identity(n=Nx_total), format="csr")


def A_matrix_off(M0, MF, D, closure_type="truncation"):
    """A matrix off-diagonal components in linear advection term

    :param M0: matrix 0th index
    :param MF: matrix final index
    :param D: matrix, diagonal matrix with Fourier derivative coefficients
    :param closure_type: str, type of closure used, e.g. "truncation"
    :return: 2D matrix, A matrix in linear advection term
    """
    A = np.zeros((MF - M0, MF - M0), dtype="complex128")
    for ii, n in enumerate(range(M0, MF)):
        if n != M0:
            # lower diagonal
            A[ii, ii - 1] = -np.sqrt(n / 2)
        if n != MF - 1:
            # upper diagonal
            A[ii, ii + 1] = -np.sqrt((n + 1) / 2)
    if closure_type == "truncation":
        return scipy.sparse.kron(A, D, format="csr")
    elif closure_type == "hammett_perkins":
        A_off = scipy.sparse.kron(A, D, format="csr")
        Nx_total = np.shape(D)[0]
        try:
            with open("../aw_hermite/optimal_q1_HP/coeff_" + str(MF) + ".txt", "rb") as outf:
                c = float(pickle.load(outf))
                print("c = ", c)
        except:
            c = -1.017234
            print("c = ", c)
        A_off[(MF - 1) * Nx_total:, (MF - 1) * Nx_total:] = - np.sqrt(MF / 2) * c * 1j * D @ k_matrix(Nx=(Nx_total - 1) // 2)
        return A_off


def k_matrix(Nx):
    """matrix for Hammett-Perkins style closure term

    :param Nx: resolution in spatial direction x
    :return: K_matrix (diagonal matrix)
    """
    K_matrix = np.zeros((2 * Nx + 1, 2 * Nx + 1))
    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        K_matrix[ii, ii] = np.sign(kk)
    return K_matrix


def B(Nx_total, M0, MF):
    """2D matrix, B matrix of acceleration term

    :param M0: 0th n index
    :param MF: final n index
    :param Nx_total: int, number of Fourier spectral expansion coefficients (total is 2Nx + 1)
    :return: 2D matrix, B matrix of acceleration term
    """
    B = np.zeros((MF - M0, MF - M0))
    for ii, n in enumerate(range(M0, MF)):
        # lower diagonal
        if ii >= 1:
            B[ii, ii - 1] = np.sqrt(2 * n)
    return scipy.sparse.kron(B, scipy.sparse.identity(Nx_total), format="csr")


def B_2(M0, MF):
    """2D matrix, B matrix of acceleration term

    :param M0: 0th n index
    :param MF: final n index
    :return: 2D matrix, B matrix of acceleration term
    """
    B = np.zeros((MF - M0, MF - M0))
    for ii, n in enumerate(range(M0, MF)):
        # lower diagonal
        if ii >= 1:
            B[ii, ii - 1] = np.sqrt(2 * n)
    return scipy.sparse.dia_matrix(B)


def nonlinear_full(E, psi, Nx_total, Nv):
    """convolution of the electric field and state coefficients

    :param Nx_total: int, number of Fourier spectral expansion coefficients
    :param Nv: int, total number of Hermite spectral expansion coefficients
    :param E: array, electric field length Nx+1
    :param psi: array, state coefficients length (Nv * (Nx+1))
    :return: array, E * psi convolved (length Nv * (Nx+1))
    """
    y = np.zeros((Nx_total * Nv), dtype="complex128")
    for nn in range(Nv):
        y[nn * Nx_total: (nn + 1) * Nx_total] = scipy.signal.convolve(in1=E,
                                                                      in2=psi[nn * Nx_total: (nn + 1) * Nx_total],
                                                                      mode="same")
    return y


def charge_density(alpha_e, alpha_i, q_e, q_i, C0_electron, C0_ions):
    """rho(t)

    :param C0_ions: array, coefficients of the charge density of ions
    :param C0_electron: array, coefficients of the charge density of electrons
    :param alpha_e: float, velocity scaling of electrons
    :param alpha_i: float, velocity scaling of ions
    :param q_e: float, charge of electrons (normalized)
    :param q_i: float, charge of ions (normalized)
    :return: rho(t)
    """
    return q_e * alpha_e * C0_electron + q_i * alpha_i * C0_ions


def charge_density_two_stream(alpha_e1, alpha_e2, alpha_i, q_e1, q_e2, q_i, C0_electron_1, C0_electron_2, C0_ions):
    """rho(t)

    :param C0_ions: array, coefficients of the charge density of ions
    :param C0_electron_1: array, coefficients of the charge density of electron species 1
    :param C0_electron_2: array, coefficients of the charge density of electron species 2
    :param alpha_e1: float, velocity scaling of electron species 1
    :param alpha_e2: float, velocity scaling of electron species 2
    :param alpha_i: float, velocity scaling of ions
    :param q_e1: float, charge of electron species 1 (normalized)
    :param q_e2: float, charge of electron species 2 (normalized)
    :param q_i: float, charge of ions (normalized)
    :return: rho(t)
    """
    return q_e1 * alpha_e1 * C0_electron_1 + q_e2 * alpha_e2 * C0_electron_2 + q_i * alpha_i * C0_ions


def total_mass(psi, alpha_e, alpha_i, Nv, Nx, L):
    """N(t)

    :param psi: array, array of all coefficients of size ((Nv)*(2Nx + 1))
    :param alpha_e: float, velocity scaling of electrons
    :param alpha_i: float, velocity scaling of ions
    :param Nv: int, total number of Hermite spectral terms (Nv)
    :param Nx: int, number of Fourier spectral terms
    :param L: float, length of spatial domain
    :return: N(t)
    """
    return L * (alpha_e * psi[Nx] + alpha_i * psi[Nv * (2 * Nx + 1) + Nx])


def total_mass_two_stream(psi, alpha_e1, alpha_e2, alpha_i, Nv, Nx, L):
    """N(t)

    :param psi: array, array of all coefficients of size (3*(Nv)*(2Nx + 1))
    :param alpha_e1: float, velocity scaling of electrons species 1
    :param alpha_e2: float, velocity scaling of electrons species 2
    :param alpha_i: float, velocity scaling of ions
    :param Nv: int, total number of Hermite spectral terms (Nv)
    :param Nx: int, number of Fourier spectral terms
    :param L: float, length of spatial domain
    :return: N(t)
    """
    return L * (alpha_e1 * psi[Nx]
                + alpha_e2 * psi[Nv * (2 * Nx + 1) + Nx]
                + alpha_i * psi[2 * Nv * (2 * Nx + 1) + Nx])


def total_momentum(psi, alpha_e, alpha_i, Nv, Nx, L, m_i, m_e, u_e, u_i):
    """P(t)

    :param psi: array, array of all coefficients of size ((Nv)*(2Nx + 1))
    :param alpha_e: float, velocity scaling of electrons
    :param alpha_i: float, velocity scaling of ions
    :param Nv: int, total number of Hermite spectral terms (Nv)
    :param L: float, length of spatial domain
    :param m_i: float, mass of ions (normalized to electron)
    :param m_e: float, mass of electrons  (normalized to electron, i.e. 1)
    :param u_e: float, velocity shifting parameter of electrons
    :param u_i: float, velocity shifting parameter of ions
    :param Nx: int, number of Fourier spectral terms
    :return: P(t)
    """
    electron_momentum = m_e * alpha_e * L * (alpha_e * psi[(2 * Nx + 1) + Nx] / np.sqrt(2) + u_e * psi[Nx])
    ion_momentum = m_i * alpha_i * L * (alpha_i * psi[Nv * (2 * Nx + 1) + (2 * Nx + 1) + Nx] / np.sqrt(2)
                                        + u_i * psi[Nv * (2 * Nx + 1) + Nx])
    return electron_momentum + ion_momentum


def total_momentum_two_stream(psi, alpha_e1, alpha_e2, alpha_i, Nv, Nx, L, m_i, m_e1, m_e2, u_e1, u_e2, u_i):
    """P(t)

    :param psi: array, array of all coefficients of size ((Nv)*(2Nx + 1))
    :param alpha_e1: float, velocity scaling of electrons species 1
    :param alpha_e2: float, velocity scaling of electrons species 2
    :param alpha_i: float, velocity scaling of ions
    :param Nv: int, total number of Hermite spectral terms (Nv)
    :param L: float, length of spatial domain
    :param m_i: float, mass of ions (normalized to electron)
    :param m_e1: float, mass of electrons species 1 (normalized to electron, i.e. 1)
    :param m_e2: float, mass of electrons species 2 (normalized to electron, i.e. 1)
    :param u_e1: float, velocity shifting parameter of electrons species 1
    :param u_e2: float, velocity shifting parameter of electrons species 2
    :param u_i: float, velocity shifting parameter of ions
    :param Nx: int, number of Fourier spectral terms
    :return: P(t)
    """
    electron1_momentum = m_e1 * alpha_e1 * L * (alpha_e1 * psi[(2 * Nx + 1) + Nx] / np.sqrt(2) + u_e1 * psi[Nx])
    electron2_momentum = m_e2 * alpha_e2 * L * (alpha_e2 * psi[Nv * (2 * Nx + 1) + (2 * Nx + 1) + Nx] / np.sqrt(2)
                                                + u_e2 * psi[Nv * (2 * Nx + 1) + Nx])
    ion_momentum = m_i * alpha_i * L * (alpha_i * psi[2 * Nv * (2 * Nx + 1) + (2 * Nx + 1) + Nx] / np.sqrt(2)
                                        + u_i * psi[2 * Nv * (2 * Nx + 1) + Nx])
    return electron1_momentum + electron2_momentum + ion_momentum


def total_energy_k(psi, alpha_e, alpha_i, Nv, Nx, L, u_e, u_i, m_e, m_i):
    """E_{k}(t)

    :param m_i: float, mass of ions (normalized to electron)
    :param m_e: float, mass of electrons  (normalized to electron, i.e. 1)
    :param psi: array, array of all coefficients of size ((Nv)*(2Nx + 1))
    :param alpha_e: float, velocity scaling of electrons
    :param alpha_i: float, velocity scaling of ions
    :param Nv: int, total number of Hermite spectral terms (Nv)
    :param Nx: int, number of Fourier spectral terms
    :param L: float, length of spatial domain
    :param u_e: float, velocity shifting parameter of electrons
    :param u_i: float, velocity shifting parameter of ions
    :return: E_{k}(t)
    """
    # electron kinetic energy
    electron_kin = 0.5 * L * alpha_e * (alpha_e ** 2 * psi[2 * (2 * Nx + 1) + Nx] / np.sqrt(2)
                                        + np.sqrt(2) * u_e * alpha_e * psi[(2 * Nx + 1) + Nx]
                                        + ((alpha_e ** 2) / 2 + u_e ** 2) * psi[Nx])

    # ion kinetic energy
    ion_kin = 0.5 * L * alpha_i * (alpha_i ** 2 * psi[(2 * Nx + 1) * Nv + 2 * (2 * Nx + 1) + Nx] / np.sqrt(2)
                                   + np.sqrt(2) * u_i * alpha_i * psi[(2 * Nx + 1) * Nv + (2 * Nx + 1) + Nx]
                                   + ((alpha_i ** 2) / 2 + u_i ** 2) * psi[(2 * Nx + 1) * Nv + Nx])
    return m_e * electron_kin + m_i * ion_kin


def total_energy_k_two_stream(psi, alpha_e1, alpha_e2, alpha_i, Nv, Nx, L, u_e1, u_e2, u_i, m_e1, m_e2, m_i):
    """E_{k}(t)

    :param m_i: float, mass of ions (normalized to electron)
    :param m_e1: float, mass of electrons species 1 (normalized to electron, i.e. 1)
    :param m_e2: float, mass of electrons species 2 (normalized to electron, i.e. 1)
    :param psi: array, array of all coefficients of size ((Nv)*(2Nx + 1))
    :param alpha_e1: float, velocity scaling of electrons species 1
    :param alpha_e2: float, velocity scaling of electrons species 1
    :param alpha_i: float, velocity scaling of ions
    :param Nv: int, total number of Hermite spectral terms (Nv)
    :param Nx: int, number of Fourier spectral terms
    :param L: float, length of spatial domain
    :param u_e1: float, velocity shifting parameter of electrons species 1
    :param u_e2: float, velocity shifting parameter of electrons species 2
    :param u_i: float, velocity shifting parameter of ions
    :return: E_{k}(t)
    """
    # electron species 1 kinetic energy
    electron1_kin = 0.5 * L * alpha_e1 * (alpha_e1 ** 2 * psi[2 * (2 * Nx + 1) + Nx] / np.sqrt(2)
                                          + np.sqrt(2) * u_e1 * alpha_e1 * psi[(2 * Nx + 1) + Nx]
                                          + ((alpha_e1 ** 2) / 2 + u_e1 ** 2) * psi[Nx])

    # electron species 2 kinetic energy
    electron2_kin = 0.5 * L * alpha_e2 * (alpha_e2 ** 2 * psi[(2 * Nx + 1) * Nv + 2 * (2 * Nx + 1) + Nx] / np.sqrt(2)
                                          + np.sqrt(2) * u_e2 * alpha_e2 * psi[(2 * Nx + 1) * Nv + (2 * Nx + 1) + Nx]
                                          + ((alpha_e2 ** 2) / 2 + u_e2 ** 2) * psi[(2 * Nx + 1) * Nv + Nx])

    # ion kinetic energy
    ion_kin = 0.5 * L * alpha_i * (alpha_i ** 2 * psi[2 * (2 * Nx + 1) * Nv + 2 * (2 * Nx + 1) + Nx] / np.sqrt(2)
                                   + np.sqrt(2) * u_i * alpha_i * psi[2 * (2 * Nx + 1) * Nv + (2 * Nx + 1) + Nx]
                                   + ((alpha_i ** 2) / 2 + u_i ** 2) * psi[2 * (2 * Nx + 1) * Nv + Nx])

    return m_e1 * electron1_kin + m_e2 * electron2_kin + m_i * ion_kin

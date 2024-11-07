"""Module to run the nonlinear Landau damping testcase

Author: Opal Issan
Date: Nov 6th, 2024
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.FOM import nonlinear_full, charge_density
from operators.implicit_midpoint_FOM import implicit_midpoint_solver_FOM
from operators.setup_FOM import SimulationSetupFOM
import time
import numpy as np


def rhs(y):
    # electric field computed
    E = setup.D_inv @ charge_density(alpha_e=setup.alpha_e, alpha_i=setup.alpha_i,
                                     q_e=setup.q_e, q_i=setup.q_i,
                                     C0_electron=y[:setup.Nx_total],
                                     C0_ions=C0_ions)

    # evolving only electrons
    return setup.A_e @ y + setup.B_e @  nonlinear_full(E=E, psi=y, Nv=setup.Nv, Nx_total=setup.Nx_total)


if __name__ == "__main__":
    setup = SimulationSetupFOM(Nx=100,  # spatial resolution
                               Nx_total=201,  # this is always 2Nx+1 in this code.
                               Nv=100,  # velocity resolution
                               epsilon=0.5,  # perturbation amplitude
                               alpha_e=np.sqrt(2),  # hermite scaling parameter for electrons
                               alpha_i=np.sqrt(2 / 1836),  # hermite scaling parameter for ions
                               u_e=0,  # hermite shifting parameter for electrons
                               u_i=0,  # hermite shifting parameter for ions
                               L=4 * np.pi,  # spatial length
                               dt=0.01,  # time step
                               T0=0,  # initial time
                               T=100,  # final time
                               nu=0,  # collisional frequency
                               hyper_rate=None,  # artificial collision order
                               col_type="collisionless",  # type of collisional operator
                               closure_type="truncation")  # closure

    # initial condition
    y0 = np.zeros(setup.Nv * setup.Nx_total, dtype="complex128")

    # electron equilibrium
    y0[setup.Nx] = 1 / setup.alpha_e

    # electron perturbation k=0.5
    y0[setup.Nx + 1] = 0.5 * setup.epsilon / setup.alpha_e
    y0[setup.Nx - 1] = 0.5 * setup.epsilon / setup.alpha_e

    # ions (unperturbed)
    C0_ions = np.zeros(setup.Nx_total)
    C0_ions[setup.Nx] = 1 / setup.alpha_i

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u, setup = implicit_midpoint_solver_FOM(y_0=y0,
                                                         right_hand_side=rhs,
                                                         r_tol=1e-10,
                                                         a_tol=1e-10,
                                                         max_iter=100,
                                                         param=setup,
                                                         verbose=True)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)

    # make directory
    if not os.path.exists("data/nonlinear_landau_1024"):
        os.makedirs("data/nonlinear_landau_1024")

    # save results every n steps
    skip = 10

    # save results
    np.save("data/nonlinear_landau_1024/sol_u_" + str(setup.Nv) + "_closure_" + str(setup.closure_type) + "_collisions_" + str(setup.col_type) + "_" + str(setup.hyper_rate) + "_nu_" + str(setup.nu), sol_midpoint_u[:, ::skip])
    np.save("data/nonlinear_landau_1024/sol_t_" + str(setup.Nv) + "_closure_" + str(setup.closure_type) + "_collisions_" + str(setup.col_type) + "_" + str(setup.hyper_rate) + "_nu_" + str(setup.nu), setup.t_vec[::skip])

    # save parameters
    np.save("data/nonlinear_landau_1024/sol_setup_" + str(setup.Nv) + "_closure_" + str(setup.closure_type) + "_collisions_" + str(setup.col_type) + "_" + str(setup.hyper_rate) + "_nu_" + str(setup.nu), setup)

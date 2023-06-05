import numpy as np
from scipy.optimize import fsolve

def solve_char_eq(
        char_eq_f_of_x_T, Ts,
        initial_guess=(.1, .1), ret_err=False,
    ):
    '''
    char_eq_f_of_x_T is the characteristic equation as a function of
    - first arg: x = (Re(lambda), Im(lambda))
    - second arg: time delay T
    '''
    ps, qs = np.zeros(len(Ts)), np.zeros(len(Ts))
    p_errs, q_errs = np.zeros(len(Ts)), np.zeros(len(Ts))
    g = initial_guess
    for i, T in enumerate(Ts):
        sol = fsolve(lambda x: char_eq_f_of_x_T(x, T), g)
        p, q = sol
        ps[i], qs[i] = p, q
        p_errs[i], q_errs[i] = np.abs(char_eq_f_of_x_T(sol, T))
        if np.isclose(p, 0) or np.isclose(q, 0):
            d_p, d_q = .1, .1
        else:
            d_p, d_q = p * .1, q * .1
        g = (p + d_p, q + d_q)
    if ret_err:
        return ps, qs, p_errs, q_errs
    return ps, qs


def ode_jacobian_eigs(jacobian_of_T, Ts):
    '''
    Returns the eigenvalues of the jacobian of the ODE approximation with
    n_kernels kernels evaluated at the fixed point fp for various values of the
    time delay T. Thus the spectrum is obtained as a function of T. The maximal
    eigenvalue is an approximation to the local LE of the DDE.
    '''
    eigs = []
    for T in Ts:
        J = jacobian_of_T(T)
        eigs.append(np.sort_complex(np.linalg.eigvals(J)))
    return np.array(eigs).T


def _find_hopf_point_bisec(p, T_min, T_max, tol, max_iter):
    '''
    p must be a python function p(T) that returns the real part of the relevant
    lyapunov exponent at the time delay T
    '''
    assert T_min < T_max
    assert p(T_min) < 0 and p(T_max) > 0
    assert max_iter > 1
    i = 1
    converged = False
    T_mid = None
    while not converged:
        if i > max_iter:
            raise Exception('Maximal iteration exceeded! tol is too small or no Hopf bifurcation in given interval!')
        T_mid = (T_max + T_min) / 2
        if p(T_mid) < 0:
            T_min = T_mid
        else:
            T_max = T_mid
        err = np.abs(T_max - T_min)
        if err < tol:
            converged = True
        i += 1
    return T_mid


def ode_find_hopf_point_bisec(jacobian_of_T, T_min, T_max, tol=1e-4, max_iter=50):
    '''
    jacobian_of_T is a function that takes the time delay T as an argument and
    returns the jacobian of the system.
    '''
    def p(T):
        J = jacobian_of_T(T)
        Re_max_LE = np.sort_complex(np.linalg.eigvals(J))[-1].real
        return Re_max_LE
    return _find_hopf_point_bisec(p, T_min, T_max, tol, max_iter)

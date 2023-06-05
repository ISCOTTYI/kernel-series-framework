import numpy as np
from scipy.signal import find_peaks
from src.util import kill_transients, x_over_xT
from src.solver import dde_solver, ode_solver

def find_period_acorr(ts, traj, tol=.02, L_frac=.1):
    '''
    L_frac is the fraction of the length of the given trajectory that should be
    considered for the period determination.
    '''
    L = int(len(traj) * L_frac)
    L_ts, L_traj = ts[:L] - ts[0], traj[:L]
    mu, var = np.mean(L_traj), np.var(L_traj)
    L_traj_ = L_traj - mu
    # https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python
    # https://stackoverflow.com/questions/47351483/autocorrelation-to-estimate-periodicity-with-numpy
    # Why the normalization is as it is
    # https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html
    # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html#numpy.corrcoef
    acf = np.correlate(L_traj_, L_traj_, 'full')[-L:] / (var * np.arange(L, 0, -1)) # normalized
    p_idxs, _ = find_peaks(acf)
    j = 0
    for i in p_idxs:
        if acf[i] > 1 - tol:
            j = i
            break
    if j == 0:
        raise Exception('No local max of ACF exceeded 1 - tol')
    period = L_ts[j]
    return period


def _pd_bisection(T_min, T_max, dT_tol, f_period, max_iter, print_final_periods=False, debug=False):
    '''
    dT_tol = |T_max - T_min|, where T_min is below the PD, while T_max is above.
    So the condition for convergence is that |T_max - T_min| < dT_tol and that
    period(T_max) / 2 > period(T_min).
    Start with given T_min, T_max and calculate the period for both - should be
    at least factor of 2. Then calculate period for T_min+T_max / 2. If the
    period is below PD, change T_min to T_mid, else change T_max. After each
    step check that period doubling still in the middle and whether the
    tolerance is already achieved.
    '''
    assert T_min < T_max
    assert max_iter > 1
    p_min, p_max = f_period(T_min), f_period(T_max)
    if debug:
        print(f'p_min = {p_min}, p_max = {p_max}')
    assert 2 * p_min < p_max
    assert p_min > 1
    iters = 1
    converged = False
    T_mid = None
    while not converged:
        if iters > max_iter:
            raise Exception('Maximal iterations exceeded! Check parameters!')
        T_mid = (T_max + T_min) / 2
        p = f_period(T_mid)
        if p * 1.75 < p_max:
            p_min = p
            T_min = T_mid
        else:
            p_max = p
            T_max = T_mid
        if debug:
            print(f'T_min = {T_min}, T_max = {T_max}')
            print(f'p_min = {p_min}, p_max = {p_max}')
        if np.abs(T_max - T_min) < dT_tol:
            converged = True
        iters += 1
    if print_final_periods:
        print(f'Period(T<T_c) = {p_min}, Period(T>T_c) = {p_max}')
    if not np.isclose(p_max / 2, p_min, atol=0, rtol=.05):
        raise Exception(f'Wrong period doubling, final periods are Period(T<T_c) = {p_min}, Period(T>T_c) = {p_max}!')
    return (T_max + T_min) / 2


def dde_find_pd_bisec(
        jitc_flow_of_T, m, t_max, T_min, T_max,
        dT_tol=1e-4, tol=.02, L_frac=.2, max_iter=500, print_final_periods=False,
        debug=False
    ):
    '''
    jitc_flow_of_T is a function with the time delay T as the only parameter
    that returns the flow of the system.
    '''
    def f_period(T):
        ts, traj = kill_transients(*dde_solver(jitc_flow_of_T(T), T, m, t_max))
        return find_period_acorr(ts, traj, tol=tol, L_frac=L_frac)
    return _pd_bisection(T_min, T_max, dT_tol, f_period, max_iter,
                         print_final_periods=print_final_periods, debug=debug)


def ode_find_pd_bisec(
        jitc_flow_of_T, m, t_max, T_min, T_max,
        dT_tol=1e-4, tol=.02, L_frac=.2, max_iter=500, print_final_periods=False,
        debug=False # , solver=ode_solver
    ):
    '''
    jitc_flow_of_T is a function with the time delay T as the only parameter
    that returns the flow of the system.
    '''
    def f_period(T):
        ts, traj = kill_transients(*ode_solver(jitc_flow_of_T(T), T, m, t_max))
        xs, _ = x_over_xT(traj)
        return find_period_acorr(ts, xs, tol=tol, L_frac=L_frac)
    return _pd_bisection(T_min, T_max, dT_tol, f_period, max_iter,
                         print_final_periods=print_final_periods, debug=debug)

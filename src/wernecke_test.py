import numpy as np
from src.util import kill_transients, group_hs, attractor_mean, attractor_size, x_over_xT
from src.solver import dde_sample_random_h0, ode_sample_random_st0, dde_solver, ode_solver, construct_ts

'''
0-1 TEST FOR DDE
----------------
'''
def dde_sample_h0s_on_attractor(system_flow, m, traj, n_h0, tr_skip):
    '''
    system_flow is a function of x, xT and returns dx (flow).
    traj: the trajectory to sample from
    n_h0: number of initial conditions to sample
    tr_skip: float in (0, 1) indicating the percent of transients
        to skip of the trajectory
    '''
    traj_cl = kill_transients(traj, tr_skip=tr_skip)[0]
    xs, xTs = x_over_xT(traj_cl, m)
    sample_idx = np.random.choice(xs.shape[0], size=n_h0, replace=False)
    h0s = np.array([xs[i:i+m] for i in sample_idx])
    dh0s = np.array([[system_flow(xs[j], xTs[j]) for j in range(i, i+m)] for i in sample_idx])
    return h0s, dh0s


def dde_norm(st, m, axis=0):
    return np.linalg.norm(st, axis=axis) / np.sqrt(m)


def dde_trajectory_distance(traj1, traj2, m):
    hs1, hs2 = group_hs(traj1, m), group_hs(traj2, m)
    ds = hs1 - hs2
    return dde_norm(ds, m, axis=1)


def dde_seperate_h0(m, delta, h0):
    n = dde_sample_random_h0(m)
    n = n/dde_norm(n, m)
    sep_h0 = h0 + delta*n
    return sep_h0


def _dde_distance_scaling(
        system_flow, jitc_flow, T, delta, m, t_max,
        n_h0, tr_skip, verbose,
        return_mean_and_size=False
    ):
    '''
    jitc_flow is the flow of the system for the given T.

    returns
    -------
    [
        [d(t) for times 0, T, 2T, ...]
        for every initial condition out of n_h0
    ]
    thus still need to average accordingly
    '''
    sample_traj = dde_solver(jitc_flow, T, 100, 1e5)[1]
    mean = attractor_mean(sample_traj[int(tr_skip*len(sample_traj)):])
    size = attractor_size(sample_traj[int(tr_skip*len(sample_traj)):])
    h0s, dh0s = dde_sample_h0s_on_attractor(system_flow, m, sample_traj, n_h0, tr_skip)
    d_over_ts = [] # d(t) for the various initial state histories
    for i, h0, dh0 in zip(range(n_h0), h0s, dh0s):
        if verbose and i % (n_h0 / 10) == 0:
            print(i, end=' ', flush=True)
        sep_h0 = dde_seperate_h0(m, delta, h0)
        traj1 = dde_solver(jitc_flow, T, m, t_max, h0=h0, dh0=dh0)[1]
        traj2 = dde_solver(jitc_flow, T, m, t_max, h0=sep_h0, dh0=dh0)[1]
        # print(np.log(dde_norm(np.abs(traj1 - traj2).reshape(n_intervals+1, n_steps), n_steps, axis=1))[:3])
        d_over_ts.append(dde_trajectory_distance(traj1, traj2, m))
    d_over_ts = np.array(d_over_ts)
    # states histories are at times 0, T, 2T, ...
    ts = group_hs(construct_ts(T, m, t_max, t0=-T), m)[:,-1]
    if return_mean_and_size:
        return ts, d_over_ts, mean, size
    return ts, d_over_ts


def dde_distance_scaling(
        system_flow, jitc_flow, T, delta, m, t_max,
        n_h0=100, tr_skip=.1, verbose=True,
        log=False
    ):
    # calculate average (over n_h0 initial conditons) cross-distance scaling
    ts, d_over_ts = _dde_distance_scaling(
        system_flow, jitc_flow, T, delta, m, t_max,
        n_h0, tr_skip, verbose
    )
    if log:
        return ts, np.log(np.mean(d_over_ts.T, axis=1))
    return ts, np.mean(d_over_ts.T, axis=1)


def dde_cross_correlation(
        system_flow, jitc_flow, T, delta, m, t_max,
        n_h0=100, tr_skip=.1, verbose=True
    ):
    '''
    Cross correlation is given by
        C(t) = 1 - d^2(t) / (2 σ^2),
    where d is the cross distance scaling and σ^2 is the
    attractor size (variance, standard deviation squared)
    '''
    ts, d_over_ts, mean, size = _dde_distance_scaling(
        system_flow, jitc_flow, T, delta, m, t_max,
        n_h0, tr_skip, verbose,
        return_mean_and_size=True
    )
    dds = np.mean((d_over_ts ** 2).T, axis=1) # d^2(t) averaged over initial conditions
    cs = 1 - dds / (2 * size)
    return ts, cs


'''
0-1 TEST FOR ODE
----------------
'''
def ode_sample_st0s_on_attractor(traj, n_st0, tr_skip):
    traj_cl = kill_transients(traj, tr_skip=tr_skip)[0]
    sample_idx = np.random.choice(traj_cl.shape[0], size=n_st0, replace=False)
    return traj_cl[sample_idx,:]


def ode_norm(st, n_kernels, axis=0):
    return np.linalg.norm(st, axis=axis) / np.sqrt(n_kernels)


def ode_seperate_st0(delta, st0):
    n_kernels = st0.shape[0] - 1
    n = ode_sample_random_st0(n_kernels)
    n = n/ode_norm(n, n_kernels)
    return st0 + delta*n


def ode_trajectory_distance(traj1, traj2):
    n_kernels = traj1[0].shape[0] - 1 # ERROR: correct?
    d = traj1 - traj2
    return ode_norm(d, n_kernels, axis=1)


def _ode_distance_scaling(
        jitc_flow, T, delta, m, t_max,
        n_st0, tr_skip, verbose,
        return_mean_and_size=False
    ):
    sample_traj = ode_solver(jitc_flow, T, 100, 1e5)[1]
    smpl_xs = sample_traj.T[0]
    mean = attractor_mean(smpl_xs[int(tr_skip*len(smpl_xs)):])
    size = attractor_size(smpl_xs[int(tr_skip*len(smpl_xs)):])
    st0s = ode_sample_st0s_on_attractor(sample_traj, n_st0, tr_skip)
    d_over_ts = []
    for i, st0 in enumerate(st0s):
        if verbose and i % (n_st0 / 10) == 0:
            print(i, end=' ', flush=True)
        st0pd = ode_seperate_st0(delta, st0)
        traj1 = ode_solver(jitc_flow, T, m, t_max, st0=st0)[1]
        traj2 = ode_solver(jitc_flow, T, m, t_max, st0=st0pd)[1]
        d_over_ts.append(ode_trajectory_distance(traj1, traj2))
    d_over_ts = np.array(d_over_ts)
    ts = construct_ts(T, m, t_max)
    if return_mean_and_size:
        return ts, d_over_ts, mean, size
    return ts, d_over_ts


def ode_distance_scaling(
        jitc_flow, T, delta, m, t_max,
        n_st0=100, tr_skip=.1, verbose=True,
        log=False
    ):
    # calculate average (over n_h0 initial conditons) cross-distance scaling
    ts, d_over_ts = _ode_distance_scaling(
        jitc_flow, T, delta, m, t_max,
        n_st0, tr_skip, verbose
    )
    if log:
        return ts, np.log(np.mean(d_over_ts.T, axis=1))
    return ts, np.mean(d_over_ts.T, axis=1)


def ode_cross_correlation(
        jitc_flow, T, delta, m, t_max,
        n_st0=100, tr_skip=.1, verbose=True
    ):
    '''
    Cross correlation is given by
        C(t) = 1 - d^2(t) / (2 σ^2),
    where d is the cross distance scaling and σ^2 is the
    attractor size (variance, standard deviation squared)
    '''
    ts, d_over_ts, mean, size = _ode_distance_scaling(
        jitc_flow, T, delta, m, t_max,
        n_st0, tr_skip, verbose,
        return_mean_and_size=True
    )
    dds = np.mean((d_over_ts ** 2).T, axis=1) # d^2(t) averaged over initial conditions
    cs = 1 - dds / (2 * size)
    return ts, cs


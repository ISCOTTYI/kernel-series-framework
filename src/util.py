import numpy as np

def power_law(x, a, k):
    return a * x**(-k)


def spaced_ints(mini, maxi, step_multiplier=1, multiplier_after=0):
    step = 1
    ints = [mini]
    while ints[-1] < maxi:
        li = ints[-1]
        if li > multiplier_after:
            ni = li + step * step_multiplier
        else:
            ni = li + step
        ints.append(ni)
        step += 1
    return ints


def kill_transients(*ts_and_trajs, tr_skip=.2):
    '''
    For given trajectories and corresponding times, remove the transient
    regime. The skipped part of the traj is given by tr_skip. This can either be
    a float between 0 and 1 indicating a percentage of the total traj length or
    a concrete index before which the transient regime lies.
    '''
    L = len(ts_and_trajs[0])
    if not all(len(arr) == L for arr in ts_and_trajs):
        raise ValueError('ts and trajs must have the same dimensions!')
    if type(tr_skip) is float and tr_skip > 1 or tr_skip < 0:
        raise ValueError('tr_skip in percent must be between 0 and 1.')
    if type(tr_skip) is float:
        return tuple(arr[int(tr_skip*L):] for arr in ts_and_trajs)
    return tuple(arr[int(tr_skip):] for arr in ts_and_trajs)

def group_hs(traj, m):
    '''
    The 1D trajectory of the DDE is bunched into state histories, each containing
    m time points.
    '''
    n_hs = np.ceil(len(traj) / m)
    return traj.reshape(int(n_hs), m)


def x_over_xT(traj, m=None):
    '''
    Returns x(t) and x(t-T) for the given traj. This can either be DDE
    states (1 dimensional) or ODE states (n_kernels + 1 dimensional).
    In case of a DDE trajectory, the first interval [t0, t0+T) is omitted.
    '''
    if len(traj.shape) == 1:
        xs = traj[m:]
        xTs = traj[:-m]
    elif len(traj.shape) == 2:
        xs = traj.T[0]
        xTs = traj.T[-1]
    else:
        raise ValueError(f'Given traj with shape {traj.shape} is not valid')
    return xs, xTs


def attractor_mean(xs):
    '''
    xs: transient free trajectory of the MG system or x0 component of the ODE
    approximation.
    '''
    return np.mean(xs, axis=0)


def attractor_size(xs):
    '''
    xs: transient free trajectory of the MG system or x0 component of the ODE
    approximation.
    '''
    return np.var(xs, axis=0)


def attractor_diameter(xs):
    return np.max(xs) - np.min(xs)

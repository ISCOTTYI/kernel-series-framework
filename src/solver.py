import numpy as np

'''
DDE SOLVER
----------
In the following we differentiate states 'st' and state histories 'h'. A state
history is given by the 'm' discrete states in the interval [t-T, t) . A system
state 'sys-st' contains the times and corresponding states, a full system state
'fsys-st' additionally contains the derivatives at the corresponding times.
'''

def construct_ts(T, m, t_max, t0=0):
    '''
    For a given time delay T, number of history steps m and maximal time t_max,
    returns equidistant discrete times, such that the time array can be bunched
    into state histories after every m steps. t0 is the start time.
    '''
    dt, n_hs = T/m, np.ceil(t_max/T)
    return np.arange(t0, T*n_hs-dt/2, dt)


def dde_sample_random_h0(m, low=.5, high=1.5):
    '''
    Samples a random state history with values at the m discrete time steps
    between low and high.
    '''
    h = np.random.randint(low=low*1000, high=high*1000, size=m) / 1000
    return h


def dde_fsys_st_from_h(ts, h):
    '''
    For a given state history and corresponding times ts, returns a full system
    state compatible to use as an initial state for the jitcdde integrator.
    '''
    dt = np.abs(ts[1] - ts[0])
    # derivatives are the slopes between states
    dh = (np.roll(h, -1) - h) / dt
    return ts, h, dh


def dde_initial_fsys_st_from_h(T, h, dh=None):
    '''
    For a given time delay T and a initial state history, construct a initial 
    fsys-st for the times [-T, 0). Optionally initial derivatives may be given.
    '''
    m = len(h)
    ts = construct_ts(T, m, 0, t0=-T)
    if dh is None:
        return dde_fsys_st_from_h(ts, h)
    return ts, h, dh


def dde_solver(
        jitc_flow, T, m, t_max,
        h0=None, dh0=None,
        incl_h0=True
    ):
    '''
    Returns sys-sts until t_max of the MG system. Time delay is T, number of
    time steps in state history is m. Optionally a initial state history h0 and
    initial derivatives dh0 may be given, if None, this is sampled randomly.
    If incl_h0 is True, the initial state history at times [-T, 0) are included
    in the output.
    '''
    from jitcdde import jitcdde
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dde = jitcdde(jitc_flow, verbose=False)
        # dde.set_integration_parameters(atol=0, rtol=1e-5)
        # sample initial state histroy
        if h0 is None:
            h0 = dde_sample_random_h0(m)
        t0s, h0, dh0 = dde_initial_fsys_st_from_h(T, h0, dh0)
        # add states in h0 as initial points
        dde.add_past_points(zip(t0s, h0, dh0))
        # construct t values
        ts = construct_ts(T, m, t_max)
        # integrate system
        dde.adjust_diff() # instead of step_on_discontinuities()
        sts = []
        for ti in ts:
            sts.append(dde.integrate(ti))
        sts = np.array(sts).flatten()
        del dde
    if incl_h0:
        return np.concatenate((t0s, ts)), np.concatenate((h0, sts))
    return ts, sts


'''
ODE SOLVER
----------
For the ODE approximation, a state st is a vector in the N + 1
dimensional state space. A state is therefore given by
    st = (x0, x1, ..., xn).
Note, that this corresponds to a approximated state history of the DDE, where xn
is the solution at time t and x0 is the solution at time t-T. The rest of the
points are the solutions at the discrete time steps of a state history with
m = N + 1 steps.
'''

def ode_sample_random_st0(size):
    '''
    Samples a random state.
    '''
    low, high = .5, 1.5
    p = 1000
    # if size is None:
    #     size = N + 1
    return np.random.randint(low=int(low*p), high=int(high*p), size=size) / p


def ode_solver(jitc_flow, T, m, t_max, st0=None):
    '''
    Returns sys-sts of the ODE approximation to the MG system with N
    kernels and time delay T until time t_max. Time discretization is done
    such that time steps are easily bunched into state histories of the DDE.
    '''
    from jitcode import jitcode
    ode = jitcode(jitc_flow, verbose=False)
    # sample initial state
    if st0 is None:
        st0 = ode_sample_random_st0(len(jitc_flow))
    assert len(st0) == len(jitc_flow)
    # construct t values
    ts = construct_ts(T, m, t_max)
    # integrate system
    ode.set_integrator('dopri5', interpolate=False)
    ode.set_initial_value(st0)
    sts = []
    for ti in ts:
        sts.append(ode.integrate(ti))
    sts = np.array(sts)
    del ode
    return ts, sts

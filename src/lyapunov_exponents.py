import numpy as np
from scipy.stats import sem
from src.solver import dde_sample_random_h0, construct_ts, dde_initial_fsys_st_from_h, ode_sample_random_st0


def dde_lyap_spec(
        jitc_flow, T, m, t_max,
        n_lyap=1, tr_skip=.1, print_est_err=False
    ):
    from jitcdde import jitcdde_lyap
    T = np.around(T, 8)
    # specify system
    dde = jitcdde_lyap(jitc_flow, n_lyap=n_lyap, verbose=False)
    # sample initial state histroy
    h0 = dde_sample_random_h0(m)
    t0s, h0, dh0 = dde_initial_fsys_st_from_h(T, h0)
    # add states in h0 as initial points
    dde.add_past_points(zip(t0s, h0, dh0))
    # integrate system
    # np.arange(0, 10000, 10)
    ts = construct_ts(T, m, t_max)
    # ts = np.arange(*integ_ts)
    dde.adjust_diff() # instead of step_on_discontinuities()
    lyaps = [] # finite time LEs
    weis = []
    for ti in ts: # dde.t+ts:
        st, ly, wei = dde.integrate(ti)
        lyaps.append(ly) # do i need weights?
        weis.append(wei)
    del dde
    lyaps = np.vstack(lyaps)
    # average to get LE according to Benettin
    Lyaps = []
    if type(tr_skip) is float and tr_skip > 0 and tr_skip < 1:
        tr_i = int(len(lyaps) * tr_skip)
    elif type(tr_skip) is int:
        tr_i = tr_skip
    else:
        raise Exception('tr_skip cannot be interpreted!')
    for i in range(n_lyap):
        val = np.average(lyaps[tr_i:, i], weights=weis[tr_i:])
        Lyaps.append(val)
        if print_est_err:
            stderr = sem(lyaps[tr_i:, i])
            print("%i. LE: %.8f +/- %.8f" % (i+1, val, stderr))
    return np.array(Lyaps)


def ode_lyap_spec(
        jitc_flow, T, m, t_max,
        n_lyap=1, tr_skip=.1, print_est_err=False
    ):
    from jitcode import jitcode_lyap
    ode = jitcode_lyap(jitc_flow, n_lyap=n_lyap, verbose=False)
    # sample initial state
    st0 = ode_sample_random_st0(len(jitc_flow))
    # integrate system
    # ts = np.arange(5000, 10000, 10)
    # ts = np.arange(*integ_ts) # ERROR: 100.000?
    ts = construct_ts(T, m, t_max)
    ode.set_integrator('RK45', interpolate=False)
    ode.set_initial_value(st0)
    lyaps = [] # finite time LEs
    for ti in ts:
        lyaps.append(ode.integrate(ti)[1])
    del ode
    lyaps = np.vstack(lyaps)
    # average to get LE according to Benettin
    Lyaps = []
    if type(tr_skip) is float and tr_skip > 0 and tr_skip < 1:
        tr_i = int(len(lyaps) * tr_skip)
    elif type(tr_skip) is int:
        tr_i = tr_skip
    else:
        raise Exception('tr_skip cannot be interpreted!')
    for i in range(n_lyap):
        val = np.average(lyaps[tr_i:, i])
        Lyaps.append(val)
        if print_est_err:
            stderr = sem(lyaps[tr_i:, i])
            print("%i. LE: %.8f +/- %.8f" % (i+1, val, stderr))
    return np.array(Lyaps)

import numpy as np

#############################
#                           #
#    MACKEY-GLASS SYSTEM    #
#                           #
#############################

# Constants
MG_const = (MG_alpha, MC_beta, MG_gamma) = (.2, .1, 10) # values from Lakshmanan book and Claudius Paper
MG_nontriv_FP = ((MG_alpha - MC_beta) / MC_beta) ** (1 / MG_gamma)

def mackey_glass_dde_flow(x, xT, const=None):
    if const is None:
        alpha, beta, gamma = MG_const
    else:
        alpha, beta, gamma = const
    return -beta * x + alpha * xT / (1 + xT ** gamma)


def mackey_glass_dde_jitc(T, const=None):
    if const is None:
        alpha, beta, gamma = MG_const
    else:
        alpha, beta, gamma = const
    from jitcdde import y, t
    f = [-beta * y(0) + alpha * y(0, t-T) / (1 + y(0, t-T) ** gamma)]
    return f


def mackey_glass_char_eq(x, T, fp=None, const=None):
    '''
    For fp=None, the non-trivial fixpoint ((alpha - beta) / beta) ** (1 / gamma)
    is assumed.
    '''
    if const is None:
        alpha, beta, gamma = MG_const
    else:
        alpha, beta, gamma = const
    if fp is None:
        fp = MG_nontriv_FP
    p, q = x
    pf = np.exp(- p * T) * (alpha * (1 + fp) - alpha * gamma * fp) / (1 + fp) ** 2
    eq = [
        p + beta - pf * np.cos(q * T),
        q + pf * np.sin(q * T)
    ]
    return eq


def mackey_glass_T_hopf(fp=None, const=None):
    '''
    For fp=None, the non-trivial fixpoint ((alpha - beta) / beta) ** (1 / gamma)
    is assumed.

    Linearized system is dx = -b_lin x(t) - a_lin x(t-T)
    '''
    if const is None:
        alpha, beta, gamma = MG_const
    else:
        alpha, beta, gamma = const
    if fp is None:
        fp = MG_nontriv_FP
    b_lin, a_lin = beta, -(alpha - alpha * (gamma - 1) * fp**gamma) / (fp**gamma + 1)**2
    return np.arccos(-b_lin/a_lin) / np.sqrt(a_lin**2 - b_lin**2)


def mackey_glass_ksf_jitc(N, T, const=None):
    from jitcode import y
    if const is None:
        alpha, beta, gamma = MG_const
    else:
        alpha, beta, gamma = const
    T_N = T / N
    f = [
        -beta * y(0) + alpha * y(N) / (1 + y(N) ** gamma)
    ]
    for i in range(1, N + 1):
        f.append(
            (y(i-1) - y(i)) / T_N
        )
    return f


def mackey_glass_ksf_jacobian(N, T, xN, const=None):
    # Returns jacobian at FP = (x*, ..., x*). xN is last component of FP.
    if const is None:
        alpha, beta, gamma = MG_const
    else:
        alpha, beta, gamma = const
    T_N = T / N
    J = np.zeros((N + 1, N + 1), dtype=np.float64)
    for row_idx in range(N + 1):
        if row_idx:
            J[row_idx][row_idx] = - 1 / T_N
            J[row_idx][row_idx - 1] = 1 / T_N
        else:
            J[row_idx][0] = - beta
            J[row_idx][-1] = alpha / (1 + xN ** gamma) - alpha * gamma * xN ** gamma / (1 + xN ** gamma) ** 2
    return J


###########################
#                         #
#    IKEDA-LIKE SYSTEM    #
#                         #
###########################

def ikeda_dde_flow(xT):
    return np.sin(xT)


def ikeda_dde_jitc(T):
    from jitcdde import y, t
    from symengine import sin
    return [sin(y(0, t-T))]


def ikeda_dde_T_hopf():
    return np.pi / 2


def ikeda_ksf_jitc(N, T):
    from jitcode import y
    from symengine import sin
    T_N = T / N
    f = [sin(y(N))]
    for i in range(1, N+1):
        f.append(
            (y(i-1) - y(i)) / T_N
        )
    return f


def ikeda_ksf_jacobian(N, T):
    # Returns jacobian at FP = k * pi
    T_N = T / N
    J = np.zeros((N + 1, N + 1), dtype=np.float64)
    for row_idx in range(N + 1):
        if row_idx:
            J[row_idx][row_idx] = - 1 / T_N
            J[row_idx][row_idx - 1] = 1 / T_N
        else:
            J[row_idx][-1] = -1
    return J

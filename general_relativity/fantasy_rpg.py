"""
This module is based on the symplectic integration technique
described in these papers:

Christian, Pierre and Chan, Chi-Kwan;
"FANTASY: User-Friendly Symplectic Geodesic Integrator
for Arbitrary Metrics with Automatic Differentiation";
`2021 ApJ 909 67 <https://doi.org/10.3847/1538-4357/abdc28>`_

Tao M. Explicit symplectic approximation of
nonseparable Hamiltonians: Algorithm and
long time performance. Phys Rev E. 2016 Oct;94(4-1):043303.
doi: 10.1103/PhysRevE.94.043303. Epub 2016 Oct 10. PMID: 27841574.
"""
import numpy as np

from functools import partial

import utils.symbolic_conversions as cnv
import utils.symbolic_differentiation_tools as dts


def g_inv_tensor(g_inv_func, Q):
    n_coords, N = Q.shape
    G = np.concatenate(
        [
            g_inv_func(*Q[:, i]).reshape((n_coords, n_coords, 1))
            for i in range(N)
        ],
        axis=2
    )
    return G


def g_inv_derivative_tensor_piece(g_inv_deriv_funcs, q):
    n_coords = q.shape[0]
    m = np.concatenate(
        [
            g_inv_deriv_funcs[i](*q).reshape((n_coords, n_coords, 1))
            for i in range(n_coords)
        ],
        axis=2
    )
    return m


def g_inv_derivative_tensor(g_inv_deriv_funcs, Q):
    n_coords, N = Q.shape
    M = np.concatenate(
        [
            g_inv_derivative_tensor_piece(
                g_inv_deriv_funcs,
                Q[:, i]
            ).reshape((n_coords, n_coords, n_coords, 1))
            for i in range(N)
        ],
        axis=3
    )
    return M


def g_inv_dot_p(G, P):
    return np.einsum('ijk,jk->ik', G, P)


def p_dot_dg_dot_p(M, P):
    return np.einsum('il,ijkl,jl->kl', P, M, P)


def psi_a(g_inv_func, g_inv_deriv_funcs, V):
    phase_space_dim, n_coords, N = V.shape
    Q = V[0, :, :]
    Y = V[3, :, :]
    f = g_inv_dot_p(g_inv_tensor(
        g_inv_func, Q),
        Y
    ).reshape((1, n_coords, N))
    h = p_dot_dg_dot_p(
        g_inv_derivative_tensor(g_inv_deriv_funcs, Q),
        Y
    ).reshape((1, n_coords, N))
    Z = np.zeros((1, n_coords, N))
    return np.concatenate([Z, -h / 2, f, Z], axis=0)


def psi_b(g_inv_func, g_inv_deriv_funcs, V):
    phase_space_dim, n_coords, N = V.shape
    X = V[2, :, :]
    P = V[1, :, :]
    f = g_inv_dot_p(g_inv_tensor(
        g_inv_func, X),
        P
    ).reshape((1, n_coords, N))
    h = p_dot_dg_dot_p(
        g_inv_derivative_tensor(g_inv_deriv_funcs, X),
        P
    ).reshape((1, n_coords, N))
    Z = np.zeros((1, n_coords, N))
    return np.concatenate([f, Z, Z, -h / 2], axis=0)


def phi_ha(g_inv_func, g_inv_deriv_funcs, delta, V):
    ''' 
    time-delta flow of H_A = H(q, y) on the extended phase space
    with symplectic 2-form dq ^ dp + dx ^ dy
    
    Only updates x and p
    '''
    return V + delta * psi_a(g_inv_func, g_inv_deriv_funcs, V)


def phi_hb(g_inv_func, g_inv_deriv_funcs, delta, V):
    ''' 
    time-delta flow of H_B = H(x, p) on the extended phase space
    with symplectic 2-form dq ^ dp + dx ^ dy
    
    Only updates q and y
    '''
    return V + delta * psi_b(g_inv_func, g_inv_deriv_funcs, V)


def phi_hc(C, V):
    return 0.5 * np.einsum('ij,jkl->ikl', C, V)


def phi_delta_2(g_inv_func, g_inv_deriv_funcs, C_delta, delta, V):
    _phi_ha_half_delta = partial(
        phi_ha,
        g_inv_func,
        g_inv_deriv_funcs,
        0.5 * delta
    )
    _phi_hb_half_delta = partial(
        phi_hb,
        g_inv_func,
        g_inv_deriv_funcs,
        0.5 * delta
    )
    _phi_hc_delta = partial(
        phi_hc,
        C_delta(delta)
    )
    return _phi_ha_half_delta(
        _phi_hb_half_delta(
            _phi_hc_delta(
                _phi_hb_half_delta(
                    _phi_ha_half_delta(V)
                )
            )
        )
    )


def gamma_lth_order(ell):
    gamma_l = 1 / (2 - 2 ** (1 / (ell + 1)))
    return gamma_l


def phi_delta_ell(g_inv_func, g_inv_deriv_funcs, C_delta, delta, ell, V):
    """
    Recursive function.  Python default recursion limit is 1000.  An order,
    or ell, greater than 2000 will exceed the python default recusion limit.
    
    for ell >= 2
    """
    if ell == 2:
        return phi_delta_2(g_inv_func, g_inv_deriv_funcs, C_delta, delta, V)
    else:
        gamma_l =  gamma_lth_order(ell)
        _phi1 = partial(
            phi_delta_ell,
            g_inv_func, 
            g_inv_deriv_funcs,
            C_delta,
            gamma_l * delta,
            ell - 2
        )
        _phi2 = partial(
            phi_delta_ell,
            g_inv_func,
            g_inv_deriv_funcs,
            C_delta,
            (1 - 2 * gamma_l) * delta,
            ell - 2
        )
        return _phi1(_phi2(_phi1(V)))


def initial_position_momentum_to_tensor(Q0, P0):
    """
    
    Parameters
    ----------
    Q0 : array like, shape (n_coords, N)
        Initial 4-postion matrix. Rows are commponents of postions and columns
        are different initial positions.
    P0 : array like, shape (n_coords, N)
        Initial 4-momentum matrix. Rows are commponents of momentum and columns
        are different initial momenta.
    
    Returns
    -------
    V0 : ndarray, shape (phase_space_dim, n_coords, N)
        numpy ndarray of double phase space initial postions and momenta for N
        different initial vectors. Rows are the phase space vectors, q, p, x, y
        and columnts are the vector components.  The depth axis is for each
        initial condtion.
    
    """
    if len(Q0.shape) == 1:
        n_coords = Q0.shape[0]
        N = 1
    else:
        n_coords, N = Q0.shape
    V0 = np.concatenate(
        [
            Q0.reshape((1, n_coords, N)),
            P0.reshape((1, n_coords, N)),
            Q0.reshape((1, n_coords, N)),
            P0.reshape((1, n_coords, N))
        ],
        axis=0
    )
    return V0


def phi_hc_transform_matrix(omega):
    def _inner(delta):
        a = np.cos(2 * omega * delta)
        b = np.sin(2 * omega * delta)
        C = np.array(
            [
                [1 + a, b, 1 - a, -b],
                [-b, 1 + a, b, 1 - a],
                [1 - a, -b, 1 + a, b],
                [b, 1 - a, -b, 1 + a]
            ]
        )
        return C
    return _inner


def geodesic_integrator(g_sym_inv, q, n_timesteps, delta, omega, Q0, P0, 
                        order=2):
    """
    
    Parameters
    ----------
    g_sym_inv : Sympy Matrix expression
        Matrix expression for the inverse metric tensor, expressed using 
        chosen symbolic coordinate variables.
    q : Sympy Matrix expression
        Vecotor contraining the symbolic variables of coordinates used in the
        provided metric.
    n_timesteps : int
        Number of timesteps to perform integration on.
    delta : float
        Timestep spacing.
    omega : float
        Hamiltonian coupling constant, used in the formula for Hamitonain 
        separation to account for non-separable Hamiltonians.
    Q0 : array like, shape (n_coords, N)
        Initial 4-postion matrix. Rows are commponents of postions and columns
        are different initial positions.
    P0 : array like, shape (n_coords, N)
        Initial 4-momentum matrix. Rows are commponents of momentum and columns
        are different initial momenta.
    order : int, optional
        Order of the sympletic integration scheme. Only even orders are
        possible, per the technique. The default is 2.

    Raises
    ------
    ValueError
        If order is not an even, positive integer raise value error on `order`.

    Returns
    -------
    result_list : list
        List of results for each timestep.  Each timestep will contain an
        array of shape (phase_space_dim, n_coords, N).
    
    """
    V0 = initial_position_momentum_to_tensor(Q0, P0)
    print(V0[:, :, 0])
    if (order % 2 != 0) or (order == 0):
        raise ValueError(
            f'{order} -- order must be a non-zero even integer!'
        )
    
    # used for phi_hc
    C_delta = phi_hc_transform_matrix(omega)
    
    g_inv_func = cnv.symbolic_to_numpy_func(g_sym_inv, q)
    
    # partial derivatives of g w.r.t. to each coord in q symbolic
    # list of functions of q array
    g_inv_deriv_funcs = [
        dts.metric_tensor_partial_derivative(g_sym_inv, q, i)
        for i in range(4)
    ]
    
    result_list = [V0]
    result = V0
    for count in range(n_timesteps):
        result = phi_delta_ell(
            g_inv_func,
            g_inv_deriv_funcs,
            C_delta,
            delta,
            order,
            result
        )
        result_list.append(result)
        
        if not count % 1000:
            print(
                f'On iteration number {count} with delta {delta}'
            )
    return result_list
  # -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 04:02:36 2021

@author: TD
"""
import timeit

import line_element_to_metric_tensor as lmt
import utils.initial_momentum as im
import utils.symbolic_conversions as cnv

from fantasy_rpg import geodesic_integrator


def geodesic_from_metric(q, dq, line_element, metric_tensor_params_sym,
                         Q0, P0, metric_tensor_params, n_timesteps,
                         delta, omega=1, order=2, timelike=True,
                         solve_p0_zeroth_term=True,
                         neg_g_inv=False):
    """
    

    Parameters
    ----------
    q : sympy expression
        sympy Matrix holding the coordinate variables.
    dq : sympy expression
        sympy Matrix holding the differential variables.
    line_element : sympy expression
        Expression for the line element in provided coordinates.
    metric_tensor_params_sym : sympy expression
        sympy Matrix holding the parameter varaibles used in the line element
        or metric tensor.
    Q0 : array like, shape (n_coords, N)
        Initial 4-postion matrix of the test particles. Rows are commponents
        of postions and columns are different initial positions.
    P0 : array like, shape (n_coords, N)
        Initial 4-momentum matrix of the test particles. Rows are commponents
        of momentum and columns are different initial momenta. If 3-momentum is 
        provided, the zeroth component will automatically be solved for using
        the metric.  Otherwise if a 4-momentum is provided there is an option
        to ignore the zeroth term and recalculte using the 
        ``solve_p0_zeroth_term`` boolean parameter.
    metric_tensor_params : list
        List of parameter values for the metric tensor.
    n_timesteps : int
        Number of timesteps used in computing the geodesics.
    delta : float
        Interval size for each timestep in the geodesic computations.
    omega : float, optional
        Coupling constant for Hamiltonian Flow.  This allows for non-separable
        Hamiltonians.  The default is 1.
    order : int, optional
        Order of the sympletic integration scheme. The default is 2.
    timelike : bool, optional
        Option to toggle between time-like and null-like geodesics. 
        The default is True.
    solve_p0_zeroth_term : bool, optional
        Option to toggle the automatic computation of the zeroth initial 
        4-momentum term.  Applicable when p0 provided is a list of length 4.
        The default is True.
    neg_g_inv : bool, optional
        Option to toggle the sign convention on the inverse metric tensor.
        Sometimes sympy inverse Matrix method may change the sign convention.
        The default is False

    Returns
    -------
    geod : list
        List of results for each timestep.  Each timestep will contain an
        array of shape (phase_space_dim, n_coords, N).  The phase space will
        contain (q, p, x, y) vectors.
        
        Note: Here q and x are the postions in the double
        phase space, constructed for the symplectic integration scheme, which
        has the 2-form dq ^ dp + dx ^ dy (where ^ means the wedge product
        of differential forms).

    """
    # metric tensor symbolic
    metric_tensor_sym = lmt.line_element_to_metric_tensor(line_element, dq)
    g_sym_inv = metric_tensor_sym.inv()
    del metric_tensor_sym
    # incase signature convention is not correct
    if neg_g_inv:
        g_sym_inv = -g_sym_inv
    
    if (len(Q0.shape) == 1) & (len(P0.shape) == 1):
        Q0 = Q0.reshape((-1, 1))
        P0 = P0.reshape((-1, 1))
    
    # calculate the initial condition for 4-momentum engery term
    if solve_p0_zeroth_term or (P0.shape[0] == 3):
        # try catch in case didnt get sign correct on g_sym_inv
        try:
            P0 = im.solve_energy_term_initial_4_momentum(
                Q0, 
                P0,
                g_sym_inv,
                metric_tensor_params,
                q, 
                metric_tensor_params_sym,
                timelike=timelike
            )
        except TypeError as e:
            print(
                f'CAUGHT: {e}\n'
                'Trying g_sym_inv = -g_sym_inv...\n'
            )
            g_sym_inv = -g_sym_inv
            P0 = im.solve_energy_term_initial_4_momentum(
                Q0, 
                P0,
                g_sym_inv,
                metric_tensor_params,
                q, 
                metric_tensor_params_sym,
                timelike=timelike
            )
    
    # contravariant metric tensor with param values subbed in
    g_sym_inv = cnv.symbolic_obj_subs(
        g_sym_inv,
        [metric_tensor_params_sym],
        [metric_tensor_params]
    )
    
    if order > 2:
        print(
            f'WARNING: order={order} > default=2, delta={delta} '
            '-- you may need to decrease delta to prevent overflow.'
        )
    # geodesic
    s = timeit.default_timer()
    geod = geodesic_integrator(
        g_sym_inv,
        q,
        n_timesteps,
        delta,
        omega,
        Q0,
        P0,
        order=order
    )
    e = timeit.default_timer()
    print(f'{e - s} -- sec to complete geodesic calculations.')
    del e, s
    return geod
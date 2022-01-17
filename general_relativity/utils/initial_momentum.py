# -*- coding: utf-8 -*-
import numpy as np
import sympy as sym

import utils.symbolic_conversions as cnv


def solve_energy_term_initial_4_momentum(Q0, P0, g_sym_inv,
                                         metric_tensor_params,
                                         q, metric_tensor_params_sym,
                                         timelike=True):
    """
    Solve the equation, where P = (p0, p1, p1, p3),
    (P^T).(g^-1).P = (m0*c)^2 for p0 when p1, p2, p3 are given.
    
    space-like not implimented only timelike and null like (timelike false)
    """
    res = []
    for q0, p0 in zip(Q0.T, P0.T):
        p0_sym = sym.zeros(4, 1)
        p0_sym[0, 0] = sym.symbols('p_0')
        p0_sym[1:, 0] = p0
        
        # solve for time component p_0, in initial 4-momentum p0
        g_sym_inv0 = cnv.symbolic_obj_subs(
            g_sym_inv,
            [metric_tensor_params_sym, q], # symbolic variables
            [metric_tensor_params, q0] # variables
        )
        # polynomial for P^2 = P_alpha g^{alpha beta} P_beta
        p_0_poly = (
            np.dot(np.dot(p0_sym.T, g_sym_inv0), p0_sym).flatten()[0]
        )
        # add 1 for timelike
        p_0 = sym.solve(p_0_poly + int(timelike), 'p_0')
        
        if isinstance(p_0[0], sym.Mul):
            print(
                'CAUGHT: p_0_poly solution is not float.\n'
                'Trying g_sym_inv = -g_sym_inv...\n'
            )
            g_sym_inv = -g_sym_inv
            g_sym_inv0 = cnv.symbolic_obj_subs(
                g_sym_inv,
                [metric_tensor_params_sym, q], # symbolic variables
                [metric_tensor_params, q0] # variables
            )
            # polynomial for P^2 = P_alpha g^{alpha beta} P_beta
            p_0_poly = (
                np.dot(np.dot(p0_sym.T, g_sym_inv0), p0_sym).flatten()[0]
            )
            # add 1 for timelike
            p_0 = sym.solve(p_0_poly + int(timelike), 'p_0')
            if isinstance(p_0[0], sym.Mul):
                print(
                    'CAUGHT: p_0_poly solution is not float.\n'
                    'Already tried g_sym_inv = -g_sym_inv!\n'
                    'The given initial 4-position and 3-momentum are '
                    f'likely not consistent with a timelike={timelike} curve.\n'
                    'Returning a nan p_0 energy component...\n'   
                )
                p0 = cnv.symbolic_const_matrix_to_numpy(
                    p0_sym.subs('p_0', np.nan)
                ).flatten()
            else:
               if p_0[0] < 0:
                   p0_sym[0] = p_0[0]
               else:
                   p0_sym[0] = p_0[1]
               p0 = cnv.symbolic_const_matrix_to_numpy(p0_sym).flatten() 
        else:
            if p_0[0] < 0:
                p0_sym[0] = p_0[0]
            else:
                p0_sym[0] = p_0[1]
            p0 = cnv.symbolic_const_matrix_to_numpy(p0_sym).flatten()
        res.append(p0)
    return np.vstack(res).T
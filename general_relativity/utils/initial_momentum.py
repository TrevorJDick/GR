# -*- coding: utf-8 -*-
import numpy as np
import sympy as sym

import utils.symbolic_conversions as cnv


def solve_energy_term_initial_4_momentum(q0, p0, g_sym_inv, params,
                                         q, params_sym, timelike=True):
    p0_sym = sym.zeros(4, 1)
    p0_sym[0, 0] = sym.symbols('p_0')
    p0_sym[1:, 0] = p0
    
    # solve for p_0 in initial 4-momentum p0
    g_sym_inv0 = cnv.symbolic_obj_subs(
        g_sym_inv,
        [params_sym, q], # symbolic variables
        [params, q0] # variables
    )
    
    # add 1 for timelike
    p_0_poly = (
        np.dot(np.dot(p0_sym.T, g_sym_inv0), p0_sym).flatten()[0]
        + int(timelike)
    )
    p_0 = sorted(sym.solve(p_0_poly, 'p_0'))[0]
    p0_sym[0] = p_0
    p0 = cnv.symbolic_const_matrix_to_numpy(p0_sym).flatten()
    return p0



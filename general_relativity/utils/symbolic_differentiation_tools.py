# -*- coding: utf-8 -*-
import sympy as sym

import utils.symbolic_conversions as cnv


def metric_tensor_partial_deriv_symbolic(g_sym, symbolic_variables, i):
    """
    differentiates in each component w.r.t. ith symbolic_variable
    """
    return sym.diff(g_sym, symbolic_variables[i])


def metric_tensor_partial_derivative(g_sym, symbolic_variables, i):
    """
    differentiates in each component w.r.t. ith symbolic_variable and returns
    a function of the variables that when called will return a numpy array.
    
    All metric tensor params must be substituted for values before converting
    to a function
    """
    dg_dqi_sym = metric_tensor_partial_deriv_symbolic(
        g_sym, 
        symbolic_variables,
        i
    )
    dg_dqi = cnv.symbolic_to_numpy_func(dg_dqi_sym, symbolic_variables)
    return dg_dqi
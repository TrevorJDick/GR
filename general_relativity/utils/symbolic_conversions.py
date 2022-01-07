# -*- coding: utf-8 -*-
import numpy as np
import sympy as sym

from scipy.special import jv, kv, iv, yv # bessel functions


def symbolic_to_numpy_func(symbolic_expr_obj, symbolic_variables):
    """
    Converts a symbolic object like a symbol or Matrix into a numpy function
    that returns a numpy array of the same shape as that object.

    Parameters
    ----------
    symbolic_expr_obj : sympy expression
        A sympy expression, list of expressions, or matrix to be evaluated.
    symbolic_variables : list of sympy variables
        A variable or a list of variables, represented in the way that
        arguments will be passed to the function.
        Includes symbols, undefined functions, or matrix symbols.

    Returns
    -------
    function
        Function that will return a numpy array of the same shape as the 
        symbolic_expr_obj.

    """
    bessel = {'besselj': jv, 'besselk':kv, 'besseli':iv, 'bessely':yv}
    libraries = [bessel, 'numpy']  
    return sym.lambdify(
        symbolic_variables,
        symbolic_expr_obj,
        modules=libraries
    )


def symbolic_obj_subs(symbolic_expr_obj, symbolic_variables,
                      variable_constants):
    """
    Converts a symbolic object like a symbol or a matrix into a constant
    version of that object, where all symbolic_variables listed have been 
    substituted with thier respective variable_constants

    Parameters
    ----------
    symbolic_expr_obj : sympy expression
        A sympy expression, list of expressions, or matrix to be evaluated.
    symbolic_variables : list of sympy variables
        A variable or a list of variables, represented in the way that
        arguments will be passed to the function.
        Includes symbols, undefined functions, or matrix symbols.
    variable_constants : list of arrays likes
        The values that will be used to substitute symbolic_variables.

    Returns
    -------
    symbolic_const_obj : numpy array
        numpy array of the sampe shape as the symbolic_expr_obj, now with
        all the provided variables substituted with constants.

    """
    symbolic_variables = np.concatenate(symbolic_variables).flatten()
    variable_constants = np.concatenate(variable_constants)
    subs_map = {
        str(k):v 
        for k, v in zip(symbolic_variables, variable_constants)
    }
    if isinstance(symbolic_expr_obj, list):
        symbolic_const_obj = [o.subs(subs_map) for o in symbolic_expr_obj]
    else:
        symbolic_const_obj = symbolic_expr_obj.subs(subs_map)
    return symbolic_const_obj


def symbolic_const_matrix_to_numpy(A):
    return np.array(A).astype(float)
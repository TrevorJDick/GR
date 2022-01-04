 # -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 03:58:01 2021

@author: TD
"""
import sympy as sym


def create_variables_and_differentials(variables_string,
                                       parameter_var_string=None):
    """
    Parameters
    ----------
    variables_string : str
        Sting of variable names separated by commas or spaces. 
        Greek variable names can be converted to their appropriate symbols
        by sympy.
    parameter_var_string : optional
        String name of the variable used to parameterize the other variables.
        e.g. `sigma` or `t`.  Useful if using q and dq in constructing a
        Lagrangian and symbolically deriving the Euler-Lagrange equations.
        The default is None.

    Raises
    ------
    ValueError
        If parameter_var_string is an empty string raise a value error that
        None should be used instead.

    Returns
    -------
    q : sympy Matrix of shape (n_variables, 1)
        Vector of coordinate variables.  Position 4-vector in whatever 
        coordinates the user decides. 
    dq : sympy Matrix of shape (n_variables, 1)
        Vector of differentials of the coordinate variables.

    """
    q = sym.symbols(variables_string) # tuple
    if parameter_var_string is None:
        q = sym.Matrix(q)
        dq = sym.Matrix(
            [sym.symbols(f'd{q[i].name}') for i in range(len(q))]
        )
    else:
        if parameter_var_string == '':
            raise ValueError(
                f'{parameter_var_string} -- parameter_var_string,'
                ' cannot be an empty string, use None!'
            )
        param_var = sym.symbols(parameter_var_string)
        q = sym.Matrix(
            [sym.Function(e.name)(param_var) for e in q]
        )
        dq = sym.diff(q, param_var)
    return q, dq


def create_param_variables(params_string, parameter_var_string_list=None):
    s = sym.symbols(params_string)
    if isinstance(s, sym.Symbol):
        params_sym = sym.Matrix([s])
    else:
        params_sym = sym.Matrix(s)
    
    if parameter_var_string_list is not None:
        if parameter_var_string_list == []:
            raise ValueError(
                f'{parameter_var_string_list} -- parameter_var_string_list,'
                ' cannot be an empty list, use None!'
            )
        param_vars = [sym.symbols(parameter_var_string) for parameter_var_string in parameter_var_string_list]
        params_sym = sym.Matrix(
            [sym.Function(e.name)(*param_var) for e, param_var in zip(params_sym, param_vars)]
        )
    return params_sym
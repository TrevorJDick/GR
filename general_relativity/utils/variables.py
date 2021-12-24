# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 03:58:01 2021

@author: TD
"""
import sympy as sym


def create_variables_and_differentials(variables_string,
                                       parameter_var_string=None):
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
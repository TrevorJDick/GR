# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 03:51:39 2021

@author: TD
"""
import sympy as sym

import line_element_to_metric_tensor as lmt


def test_line_elem_metric_inverses():
    bools = []
    
    # create variables and dq
    c, dt, dx, dy, dz = sym.symbols('c dt dx dy dz')
    dq = sym.Matrix([dt, dx, dy, dz])
    
    # forward direction
    line_element = sym.expand(-c ** 2 * dt ** 2 + dx ** 2 + dy ** 2 + dz ** 2 + 2 * dt * dz)
    g = lmt.line_element_to_metric_tensor(line_element, dq)
    bools.append(
        sym.Equality(line_element, lmt.metric_tensor_to_line_element(g, dq))
    )
    # backward direction, choosing a different metric tensor that is not symmetric
    g = sym.Matrix(
        [
            [-c ** 2, 0, 0, 2],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    )
    g = (g + g.T) / 2 # make symmetric
    line_element = lmt.metric_tensor_to_line_element(g, dq)
    bools.append(
        sym.Equality(g, lmt.line_element_to_metric_tensor(line_element, dq))
    )
    
    print(
        f'{all(bools)} -- '
        'line element to metric tensor, and metric tensor to line element'
        ' are inverse methods.'
    )
    return


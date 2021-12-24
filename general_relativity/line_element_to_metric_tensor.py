# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 03:51:03 2021

@author: TD
"""
import sympy as sym


def line_element_to_metric_tensor(ds_sqrd, dq):
    n = len(dq)
    dq_dq_permutations = sym.tensorproduct(dq, dq).reshape(n ** 2, 1)
    # must expand so coeff method will work properly!
    g = sym.Matrix(
        [sym.expand(ds_sqrd).coeff(e[0], 1) for e in dq_dq_permutations]
    )
    return g.reshape(n, n)


def metric_tensor_to_line_element(g, dq):
    return sym.expand(sym.Mul(sym.Mul(dq.T, g), dq)[0])
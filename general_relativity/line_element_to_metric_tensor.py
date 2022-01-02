# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 03:51:03 2021

@author: TD
"""
import numpy as np
import sympy as sym


def symmetric_matrix(A):
    return (A + A.T) / 2


def diag_matrix(A):
    return sym.Matrix(np.diag(np.diag(A)))


def line_element_to_metric_tensor(ds_sqrd, dq):
    n = len(dq)
    dq_dq_permutations = sym.tensorproduct(dq, dq).reshape(n ** 2, 1)
    # must expand so coeff method will work properly!
    g = sym.Matrix(
        [sym.expand(ds_sqrd).coeff(e[0], 1) for e in dq_dq_permutations]
    )
    g = g.reshape(n, n)
    diag_g = diag_matrix(g)
    g = (g + diag_g) / 2
    return g


def metric_tensor_to_line_element(g, dq):
    return sym.expand(np.dot(np.dot(dq.T, g), dq).flatten()[0])
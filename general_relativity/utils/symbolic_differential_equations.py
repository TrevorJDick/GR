# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 13:54:00 2021

@author: TD
"""
import sympy as sym


def euler_lagrange_equations(lagrangian, q, dq, param_var):
    euler_lagrange = sym.Matrix(
        [sym.diff(lagrangian, q[i]) - sym.diff(sym.diff(lagrangian, dq[i]), param_var)
         for i in range(len(q))]
    )
    return euler_lagrange
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 04:02:36 2021

@author: TD
"""
import sympy as sym

import FANTASY as ftsy

import line_element_to_metric_tensor as lmt
import utils.variables as vrs



##### Schwartzchild Metric #####
q, dq = vrs.create_variables_and_differentials(
    't, r, theta, phi',
    parameter_var_string=None
)
m = sym.symbols('m')
t, r, theta, phi = q
dt, dr, dtheta, dphi = dq

# metric ds^2
line_element = (
    (1 - 2 * m / r) * dt ** 2 
    - (1 - 2 * m / r) ** -1 * dr ** 2 
    - r ** 2 * (dtheta ** 2 + sym.sin(theta) ** 2 * dphi ** 2)
)

# metric tensor
g = lmt.line_element_to_metric_tensor(line_element, dq)
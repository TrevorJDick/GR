# -*- coding: utf-8 -*-
import sympy as sym

import utils.symbolic_variables as vrs


def schwarzchild_metric():
    q, dq = vrs.create_variables_and_differentials(
        't, r, theta, phi',
        parameter_var_string=None
    )
    params_sym = vrs.create_param_variables('M')
    
    # metric ds^2
    line_element = (
        (1 - 2 * params_sym[0] / q[1]) * dq[0] ** 2 
        - (1 - 2 * params_sym[0] / q[1]) ** -1 * dq[1] ** 2 
        - q[1] ** 2 * (dq[2] ** 2 + sym.sin(q[2]) ** 2 * dq[3] ** 2)
    )
    return q, dq, line_element, params_sym


def kerr_metric():
    q, dq = vrs.create_variables_and_differentials(
        't, r, theta, phi',
        parameter_var_string=None
    )
    params_sym = vrs.create_param_variables('M a')
    
    # metric ds^2
    rs = 2 * params_sym[0]
    rho_sqrd = q[1] ** 2 + params_sym[1] ** 2 * sym.cos(q[2]) ** 2
    delta = q[1] ** 2 - rs * q[1] + params_sym[1] ** 2
    line_element = (
        -(1 - ((rs * q[1]) / rho_sqrd)) * dq[0] ** 2
        + rho_sqrd / delta * dq[1] ** 2
        + rho_sqrd * dq[2] ** 2
        + (q[1] ** 2 + params_sym[1] ** 2 + ((rs * q[1] * params_sym[1] ** 2) / rho_sqrd) * sym.sin(q[2]) ** 2) * dq[3] ** 2
        - (2 * rs * q[1] * params_sym[1] * sym.sin(q[2]) ** 2) / rho_sqrd * dq[0] * dq[3]
    )
    return q, dq, line_element, params_sym


def einstein_rosen_metric():
    q, dq = vrs.create_variables_and_differentials(
        't, rho, phi z',
        parameter_var_string=None
    )
    params_sym = vrs.create_param_variables('Dummy')
    psi = sym.besselj(0, q[1]) * sym.cos(q[0])
    gamma = (
        (1 / 2) * q[1] ** 2 * (sym.besselj(0, q[1]) ** 2 + sym.besselj(1, q[1]) ** 2)
        - q[1] * sym.besselj(0, q[1]) * sym.besselj(1, q[1]) * sym.cos(q[0]) ** 2
    )
    line_element = (
        sym.exp(2 * gamma - 2 * psi) * (-dq[0] ** 2 + dq[1] ** 2)
        + sym.exp(-2 * psi) * q[1] ** 2 * dq[2] ** 2
        + sym.exp(2 * psi) * dq[3] ** 2
    )
    return q, dq, line_element, params_sym
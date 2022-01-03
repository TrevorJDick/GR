# -*- coding: utf-8 -*-
import numpy as np
import sympy as sym

import utils.basic_geodesic_plotter as bgp
import utils.symbolic_variables as vrs

from geodesics_from_metric import geodesic_from_metric


##### Kerr Metric #####
q, dq = vrs.create_variables_and_differentials(
    't, r, theta, phi',
    parameter_var_string=None
)
params_sym = sym.Matrix(sym.symbols('M a'))

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

# intial conditions
metric_tensor_params = [1, 0.5] # M=1, a=0.5
q0 = [0, 20, np.pi / 2, 0]
# initial 3-momentum
p0 = [0, 3.8, 3]

# geodesic
n_timesteps = 10000
delta = 1
geod = geodesic_from_metric(
    q, 
    dq,
    line_element,
    params_sym,
    q0, 
    p0, 
    metric_tensor_params,
    n_timesteps,
    delta, 
    omega=1,
    order=2, 
    timelike=True,
    solve_p0_zeroth_term=True,
    neg_g_inv=False
)


### for plotting ###

# two phase space should convege so will just pick first phase space
eqns_motion = np.array([i[0, :] for i in geod[1:]])

# stock coords
# x = eqns_motion[:, 1]
# y = eqns_motion[:, 2]
# z = eqns_motion[:, 3]

# convert to cartesian for Schwarzchild/Kerr
a = metric_tensor_params[1] # kerr
x = np.sqrt(a ** 2 + eqns_motion[:, 1] ** 2) * np.sin(eqns_motion[:, 2]) * np.cos(eqns_motion[:, 3])
y = np.sqrt(a ** 2 + eqns_motion[:, 1] ** 2) * np.sin(eqns_motion[:, 2]) * np.sin(eqns_motion[:, 3])
z = eqns_motion[:, 1]  * np.cos(eqns_motion[:, 2])


bgp.geodesic_plotter_3d(x, y, z, axes_names=['X', 'Y', 'Z'])

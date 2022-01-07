# -*- coding: utf-8 -*-
import numpy as np

import metrics.metrics as met
import utils.basic_geodesic_plotter as bgp

from geodesics_from_metric import geodesic_from_metric


##### Kerr Metric #####
q, dq, line_element, params_sym = met.kerr_metric()

# intial conditions
metric_tensor_params = [1, 0.5] # M=1, a=0.5
q0 = [0, 20, np.pi / 2, 0]
# initial 3-momentum
p0 = [0, 3.8, 3]

# geodesic
n_timesteps = 11000
delta = 0.25
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
    order=4, 
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

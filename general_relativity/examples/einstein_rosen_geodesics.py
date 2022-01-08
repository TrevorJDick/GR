# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:41:38 2022

@author: TD
"""
import numpy as np

import metrics.metrics as met
import utils.basic_geodesic_plotter as bgp

from geodesics_from_metric import geodesic_from_metric


##### Schwarzchild Metric #####
q, dq, line_element, params_sym = met.einstein_rosen_metric()

# intial conditions
metric_tensor_params = [0] # dummy var
q0 = [0, 10, np.pi / 4, 1]
# initial 3-momentum
p0 = [0, 2, 1]

# geodesic
n_timesteps = 5500
delta = 0.5
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
    neg_g_inv=True
)


### for plotting ###

# two phase space should convege so will just pick first phase space
eqns_motion = np.array([i[0, :] for i in geod[1:]])

# stock coords
# x = eqns_motion[:, 1]
# y = eqns_motion[:, 2]
# z = eqns_motion[:, 3]

# convert to cartesian
x = eqns_motion[:, 1] * np.cos(eqns_motion[:, 2])
y = eqns_motion[:, 1] * np.sin(eqns_motion[:, 2])
z =  eqns_motion[:, 3]


bgp.geodesic_plotter_3d(x, y, z, axes_names=['rho', 'phi', 'z'])

# -*- coding: utf-8 -*-
import numpy as np

import metrics.metrics as met
import utils.basic_geodesic_plotter as bgp

from geodesics_from_metric import geodesic_from_metric


##### Morris-Thorne Metric #####
q, dq, line_element, params_sym = met.morris_thorne_metric()

# intial conditions
metric_tensor_params = [2] # size of wormhole throat
# q0 = [0, 8, np.pi / 2, 5 * (np.pi / 180)] # need to try 0, 5, 10, 15, 20 degrees
# # initial 3-momentum
# p0 = [0, 2.0, .1]
q0 = np.random.normal(loc=0, scale=10, size=4)
q0[0] = 0
# initial 3-momentum
p0 = np.random.normal(loc=0, scale=2, size=3)
print(
    f'{q0} -- initial 4 position\n'
    f'{p0} -- initial 3 momentum'
)

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
print(
    f'{geod[0][0]} -- initial 4 position\n'
    f'{geod[0][1]} -- initial 4 momentum'
)

### for plotting ###

# two phase space should convege so will just pick first phase space
eqns_motion = np.array([i[0, :] for i in geod[1:]])

# convert to cartesian
ell = eqns_motion[:, 1]
theta = eqns_motion[:, 2]
phi = eqns_motion[:, 3]
b0 = metric_tensor_params[0]

r = np.sqrt(b0 ** 2 + ell ** 2)
z = b0 * np.log((r / b0) + np.sqrt((r / b0) ** 2 - 1))
                
x = r * np.cos(theta)
y = r * np.sin(theta)


bgp.geodesic_plotter_3d(x, y, z, axes_names=['x', 'y', 'z'])

# -*- coding: utf-8 -*-
import numpy as np

import metrics.metrics as met
import utils.basic_geodesic_plotter as bgp

from geodesics_from_metric import geodesic_from_metric


##### Schwarzchild Metric #####
q, dq, line_element, params_sym = met.schwarzchild_metric()

# intial conditions
metric_tensor_params = [1] # M=1
Q0 = np.array(
    [
     [0, 40, np.pi / 2, 0],
      [0, 40, np.pi / 2, 0]
     ]
).T
Q0 = np.concatenate(
    [
        Q0,
        np.concatenate(
            [
                np.zeros(10).reshape(1, -1),
                np.random.normal(loc=0, scale=10, size=(3, 10))
            ],
            axis=0
        )
    ],
    axis=1
)

# initial 3-momentum
P0 = np.array(
    [
     [0, 0, 3.83405],
      [0, 0, 4.83405]
     ]
).T
P0 = np.concatenate(
    [
        P0,
        np.random.normal(loc=0, scale=2, size=(3, 10))
    ],
    axis=1
)

# geodesic
n_timesteps = 5500 * 1
delta = 0.5 ** 1
geod = geodesic_from_metric(
    q, 
    dq,
    line_element,
    params_sym,
    Q0, 
    P0, 
    metric_tensor_params,
    n_timesteps,
    delta, 
    omega=1,
    order=2 * 1, 
    timelike=True,
    solve_p0_zeroth_term=True,
    neg_g_inv=False
)


### for plotting ###

# two phase space should convege so will just pick first phase space
eqns_motion = np.array([g[0, :, :] for g in geod]) # shape (n_timesteps, n_coords, n_particles)

# convert to cartesian for Schwarzchild/Kerr
def schwarzchild_to_cartesian(eqns_motion, a):
    cart_eqns_motion = np.zeros_like(eqns_motion)
    cart_eqns_motion[:, 0, :] = eqns_motion[:, 0, :]
    # Eddington–Finkelstein coordinates
    cart_eqns_motion[:, 1, :] = (
        np.sqrt(a ** 2 + eqns_motion[:, 1, :] ** 2) 
        * np.sin(eqns_motion[:, 2, :]) 
        * np.cos(eqns_motion[:, 3, :])
    )
    cart_eqns_motion[:, 2, :] = (
        np.sqrt(a ** 2 + eqns_motion[:, 1, :] ** 2)
        * np.sin(eqns_motion[:, 2, :]) 
        * np.sin(eqns_motion[:, 3, :])
    )
    cart_eqns_motion[:, 3, :] = eqns_motion[:, 1, :]  * np.cos(eqns_motion[:, 2, :])
    return cart_eqns_motion

a = 0 # schwarzchild
cart_eqns_motion = schwarzchild_to_cartesian(eqns_motion, a)
print(f'{cart_eqns_motion.shape} -- shape of equations of motion')

# plot single particle
i = 0
bgp.geodesic_plotter_3d(
    cart_eqns_motion[:, 1, i],
    cart_eqns_motion[:, 2, i],
    cart_eqns_motion[:, 3, i],
    axes_names=['X', 'Y', 'Z']
)


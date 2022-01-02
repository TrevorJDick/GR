# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import utils.symbolic_variables as vrs

from geodesics_from_metric import geodesic_from_metric


##### Schwarzchild Metric #####
q, dq = vrs.create_variables_and_differentials(
    't, r, theta, phi',
    parameter_var_string=None
)
params_sym = sym.Matrix([sym.symbols('M')])

# metric ds^2
line_element = (
    (1 - 2 * params_sym[0] / q[1]) * dq[0] ** 2 
    - (1 - 2 * params_sym[0] / q[1]) ** -1 * dq[1] ** 2 
    - q[1] ** 2 * (dq[2] ** 2 + sym.sin(q[2]) ** 2 * dq[3] ** 2)
)

# intial conditions
metric_tensor_params = [1] # M=1
q0 = [0, 40, np.pi / 2, 0]
# initial 3-momentum
p0 = [0, 0, 3.83405]

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
    solve_p0_zeroth_term=True
)


### for plotting ###

# two phase space should convege so will just pick first phase space
eqns_motion = np.array([i[0, :] for i in geod[1:]])

# stock coords
# x = eqns_motion[:, 1]
# y = eqns_motion[:, 2]
# z = eqns_motion[:, 3]

# convert to cartesian for Schwarzchild/Kerr
a = 0 # schwarzchild
x = np.sqrt(a ** 2 + eqns_motion[:, 1] ** 2) * np.sin(eqns_motion[:, 2]) * np.cos(eqns_motion[:, 3])
y = np.sqrt(a ** 2 + eqns_motion[:, 1] ** 2) * np.sin(eqns_motion[:, 2]) * np.sin(eqns_motion[:, 3])
z = eqns_motion[:, 1]  * np.cos(eqns_motion[:, 2])


fig, axs = plt.subplots(1, 2, sharex=False, sharey=False)

axs[0].scatter(x, y)
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')

axs[1].scatter(x, z)
axs[1].set_xlabel('X')
axs[1].set_ylabel('Z')

plt.tight_layout()
plt.show()
del fig, axs


fig = plt.figure(figsize=(18, 18))
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z, s=80)

ax.plot(x, z, 'r', zdir='y', zs=y.max(), lw=1, alpha=0.75)
ax.plot(y, z, 'g', zdir='x', zs=x.min(), lw=1, alpha=0.75)
ax.plot(x, y, 'k', zdir='z', zs=z.min(), lw=1, alpha=0.75)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
del fig, ax
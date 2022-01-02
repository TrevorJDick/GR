# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 04:02:36 2021

@author: TD
"""
import timeit

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import line_element_to_metric_tensor as lmt
import utils.initial_momentum as im
import utils.symbolic_conversions as cnv
import utils.symbolic_variables as vrs

from fantasy_rpg import geodesic_integrator



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
##### line element #####


### metric tensor symbolic
g_sym = lmt.line_element_to_metric_tensor(line_element, dq)
# g_sym = -g_sym
g_sym_inv = g_sym.inv()
### TODO way to check if signature convention is still correct
g_sym_inv = -g_sym_inv 


### initial conditions
params = np.array([1,])
# initial 4-position
q0 = [0, 40, np.pi / 2, 0]
# initial 3-momentum
p0 = [0, 0, 3.83405]
p0 = im.solve_energy_term_initial_4_momentum(q0, p0, g_sym_inv, params,
                                         q, params_sym, timelike=True)


# contravariant metric function of params and coords
g_inv_func = cnv.symbolic_to_numpy_func(
    g_sym_inv, 
    [params_sym, q]
)
del q, dq, params_sym, line_element, g_sym, g_sym_inv


### geodesics ###
nsteps = 5500
delta = 0.5
omega=1
order=2


s = timeit.default_timer()
geod = geodesic_integrator(
    g_inv_func,
    N=nsteps,
    delta=delta,
    omega=omega,
    q0=q0,
    p0=p0,
    params=params,
    order=order
)
e = timeit.default_timer()
print(f'{e - s} -- sec to complete geodesic calculations.')
del e, s

# two phase space should convege so will just pick first phase space
eqns_motion = np.array([i[0, :] for i in geod[1:]])

# stock coords
# x = eqns_motion[:, 1]
# y = eqns_motion[:, 2]
# z = eqns_motion[:, 3]

# convert to cartesian for Kerr
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
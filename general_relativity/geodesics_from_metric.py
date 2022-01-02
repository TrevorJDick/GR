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
import utils.symbolic_conversions as cnv
import utils.symbolic_variables as vrs

from FANTASYRPG import geodesic_integrator



##### Schwarzchild Metric #####
q, dq = vrs.create_variables_and_differentials(
    't, r, theta, phi',
    parameter_var_string=None
)
params = sym.Matrix([sym.symbols('M')])

# metric ds^2
line_element = (
    (1 - 2 * params[0] / q[1]) * dq[0] ** 2 
    - (1 - 2 * params[0] / q[1]) ** -1 * dq[1] ** 2 
    - q[1] ** 2 * (dq[2] ** 2 + sym.sin(q[2]) ** 2 * dq[3] ** 2)
)

# metric tensor
g_sym = lmt.line_element_to_metric_tensor(line_element, dq)
# g_sym = -g_sym
g_sym_inv = g_sym.inv()
### TODO way to check if signature convention is still correct
g_sym_inv = -g_sym_inv 


# initial conditions, solve for p_0 in initial 4-momentum p0
params_const = np.array([1,])
q0 = np.array([0, 40, np.pi / 2, 0])
p0 = sym.Matrix([sym.symbols('p_0'), 0, 0, 3.83405])

g_sym_inv_0 = cnv.symbolic_obj_subs(
    g_sym_inv,
    [params, q],
    [params_const, q0]
)


p_0_poly = np.dot(np.dot(p0.T, g_sym_inv_0), p0).flatten()[0] + 1  # add 1 for timelike
print(p_0_poly)
p_0 = sorted(sym.solve(p_0_poly, 'p_0'))[0]
p0[0] = p_0
p0 = cnv.symbolic_const_matrix_to_numpy(p0).flatten()


### geodesics ###
nsteps = 5500
delta = 0.5
omega=1
order=4


s = timeit.default_timer()
geod = geodesic_integrator(
    N=nsteps,
    delta=delta,
    omega=omega,
    q0=q0,
    p0=p0,
    param=params_const, # use FANTASYRPG
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
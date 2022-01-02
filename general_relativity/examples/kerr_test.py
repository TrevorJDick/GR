# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:27:04 2021

@author: TD
"""
import timeit

import matplotlib.pyplot as plt
import numpy as np

from einsteinpy.geodesic import Geodesic
from einsteinpy.plotting.geodesic import GeodesicPlotter
### choose one to import for testing
# from FANTASY import geodesic_integrator
from FANTASYRPG import geodesic_integrator

# t, r, theta, phi -- Boyer-Lindquist coords
### Kerr
# M = 1
# a = 0.5
# q0 = [0, 20, np.pi/2, 0]
# p0 = [-0.9764550153430405, 0, 3.8, 3]
# delta = 0.5

### Schwartzchild
nsteps = 5500
M = 1
a = 0
q0 = [0, 40, np.pi / 2, 0]
p0 = [-0.9791466126276669, 0, 0, 3.83405]
# p0 = [0, 0, 0, 3.83405]


### testing
from FANTASYRPG import _metric_tensor_list
g_p0 = np.zeros((4, 4))
for i, row in enumerate(_metric_tensor_list):
    for j, e in enumerate(row):
        if isinstance(e, (int, float)):
            g_p0[i, j] = e
        else:
            g_p0[i, j] = e([M, a], q0)
    del j, e
del i, row

C = np.dot(np.dot(np.array(p0[1:]).T, g_p0[1:, 1:]), np.array(p0[1:])) + 1 # add 1 for timelike dunno why
B = np.dot(g_p0[0, 1:], np.array(p0[1:])) + np.dot(np.array(p0[1:]).T, g_p0[1:, 0])
A = g_p0[0, 0]
p_0 = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
p0[0] = p_0

# p0 = [-.97915, 0, 0, 3.83405]
# q0 = [0, 20, 2, 0]
# p0 = [-.97, 0, 0, 3.4]
delta = 0.5
omega=1
order=2

# FANTASY about 91 delta = 0.5


### load up geodesic_integrator from FANTASY or FANTASYRPG to compare
s = timeit.default_timer()
geod = geodesic_integrator(
    N=nsteps,
    delta=delta,
    omega=omega,
    q0=q0,
    p0=p0,
    # Param=[M, a], #use for FANTASY changed api in FANTASYRPG
    param=[M, a], # use FANTASYRPG
    order=order
)
e = timeit.default_timer()
print(f'{e - s} -- sec to complete geodesic calculations.')
del e, s
# FANTASYRPG about 60 sec

# two phase space should convege so will just pick first phase space
eqns_motion = np.array([i[0, :] for i in geod[1:]])

# stock coords
# x = eqns_motion[:, 1]
# y = eqns_motion[:, 2]
# z = eqns_motion[:, 3]

# convert to cartesian for Kerr
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



s = timeit.default_timer()
einpy_geod = Geodesic(
    metric="Schwarzschild",
    metric_params=(a,),
    position=q0[1:],
    momentum=p0[1:],
    steps=nsteps,
    delta=delta,
    return_cartesian=True,
    omega=omega,
    order=order,
    time_like=True
)
e = timeit.default_timer()
print(f'{e - s} -- sec to complete einsteinpy geodesic calculations.')
del e, s
## 222.6 sec
print(
    f'{einpy_geod.momentum} -- 4-momentum'
)

gpl = GeodesicPlotter()
gpl.plot(einpy_geod)
gpl.show()


einpy_traj = einpy_geod.trajectory[1]
einpy_eqns_motion = einpy_traj[:, [1,2,3]]
ex = einpy_eqns_motion[:, 0]
ey = einpy_eqns_motion[:, 1]
ez = einpy_eqns_motion[:, 2]

fig = plt.figure(figsize=(18, 18))
ax = fig.add_subplot(projection='3d')
ax.scatter(ex, ey, ez, s=80)

ax.plot(ex, ez, 'r', zdir='y', zs=ey.max(), lw=1, alpha=0.75)
ax.plot(ey, ez, 'g', zdir='x', zs=ex.min(), lw=1, alpha=0.75)
ax.plot(ex, ey, 'k', zdir='z', zs=ez.min(), lw=1, alpha=0.75)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
del fig, ax

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:27:04 2021

@author: TD
"""
import timeit

import matplotlib.pyplot as plt
import numpy as np

### choose one to import for testing
# from FANTASY import geodesic_integrator
from FANTASYRPG import geodesic_integrator

# t, r, theta, phi -- Boyer-Lindquist coords
M = 1
a = 0.5
### load up geodesic_integrator from FANTASY or FANTASYRPG to compare
s = timeit.default_timer()
geod = geodesic_integrator(
    N=10000,
    delta=0.5,
    omega=1,
    q0=[0, 20, np.pi/2, 0],
    p0=[-0.9764550153430405, 0, 3.8, 3],
    # Param=[M, a], #use for FANTASY changed api in FANTASYRPG
    param=[M, a], # use FANTASYRPG
    order=2
)
e = timeit.default_timer()
print(f'{e - s} -- sec to complete geodesic calculations.')
del e, s
# FANTASY about 91 sec
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
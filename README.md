# GR: Geodesics From Custom Metric

The philosophy behind this project is building code that is as simple as possible, so that it can be modified and adapted to suit the needs of those wanting to play around with constructing thier own geodesics from custom metrics.  In order to provide a reasonable service of such a goal the methods for computing the space time trajectories needed to already be both resource light and fairly accurate for long time intervals.  Thus, the desicion to use both a symbolic approach to python via sympy and a more accurate sympletic integration scheme for finding geodesics makes the most sense.   The choice to incorporate more abstract elements may be a geat boon or somewhat cumbersome.  We shall see as the utility of the project is futher explored.

## Requirements
The project only relies on `numpy` and `sympy`

### Example Schwarzchild Metric
Defining a metric is both simple and easy to read in a jupyter notebook

1) Define the variables using sympy and create the line element
```
import sympy as sym
import utils.symbolic_variables as vrs

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
```

2) Provide initial condictions and parameter values
```
# intial conditions
metric_tensor_params = [1] # M=1
q0 = [0, 40, np.pi / 2, 0]
# initial 3-momentum
p0 = [0, 0, 3.83405]
```

3) Call up the geodesic_from_metric function with specified timesteps, delta, omega, etc
```
from geodesics_from_metric import geodesic_from_metric

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
```

4) Extract the equations of motion
```
# two phase spaces should convege so will just pick first phase space
eqns_motion = np.array([i[0, :] for i in geod[1:]])
```

5) Do some coordinate transformations if necessary
```
# convert to cartesian for Schwarzchild/Kerr
a = 0 # schwarzchild
x = np.sqrt(a ** 2 + eqns_motion[:, 1] ** 2) * np.sin(eqns_motion[:, 2]) * np.cos(eqns_motion[:, 3])
y = np.sqrt(a ** 2 + eqns_motion[:, 1] ** 2) * np.sin(eqns_motion[:, 2]) * np.sin(eqns_motion[:, 3])
z = eqns_motion[:, 1]  * np.cos(eqns_motion[:, 2])
```

6) Plotting
```
import utils.basic_geodesic_plotter as bgp

bgp.geodesic_plotter_3d(x, y, z, axes_names=['X', 'Y', 'Z'])
```


#### References
```
[1] Christian, Pierre and Chan, Chi-Kwan;
    "FANTASY: User-Friendly Symplectic Geodesic Integrator
    for Arbitrary Metrics with Automatic Differentiation";
    `2021 ApJ 909 67 <https://doi.org/10.3847/1538-4357/abdc28>`__
[2] The EinsteinPy Project (2021).
    EinsteinPy: Python library for General Relativity
    URL https://einsteinpy.org
```

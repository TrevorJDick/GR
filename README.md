# GR: Geodesics From Custom Metric
The project goal is simple: given a metric, through a provided line element of the general form <img src="https://latex.codecogs.com/svg.image?ds^2&space;=&space;g_{ij}&space;dq^i&space;dq^j" title="ds^2 = g_{ij} dq^i dq^j" />, return the geodesics under some initial conditions of postion and momentum.

The underpinning idea for solving the geodesic equations is given by [1] and [2].  A dynamical Hamiltonian system can be solved via a symplectic integration technique that seeks solutions of 4-postion and 4-momentum in an extended phase space.  The technique offers arbitary order local error with the builtin 2n-order for updating each step.  The lowest 2nd order symplectic integration technique already provides realatively low error solutions for large times and also accounts for non-separable Hamiltonians through a coupling term <img src="https://latex.codecogs.com/svg.image?\omega" title="\omega" />.  Thus, it can be necesary at times to tune the parameter <img src="https://latex.codecogs.com/svg.image?\omega" title="\omega" />, in order to achieve well behaved trajectories.

## Cool Plots
#### Schwarzchild M=1, order 4 with 11000 timesteps
![schwarzchild_metric_order_4_2d](general_relativity/images/schwarzchild_metric_order_4_2d.png)
![schwarzchild_metric_order_4_3d](general_relativity/images/schwarzchild_metric_order_4_3d.png)
#### Kerr M=1, a=0.5, order 4 with 6000 timesteps
![kerr_metric_order_4_2d](general_relativity/images/kerr_metric_order_4_2d.png)
![kerr_metric_order_4_3d](general_relativity/images/kerr_metric_order_4_3d.png)
#### Einstein-Rosen Wave, order 2 with 5500 timesteps
coordinates (t, rho, phi, z)
initial 4-position
q0 = [0, 10, np.pi / 4, 1]
initial 3-momentum
p0 = [0, 2, 1]
(note: mainly tesing this more complex metrix works, not entirely sure of good intial conditions to test with)
![image](https://user-images.githubusercontent.com/34322886/148498027-db0dd481-0cfe-4479-8513-5d8a037c7845.png)
![image](https://user-images.githubusercontent.com/34322886/148498212-3d5c0eb6-3327-4ae6-85a5-432b65e89abb.png)

## Philosophy
The philosophy behind this project is building code that is as simple as possible, so that it can be modified and adapted to suit the needs of those wanting to play around with constructing thier own geodesics from custom metrics.  In order to provide a reasonable service of such a goal the methods for computing the space time trajectories needed to already be both resource light and fairly accurate for long time intervals.  Thus, the desicion to use both a symbolic approach to python, via sympy, and a more accurate sympletic integration scheme for finding geodesics made the most sense.   The choice to incorporate more abstract elements may be a geat boon or somewhat cumbersome.  We shall see as the utility of the project is futher explored.

## Requirements
The project only relies on `numpy`, `sympy`, and `matplotlib` for plotting

## Best Way to Contribute
Reference what you use and give credit.  This project was not built in a vacuum.  It has been iterated on and the more the source material of mathematics is understood the more this project will evolve into its own.  Follow the project philosphy, and see how that is reflected in code.  This tool should only become more useful to scientists, students, or anyone wanting to learn.

We need more metrics!  See the the metric folder and metric.py.  One can always add thier own metric as per the goal of the project, but building up a library of metrics would be nice too.

### Example Schwarzchild Metric
Defining a metric is both simple and easy to explore in jupyter notebook first (see example_notebooks/schwarzchild_metric.ipynb)

1) Define the variables using sympy and create the line element
```
import sympy as sym
import utils.symbolic_variables as vrs

##### Schwarzchild Metric #####
q, dq = vrs.create_variables_and_differentials(
    't, r, theta, phi',
    parameter_var_string=None
)
params_sym = vrs.create_param_variables('M')

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

3) Call up the geodesic_from_metric function with specified timesteps, delta, omega, etc.  The geodesic results will be a list of numpy arrays.  The first element of the list is just the intial conditions for the 4-vectors (q0, p0, q0, p0).  The numpy arrays are of shape (4, 4), where the rows are the q, p solutions in the double phase space, so (q1, p1, q2, p2) and the columns are the components of the 4-vectors.
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
# the two phase spaces should converge, so we just pick first phase space postion component
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
Special thanks to https://github.com/GeoffCope/ for sparking this project. Also for work in finding interesting metrics and reducing them to explicit forms, which often involves solving systems of pdes!

```
[1] Christian, Pierre and Chan, Chi-Kwan;
    "FANTASY: User-Friendly Symplectic Geodesic Integrator
    for Arbitrary Metrics with Automatic Differentiation";
    `2021 ApJ 909 67 <https://doi.org/10.3847/1538-4357/abdc28>`__
[2] Tao M. Explicit symplectic approximation of
    nonseparable Hamiltonians: Algorithm and
    long time performance. Phys Rev E. 2016 Oct;94(4-1):043303.
    doi: 10.1103/PhysRevE.94.043303. Epub 2016 Oct 10. PMID: 27841574.
[3] The EinsteinPy Project (2021).
    EinsteinPy: Python library for General Relativity
    URL https://einsteinpy.org
```

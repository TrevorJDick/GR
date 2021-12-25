# Copyright (C) 2020 Pierre Christian and Chi-kwan Chan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''
################### USER GUIDE ###################
FANTASY is a geodesic integration code for arbitrary metrics with automatic differentiation. Please refer to Christian and Chan, 2021 for details.

################### Inputing the Metric ###################
Components of the metric are stored in the functions g00, g01, g02, etc that can be found under the heading "Metric Components". Each of these take as input a list called param, which contains the fixed parameters of the metric (e.g., 'M' and 'a' for the Kerr metric in Boyer-Lindquist coordinates) and a list called Coord, which contains the coordinates (e.g., 'r' and 't' for the Kerr metric in Boyer-Lindquist coordinates). In order to set up a metric,
Step 1) Write down the fixed parameters in a list
Step 2) Write down the coordinates in a list
Step 3) Type the metric into the functions under "Metric Components".

Example: Kerr in Boyer-Lindquist coordinates
Step 1) The fixed parameters are listed as [M,a]
Step 2) The coordinates are listed as [t,r,theta,phi]
Step 3) Type in the metric components, for example, the g11 function becomes:

def g11(param,Coord):
    return (param[1]**2.-2.*param[0]*Coord[1]+Coord[1]**2.)/(Coord[1]**2.+param[1]**2.*cos(Coord[2])**2.)

Extra step) To make your code more readable, you can redefine variables in place of param[i] or Coord[i], for example, the g11 function can be rewritten as:
def g11(param,Coord):
    M = param[0]
    a = param[1]
    r = Coord[1]
    theta = Coord[2]
    return (a**2.-2.*M*r+r**2.)/(r**2.+a**2.*cos(theta)**2.)

################### A Guide on Choosing omega ###################
The parameter omega determines how much the two phase spaces interact with each other. The smaller omega is, the smaller the integration error, but if omega is too small, the equation of motion will become non-integrable. Thus, it is important to find an omega that is appropriate for the problem at hand. The easiest way to choose an omega is through trial and error:

Step 1) Start with omega=1; if you are working in geometric/code units in which all relevant factors are ~unity, this is usually already a good choice of omega
Step 2) If the trajectory varies wildly with time (this indicates highly chaotic, non-integrable behavior), increase omega and redo integration
Step 3) Repeat Step 2) until trajectory converges

################### Running the Code ###################
To run the code, run the function geodesic_integrator(N,delta,omega,q0,p0,param,order). N is the number of steps, delta is the timestep, omega is the interaction parameter between the two phase spaces, q0 is a list containing the initial position, p0 is a list containing the initial momentum, param is a list containing the fixed parameters of the metric (e.g., [M,a] for Kerr metric in Boyer-Lindquist coordinates), and order is the integration order. You can choose either order=2 for a 2nd order scheme or order=4 for a 4th order scheme.

################### Reading the Output ###################
The output is a numpy array indexed by timestep. For each timestep, the output contains four lists:

output[timestep][0] = a list containing the position of the particle at said timestep in the 1st phase space
output[timestep][1] = a list containing the momentum of the particle at said timestep in the 1st phase space
output[timestep][2] = a list containing the position of the particle at said timestep in the 2nd phase space
output[timestep][3] = a list containing the momentum of the particle at said timestep in the 2nd phase space

As long as the equation of motion is integrable (see section "A Guide on Choosing omega"), the trajectories in the two phase spaces will quickly converge, and you can choose either one as the result of your calculation.

################### Automatic Jacobian ###################

Input coordinate transformations for the 0th, 1st, 2nd, 3rd coordinate in functions CoordTrans0, CoordTrans1, CoordTrans2, CoordTrans3. As an example, coordinate transformation from Spherical Schwarzschild to Cartesian Schwarzschild has been provided.

'''

################### Code Preamble ###################

# from pylab import *
import numpy as np
# from IPython.display import clear_output, display


class dual:
    # see https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
    # for more info on classes of numerics
    
    def __init__(self, first, second):
        '''
        Parameters
        ----------
        first
            function value.
        second
            auto derivative value.
        '''
        self.f = first
        self.s = second


    def __mul__(self, other):
        if isinstance(other, dual):
          return dual(self.f * other.f, self.s * other.f + self.f * other.s)
        else:
          return dual(self.f * other, self.s * other)
    
    
    def __rmul__(self, other):
        if isinstance(other, dual):
          return dual(self.f * other.f, self.s * other.f + self.f * other.s)
        else:
          return dual(self.f * other, self.s * other)
    
    
    def __add__(self, other):
        if isinstance(other, dual):
          return dual(self.f + other.f, self.s + other.s)
        else:
          return dual(self.f + other, self.s)
    
    
    def __radd__(self, other):
        if isinstance(other, dual):
          return dual(self.f + other.f, self.s + other.s)
        else:
          return dual(self.f + other, self.s)
      
      
    def __sub__(self, other):
        if isinstance(other, dual):
          return dual(self.f - other.f, self.s - other.s)
        else:
          return dual(self.f - other, self.s)
    
    
    def __rsub__(self, other):
        return dual(other, 0) - self
    
    
    def __truediv__(self, other):
        '''
        when the first component of the divisor is not 0
        '''
        if isinstance(other, dual):
          return dual(
              self.f / other.f,
              (self.s * other.f - self.f * other.s) / (other.f ** 2.0)
          )
        else:
          return dual(self.f / other, self.s / other)
      
    
    def __rtruediv__(self, other):
        return dual(other, 0).__truediv__(self)
    
    
    def __neg__(self):
        return dual(-self.f, -self.s)
    
    
    def __pow__(self, power):
        return dual(self.f ** power, self.s * power * self.f ** (power - 1))
    
    
    def sin(self):
        return dual(np.sin(self.f), self.s * np.cos(self.f))
    
    
    def cos(self):
        return dual(np.cos(self.f), -self.s * np.sin(self.f))
    
    
    def tan(self):
        return self.sin / self.cos
    
    
    def log(self):
        return dual(np.log(self.f), self.s / self.f)
    
    
    def exp(self):
       return dual(np.exp(self.f), self.s * np.exp(self.f))


def dif(func, x):
    funcdual = func(dual(x, 1))
    if isinstance(funcdual, dual):
        return funcdual.s
    else:
        # this is for when the function is a constant, e.g. gtt:=0
        return 0


################### Metric Components ###################

# Diagonal components of the metric
def g00(param, coord):
    M = param[0]
    a = param[1]
    r = coord[1]
    theta = coord[2]
    delta = r ** 2 - 2 * M * r + a ** 2
    rhosq = r ** 2 + a ** 2 * np.cos(theta) ** 2
    return -(r ** 2 + a ** 2 + 2 * M * r * a ** 2 * np.sin(theta) ** 2 / rhosq) / delta


def g11(param, coord):
    M = param[0]
    a = param[1]
    r = coord[1]
    theta = coord[2]
    return (a ** 2 - 2 * M * r + r ** 2) / (r ** 2 + a ** 2 * np.cos(theta) ** 2)


def g22(param, coord):
    M = param[0]
    a = param[1]
    r = coord[1]
    theta = coord[2]
    return 1 / (r ** 2 + a ** 2 * np.cos(theta) ** 2)


def g33(param, coord):
    M = param[0]
    a = param[1]
    r = coord[1]
    theta = coord[2]
    delta = r ** 2 - 2 * M * r + a ** 2
    rhosq = r ** 2 + a ** 2 * np.cos(theta) ** 2
    return (1 / (delta * np.sin(theta) ** 2)) * (1 - 2 * M * r / rhosq)


# Off-diagonal components of the metric
def g01(param, coord):
    return 0


def g02(param, coord):
    return 0


def g03(param, coord):
    M = param[0]
    a = param[1]
    r = coord[1]
    theta = coord[2]
    delta = r ** 2 - 2 * M * r + a ** 2
    rhosq = r ** 2 + a ** 2 * np.cos(theta) ** 2
    return -(2 * M * r * a) / (rhosq * delta)


def g12(param, coord):
    return 0


def g13(param, coord):
    return 0


def g23(param, coord):
    return 0


_metric_tensor_list = [
    [g00, g01, g02, g03],
    [0, g11, g12, g13],
    [0, 0, g22, g23],
    [0, 0, 0, g33]
]
################### Metric Derivatives ###################

def _dm(param, coord, metric_tensor_elem, wrt):
    gij = metric_tensor_elem
    points_a = [dual(coord[i], 0) for i in range(wrt)]
    points_b = [dual(coord[i], 0) for i in range(wrt + 1, len(coord))]
    # wrt is assumed to be an int index value of the coord dif is w.r.t.
    gij_p = lambda p: gij(param, points_a + [p] + points_b)
    return dif(gij_p, coord[wrt])


def dm(param, coord, metric, wrt):
    i, j = metric
    metric_tensor_elem = _metric_tensor_list[i][j]
    return _dm(param, coord, metric_tensor_elem, wrt)

################### Automatic Coordinate Transformation ###################

def CoordTrans0(param, coord):

    M = param[0]
    a = param[1]
    t = coord[0]
    
    return t
        

def CoordTrans1(param, coord):

    M = param[0]
    a = param[1]
    r = coord[1]
    theta = coord[2]
    phi = coord[3]
    
    x = r * np.sin(theta) * np.cos(phi)

    return x


def CoordTrans2(param, coord):

    M = param[0]
    a = param[1]
    r = coord[1]
    theta = coord[2]
    phi = coord[3]
    
    y = r * np.sin(theta) * np.sin(phi)

    return y


def CoordTrans3(param, coord):

    M = param[0]
    a = param[1]
    r = coord[1]
    theta = coord[2]
    
    z = r * np.cos(theta)

    return z


def AutoJacob(param, coord, i, wrt):
    
    point_d = coord[wrt]

    point_0 = dual(coord[0],0)
    point_1 = dual(coord[1],0)
    point_2 = dual(coord[2],0)
    point_3 = dual(coord[3],0)

    if i == 0:
        if wrt == 0:
            return dif(lambda p:CoordTrans0(param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:CoordTrans0(param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:CoordTrans0(param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:CoordTrans0(param,[point_0,point_1,point_2,p]),point_d)

    if i == 1:
        if wrt == 0:
            return dif(lambda p:CoordTrans1(param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:CoordTrans1(param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:CoordTrans1(param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:CoordTrans1(param,[point_0,point_1,point_2,p]),point_d)

    if i == 2:
        if wrt == 0:
            return dif(lambda p:CoordTrans2(param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:CoordTrans2(param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:CoordTrans2(param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:CoordTrans2(param,[point_0,point_1,point_2,p]),point_d)

    if i == 3:
        if wrt == 0:
            return dif(lambda p:CoordTrans3(param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:CoordTrans3(param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:CoordTrans3(param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:CoordTrans3(param,[point_0,point_1,point_2,p]),point_d)
    
        
################### Integrator ###################

def hamil_inside(q, p, param, wrt):
    return (
        p[0] * p[0] * dm(param, q, (0, 0), wrt) 
        + p[1] * p[1] * dm(param, q, (1, 1), wrt) 
        + p[2] * p[2] * dm(param, q, (2, 2), wrt) 
        + p[3] * p[3] * dm(param, q, (3, 3), wrt) 
        + 2 * p[0] * p[1] * dm(param, q, (0, 1), wrt)
        + 2 * p[0] * p[2] * dm(param, q, (0, 2), wrt)
        + 2 * p[0] * p[3] * dm(param, q, (0, 3), wrt) 
        + 2 * p[1] * p[2] * dm(param, q, (1, 2), wrt)
        + 2 * p[1] * p[3] * dm(param, q, (1, 3), wrt)
        + 2 * p[2] * p[3] * dm(param, q, (2, 3), wrt)
    )


def phi_ha(delta, omega, q1, p1, q2, p2, param):
    ''' 
    q1=(t1,r1,theta1,phi1),
    p1=(pt1,pr1,ptheta1,pphi1),
    q2=(t2,r2,theta2,phi2), 
    p2=(pt2,pr2,ptheta2,pphi2)
    '''
    dq1H_p1_0 = 0.5 * (hamil_inside(q1, p2, param, 0))
    dq1H_p1_1 = 0.5 * (hamil_inside(q1, p2, param, 1))
    dq1H_p1_2 =  0.5 * (hamil_inside(q1, p2, param, 2))
    dq1H_p1_3 =  0.5 * (hamil_inside(q1, p2, param, 3))

    p1_update_array = np.array([dq1H_p1_0, dq1H_p1_1, dq1H_p1_2, dq1H_p1_3])
    p1_updated = p1 - delta * p1_update_array

    dp2H_q2_0 = (
        g00(param, q1) * p2[0] 
        + g01(param, q1) * p2[1] 
        + g02(param, q1) * p2[2] 
        + g03(param, q1) * p2[3]
    )
    dp2H_q2_1 = (
        g01(param, q1) * p2[0]
        + g11(param, q1) * p2[1]
        + g12(param, q1) * p2[2]
        + g13(param, q1) * p2[3]
    )
    dp2H_q2_2 = (
        g02(param, q1) * p2[0]
        + g12(param, q1) * p2[1]
        + g22(param, q1) * p2[2]
        + g23(param, q1) * p2[3]
    )
    dp2H_q2_3 = (
        g03(param, q1) * p2[0]
        + g13(param, q1) * p2[1]
        + g23(param, q1) * p2[2]
        + g33(param, q1) * p2[3]
    )

    q2_update_array = np.array([dp2H_q2_0, dp2H_q2_1, dp2H_q2_2, dp2H_q2_3])
    q2_updated = q2 + delta * q2_update_array

    return (q2_updated, p1_updated)


def phi_hb(delta,omega,q1,p1,q2,p2,param):
    ''' 
    q1=(t1,r1,theta1,phi1),
    p1=(pt1,pr1,ptheta1,pphi1),
    q2=(t2,r2,theta2,phi2), 
    p2=(pt2,pr2,ptheta2,pphi2)
    '''
    dq2H_p2_0 = 0.5 * (hamil_inside(q2, p1, param, 0))
    dq2H_p2_1 = 0.5 * (hamil_inside(q2, p1, param, 1))
    dq2H_p2_2 =  0.5 * (hamil_inside(q2, p1, param, 2))
    dq2H_p2_3 =  0.5 * (hamil_inside(q2 ,p1, param, 3))

    p2_update_array = np.array([dq2H_p2_0, dq2H_p2_1, dq2H_p2_2, dq2H_p2_3])
    p2_updated = p2 - delta * p2_update_array

    dp1H_q1_0 = (
        g00(param, q2) * p1[0]
        + g01(param, q2) * p1[1]
        + g02(param, q2) * p1[2]
        + g03(param, q2) * p1[3]
    )
    dp1H_q1_1 = (
        g01(param, q2) * p1[0]
        + g11(param, q2) * p1[1]
        + g12(param, q2) * p1[2]
        + g13(param, q2) * p1[3]
    )
    dp1H_q1_2 = (
        g02(param, q2) * p1[0]
        + g12(param, q2) * p1[1]
        + g22(param, q2) * p1[2]
        + g23(param, q2) * p1[3]
    )
    dp1H_q1_3 = (
        g03(param, q2) * p1[0]
        + g13(param, q2) * p1[1]
        + g23(param, q2) * p1[2]
        + g33(param, q2) * p1[3]
    )

    q1_update_array = np.array([dp1H_q1_0, dp1H_q1_1, dp1H_q1_2, dp1H_q1_3])
    q1_updated = q1 + delta * q1_update_array

    return (q1_updated, p2_updated)


def phi_hc(delta, omega, q1, p1, q2, p2, param):
    q1 = np.array(q1)
    q2 = np.array(q2)
    p1 = np.array(p1)
    p2 = np.array(p2)

    q1_updated = 0.5 * (
        q1 
        + q2 
        + (q1 - q2) * np.cos(2 * omega * delta) 
        + (p1 - p2) * np.sin(2 * omega * delta)
    )
    p1_updated = 0.5 * (
        p1
        + p2 
        + (p1 - p2) * np.cos(2 * omega * delta)
        - (q1 - q2) * np.sin(2 * omega * delta)
    )
    
    q2_updated = 0.5 * (
        q1
        + q2 
        - (q1 - q2) * np.cos(2 * omega * delta)
        - (p1 - p2) * np.sin(2 * omega * delta) 
    )
    p2_updated = 0.5 * (
        p1 
        + p2 
        - (p1 - p2) * np.cos(2 * omega * delta) 
        + (q1 - q2) * np.sin(2 *omega * delta) 
    )

    return (q1_updated, p1_updated, q2_updated, p2_updated)


def updator(delta, omega, q1, p1, q2, p2, param):
    first_ha_step = np.array(
        [
            q1,
            phi_ha(
                0.5 * delta,
                omega,
                q1,
                p1,
                q2,
                p2,
                param
            )[1],
            phi_ha(
                0.5 * delta,
                omega,
                q1,
                p1,
                q2,
                p2,
                param
            )[0],
            p2
        ]
    )
    
    first_hb_step = np.array(
        [
            phi_hb(
                0.5 * delta,
                omega, 
                first_ha_step[0],
                first_ha_step[1],
                first_ha_step[2],
                first_ha_step[3],
                param
            )[0],
            first_ha_step[1],
            first_ha_step[2],
            phi_hb(
                0.5 * delta,
                omega,
                first_ha_step[0],
                first_ha_step[1],
                first_ha_step[2],
                first_ha_step[3],
                param
            )[1]
        ]
    )
    
    hc_step = phi_hc(
        delta,
        omega,
        first_hb_step[0],
        first_hb_step[1],
        first_hb_step[2],
        first_hb_step[3],
        param
    )
    
    second_hb_step = np.array(
        [
            phi_hb(
                0.5 * delta,
                omega,
                hc_step[0],
                hc_step[1],
                hc_step[2],
                hc_step[3],
                param
            )[0],
            hc_step[1],
            hc_step[2],
            phi_hb(
                0.5 * delta,
                omega,
                hc_step[0],
                hc_step[1],
                hc_step[2],
                hc_step[3],
                param
            )[1]
        ]
    )
    
    second_ha_step = np.array(
        [
            second_hb_step[0],
            phi_ha(
                0.5 * delta,
                omega,
                second_hb_step[0],
                second_hb_step[1],
                second_hb_step[2],
                second_hb_step[3],
                param
            )[1],
            phi_ha(
                0.5 * delta,
                omega,
                second_hb_step[0],
                second_hb_step[1],
                second_hb_step[2],
                second_hb_step[3],
                param
            )[0],
            second_hb_step[3]
        ]
    )
    return second_ha_step


def updator_4th_ord(delta, omega, q1, p1, q2, p2, param):
    z14 = 1.3512071919596578
    z04 = -1.7024143839193155
    step1 = updator(delta * z14, omega, q1, p1, q2, p2, param)
    step2 = updator(delta * z04, omega, step1[0], step1[1], step1[2], step1[3], param)
    step3 = updator(delta * z14, omega, step2[0], step2[1], step2[2], step2[3], param)
    return step3


def geodesic_integrator(N, delta, omega, q0, p0, param, order=2):
    q1, q2, p1, p2 = (q0, q0, p0, p0)
    
    result_list = [[q1,p1,q2,p2]]
    result = (q1,p1,q2,p2)
    
    if order == 2:
        updator_func = updator
    elif order == 4:
        updator_func = updator_4th_ord
    else:
        raise ValueError(
            f'{order} -- not supported integration order scheme!'
        )
    
    for count, timestep in enumerate(range(N)):
        result = updator_func(
            delta,
            omega,
            result[0],
            result[1],
            result[2],
            result[3],
            param
        )
        result_list += [result]
        
        if not count % 1000:
            print(
                f'On iteration number {count} with delta {delta}'
            )
    return result_list

import numpy as np
import sympy as sym

import dual_number as dn
import line_element_to_metric_tensor as lmt
import utils.symbolic_conversions as cnv
import utils.symbolic_variables as vrs



def dif(func, x):
    """
    Calculates first (partial) derivative of ``func`` at ``x``
    Parameters
    ----------
    func : callable
        Function to differentiate
    x : float
        Point, at which, the derivative will be evaluated
    Returns
    _______
    float
        First partial derivative of ``func`` at ``x``
    """
    funcdual = func(dn.DualNumber(x, 1.0))

    if isinstance(funcdual, dn.DualNumber):
        return funcdual.deriv

    return 0.0


# def dm(g, g_prms, coords, indices, wrt):
#     """
#     Computes derivative of metric elements
#     Parameters
#     ----------
#     g : callable
#         Metric (Contravariant) Function
#     g_prms : array_like
#         Tuple of parameters to pass to the metric
#         E.g., ``(a,)`` for Kerr
#     coords : array_like
#         4-Position
#     indices : array_like
#         2-tuple, containing indices, indexing a metric
#         element, whose derivative will be calculated
#     wrt : int
#         Coordinate, with respect to which, the derivative
#         will be calculated
#         Takes values from ``[0, 1, 2, 3]``
#     Returns
#     -------
#     float
#         Value of derivative of metric element at ``coords``
#     Raises
#     ------
#     ValueError
#         If ``wrt`` is not in [1, 2, 3, 4]
#         or ``len(indices) != 2``
#     """
#     if wrt not in [0, 1, 2, 3]:
#         raise ValueError(f"wrt takes values from [0, 1, 2, 3]. Supplied value: {wrt}")

#     if len(indices) != 2:
#         raise ValueError("indices must be a 2-tuple containing indices for the metric.")

#     dual_coords = [
#         dn.DualNumber(coords[0], 0.0),
#         dn.DualNumber(coords[1], 0.0),
#         dn.DualNumber(coords[2], 0.0),
#         dn.DualNumber(coords[3], 0.0),
#     ]

#     # Coordinate, against which, derivative will be propagated
#     dual_coords[wrt].deriv = 1.0

#     return dif(lambda q: g(g_prms, dual_coords)[indices], coords[wrt])


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
g_sym_inv = g_sym.inv()
### TODO way to check if signature convention is still correct
g_sym_inv = -g_sym_inv

# contravariant metric function of params and coords
g_sym_inv_func = cnv.symbolic_to_numpy_func(
    g_sym_inv, 
    [params, q]
)
del q, dq, params, line_element, g_sym, g_sym_inv

################### Metric Derivatives ###################

def dm(g, params, coords, indices, wrt):
    q_a = [dn.DualNumber(coords[i], 0) for i in range(wrt)]
    q_b = [dn.DualNumber(coords[i], 0) for i in range(wrt + 1, len(coords))]
    # wrt is assumed to be an int index value of the coord dif is w.r.t.
    gij_p = lambda p: g_sym_inv_func(params, q_a + [p] + q_b)[indices]
    return dif(gij_p, coords[wrt])


# def dm(g, params, coords, indices, wrt):
#     return _dm(g, params, coords, indices, wrt)

################### Automatic Coordinate Transformation ###################

# def CoordTrans0(param, coord):

#     M = param[0]
#     a = param[1]
#     t = coord[0]
    
#     return t
   

# def CoordTrans1(param, coord):

#     M = param[0]
#     a = param[1]
#     r = coord[1]
#     theta = coord[2]
#     phi = coord[3]
    
#     x = r * np.sin(theta) * np.cos(phi)

#     return x


# def CoordTrans2(param, coord):

#     M = param[0]
#     a = param[1]
#     r = coord[1]
#     theta = coord[2]
#     phi = coord[3]
    
#     y = r * np.sin(theta) * np.sin(phi)

#     return y


# def CoordTrans3(param, coord):

#     M = param[0]
#     a = param[1]
#     r = coord[1]
#     theta = coord[2]
    
#     z = r * np.cos(theta)

#     return z


# def AutoJacob(param, coord, i, wrt):
    
#     point_d = coord[wrt]

#     point_0 = DualNumber(coord[0],0)
#     point_1 = DualNumber(coord[1],0)
#     point_2 = DualNumber(coord[2],0)
#     point_3 = DualNumber(coord[3],0)

#     if i == 0:
#         if wrt == 0:
#             return dif(lambda p:CoordTrans0(param,[p,point_1,point_2,point_3]),point_d)
#         elif wrt == 1:
#             return dif(lambda p:CoordTrans0(param,[point_0,p,point_2,point_3]),point_d)
#         elif wrt == 2:
#             return dif(lambda p:CoordTrans0(param,[point_0,point_1,p,point_3]),point_d)
#         elif wrt == 3:
#             return dif(lambda p:CoordTrans0(param,[point_0,point_1,point_2,p]),point_d)

#     if i == 1:
#         if wrt == 0:
#             return dif(lambda p:CoordTrans1(param,[p,point_1,point_2,point_3]),point_d)
#         elif wrt == 1:
#             return dif(lambda p:CoordTrans1(param,[point_0,p,point_2,point_3]),point_d)
#         elif wrt == 2:
#             return dif(lambda p:CoordTrans1(param,[point_0,point_1,p,point_3]),point_d)
#         elif wrt == 3:
#             return dif(lambda p:CoordTrans1(param,[point_0,point_1,point_2,p]),point_d)

#     if i == 2:
#         if wrt == 0:
#             return dif(lambda p:CoordTrans2(param,[p,point_1,point_2,point_3]),point_d)
#         elif wrt == 1:
#             return dif(lambda p:CoordTrans2(param,[point_0,p,point_2,point_3]),point_d)
#         elif wrt == 2:
#             return dif(lambda p:CoordTrans2(param,[point_0,point_1,p,point_3]),point_d)
#         elif wrt == 3:
#             return dif(lambda p:CoordTrans2(param,[point_0,point_1,point_2,p]),point_d)

#     if i == 3:
#         if wrt == 0:
#             return dif(lambda p:CoordTrans3(param,[p,point_1,point_2,point_3]),point_d)
#         elif wrt == 1:
#             return dif(lambda p:CoordTrans3(param,[point_0,p,point_2,point_3]),point_d)
#         elif wrt == 2:
#             return dif(lambda p:CoordTrans3(param,[point_0,point_1,p,point_3]),point_d)
#         elif wrt == 3:
#             return dif(lambda p:CoordTrans3(param,[point_0,point_1,point_2,p]),point_d)
    
        
################### Integrator ###################
# _PartHamFlow
# q and p are just single 4,1 vectors
def hamil_inside(q, p, param, wrt):
    g = g_sym_inv_func
    # print(q)
    return (
        p[0] * p[0] * dm(g, param, q, (0, 0), wrt) 
        + p[1] * p[1] * dm(g, param, q, (1, 1), wrt) 
        + p[2] * p[2] * dm(g, param, q, (2, 2), wrt) 
        + p[3] * p[3] * dm(g, param, q, (3, 3), wrt) 
        + 2 * p[0] * p[1] * dm(g, param, q, (0, 1), wrt)
        + 2 * p[0] * p[2] * dm(g, param, q, (0, 2), wrt)
        + 2 * p[0] * p[3] * dm(g, param, q, (0, 3), wrt) 
        + 2 * p[1] * p[2] * dm(g, param, q, (1, 2), wrt)
        + 2 * p[1] * p[3] * dm(g, param, q, (1, 3), wrt)
        + 2 * p[2] * p[3] * dm(g, param, q, (2, 3), wrt)
    )


#_flow_A
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
    
    g_inv_q1 = g_sym_inv_func(param, q1)
    dp2H_q2_0 = np.dot(g_inv_q1[0, :], p2)
    dp2H_q2_1 = np.dot(g_inv_q1[1, :], p2)
    dp2H_q2_2 = np.dot(g_inv_q1[2, :], p2)
    dp2H_q2_3 = np.dot(g_inv_q1[3, :], p2)

    # dp2H_q2_0 = (
    #     g00(param, q1) * p2[0] 
    #     + g01(param, q1) * p2[1] 
    #     + g02(param, q1) * p2[2] 
    #     + g03(param, q1) * p2[3]
    # )
    # dp2H_q2_1 = (
    #     g01(param, q1) * p2[0]
    #     + g11(param, q1) * p2[1]
    #     + g12(param, q1) * p2[2]
    #     + g13(param, q1) * p2[3]
    # )
    # dp2H_q2_2 = (
    #     g02(param, q1) * p2[0]
    #     + g12(param, q1) * p2[1]
    #     + g22(param, q1) * p2[2]
    #     + g23(param, q1) * p2[3]
    # )
    # dp2H_q2_3 = (
    #     g03(param, q1) * p2[0]
    #     + g13(param, q1) * p2[1]
    #     + g23(param, q1) * p2[2]
    #     + g33(param, q1) * p2[3]
    # )

    q2_update_array = np.array([dp2H_q2_0, dp2H_q2_1, dp2H_q2_2, dp2H_q2_3])
    q2_updated = q2 + delta * q2_update_array

    return (q2_updated, p1_updated)


def phi_hb(delta, omega, q1, p1, q2, p2, param):
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
    
    g_inv_q2 = g_sym_inv_func(param, q2)
    dp1H_q1_0 = np.dot(g_inv_q2[0, :], p1)
    dp1H_q1_1 = np.dot(g_inv_q2[1, :], p1)
    dp1H_q1_2 = np.dot(g_inv_q2[2, :], p1)
    dp1H_q1_3 = np.dot(g_inv_q2[3, :], p1)
    
    # dp1H_q1_0 = (
    #     g00(param, q2) * p1[0]
    #     + g01(param, q2) * p1[1]
    #     + g02(param, q2) * p1[2]
    #     + g03(param, q2) * p1[3]
    # )
    # dp1H_q1_1 = (
    #     g01(param, q2) * p1[0]
    #     + g11(param, q2) * p1[1]
    #     + g12(param, q2) * p1[2]
    #     + g13(param, q2) * p1[3]
    # )
    # dp1H_q1_2 = (
    #     g02(param, q2) * p1[0]
    #     + g12(param, q2) * p1[1]
    #     + g22(param, q2) * p1[2]
    #     + g23(param, q2) * p1[3]
    # )
    # dp1H_q1_3 = (
    #     g03(param, q2) * p1[0]
    #     + g13(param, q2) * p1[1]
    #     + g23(param, q2) * p1[2]
    #     + g33(param, q2) * p1[3]
    # )

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

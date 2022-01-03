"""
This module is based on https://github.com/pierrechristian/FANTASY

More detail however can be found in the paper:

Tao M. Explicit symplectic approximation of
nonseparable Hamiltonians: Algorithm and
long time performance. Phys Rev E. 2016 Oct;94(4-1):043303.
doi: 10.1103/PhysRevE.94.043303. Epub 2016 Oct 10. PMID: 27841574.
"""

import numpy as np

import utils.differentiation_tools as dts


def hamil_inside(g_inv_func, q, p, params, wrt):
    """
    On Schwarzchild and Kerr this method runs between
    100 - 500 µs depending on if wrt is 0, 1, 2, or 3
    """
    if isinstance(q, np.ndarray):
        q = q.tolist()
    # part of the hamiltonian flow
    # q and p are just single (4,1) vectors
    partial_g_inv = dts.partial_deriv_metric_tensor(g_inv_func, params, q, wrt)
    # print(q)
    return np.dot(np.dot(p.T, partial_g_inv), p)

 
def phi_ha(g_inv_func, delta, omega, q1, p1, q2, p2, param):
    ''' 
    time-delta flow of H_A = H(q1, p2) on the extended phase space
    with symplectic 2-form dq1 ^ dp1 + dq2 ^ dp2
    '''
    dq1H_p1_0 = 0.5 * (hamil_inside(g_inv_func, q1, p2, param, 0))
    dq1H_p1_1 = 0.5 * (hamil_inside(g_inv_func, q1, p2, param, 1))
    dq1H_p1_2 =  0.5 * (hamil_inside(g_inv_func, q1, p2, param, 2))
    dq1H_p1_3 =  0.5 * (hamil_inside(g_inv_func, q1, p2, param, 3))

    p1_update_array = np.array([dq1H_p1_0, dq1H_p1_1, dq1H_p1_2, dq1H_p1_3])
    p1_updated = p1 - delta * p1_update_array
    
    g_inv_q1 = g_inv_func(param, q1)
    dp2H_q2_0 = np.dot(g_inv_q1[0, :], p2)
    dp2H_q2_1 = np.dot(g_inv_q1[1, :], p2)
    dp2H_q2_2 = np.dot(g_inv_q1[2, :], p2)
    dp2H_q2_3 = np.dot(g_inv_q1[3, :], p2)

    q2_update_array = np.array([dp2H_q2_0, dp2H_q2_1, dp2H_q2_2, dp2H_q2_3])
    q2_updated = q2 + delta * q2_update_array

    return (q2_updated, p1_updated)


def phi_hb(g_inv_func, delta, omega, q1, p1, q2, p2, param):
    ''' 
    time-delta flow of H_B = H(q2, p1) on the extended phase space
    with symplectic 2-form dq1 ^ dp1 + dq2 ^ dp2
    '''
    dq2H_p2_0 = 0.5 * (hamil_inside(g_inv_func, q2, p1, param, 0))
    dq2H_p2_1 = 0.5 * (hamil_inside(g_inv_func, q2, p1, param, 1))
    dq2H_p2_2 =  0.5 * (hamil_inside(g_inv_func, q2, p1, param, 2))
    dq2H_p2_3 =  0.5 * (hamil_inside(g_inv_func, q2 ,p1, param, 3))

    p2_update_array = np.array([dq2H_p2_0, dq2H_p2_1, dq2H_p2_2, dq2H_p2_3])
    p2_updated = p2 - delta * p2_update_array
    
    g_inv_q2 = g_inv_func(param, q2)
    dp1H_q1_0 = np.dot(g_inv_q2[0, :], p1)
    dp1H_q1_1 = np.dot(g_inv_q2[1, :], p1)
    dp1H_q1_2 = np.dot(g_inv_q2[2, :], p1)
    dp1H_q1_3 = np.dot(g_inv_q2[3, :], p1)
    
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


def updator_2nd_ord(g_inv_func, delta, omega, q1, p1, q2, p2, params):
    first_ha_step = np.array(
        [
            q1,
            phi_ha(
                g_inv_func,
                0.5 * delta,
                omega,
                q1,
                p1,
                q2,
                p2,
                params
            )[1],
            phi_ha(
                g_inv_func,
                0.5 * delta,
                omega,
                q1,
                p1,
                q2,
                p2,
                params
            )[0],
            p2
        ]
    )
    
    first_hb_step = np.array(
        [
            phi_hb(
                g_inv_func,
                0.5 * delta,
                omega, 
                *first_ha_step,
                params
            )[0],
            first_ha_step[1],
            first_ha_step[2],
            phi_hb(
                g_inv_func,
                0.5 * delta,
                omega,
                *first_ha_step,
                params
            )[1]
        ]
    )
    
    hc_step = phi_hc(
        delta,
        omega,
        *first_hb_step,
        params
    )
    
    second_hb_step = np.array(
        [
            phi_hb(
                g_inv_func,
                0.5 * delta,
                omega,
                *hc_step,
                params
            )[0],
            hc_step[1],
            hc_step[2],
            phi_hb(
                g_inv_func,
                0.5 * delta,
                omega,
                *hc_step,
                params
            )[1]
        ]
    )
    
    second_ha_step = np.array(
        [
            second_hb_step[0],
            phi_ha(
                g_inv_func,
                0.5 * delta,
                omega,
                *second_hb_step,
                params
            )[1],
            phi_ha(
                g_inv_func,
                0.5 * delta,
                omega,
                *second_hb_step,
                params
            )[0],
            second_hb_step[3]
        ]
    )
    return second_ha_step


def updator_4th_ord(g_inv_func, delta, omega, q1, p1, q2, p2, params):
    z14 = 1.3512071919596578
    z04 = -1.7024143839193155
    step1 = updator_2nd_ord(
        g_inv_func,
        delta * z14,
        omega,
        q1,
        p1,
        q2,
        p2,
        params
    )
    step2 = updator_2nd_ord(
        g_inv_func,
        delta * z04,
        omega,
        *step1,
        params
    )
    step3 = updator_2nd_ord(
        g_inv_func,
        delta * z14,
        omega,
        *step2,
        params
    )
    return step3


def geodesic_integrator(g_inv_func, N, delta, omega, q0, p0, params, order=2):
    q1, q2, p1, p2 = (q0, q0, p0, p0)
    
    result_list = [[q1,p1,q2,p2]]
    result = (q1,p1,q2,p2)
    
    if order == 2:
        updator_func = updator_2nd_ord
    elif order == 4:
        updator_func = updator_4th_ord
    else:
        raise ValueError(
            f'{order} -- not supported integration order scheme!'
        )
    
    for count, timestep in enumerate(range(N)):
        result = updator_func(
            g_inv_func,
            delta,
            omega,
            *result,
            params
        )
        result_list += [result]
        
        if not count % 1000:
            print(
                f'On iteration number {count} with delta {delta}'
            )
    return result_list

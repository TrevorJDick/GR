"""
This module is based on https://github.com/pierrechristian/FANTASY

More detail however can be found in these papers:

Christian, Pierre and Chan, Chi-Kwan;
"FANTASY: User-Friendly Symplectic Geodesic Integrator
for Arbitrary Metrics with Automatic Differentiation";
`2021 ApJ 909 67 <https://doi.org/10.3847/1538-4357/abdc28>`_

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
    # q and p are just single (4,1) vectors
    # q must be list for partial_deriv_metric_tensor
    q = q.tolist()
    # dg/dqi
    partial_g_inv = dts.partial_deriv_metric_tensor(g_inv_func, params, q, wrt)
    return np.dot(np.dot(p.T, partial_g_inv), p)

 
def phi_ha(g_inv_func, delta, q1, p1, q2, p2, params):
    ''' 
    time-delta flow of H_A = H(q1, p2) on the extended phase space
    with symplectic 2-form dq1 ^ dp1 + dq2 ^ dp2
    
    Only updates q2 and p1
    '''
    dha_dq1 = np.array(
        [hamil_inside(g_inv_func, q1, p2, params, i) for i in range(4)]
    ) / 2
    p1_updated = p1 - delta * dha_dq1
    
    dha_dp2 = np.dot(g_inv_func(params, q1), p2)
    q2_updated = q2 + delta * dha_dp2
    return (q1, p1_updated, q2_updated, p2)


def phi_hb(g_inv_func, delta, q1, p1, q2, p2, params):
    ''' 
    time-delta flow of H_B = H(q2, p1) on the extended phase space
    with symplectic 2-form dq1 ^ dp1 + dq2 ^ dp2
    
    Only updates q1 and p2
    '''
    dhb_dq2 = np.array(
        [hamil_inside(g_inv_func, q2, p1, params, i) for i in range(4)]
    ) / 2
    p2_updated = p2 - delta * dhb_dq2
    
    dhb_dq1 = np.dot(g_inv_func(params, q2), p1)
    q1_updated = q1 + delta * dhb_dq1
    return (q1_updated, p1, q2, p2_updated)


def phi_hc(R_delta, q1, p1, q2, p2):
    u = np.concatenate((q1 + q2, p1 + p2))
    v = np.concatenate((q1 - q2, p1 - p2))
    
    q1p1 = (u + np.dot(R_delta, v)) / 2
    q2p2 = (u - np.dot(R_delta, v)) / 2
    
    q1_updated = q1p1[:4]
    p1_updated = q1p1[4:]
    q2_updated = q2p2[:4]
    p2_updated = q2p2[4:]
    return (q1_updated, p1_updated, q2_updated, p2_updated)


def updator_2nd_ord(g_inv_func, delta, R_delta, q1, p1, q2, p2, params):
    first_ha_step = phi_ha(
        g_inv_func,
        0.5 * delta,
        q1,
        p1,
        q2,
        p2,
        params
    )
    
    first_hb_step = phi_hb(
        g_inv_func,
        0.5 * delta,
        *first_ha_step,
        params
    )
    
    hc_step = phi_hc(
        R_delta,
        *first_hb_step
    )
    
    second_hb_step = phi_hb(
        g_inv_func,
        0.5 * delta,
        *hc_step,
        params
    )
    
    second_ha_step = phi_ha(
        g_inv_func,
        0.5 * delta,
        *second_hb_step,
        params
    )
    return np.array(second_ha_step)


def updator_4th_ord(g_inv_func, delta, R_delta, q1, p1, q2, p2, params):
    z14 = 1.3512071919596578
    z04 = -1.7024143839193155
    step1 = updator_2nd_ord(
        g_inv_func,
        delta * z14,
        R_delta,
        q1,
        p1,
        q2,
        p2,
        params
    )
    step2 = updator_2nd_ord(
        g_inv_func,
        delta * z04,
        R_delta,
        *step1,
        params
    )
    step3 = updator_2nd_ord(
        g_inv_func,
        delta * z14,
        R_delta,
        *step2,
        params
    )
    return step3


def geodesic_integrator(g_inv_func, N, delta, omega, q0, p0, params, order=2):
    q0 = np.array(q0)
    p0 = np.array(p0)
    q1, q2, p1, p2 = (q0, q0, p0, p0)
    
    result_list = [[q1, p1, q2, p2]]
    result = (q1, p1, q2, p2)
    
    if order == 2:
        updator_func = updator_2nd_ord
    elif order == 4:
        updator_func = updator_4th_ord
    else:
        raise ValueError(
            f'{order} -- not supported integration order scheme!'
        )
      
    # used for phi_hc
    I = np.identity(4)
    a = np.cos(2 * omega * delta) * I
    b = np.sin(2 * omega * delta) * I
    R_delta = np.zeros((8, 8))
    R_delta[:4, :4] = a
    R_delta[:4, 4:] = b
    R_delta[4:, :4] = -b
    R_delta[4:, 4:] = a
    
    for count, timestep in enumerate(range(N)):
        result = updator_func(
            g_inv_func,
            delta,
            R_delta,
            *result,
            params
        )
        result_list += [result]
        
        if not count % 1000:
            print(
                f'On iteration number {count} with delta {delta}'
            )
    return result_list

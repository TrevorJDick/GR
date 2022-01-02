# -*- coding: utf-8 -*-
import itertools
import numpy as np

import dual_number as dn


def derivative(func, x):
    """
    First derivative of ``func`` as ``x`` calculated through the numerics
    class DualNumber.

    Parameters
    ----------
    func : callable
        Function to differenciate.
    x : float
        Point, at, which, the derivative will be evaluated.

    Returns
    -------
    float
        First derivative of ``func`` as ``x``.

    """
    funcdual = func(dn.DualNumber(x, 1.0))

    if isinstance(funcdual, dn.DualNumber):
        return funcdual.deriv

    return 0.0


def partial_deriv_func_elements(func, params, coords, indices, wrt):
    """
    Computes the first partial derivative according to the index of the 
    component given by ``indices``.
    

    Parameters
    ----------
    func : callable
        Function to take partial derivative.
    params : list
        List of parameters for the function.
    coords : array like
        List of coordinate variables the partial derivative will be w.r.t..
    indices : float or tuple
        If the function is an array or matrix, the indicies of the component
        to calculate the partial derivative of.
    wrt : int
        Index of the coordinate variable to take the partial derivative w.r.t..

    Returns
    -------
    float
        First partial derivative of the component of the function.

    """
    func_ij_p = lambda p: func(
        params,
        coords[: wrt] + [p] + coords[wrt + 1:]
    )[indices]
    return derivative(func_ij_p, coords[wrt])


def partial_deriv_metric_tensor(metric_tensor_func, params, coords, wrt):
    """
    Parameters
    ----------
    metric_tensor_func : callable
        Function for the metric or inverse metric tensor that returns
        an array.
    params : list
        List of parameters for the function.
    coords : array like
        List of coordinate variables the partial derivative will be w.r.t..
    wrt : int
        Index of the coordinate variable to take the partial derivative w.r.t..


    Returns
    -------
    partial : array
        The array of partial derivatives w.r.t. one of the coordinates

    """
    n = len(coords)
    indices_gen = itertools.product(range(n), repeat=2)
    partial = np.array(
        [
            partial_deriv_func_elements(
                metric_tensor_func, 
                params, 
                coords,
                (i, j),
                wrt
            )
            for i, j in indices_gen
        ]
    ).reshape((n, n))
    return partial
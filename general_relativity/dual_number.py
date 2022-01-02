# -*- coding: utf-8 -*-
import numpy as np


class DualNumber:
    """
    Numbers of the form, :math:`a + b\\epsilon`, where
    :math:`\\epsilon^2 = 0` and :math:`\\epsilon \\ne 0`.
    Their addition and multiplication properties make them
    suitable for Automatic Differentiation (AD).
    This module is based on [1]_.
    
    ### TODO add more function support and how to
    
    References
    ----------
    .. [1] Christian, Pierre and Chan, Chi-Kwan;
        "FANTASY: User-Friendly Symplectic Geodesic Integrator
        for Arbitrary Metrics with Automatic Differentiation";
        `2021 ApJ 909 67 <https://doi.org/10.3847/1538-4357/abdc28>`__
    """

    def __init__(self, val, deriv):
        """
        Constructor
        Parameters
        ----------
        val : float
            Value or function value.
        deriv : float
            Directional Derivative
        """
        self.val = val
        self.deriv = deriv


    def __str__(self):
        return f"DualNumber({self.val}, {self.deriv})"


    def __repr__(self):
        return self.__str__()


    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.val + other.val, self.deriv + other.deriv)

        return DualNumber(self.val + other, self.deriv)


    __radd__ = __add__ # right addition = left addition since commutative


    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.val - other.val, self.deriv - other.deriv)

        return DualNumber(self.val - other, self.deriv)


    def __rsub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(other.val - self.val, other.deriv - self.deriv)

        return DualNumber(other, 0) - self


    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(
                self.val * other.val,
                self.deriv * other.val + self.val * other.deriv
            )

        return DualNumber(self.val * other, self.deriv * other)


    __rmul__ = __mul__


    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            if self.val == 0 and other.val == 0:
                return DualNumber(self.deriv / other.deriv, 0.0)

            return DualNumber(
                self.val / other.val,
                (self.deriv * other.val - self.val * other.deriv) / (other.val ** 2),
            )

        return DualNumber(self.val / other, self.deriv / other)


    def __rtruediv__(self, other):
        if isinstance(other, DualNumber):
            if self.val == 0 and other.val == 0:
                return DualNumber(other.deriv / self.deriv, 0.0)

            return DualNumber(
                other.val / self.val,
                (other.deriv * self.val - other.val * self.deriv) / (self.val ** 2),
            )

        return DualNumber(other, 0).__truediv__(self)


    def __eq__(self, other):
        return (self.val == other.val) and (self.deriv == other.deriv)


    def __ne__(self, other):
        return not (self == other)


    def __neg__(self):
        return DualNumber(-self.val, -self.deriv)


    def __pow__(self, power):
        val = self.val ** power
        deriv = self.deriv * power * self.val ** (power - 1)
        return DualNumber(val, deriv)


    def sin(self):
        val = np.sin(self.val)
        deriv = self.deriv * np.cos(self.val)
        return DualNumber(val, deriv)


    def cos(self):
        val = np.cos(self.val)
        deriv = -self.deriv * np.sin(self.val)
        return DualNumber(val, deriv)


    def tan(self):
        val = np.tan(self.val)
        deriv = self.deriv * (1 / np.cos(self.val)) ** 2
        return DualNumber(val, deriv)


    def log(self):
        val = np.log(self.val)
        deriv =  self.deriv / self.val
        return DualNumber(val, deriv)
    
    
    def exp(self):
        val = np.exp(self.val)
        deriv = self.deriv * val
        return DualNumber(val, deriv)
    
    
    def sqrt(self):
        val = np.sqrt(self.val)
        deriv = self.deriv / (2 * np.sqrt(self.val))
        return DualNumber(val, deriv)
    
    
    def arcsin(self):
        val = np.arcsin(self.val)
        deriv = self.deriv / np.sqrt(1 - self.val ** 2)
        return DualNumber(val, deriv)
    
    
    def arccos(self):
        val = np.arccos(self.val)
        deriv = -self.deriv / np.sqrt(1 - self.val ** 2)
        return DualNumber(val, deriv)
    
    
    def arctan(self):
        val = np.arctan(self.val)
        deriv = self.deriv / (1 + self.val ** 2)
        return DualNumber(val, deriv)
    
    
    def absolute(self):
        val = np.absolute(self.val)
        deriv = self.deriv * (self.val / val)
        return DualNumber(val, deriv)
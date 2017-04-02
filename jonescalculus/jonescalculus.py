#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
https://en.wikipedia.org/wiki/Jones_calculus
"""

import numpy as np
import math
import cmath

__author__ = 'Christian Noelleke (https://github.com/Kricki)'


class JonesMatrix(np.matrix):
    """
    Base class for a Jones matrix.
    
    Parameters
    ==========
    :param complex a, b, c, d: The elements of the 2x2 matrix or 2x2 matrix ([[a, b], [c, d]])
    """
    def __new__(cls, *args, **kwargs):
        if len(args) == 4:
            m = [[args[0], args[1]], [args[2], args[3]]]
        elif len(args) == 1 and isinstance(args[0], np.matrix) and args[0].shape == (2, 2):
            m = [[args[0][0, 0], args[0][0, 1]], [args[0][1, 0], args[0][1, 1]]]
            # unpack the numpy matrix to make sure that the returned type is JonesMatrix not numpy.matrix
        else:
            raise ValueError('Expecting 2x2 Matrix or the 4 elements of the Matrix but got %s''' % str(args))
        return np.matrix.__new__(cls, m, dtype=complex)

    def __mul__(self, other):
        if isinstance(other, JonesMatrix):
            return JonesMatrix(np.matrix.__mul__(self, other))
        elif isinstance(other, JonesVector):
            return JonesVector(np.matmul(self, other))
        else:
            return np.matmul(self, other)


class PolarizationFilter(JonesMatrix):
    def __new__(cls, angle):
        j11 = math.cos(angle)**2
        j12 = math.cos(angle)*math.sin(angle)
        j21 = math.cos(angle)*math.sin(angle)
        j22 = math.sin(angle)**2
        return JonesMatrix.__new__(cls, j11, j12, j21, j22)


class HalfWavePlate(JonesMatrix):
    def __new__(cls, angle):
        j11 = math.cos(2*angle)
        j12 = math.sin(2*angle)
        j21 = math.sin(2*angle)
        j22 = -math.cos(2*angle)
        return JonesMatrix.__new__(cls, j11, j12, j21, j22)


class QuarterWavePlate(JonesMatrix):
    def __new__(cls, angle):
        j11 = cmath.exp(1j*cmath.pi/4)*(math.cos(angle)**2 + 1j*math.sin(angle)**2)
        j12 = cmath.exp(1j*cmath.pi/4)*((1-1j)*math.sin(angle)*math.cos(angle))
        j21 = cmath.exp(1j*cmath.pi/4)*((1-1j)*math.sin(angle)*math.cos(angle))
        j22 = cmath.exp(1j*cmath.pi/4)*(math.sin(angle)**2 + 1j*math.cos(angle)**2)
        return JonesMatrix.__new__(cls, j11, j12, j21, j22)


class JonesVector(np.matrix):
    """
    Base class for a Jones vector.
    
    Parameters
    ==========
    :param float x, y: Optional. Elements of the 2x1 matrix or 2x1 matrix ([[x], [y]]).
    :param str preset: Optional. Jones vector is set to a pre-defined state, that can be one of the following:
        'H': Horizontal, 'V': Vertical, 'D': Diagonal, 'A': Anti-diagonal,
        'R': Right-hand circular, 'L': Left-hand circular.
        If "preset" is given, then x,y, normalize are ignored.
    :param bool normalize: If True (default) power of vector is normalized to 1.
    """
    def __new__(cls, *args, preset=None, normalize=True):
        if preset:
            m = [[0.0], [0.0]]  # initialization of presets is done in __init__
        elif len(args) == 2:
            m = [[args[0]], [args[1]]]
        elif len(args) == 1 and isinstance(args[0], np.matrix) and args[0].shape == (2, 1):
            m = [[args[0][0, 0]], [args[0][1, 0]]]
            # unpack the numpy matrix to make sure that the returned type is JonesVector not numpy.matrix
        else:
            raise ValueError('Expecting 2x1 Matrix or the 2 elements of the Matrix but got %s''' % str(args))
        return np.matrix.__new__(cls, m, dtype=complex)

    def __init__(self, *args, preset=None, normalize=True):
        if preset is not None:
            if preset == 'H':
                x, y = 1, 0
            elif preset == 'V':
                x, y = 0, 1
            elif preset == 'D':
                x, y = math.sqrt(2)*1, math.sqrt(2)*1
            elif preset == 'A':
                x, y = math.sqrt(2)*1, -math.sqrt(2)*1
            elif preset == 'R':
                x, y = math.sqrt(2)*1, -math.sqrt(2)*1j
            elif preset == 'L':
                x, y = math.sqrt(2)*1, math.sqrt(2)*1j
            else:
                raise ValueError('Invalid preset')

            self[0, 0] = x
            self[1, 0] = y

        if preset is None and normalize:
            self.normalize()

    @property
    def x(self):
        """
        :return: The x-component of the Jones vector 
        """
        return self[0, 0]

    @property
    def y(self):
        """
        :return: The y-component of the Jones vector 
        """
        return self[1, 0]

    @property
    def power(self):
        """      
        :return: The "power" of the vector, i.e. the sqrt of the sum of the squares of x and y. 
        """
        return math.sqrt(abs(self.x)**2 + abs(self.y)**2)

    @property
    def x_angle(self):
        """
        :return: The angle (in rad) of the polarization vector relative to the x-axis.
        """
        # Only works for linear polarized light. What about rotated ellipses?
        #return math.acos(self.x.real/self.x)
        raise NotImplementedError()

    def normalize(self):
        norm = self.power
        self[0, 0] = self[0, 0]/norm
        self[1, 0] = self[1, 0]/norm


if __name__ == '__main__':
    jv1 = JonesVector(preset='H')
    hwp = HalfWavePlate(math.radians(22.5))
    qwp = QuarterWavePlate(math.radians(45))

    jv2 = qwp*hwp*jv1
    print(jv2)
    print(jv2.power)


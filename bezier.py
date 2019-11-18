#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:20:47 2019

Note that all sources titled "Paper Reference: ..." refer to the paper
"Bernstein Polynomial Toolkit for Trajectory Generation in Multiple Autonomous
Vehicle Missions" written by Calvin Kielas-Jensen and Venanzio Cichella.

@author: ckielasjensen
"""

from collections import defaultdict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, jit
import numpy as np
from scipy.special import binom

from gjk.gjk import gjkNew
#from gjk import gjkNew


#TODO:
#   Implement curve using Bernstein basis instead of de cast
#   Precompute function to be called before optimizer
#   Min dist 3D
#   JIT ahead of time compiling
#   Priorities:
# 1. GJK for 3D
# 2. Speed
# 3.
class BezierParams:
    """Parent class used for storing Bezier parameters

    :param cpts: Control points used to define the Bezier curve. The degree of
        the Bezier curve is equal to the number of columns -1. The dimension of
        the curve is equal to the number of rows.
    :type cpts: numpy.ndarray or None
    :param tau: Values at which to evaluate the Bezier curve.
    :type tau: numpy.ndarray or None
    :param tf: Final time of the Bezier curve trajectory.
    :type tf: float
    :param fastInit: If true, init skips data type checks.
    "type fastInit: bool
    """
    splitCache = defaultdict(dict)
    elevationMatrixCache = defaultdict(dict)
    productMatrixCache = defaultdict(dict)
    diffMatrixCache = defaultdict(dict)
    bezCoefCache = dict()

    def __init__(self, cpts=None, tau=None, tf=1.0, fastInit=False):
        self._tau = tau
        self._tf = float(tf)
        self._curve = None

        if cpts is not None:
            # Checking to see if the cpts are in the desired format. If they
            # are, don't call np.array since it causes a bottleneck in certain
            # iterative procedures.
            if (isinstance(cpts, np.ndarray) and
                    cpts.dtype == 'float64' and
                    cpts.ndim == 2):
                self._cpts = cpts
            else:
                self._cpts = np.array(cpts, ndmin=2, dtype=float)
            self._dim = self._cpts.shape[0]
            self._deg = self._cpts.shape[1] - 1
        else:
            self._dim = None
            self._deg = None

    @property
    def cpts(self):
        return self._cpts

    @cpts.setter
    def cpts(self, value):
        self._curve = None

        if (isinstance(value, np.ndarray) and
                value.ndim == 2 and
                value.dtype == 'float64'):
            newCpts = value
        else:
            newCpts = np.array(value, ndmin=2, dtype=float)

        self._dim = newCpts.shape[0]
        self._deg = newCpts.shape[1] - 1
        self._cpts = newCpts

    @property
    def deg(self):
        return self._deg

    @property
    def degree(self):
        return self._deg

    @property
    def dim(self):
        return self._dim

    @property
    def dimension(self):
        return self._dim

    @property
    def tf(self):
        return self._tf

    @tf.setter
    def tf(self, value):
        self._tf = float(value)
        self._tau = None

    @property
    def tau(self):
        if self._tau is None:
            self._tau = np.linspace(0, self._tf, 1001)
        elif not isinstance(self._tau, np.ndarray):
            self._tau = np.array(self._tau)
        return self._tau

    @tau.setter
    def tau(self, val):
        self._curve = None
        self._tf = val[-1]
        self._tau = val


class Bezier(BezierParams):
    """Bezier curve for trajectory generation

    Allows the user to construct Bezier curves of arbitrary dimension and
    degrees.

    :param cpts: Control points used to define the Bezier curve. The degree of
        the Bezier curve is equal to the number of columns -1. The dimension of
        the curve is equal to the number of rows.
    :type cpts: numpy.ndarray or None
    :param tau: Values at which to evaluate the Bezier curve.
    :type tau: numpy.ndarray or None
    :param tf: Final time of the Bezier curve trajectory.
    :type tf: float

    """

    def __init__(self, cpts=None, tau=None, tf=1.0):
        super().__init__(cpts=cpts, tau=tau, tf=tf)

    def __add__(self, curve):
        return self.add(curve)

    def __sub__(self, curve):
        return self.sub(curve)

    def __mul__(self, curve):
        return self.mul(curve)

    def __truediv__(self, curve):
        return self.div(curve)

    def __pow__(self, power):
        pass

    def __repr__(self):
        return 'Bezier({}, {}, {})'.format(self.cpts, self.tau, self.tf)

#    def __call__(self, t):
#        """Calling the curve returns a single value at the passed in t value
#        """
#        pt = np.empty((self.dim, 1))
#        for i, pts in enumerate(self.cpts):
#            pt[i] = deCasteljauCurve(pts, np.array([t]), self.tf)
#
#        return pt

    def __call__(self, t):
        """Calling the object returns the values of the curve at the t values

        Note that unlike the curve property, this will NOT cache the computed
        values. This is meant to be a convenience function to quickly peek at
        the values of the curve.

        :param t: Single value or numpy array of values at which to compute the
            curve.
        :type t: float or numpy.ndarray
        """
        tau = np.atleast_1d(t)
        curve = np.empty((self.dim, tau.size))
        for i, pts in enumerate(self.cpts):
            curve[i] = deCasteljauCurve(pts, tau, self.tf)

        return curve

    @property
    def x(self):
        """
        Returns a Bezier object whose control points are the 0th row of the
        original object's control points.
        """
        return Bezier(self.cpts[0], tau=self.tau, tf=self.tf)

    @property
    def y(self):
        """
        Returns a Bezier object whose control points are the 1st row of the
        original object's control points. If the original object is less than
        2 dimensions, this returns None.
        """
        if self.dim > 1:
            return Bezier(self.cpts[1], tau=self.tau, tf=self.tf)
        else:
            return None

    @property
    def z(self):
        """
        Returns a Bezier object whose control points are the 2nd row of the
        original object's control points. If the original object is less than
        3 dimensions, this returns None.
        """
        if self.dim > 2:
            return Bezier(self.cpts[2], tau=self.tau, tf=self.tf)
        else:
            return None

    @property
    def curve(self):
#        if self._tau is None:
#            self._tau = np.arange(0, 1.01, 0.01)

        if self._curve is None:
            self._curve = np.zeros([self.dim, len(self.tau)])
            for i, pts in enumerate(self.cpts):
                self._curve[i] = deCasteljauCurve(pts, self.tau, self.tf)

        return self._curve

    def copy(self):
        """Creates an exact, deep copy of the current Bezier object

        :return: Deep copy of Bezier object
        :rtype: Bezier
        """
        return Bezier(self.cpts, self.tau, self.tf)
#        return Bezier(self.cpts, None, self.tf)

    def plot(self, axisHandle=None, showCpts=True, **kwargs):
        """Plots the Bezier curve in 1D or 2D

        Note: Currently only supports plotting in 1D or 2D.

        :param axisHandle: Handle to the figure axis. If it is None, a new
            figure will be plotted.
        :type axisHandle: matplotlib.axes._subplots.AxesSubplot or None
        :param showCpts: Flag that decides whether to show the control points
            in the plot. Default is True.
        :type showCpts: bool
        :param **kwargs: Keyword arguments passed into the plot command. Note
            that the arguments are only passed into the plot command that
            plots the curve and not the command that plots the control points.
        :type **kwargs: dict
        :return: Axis object where the curve was plotted.
        :rtype: matplotlib.axes._subplots.AxesSubplot
        """
        if axisHandle is None:
            fig, ax = plt.subplots()
        else:
            ax = axisHandle

        cpts = np.asarray(self.cpts)

        if self.dim == 1:
            ax.plot(self.tau, self.curve[0], **kwargs)
            if showCpts:
                ax.plot(np.linspace(0, self.tf, self.deg+1),
                        self.cpts.squeeze(), '.--')
        elif self.dim == 2 or (self.cpts[2, :] == 0).all():
            ax.plot(self.curve[0], self.curve[1], **kwargs)
            if showCpts:
                ax.plot(cpts[0], cpts[1], '.--')
        else:
            # Check whether ax is 3D
            if not hasattr(ax, 'get_zlim'):
                parent = ax.get_figure()
                ax.remove()
                ax = parent.add_subplot(111, projection='3d')
            ax.plot(self.curve[0], self.curve[1], self.curve[2], **kwargs)
            if showCpts:
                ax.plot(self.cpts[0], self.cpts[1], self.cpts[2], '.--')

        return ax

    def add(self, other):
        """Adds two Bezier curves

        Paper Reference: Property 7: Arithmetic Operations

        :param other: Other Bezier curve to be added
        :type other: Bezier
        :return: Sum of the two Bezier curves
        :rtype: Bezier
        """
        addedCpts = self.cpts + other.cpts
        newCurve = self.copy()
        newCurve.cpts = addedCpts
        return newCurve

    def sub(self, other):
        """Subtracts two Bezier curves

        If the final times of the two curves do not match, the longer curve is
        split at the final time of the shorter curve. The resulting curve will
        have a final time equal to that of the shorter curve. Note that the
        initial time is always assumed to be 0 for Bezier objects.

        Paper Reference: Property 7: Arithmetic Operations

        :param other: Bezier curve to subtract from the original
        :type other: Bezier
        :return: Original curve - Other curve
        :rtype: Bezier
        """
        if self.tf == other.tf:
            subCpts = self.cpts - other.cpts
            newCurve = self.copy()

        elif self.tf > other.tf:
            tsplit = other.tf
            tempCurve, _ = self.split(tsplit)
            subCpts = tempCurve.cpts - other.cpts
            newCurve = tempCurve.copy()
            newCurve.tf = other.tf

        else:
            tsplit = self.tf
            tempCurve, _ = other.split(tsplit)
            subCpts = self.cpts - tempCurve.cpts
            newCurve = self.copy()
            newCurve.tf = self.tf

        newCurve.cpts = subCpts
        return newCurve

    def mul(self, multiplicand):
        """Computes the product of two Bezier curves.

        Paper Reference: Property 7: Arithmetic Operations

        Source: Section 5.1 of "The Bernstein Polynomial Basis: A Centennial
        Retrospective" by Farouki.

        :param multiplicand: Multiplicand
        :type multiplicand: Bezier
        :return: Product of the two curve
        :rtype: Bezier
        """
        if not isinstance(multiplicand, Bezier):
            msg = 'The multiplicand must be a {} object, not a {}'.format(
                    Bezier, type(multiplicand))
            raise TypeError(msg)

        dim = self.dim
        if multiplicand.dim != dim:
            msg = ('The dimension of both Bezier curves must be the same.\n'
                   'The first dimension is {} and the second is {}'.format(
                           dim, multiplicand.dim))
            raise ValueError(msg)

        a = np.array(self.cpts, ndmin=2)
        b = np.array(multiplicand.cpts, ndmin=2)
        m = self.deg
        n = multiplicand.deg

        c = np.empty((dim, m+n+1))

        try:
            coefMat = Bezier.productMatrixCache[m][n]
        except KeyError:
            coefMat = bezProductCoefficients(m, n)
            Bezier.productMatrixCache[m][n] = coefMat

        for d in range(dim):
            c[d] = multiplyBezCurves(a[d], b[d], coefMat)

#   This code uses Farouki's method for multiplication but does not simplify
#   the problem using matrices.
#        for d in range(dim):
#            for k in np.arange(0, m+n+1):
#                summation = 0
#                for j in np.arange(max(0, k-n), min(m, k)+1):
#                    summation += binom(m, j)  \
#                                 * binom(n, k-j)  \
#                                 * a[d, j]  \
#                                 * b[d, k-j]
#                c[d, k] = summation / binom(m+n, k)

        newCurve = self.copy()
        newCurve.cpts = c

        return newCurve

    def div(self, denominator):
        """Divides one Bezier curve by another.

        The division of two Bezier curves results in a rational Bezier curve.

        Paper Reference: Property 7: Arithmetic Operations

        :param denominator: Denominator of the division
        :type denominator: Bezier
        :return: Rational Bezier curve representing the division of the two
            curves.
        :rtype: RationalBezier
        """
        if not isinstance(denominator, Bezier):
            msg = ('The denominator must be a Bezier object, not a {}. '
                   'Or the module has been reloaded.').format(
                           type(denominator))
            raise TypeError(msg)

        cpts = np.empty((self.dim, self.deg+1))
        for i in range(self.dim):
            for j in range(self.deg+1):
                if self.cpts[i, j] == 0:
                    cpts[i, j] = 0
                elif denominator.cpts[i, j] == 0:
                    cpts[i, j] = np.inf
                else:
                    cpts[i, j] = self.cpts[i, j] / denominator.cpts[i, j]

        weights = denominator.cpts

        return RationalBezier(cpts.astype(np.float64),
                              weights.astype(np.float64),
                              tau=self.tau, tf=self.tf)

    def elev(self, R=1):
        """Elevates the degree of the Bezier curve

        Elevates the degree of the Bezier curve by R (default is 1) and returns
        a new, higher degree Bezier object.

        :param R: Number of degrees to elevate the curve
        :type R: int
        :return: Elevated Bezier curve
        :rtype: Bezier
        """
        try:
            elevMat = Bezier.elevationMatrixCache[self.deg][R]
        except KeyError:
            elevMat = elevMatrix(self.deg, R)
            Bezier.elevationMatrixCache[self.deg][R] = elevMat

        elevPts = []
        for cpts in self.cpts:
            elevPts.append(np.dot(cpts, elevMat))

        elevPts = np.vstack(elevPts)

        curveElev = self.copy()
        curveElev.cpts = elevPts

        return curveElev

    def diff(self):
        """Calculates the derivative of the Bezier curve

        Note that this does not affect the object. Instead it returns the
        derivative.

        :return: Derivative of the Bezier curve
        :rtype: Bezier
        """
        try:
            Dm = Bezier.diffMatrixCache[self.deg][self.tf]
        except KeyError:
            Dm = diffMatrix(self.deg, self.tf)
            Bezier.diffMatrixCache[self.deg][self.tf] = Dm

        cptsDot = []
        for i in range(self.dim):
            cptsDot.append(np.dot(self.cpts[i, :], Dm))

        curveDot = self.copy()
        curveDot.cpts = cptsDot

        return curveDot.elev()

    def integrate(self):
        """Calculates the area under the curve in each dimension

        :return: Area under the curve in each dimension.
        :rtype: numpy.ndarray
        """
        areas = np.empty(self.dim)
        for d in range(self.dim):
            areas[d] = self.tf * sum(self.cpts[d]) / (self.deg+1)

        return areas

    def split(self, tDiv):
        """Splits the curve into two curves at point tDiv

        Note that the two returned curves will have the SAME tf value as the
        original curve. This may result in slightly unexpected behavior for a
        1D curve when plotting since both slices of the original curve will
        also be plotted on [0, tf]. The behavior should work as expected when
        plotting in 2D though.

        Paper Reference: Property 5: The de Casteljau Algorithm

        :param tDiv: Point at which to split the curve
        :type tDiv: float
        :return: Tuple of curves. One before the split point and one after.
        :rtype: tuple(Bezier, Bezier)
        """
        c1 = self.copy()
        c2 = self.copy()

        cpts1 = []
        cpts2 = []

        if np.isnan(tDiv):
            print('[!] Warning, tDiv is {}, changing to 0.'.format(tDiv))
            tDiv = 0

        for d in range(self.dim):
            left, right = deCasteljauSplit(self.cpts[d, :], tDiv, self.tf)
            cpts1.append(left)
            cpts2.append(right[::-1])

        c1.cpts = cpts1
        c2.cpts = cpts2

        return c1, c2

#    def min(self, dim=0, tol=1e-6, maxIter=1000):
#        """Returns the minimum value of the Bezier curve in a single dimension
#
#        Finds the minimum value of the Bezier curve. This is done by first
#        checking the first and last control points since the first and last
#        point lie on the curve. If the first or last control point is not the
#        minimum value, the curve is split at the lowest control point. The new
#        minimum value is then defined as the lowest control point of the two
#        new curves. This continues until the difference between the new minimum
#        and old minimum values is within the desired tolerance.
#
#        :param dim: Which dimension to return the minimum of.
#        :type dim: int
#        :param tol: Tolerance of the minimum value.
#        :type tol: float
#        :param maxIter: Maximum number of iterations to search for the minimum.
#        :type maxIter: int
#        :return: Minimum value of the Bezier curve. None if maximum iterations
#            is met.
#        :rtype: float or None
#        """
#        minVal = min(self.cpts[dim, :])
#        tol = np.abs(tol*np.mean(self.cpts))
#
#        if self.cpts[dim, 0] == minVal:
#            return self.cpts[dim, 0]
#
#        elif self.cpts[dim, -1] == minVal:
#            return self.cpts[dim, -1]
#
#        else:
#            lastMin = np.inf
#            newCurve = self.copy()
#            for _ in range(maxIter):
#                splitPoint = (np.argmin(newCurve.cpts[dim, :])
#                              / (newCurve.deg+1.0))
#                c1, c2 = newCurve.split(splitPoint)
#
#                min1 = min(c1.cpts[dim, :])
#                min2 = min(c2.cpts[dim, :])
#
#                if min1 < min2:
#                    newCurve = c1
#                    newMin = min1
#
#                else:
#                    newCurve = c2
#                    newMin = min2
#
#                if np.abs(newMin-lastMin) < tol:
#                    return newMin
#                else:
#                    lastMin = newMin
#
#            print('Maximum number of iterations met')
#            return None

    def min(self, dim=0, globMin=-np.inf, tol=1e-6):
        """Returns the minimum value of the Bezier curve in a single dimension

        Finds the minimum value of the Bezier curve. This is done by first
        checking the first and last control points since the first and last
        point lie on the curve. If the first or last control point is not the
        minimum value, the curve is split at the lowest control point. The new
        minimum value is then defined as the lowest control point of the two
        new curves. This continues until the difference between the new minimum
        and old minimum values is within the desired tolerance.

        :param dim: Which dimension to return the minimum of.
        :type dim: int
        :param tol: Tolerance of the minimum value.
        :type tol: float
        :param maxIter: Maximum number of iterations to search for the minimum.
        :type maxIter: int
        :return: Minimum value of the Bezier curve. None if maximum iterations
            is met.
        :rtype: float or None
        """
        minIdx = np.argmin(self.cpts[dim, :])
        newMin = min(self.cpts[dim, :])

        error = np.abs(globMin-newMin)

        if error < tol:
            return newMin
        elif minIdx != 0 and minIdx != self.deg:
            splitPoint = minIdx / self.deg
            c1, c2 = self.split(splitPoint)
            c1min = c1.min(dim=dim, globMin=newMin, tol=tol)
            c2min = c2.min(dim=dim, globMin=newMin, tol=tol)

            newMin = min((c1min, c2min))

        return newMin

#    def max4(self, dim=0, tol=1e-6, maxIter=1000):
#        """Returns the maximum value of the Bezier curve in a single dimension
#
#        Finds the maximum value of the Bezier curve. This is done by first
#        checking the first and last control points since the first and last
#        point lie on the curve. If the first or last control point is not the
#        maximum value, the curve is split at the highest control point. The new
#        maximum value is then defined as the highest control point of the two
#        new curves. This continues until the difference between the new maximum
#        and old maximum values is within the desired tolerance.
#
#        :param dim: Which dimension to return the maximum of.
#        :type dim: int
#        :param tol: Tolerance of the maximum value.
#        :type tol: float
#        :param maxIter: Maximum number of iterations to search for the minimum.
#        :type maxIter: int
#        :return: Maximum value of the Bezier curve. None if maximum iterations
#            is met.
#        :rtype: float or None
#        """
#        maxVal = max(self.cpts[dim, :])
#
#        if self.cpts[dim, 0] == maxVal:
#            return self.cpts[dim, 0]
#
#        elif self.cpts[dim, -1] == maxVal:
#            return self.cpts[dim, -1]
#
#        else:
#            lastMax = np.inf
#            newCurve = self.copy()
#            for _ in range(maxIter):
#                splitPoint = (np.argmax(newCurve.cpts[dim, :])
#                              / (newCurve.deg+1.0))
#                c1, c2 = newCurve.split(splitPoint)
#
#                max1 = max(c1.cpts[dim, :])
#                max2 = max(c2.cpts[dim, :])
#
#                if max1 > max2:
#                    newCurve = c1
#                    newMax = max1
#
#                else:
#                    newCurve = c2
#                    newMax = max2
#
#                if np.abs(newMax-lastMax)/newMax < tol:
#                    return newMax
#                else:
#                    lastMax = newMax
#
#            print('Maximum number of iterations met')
#            return None

# TODO:
#   Change error to be absolute not normalized (look @ paper)
    def max(self, dim=0, globMax=np.inf, tol=1e-6):  # , maxIter=1000):
        """Returns the maximum value of the Bezier curve in a single dimension

        Finds the maximum value of the Bezier curve. This is done by first
        checking the first and last control points since the first and last
        point lie on the curve. If the first or last control point is not the
        maximum value, the curve is split at the highest control point. The new
        maximum value is then defined as the highest control point of the two
        new curves. This continues until the difference between the new maximum
        and old maximum values is within the desired tolerance.

        :param dim: Which dimension to return the maximum of.
        :type dim: int
        :param tol: Tolerance of the maximum value.
        :type tol: float
        :param maxIter: Maximum number of iterations to search for the minimum.
        :type maxIter: int
        :return: Maximum value of the Bezier curve. None if maximum iterations
            is met.
        :rtype: float or None
        """
        maxIdx = np.argmax(self.cpts[dim, :])
        newMax = max(self.cpts[dim, :])

        error = np.abs(globMax-newMax)

        if error < tol:
            return newMax
        elif maxIdx != 0 and maxIdx != self.deg:
            splitPoint = maxIdx / self.deg
            c1, c2 = self.split(splitPoint)
            c1max = c1.max(dim=dim, globMax=newMax, tol=tol)
            c2max = c2.max(dim=dim, globMax=newMax, tol=tol)

            newMax = max((c1max, c2max))

        return newMax

#    def max3(self, dim=0, tol=1e-3, maxIter=1000):
#        maxIdx = np.argmax(self.cpts[dim, :])
#        oldMax = max(self.cpts[dim, :])
#
#        oldCurve = self.copy()
#
#        if maxIdx == 0 or maxIdx == self.deg:
#            newMax = oldMax
#        else:
#            for _ in range(maxIter):
#                newCurve = oldCurve.elev(oldCurve.deg+10)
#                newMax = max(newCurve.cpts[dim, :])
#
#                error = np.abs(newMax-oldMax) / newMax
#
#                if error < tol:
#                    break
#
#                oldMax = newMax
#                oldCurve = newCurve.copy()
#
#        return newMax
#
#    def min2(self, dim=0, tol=1e-6, maxIter=1000):
#        """Uses scipy's fminbound to find the minimum value of the Bezier curve
#
#        This method is slower than min because it does not exploit the useful
#        properties of a Bezier curve.
#
#        :param dim: Which dimension to return the minimum of.
#        :type dim: int
#        :param tol: Tolerance of the minimum value.
#        :type tol: float
#        :param maxIter: Maximum number of iterations to search for the minimum.
#        :type maxIter: int
#        :return: Minimum value of the Bezier curve. None if maximum iterations
#            is met.
#        :rtype: float or None
#        """
#        def fun(x): return bezierCurve(self.cpts[dim, :], x, tf=self._tf)
#        _, minVal, status, _ = scipy.optimize.fminbound(fun,
#                                                        x1=0,
#                                                        x2=1,
#                                                        xtol=tol,
#                                                        maxfun=maxIter,
#                                                        full_output=True,
#                                                        disp=1)
#        return minVal[0] if status == 0 else None
#
#    def max2(self, dim=0, tol=1e-6, maxIter=1000):
#        """Uses scipy's fminbound to find the maximum value of the Bezier curve
#
#        This method is slower than max because it does not exploit the useful
#        properties of a Bezier curve.
#
#        :param dim: Which dimension to return the minimum of.
#        :type dim: int
#        :param tol: Tolerance of the minimum value.
#        :type tol: float
#        :param maxIter: Maximum number of iterations to search for the minimum.
#        :type maxIter: int
#        :return: Maximum value of the Bezier curve. None if maximum iterations
#            is met.
#        :rtype: float or None
#        """
#        def fun(x): return -bezierCurve(self.cpts[dim, :], x, tf=self._tf)
#        _, maxVal, status, _ = scipy.optimize.fminbound(fun,
#                                                        x1=0,
#                                                        x2=1,
#                                                        xtol=tol,
#                                                        maxfun=maxIter,
#                                                        full_output=True,
#                                                        disp=1)
#        return -maxVal[0] if status == 0 else None

    def minDist(self, otherCurve):
        """
        """
#        if self.dim != 2 or otherCurve.dim != 2:
#            err = ('Both curves must be 2D only, not {} and {}.'
#                   ).format(self.dim, otherCurve.dim)
#            raise ValueError(err)
        if (self.dim < 2 or self.dim > 3 or
                otherCurve.dim < 2 or otherCurve.dim > 3):
            err = ('Both curves must be either 2D or 3D, not {}D and {}D.'
                   ).format(self.dim, otherCurve.dim)
            raise ValueError(err)
        return _minDist(self, otherCurve)

    def minDist2Poly(self, poly):
        """
        """
        return _minDist2Poly(self, poly)

    def collCheck(self, otherCurve):
        """
        """
        return _collCheckBez2Bez(self, otherCurve)

    def collCheck2Poly(self, poly):
        """
        """
        return _collCheckBez2Poly(self, poly)

    def normSquare(self):
        """Calculates the norm squared of the Bezier curve

        Returns a Bezier object for the norm squared result of the current
        Bezier curve.

        :return: Norm squared of the Bezier curve
        :rtype: Bezier
        """
        try:
            prodM = Bezier.productMatrixCache[self.deg][self.deg]
        except KeyError:
            prodM = prodMatrix(self.deg).T
            Bezier.productMatrixCache[self.deg][self.deg] = prodM

        normCpts = _normSquare(self.cpts, 1, self.dim, prodM.T)/2

        newCurve = self.copy()
        newCurve.cpts = normCpts

        return newCurve
#        return Bezier(_normSquare(self.cpts, 1, self.dim, prodM.T),
#                      tau=self.tau, tf=self.tf)


class RationalBezier(BezierParams):
    """Rational Bezier curve for trajectory generation

    """
    def __init__(self, cpts=None, weights=None, tau=None, tf=1.0):
        super().__init__(cpts=cpts, tau=tau, tf=tf)
        self._weights = np.array(weights, ndmin=2)


@njit(cache=True)
def deCasteljauCurve(cpts, tau, tf=1.0):
    """Returns a Bezier curve using the de Casteljau algorithm

    Uses the de Casteljau algorithm to generate the Bezier curve defined by
    the provided control points. Note that the datatypes are important due to
    the nature of the numba library.

    Paper Reference: Property 5: The de Casteljau Algorithm

    :param cpts: Control points defining the 1D Bezier curve.
    :type cpts: numpy.ndarray(dtype=numpy.float64)
    :param tau: Values at which to evaluate Bezier curve. Must be within the
        range of [0,tf].
    :type tau: numpy.ndarray(dtype=numpy.float64)
    :param tf: Final tau value for the 1D curve. Default is 1.0. Note that the
        Bezier curve is defined on the range of [0, tf].
    :type tf: float
    :return: Numpy array of length tau of the Bezier curve evaluated at each
        value of tau.
    :rtype: numpy.ndarray(dtype=numpy.float64)
    """
    tau = tau/tf
    curveLen = tau.size
    curve = np.empty(curveLen)
    curveIdx = 0

    for t in tau:
        newCpts = cpts.copy()
        while newCpts.size > 1:
            cptsTemp = np.empty(newCpts.size-1)
            for i in range(cptsTemp.size):
                cptsTemp[i] = (1-t)*newCpts[i] + t*newCpts[i+1]
            newCpts = cptsTemp.copy()
        curve[curveIdx] = newCpts[0]
        curveIdx += 1

    return curve


@njit(cache=True)
def deCasteljauSplit(cpts, tDiv, tf=1.0):
    """Uses the de Casteljau algorithm to split the curve at tDiv

    This function is similar to the de Casteljau curve function but instead of
    drawing a curve, it returns two sets of control points which define the
    curve to the left and to the right of the split point, tDiv.

    Paper Reference: Property 5: The de Casteljau Algorithm

    :param cpts: Control points defining the 1D Bezier curve.
    :type cpts: numpy.ndarray(dtype=numpy.float64)
    :param tDiv: Point at which to divide the curve.
    :type tDiv: float
    :param tf: Final tau value for the 1D curve. Default is 1.0. Note that the
        Bezier curve is defined on the range of [0, tf].
    :type tf: float
    :return: Returns a tuple of numpy arrays. The zeroth element is the
        control points defining the curve to the left of tDiv. The first
        element is the control points defining the curve to the right of tDiv.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    tDiv = tDiv/tf
    cptsLeft = np.zeros(cpts.size)
    cptsRight = np.zeros(cpts.size)
    idx = 0

    newCpts = cpts.copy()
    cptsTemp = cpts.copy()
    while newCpts.size > 1:
        cptsLeft[idx] = cptsTemp[0]
        cptsRight[idx] = cptsTemp[-1]
        idx += 1

        cptsTemp = np.empty(newCpts.size-1)
        for i in range(cptsTemp.size):
            cptsTemp[i] = (1-tDiv)*newCpts[i] + tDiv*newCpts[i+1]

        newCpts = cptsTemp.copy()

    cptsLeft[-1] = cptsRight[-1] = newCpts[0]

    return cptsLeft, cptsRight


def bezierCurve(cpts, tau, tf=1.0):
    """Computes the values of a 1D Bezier curve defined by the control points.

    Creates a 1 dimensional Bezier curve using the designated control points
    and values of tau.

    Effectively evaluates the following expression:
        T*M*P
    Where
    T is the power basis vector [1 t t^2 t^3 ... t^N]
    M is the binomial matrix (more information found in buildBezMatrix)
    P is a vector of Bezier weights (i.e. control points)

    :param cpts: Single row matrix of N+1 control points for
        a one dimensional Bezier curve.
    :type cpts: numpy.ndarray
    :param tau: Values at which to evaluate Bezier curve. Should
        typically only be on the range of [0,tf] but it should work if it's not
        on that range.
    :type tau: numpy.ndarray
    :return: Numpy array of length tau of the Bezier curve evaluated at each
        value of tau.
    :rtype: numpy.ndarray
    """
    cpts = np.array(cpts)
    tau = np.array(tau, dtype=np.float64, ndmin=1)/float(tf)
    tauLen = tau.size
    n = cpts.size-1
    curve = np.empty(tauLen)

    coeffs = buildBezMatrix(n)
    for i, t in enumerate(tau):
        powerBasis = np.power(t, range(n+1))
        curve[i] = np.dot(powerBasis, np.dot(coeffs, cpts.T))

    return curve


@jit(cache=True, forceobj=True)
def buildBezMatrix(n):
    """Builds a matrix of coefficients of the power basis to a Bernstein
    polynomial.

    The coefficients matrix allows us to represent a Bezier curve of arbitrary
    degree as a matrix. This is useful for computational efficiency and ease of
    calculations.

    Taken from the source:
    "Since the power basis {1, t, t^2, t^3, ...} forms a basis for the space of
     polynomials of degree less than or equal to n, any Bernstein polynomial of
     degree n can be written in terms of the power basis."

    Source:
        http://graphics.cs.ucdavis.edu/education/CAGDNotes/CAGDNotes/
        Bernstein-Polynomials/Bernstein-Polynomials.html#conversion

    :param n: Degree of the Bezier curve
    :type n: int
    :return: Power basis matrix of coefficients for a Bezier curve
    :rtype: numpy.ndarray
    """
    bezMatrix = np.zeros((n+1, n+1))

    for k in np.arange(0, n+1):
        for i in np.arange(k, n+1):
            bezMatrix[i, k] = (-1)**(i-k) * binom(n, i) * binom(i, k)

    return bezMatrix


@njit(cache=True)
def diffMatrix(n, tf=1.0):
    """Generates the differentiation matrix to find the derivative

    Takes the derivative of the control points for a Bezier curve. The
    resulting control points can be used to construct a Bezier curve that is
    the derivative of the original.

    Paper Reference: Property 3: Derivatives

    :param n: Degree of the Bezier curve
    :type n: int
    :param tf: Final time for the Bezier trajectory
    :type tf: float
    :return: Differentiation matrix for a Bezier curve of degree n
    :rtype: numpy.ndarray
    """
    val = n/tf
    Dm = np.zeros((n+1, n))
    for i in range(n):
        Dm[i, i] = -val
        Dm[i+1, i] = val

    return Dm


#@jit(cache=True, forceobj=True)
def elevMatrix(N, R=1):
    """Creates an elevation matrix for a Bezier curve.

    Creates a matrix to elevate a Bezier curve of degree N to degree N+R.
    The elevation is performed as such:
        B_(N)*T = B_(N+1) where * is the dot product.

    :param N: Degree of the Bezier curve being elevated
    :type N: int
    :param R: Number of degrees to elevate the Bezier curve
    :type R: int
    :return: Elevation matrix to raise a Bezier curve of degree N by R degrees
    :rtype: numpy.ndarray
    """
    T = np.zeros((N+1, N+R+1))
    for i in range(N+R+1):
        den = binom(N+R, i)
        for j in range(N+1):
            T[j, i] = binom(N, j) * binom(R, i-j) / den

    return T


@jit(cache=True, forceobj=True)
def prodMatrix(N):
    """Produces a product matrix for obtaining the norm of a Bezier curve

    This function produces a matrix which can be used to compute ||x dot x||^2
    i.e. xaug = x'*x;
    xaug = reshape(xaug',[length(x)^2,1]);
    y = Prod_T*xaug;
    or simply norm_square(x)
    prodM is the coefficient of bezier multiplication.

    Code ported over from Venanzio Cichella's MATLAB Prod_Matrix function.

    :param N: Degree of the Bezier curve
    :type N: int
    :return: Product matrix
    :rtype: numpy.ndarray
    """
    T = np.zeros((2*N+1, (N+1)**2))

    for j in np.arange(2*N+1):
        den = binom(2*N, j)
        for i in np.arange(max(0, j-N), min(N, j)+1):
            if N >= i and N >= j-i and 2*N >= j and j-i >= 0:
                T[j, N*i+j] = binom(N, i)*binom(N, j-i) / den

    return T


# TODO:
#    Change this function name to prodM.
#    Clean up and slightly change _normSquare to accommodate this change
#@jit(cache=True, forceobj=True)
def bezProductCoefficients(m, n=None):
    """Produces a product matrix for obtaining the product of two Bezier curves

    This function computes the matrix of coefficients for multiplying two
    Bezier curves. This function exists so that the coefficients matrix can be
    computed ahead of time when performing many multiplications.

    :param m: Degree of the first Bezier curve
    :type m: int
    :param n: Degree of the second Bezier curve
    :type n: int
    :return: Product matrix
    :rtype: numpy.ndarray
    """

    if n is None:
        n = m

    coefMat = np.zeros(((m+1)*(n+1), m+n+1))

    for k in range(m+n+1):
        den = binom(m+n, k)
        for j in range(max(0, k-n), min(m, k)+1):
            coefMat[m*j+k, k] = binom(m, j)*binom(n, k-j)/den

    return coefMat


@jit(cache=True, forceobj=True)
def multiplyBezCurves(multiplier, multiplicand, coefMat=None):
    """Multiplies two Bezier curves together

    The product of two Bezier curves can be computed directly from their
    control points. This function specifically uses matrix multiplication to
    increase the speed of the multiplication.

    Note that this function is made for 1D control points.

    One can pass in a matrix of product coefficients to compute the product
    about 10x faster. It is recommended that the coefficient matrix is
    precomputed and saved in memory when performing multiplication many times.
    The function bezProductCoefficients will produce this matrix.

    :param multiplier: Control points of the multiplier curve. Single dimension
    :type multiplier: numpy.ndarray
    :param multiplicand: Control points of the multiplicand curve.
    :type multiplicand: numpy.ndarray
    :param coefMat: Precomputed coefficient matrix from bezProductCoefficients
    :type coefMat: numpy.ndarray or None
    :return: Product of two Bezier curves
    :rtype: numpy.ndarray
    """
    multiplier = np.atleast_2d(multiplier)
    multiplicand = np.atleast_2d(multiplicand)
    m = multiplier.shape[1] - 1
    n = multiplicand.shape[1] - 1

    augMat = np.dot(multiplier.T, multiplicand)
    newMat = augMat.reshape((1, -1))

    if coefMat is None:
        coefMat = bezProductCoefficients(m, n)

    return np.dot(newMat, coefMat)


@jit(cache=True, forceobj=True)
def splitCurveMat(deg, z, coefMat=None):
    """Creates matrices Q and Qp that are used to compute control points for a
        split curve.

    :param deg: Degree of the Bezier curve
    :type deg: int
    :param z: Point on the curve at which to split the curve [0,1]
    :type z: float
    :param coefMat: Matrix of Binomial coefficients for a Bezier curve. Passing
        in a precomputed matrix will significantly increase the speed of the
        function.
    :type coefMat: numpy.ndarray or None
    :return: Returns a tuple of matrices. The zeroth element is the Q matrix
        for computing control points before the point z and the first element
        is the Q matrix for computing after the point z.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    powMat = np.diag(np.power(z, range(deg+1)))

    if coefMat is None:
        coefMat = buildBezMatrix(deg)

    # Q = M^-1 * Z * M
    Q = np.linalg.inv(coefMat).dot(powMat).dot(coefMat)

    # Qp is just Q but rolled and flipped
    Qp = np.empty((deg+1, deg+1))
    for i, row in enumerate(Q):
        Qp[deg-i, :] = np.roll(row, deg-i)

    return Q, Qp


def _minDist(c1, c2, cnt=0, alpha=np.inf, eps=1e-9,
             t1_l=0, t1_h=1, t2_l=0, t2_h=1):
    """
    Source: Computation of the minimum distance between two Bezier
    curves/surfaces
    """
    x1 = c1.cpts[0, :]
    y1 = c1.cpts[1, :]
    x2 = c2.cpts[0, :]
    y2 = c2.cpts[1, :]

    if c1.dim == 3:
        z1 = c1.cpts[2, :]
    else:
        z1 = [0]*x1.size

    if c2.dim == 3:
        z2 = c2.cpts[2, :]
    else:
        z2 = [0]*x1.size

    c1 = Bezier([x1, y1, z1])
    c2 = Bezier([x2, y2, z2])

    poly1 = np.array(tuple(zip(x1, y1, z1)))
    poly2 = np.array(tuple(zip(x2, y2, z2)))

    cnt += 1
    if cnt > 1000:
        return (-1, -1, -1)

    flag, info = gjkNew(poly1, poly2)
    if flag > 0:
        closest1 = info[0]
        closest2 = info[1]
        lb = info[2]

        # Check to see if the closest point on the shape is a control point
        p1idx = np.where((poly1 == closest1).all(axis=1))[0]
        p2idx = np.where((poly2 == closest2).all(axis=1))[0]
        if p1idx.size > 0:
            t1 = p1idx[0]/c1.deg

        # If the closest point is not a control point, find t by weighting all
        # the control points by their distance from the closest point
        else:
            eucDist1 = np.linalg.norm(closest1-c1.cpts.T, axis=1)
            N = eucDist1.size
            W = np.empty(N)
            for i in range(N):
                W[i] = 1 / (1 +
                            (eucDist1[i]/eucDist1[:i]).sum() +
                            (eucDist1[i]/eucDist1[i+1:]).sum())

            t1 = (W*range(N)/N).sum()

        if p2idx.size > 0:
            t2 = p2idx[0]/c2.deg

        else:
            eucDist2 = np.linalg.norm(closest2-c2.cpts.T, axis=1)
            N = eucDist2.size
            W = np.empty(N)
            for i in range(N):
                W[i] = 1 / (1 +
                            (eucDist2[i]/eucDist2[:i]).sum() +
                            (eucDist2[i]/eucDist2[i+1:]).sum())

            t2 = (W*range(N)/N).sum()

    else:
        t1 = 0.5
        t2 = 0.5
        lb = eps

    t1len = t1_h - t1_l
    t2len = t2_h - t2_l

    ub, t1local, t2local = _upperbound(c1.cpts, c2.cpts)

    if ub <= alpha:
        alpha = ub
        newT1 = (1-t1local)*t1_l + t1local*t1_h
        newT2 = (1-t2local)*t2_l + t2local*t2_h
    else:
        newT1 = -1
        newT2 = -1

    retval = (alpha, newT1, newT2)

    if lb >= alpha*(1-eps):
        return retval

    else:
        c3, c4 = c1.split(t1)
        c5, c6 = c2.split(t2)

        newAlpha, newT1, newT2 = _minDist(c3, c5, cnt=cnt, alpha=retval[0],
                                          t1_l=t1_l, t1_h=t1_l+t1*t1len,
                                          t2_l=t2_l, t2_h=t2_l+t2*t2len)

        if newAlpha < retval[0]:
            retval = (newAlpha, newT1, newT2)

        newAlpha, newT1, newT2 = _minDist(c3, c6, cnt=cnt, alpha=retval[0],
                                          t1_l=t1_l, t1_h=t1_l+t1*t1len,
                                          t2_l=t2_l+t2*t2len, t2_h=t2_h)

        if newAlpha < retval[0]:
            retval = (newAlpha, newT1, newT2)

        newAlpha, newT1, newT2 = _minDist(c4, c5, cnt=cnt, alpha=retval[0],
                                          t1_l=t1_l+t1*t1len, t1_h=t1_h,
                                          t2_l=t2_l, t2_h=t2_l+t2*t2len)

        if newAlpha < retval[0]:
            retval = (newAlpha, newT1, newT2)

        newAlpha, newT1, newT2 = _minDist(c4, c6, cnt=cnt, alpha=retval[0],
                                          t1_l=t1_l+t1*t1len, t1_h=t1_h,
                                          t2_l=t2_l+t2*t2len, t2_h=t2_h)

        if newAlpha < retval[0]:
            retval = (newAlpha, newT1, newT2)

    return retval


def _minDist2Poly(c1, poly2, cnt=0, alpha=np.inf, eps=1e-6, t1_l=0, t1_h=1):
    """Similar to _minDist but finds the distance between a curve and a polygon
    """
    x1 = c1.cpts[0, :]
    y1 = c1.cpts[1, :]

    if c1.dim == 3:
        z1 = c1.cpts[2, :]
    else:
        z1 = [0]*x1.size

    c1 = Bezier([x1, y1, z1])

    poly1 = np.array(tuple(zip(x1, y1, z1)))
    poly2 = poly2.astype(float)

    cnt += 1
    if cnt > 1000:
        return (-1, -1, -1)

    flag, info = gjkNew(poly1, poly2)
    if flag > 0:
        closest1 = info[0]
        closest2 = info[1]
        lb = info[2]

        # Check to see if the closest point on the shape is a control point
        p1idx = np.where((poly1 == closest1).all(axis=1))[0]
        if p1idx.size > 0:
            t1 = p1idx[0]/c1.deg

        # If the closest point is not a control point, find t by weighting all
        # the control points by their distance from the closest point
        else:
            eucDist1 = np.linalg.norm(closest1-c1.cpts.T, axis=1)
            N = eucDist1.size
            W = np.empty(N)
            for i in range(N):
                W[i] = 1 / (1 +
                            (eucDist1[i]/eucDist1[:i]).sum() +
                            (eucDist1[i]/eucDist1[i+1:]).sum())

            t1 = (W*range(N)/N).sum()

        # Upper bound check
        ub, t1local = _upperboundPoly(c1.cpts, closest2)
        if ub <= alpha:
            alpha = ub
            newT1 = (1-t1local)*t1_l + t1local*t1_h
        else:
            newT1 = -1

    else:
        t1 = 0.5
        newT1 = -1
        closest2 = -1
        lb = eps**3
        ub = np.inf

    t1len = t1_h - t1_l

    retval = (alpha, newT1, closest2)

    if lb >= alpha*(1-eps):
        return retval

    else:
        c3, c4 = c1.split(t1)

        newAlpha, newT1, closest2 = _minDist2Poly(c3, poly2, cnt=cnt,
                                                  alpha=retval[0],
                                                  t1_l=t1_l,
                                                  t1_h=t1_l+t1*t1len)

        if newAlpha < retval[0]:
            retval = (newAlpha, newT1, closest2)

        newAlpha, newT1, closest2 = _minDist2Poly(c4, poly2, cnt=cnt,
                                                  alpha=retval[0],
                                                  t1_l=t1_l+t1*t1len,
                                                  t1_h=t1_h)

        if newAlpha < retval[0]:
            retval = (newAlpha, newT1, closest2)

    return retval


@njit(cache=True)
def _upperbound(c1, c2):
    """
    """
    distances = np.empty(4)
    tvals = np.array([(0., 0.),
                      (0., 1.),
                      (1., 0.),
                      (1., 1.)])

    distances[0] = _norm(c1[:, 0] - c2[:, 0])
    distances[1] = _norm(c1[:, 0] - c2[:, -1])
    distances[2] = _norm(c1[:, -1] - c2[:, 0])
    distances[3] = _norm(c1[:, -1] - c2[:, -1])

    t1, t2 = tvals[distances.argmin()]

    return distances.min(), t1, t2


#@njit(cache=True)
#def _upperboundPoly(c1, poly):
#    """
#    """
#    distances = np.empty(2*poly.shape[0])
#    tvals = np.array((0., 1.))
#
#    for i in range(poly.shape[0]):
#        distances[2*i] = _norm(c1[:, 0] - poly[i])
#        distances[2*i+1] = _norm(c1[:, -1] - poly[i])
#
#    t1 = tvals[distances.argmin() % 2]
#
#    return distances.min(), t1


@njit(cache=True)
def _upperboundPoly(c1, pt):
    """
    """
    distances = np.empty(2)
    tvals = np.array((0., 1.))

    distances[0] = _norm(c1[:, 0] - pt)
    distances[1] = _norm(c1[:, -1] - pt)

    t1 = tvals[distances.argmin()]

    return distances.min(), t1


@njit(cache=True)
def _norm(x):
    """
    """
    summation = 0.
    for val in x:
        summation += val*val

    return np.sqrt(summation)


def _collCheckBez2Bez(c1, c2, cnt=0, alpha=np.inf, eps=1e-9):
    """
    Source: Computation of the minimum distance between two Bezier
    curves/surfaces
    """
    x1 = c1.cpts[0, :]
    y1 = c1.cpts[1, :]
    x2 = c2.cpts[0, :]
    y2 = c2.cpts[1, :]

    if c1.dim == 3:
        z1 = c1.cpts[2, :]
    else:
        z1 = [0]*x1.size

    if c2.dim == 3:
        z2 = c2.cpts[2, :]
    else:
        z2 = [0]*x1.size

    c1 = Bezier([x1, y1, z1])
    c2 = Bezier([x2, y2, z2])

    poly1 = np.array(tuple(zip(x1, y1, z1)))
    poly2 = np.array(tuple(zip(x2, y2, z2)))

    cnt += 1
    if cnt > 100:
        return -1

    ub, t1local, t2local = _upperbound(c1.cpts, c2.cpts)

    flag, info = gjkNew(poly1, poly2)
    if flag > 0:
        return 1
    else:
        t1 = 0.5
        t2 = 0.5
        lb = 0

    if ub <= alpha:
        alpha = ub

    if lb >= alpha*(1-eps):
        return alpha

    else:
        c3, c4 = c1.split(t1)
        c5, c6 = c2.split(t2)
        alpha = min(alpha, _collCheckBez2Bez(c3, c5, cnt=cnt, alpha=alpha))
        alpha = min(alpha, _collCheckBez2Bez(c3, c6, cnt=cnt, alpha=alpha))
        alpha = min(alpha, _collCheckBez2Bez(c4, c5, cnt=cnt, alpha=alpha))
        alpha = min(alpha, _collCheckBez2Bez(c4, c6, cnt=cnt, alpha=alpha))

    return alpha


def _collCheckBez2Poly(c1, poly2, cnt=0, alpha=np.inf, eps=1e-9):
    """
    Source: Computation of the minimum distance between two Bezier
    curves/surfaces
    """
    x1 = c1.cpts[0, :]
    y1 = c1.cpts[1, :]

    if c1.dim == 3:
        z1 = c1.cpts[2, :]
    else:
        z1 = [0]*x1.size

    c1 = Bezier([x1, y1, z1])

    poly1 = np.array(tuple(zip(x1, y1, z1)))

    cnt += 1
    if cnt > 100:
        return -1

    flag, info = gjkNew(poly1, poly2)
    if flag > 0:
        return 1
    else:
#        t1 = 0.5
#        lb = eps

        c3, c4 = c1.split(0.5)
        if (_collCheckBez2Poly(c3, poly2, cnt=cnt) == 1 and
            _collCheckBez2Poly(c4, poly2, cnt=cnt) == 1):
            return 1

    return 0

#    ub = np.inf
#
#    flag1, info1 = gjkNew(poly2, np.atleast_2d(poly1[0]))
#    flag2, info2 = gjkNew(poly2, np.atleast_2d(poly1[-1]))
#
#    if flag1 > 0:
#        ub = info1[2]
#
#    if flag2 > 0 and info2[2] < ub:
#        ub = info2[2]
#
#    if ub <= alpha:
#        alpha = ub
#
#    if lb >= alpha*(1-eps):
#        return round(alpha, int(round(np.log(eps)/np.log(10))))
#
#    else:
#        c3, c4 = c1.split(t1)
#        alpha = min(alpha, _collCheckBez2Poly(c3, poly2, cnt=cnt, alpha=alpha))
#        alpha = min(alpha, _collCheckBez2Poly(c4, poly2, cnt=cnt, alpha=alpha))
#
#    return alpha


# TODO
#   * Find a fast implementation for the calculation of binomial coefficients
#     that doesn't break for large numbers. Try fastBinom for 70 choose 20 and
#     it returns 0 which it shouldn't.
# @numba.jit(nopython=True)
# def fastBinom(n, k):
#    """Quickly computes the binomial coefficients.
#
#    A fast way to calculate binomial coefficients by Andrew Dalke.
#    See http://stackoverflow.com/questions/3025162/
#        statistics-combinations-in-python
#
#    :param n: n portion of "n choose k"
#    :type n: int
#    :param k: k portion of "n choose k"
#    :type k: int
#    :return: binomial coefficient of n choose k
#    :rtype: int
#    """
#    if 0 <= k <= n:
#        ntok = 1
#        ktok = 1
#        for t in range(1, min(k, n-k) + 1):
#            ntok *= n
#            ktok *= t
#            n -= 1
#        return ntok // ktok
#    else:
#        return 0
#
#
# @numba.jit(nopython=True)
# def binomialCoefficient(n, k):
#    # since C(n, k) = C(n, n - k)
#    if(k > n - k):
#        k = n - k
#    # initialize result
#    res = 1
#    # Calculate value of
#    # [n * (n-1) *---* (n-k + 1)] / [k * (k-1) *----* 1]
#    for i in range(k):
#        res = res * (n - i)
#        res = res // (i + 1)
#    return res


@njit(cache=True)
def _normSquare(x, Nveh, Ndim, prodM):
    """Compute the control points of the square of the norm of a vector

    normSquare(x, Nveh, Ndim, prodM)

    INPUT: Ndim*Nveh by N matrix x = [x1,...,x_Nveh)], x_i in R^Ndim
    OUTPUT: control points of ||x_i||^2 ... Nveh by N matrix

    Code ported over from Venanzio Cichella's MATLAB norm_square function.
    NOTE: This only works on 1D or 2D matricies. It will fail for 3 or more.
    """
#    x = np.array(x)
#    if x.ndim == 1:
#        x = x[None]

    m, N = x.shape

    xsquare = np.zeros((m, prodM.shape[0]))

    for i in range(m):
#        xaug = np.dot(x[i, None].T, x[i, None])
        xaug = np.dot(x.T, x)
        xnew = xaug.reshape((N**2, 1))
        xsquare[i, :] = np.dot(prodM, xnew).T[0]

    S = np.zeros((Nveh, Nveh*Ndim))

    for i in range(Nveh):
        for j in range(Ndim):
            S[i, Ndim*i+j] = 1

    return np.dot(S, xsquare)


def plotPoly(poly, ax):
    from scipy.spatial import ConvexHull

    pts = poly.copy()
    hull = ConvexHull(pts, qhull_options='QJ')

    ax.plot(pts.T[0], pts.T[1], pts.T[2], 'ko')

    for s in hull.simplices:
        s = np.append(s, s[0])
        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], 'b-')


if __name__ == '__main__':
    COMPARE_DISC_DIST = False
    cpts1 = np.array([(0, 1, 2, 3, 4, 5),
                      (1, 2, 0, 0, 2, 1),
                      (0, 1, 2, 3, 4, 5)])

    cpts2 = np.array([(0, 1, 2, 3, 4, 5),
                      (3, 2, 0, 0, 2, 3),
                      (5, 4, 3, 2, 1, 0)])

    cpts3 = np.array([(0, 1, 2, 3, 4, 5),
                      (0, 1, 2, 3, 4, 5),
                      (0, 0, 0, 0, 0, 0)])

    cpts4 = np.array([(5, 4, 3, 2, 1, 0),
                      (0, 1, 2, 3, 4, 5),
                      (0, 0, 0, 0, 0, 0,)])

    cpts4[1, :] -= 1

    cpts5 = cpts1 + 3

    poly1 = np.array([(1, 1, 3),
                      (1, 1, 2),
                      (1, 2, 1),
                      (3, 1, 3),
                      (1, 3, 1)])

    poly2 = np.array([(1, 1, 3),
                      (1, 1, 2),
                      (1, 2, 1),
                      (3, -1, 3),
                      (1, 3, 1)])

    c1 = Bezier(cpts1, tau=np.linspace(0, 1, 1001))
    c2 = Bezier(cpts2, tau=np.linspace(0, 1, 1001))
    c3 = Bezier(cpts3)
    c4 = Bezier(cpts4)
    c5 = Bezier(cpts5)

    print('Checking for collisions, 1 means none, 0 means collision')
    print('C1 and C2')
    print(_collCheckBez2Bez(c1, c2))
    print('C3 and C4')
    print(_collCheckBez2Bez(c3, c4))
    print('C1 and poly2')
    print(_collCheckBez2Poly(c1, poly2))
    print('C5 and poly1')
    print(_collCheckBez2Poly(c5, poly1))

    print('---')
    print('Minimum distances')
    print('C1 and C2')
    dist, t1, t2 = c1.minDist(c2)
    print('MinDist: {}, t1: {}, t2: {}'.format(dist, t1, t2))
    pt1 = c1(t1)
    pt2 = c2(t2)

    plt.close('all')
    ax = c1.plot()
    c2.plot(ax)

    plt.plot(np.array((pt1[0], pt2[0])).squeeze(),
             np.array((pt1[1], pt2[1])).squeeze(),
             np.array((pt1[2], pt2[2])).squeeze(), 'g-')

    if COMPARE_DISC_DIST:
        discreteMinDist = np.inf
        for i, pt1 in enumerate(c1.curve.T):
            for j, pt2 in enumerate(c2.curve.T):
                temp = np.linalg.norm(pt1-pt2)
                if temp < discreteMinDist:
                    idx1 = i
                    idx2 = j
                    discPt1 = pt1
                    discPt2 = pt2
                    discreteMinDist = temp

        print('Disc Min Dist: {}, idx1: {}, idx2: {}'.format(discreteMinDist,
              idx1, idx2))
        print('+++++++++++++++++++++++++++++++++++++')
        plt.plot(c1.curve[0, :], c1.curve[1, :], c1.curve[2, :], '-.')
        plt.plot(c2.curve[0, :], c2.curve[1, :], c2.curve[2, :], '-.')
        plt.plot(np.array((discPt1[0], discPt2[0])).squeeze(),
                 np.array((discPt1[1], discPt2[1])).squeeze(),
                 np.array((discPt1[2], discPt2[2])).squeeze(), 'r--')

    print('C1 and poly1')
    shapeDist, t1, pt = _minDist2Poly(c1, poly1)
    print('Shape Dist: {}, t1: {}, pt: {}'.format(shapeDist, t1, pt))
    ax2 = c1.plot()
    plotPoly(poly1, ax2)
    plt.plot(np.array((c1(t1)[0], pt[0])),
             np.array((c1(t1)[1], pt[1])),
             np.array((c1(t1)[2], pt[2])))

    plt.show()

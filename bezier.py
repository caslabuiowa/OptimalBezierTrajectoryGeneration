#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:20:47 2019

@author: ckielasjensen
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy.special import binom


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
    """
    def __init__(self, cpts=None, tau=None, tf=1.0):
        self._tau = tau
        self._tf = float(tf)
        self._curve = None

        if cpts is not None:
            self._cpts = np.array(cpts, ndmin=2)
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

        newCpts = np.array(value, ndmin=2)

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

    @property
    def tau(self):
        if self._tau is None:
            self._tau = np.arange(0, 1.01, 0.01)
        else:
            self._tau = np.array(self._tau)
        return self._tau

    @tau.setter
    def tau(self, val):
        self._curve = None
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
    splitCache = dict()
    elevationMatrixCache = defaultdict(dict)
    productMatrixCache = dict()
    diffMatrixCache = defaultdict(dict)
    bezCoefCache = defaultdict(dict)

    def __init__(self, cpts=None, tau=None, tf=1.0):
        super().__init__(cpts=cpts, tau=tau, tf=tf)

    def __add__(self, curve):
        return self.add(curve)

    def __sub__(self, curve):
        return self.sub(curve)

    def __mul__(self, curve):
        return self.mul(curve)

    def __div__(self, curve):
        return self.div(curve)

    def __pow__(self, power):
        pass

    def __repr__(self):
        return 'Bezier({}, {}, {})'.format(self.cpts, self.tau, self.tf)

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
        if self._curve is None:
            self._curve = np.zeros([self.dim, len(self.tau)])
            for i, pts in enumerate(self.cpts):
                self._curve[i] = bezierCurve(pts, self.tau)

        return self._curve

    def copy(self):
        """Creates an exact, deep copy of the current Bezier object

        :return: Deep copy of Bezier object
        :rtype: Bezier
        """
        return Bezier(self.cpts, self.tau, self.tf)

    def plot(self, axisHandle=None):
        """Plots the Bezier curve in 1D or 2D

        Note: Currently only supports plotting in 1D or 2D.

        :param axisHandle: Handle to the figure axis. If it is None, a new
            figure will be plotted.
        :type axisHandle: matplotlib.axes._subplots.AxesSubplot or None
        :return: Axis object where the curve was plotted.
        :rtype: matplotlib.axes._subplots.AxesSubplot
        """
        if axisHandle is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = axisHandle

        cpts = np.asarray(self.cpts)

        if self.dim == 1:
            ax.plot(self.tau, self.curve[0])
            ax.plot(np.linspace(0, self.tf, self.deg+1), self.cpts.squeeze())
        elif self.dim == 2:
            ax.plot(self.curve[0], self.curve[1])
            ax.plot(cpts[0], cpts[1], '.--')
        else:
            print('WARNING: Only 1D and 2D Plotting are Supported for Now')

        return ax

    def add(self, other):
        """Adds two Bezier curves

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

        :param other: Bezier curve to subtract from the original
        :type other: Bezier
        :return: Original curve - Other curve
        :rtype: Bezier
        """
        subCpts = self.cpts - other.cpts
        newCurve = self.copy()
        newCurve.cpts = subCpts
        return newCurve

    def mul(self, multiplicand):
        """Computes the product of two Bezier curves.

        Source: Section 5.1 of "The Bernstein Polynomial Basis: A Centennial
        Retrospective" by Farouki.

        :param multiplicand: Multiplicand
        :type multiplicand: Bezier
        :return: Product of the two curve
        :rtype: Bezier
        """
        if not isinstance(multiplicand, Bezier):
            msg = 'The multiplicand must be a Bezier object, not a {}'.format(
                    type(multiplicand))
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
            coefMat = Bezier.bezCoefCache[m][n]
        except KeyError:
            coefMat = bezProductCoefficients(m, n)
            Bezier.bezCoefCache[m][n] = coefMat

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

        RETURNS:
            RationalBezier object resulting in the division of two Bezier
                curves.
        :param denominator: Denominator of the division
        :type denominator: Bezier
        :return: Rational Bezier curve representing the division of the two
            curves.
        :rtype: RationalBezier
        """
        if not isinstance(denominator, Bezier):
            msg = ('The denominator must be a Bezier object, not a %s. '
                   'Or the module has been reloaded.').format(
                           type(denominator))
            raise TypeError(msg)

        cpts = self.cpts.astype(np.float64) / denominator.cpts
        weights = denominator.cpts

        return RationalBezier(cpts, weights, tau=self.tau, tf=self.tf)

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
            elevMat = elevBez(self.deg, R)
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
            Dm = diffBez(self.deg, self.tf)
            Bezier.diffMatrixCache[self.deg][self.tf] = Dm

        cptsDot = []
        for i in range(self.dim):
            cptsDot.append(np.dot(self.cpts[i, :], Dm))

        curveDot = self.copy()
        curveDot.cpts = cptsDot

        return curveDot.elev()

    def normSquare(self):
        """Calculates the norm squared of the Bezier curve

        Returns a Bezier object for the norm squared result of the current
        Bezier curve.

        :return: Norm squared of the Bezier curve
        :rtype: Bezier
        """
        try:
            prodM = Bezier.productMatrixCache[self.deg]
        except KeyError:
            prodM = prodMatrix(self.deg)
            Bezier.productMatrixCache[self.deg] = prodM

        return Bezier(_normSquare(self.cpts, 1, self.dim, prodM),
                      tau=self.tau, tf=self.tf)

# TODO: Create a split function that doesn't use the symbolic library so that
#        it runs quicker.
#
#    def split(self, splitPoint):
#        """Splits the Bezier curve at tau = splitPoint.
#
#        This method uses the matrix representation of Bezier curves and can be
#        found at the following source:
#        https://pomax.github.io/bezierinfo/#matrixsplit
#        """
#        if splitPoint < 0 or splitPoint > 1:
#            errorMsg = (
#                    'Can only split the curve at a tau '
#                    'value between 0 and 1.'
#                    )
#            raise ValueError(errorMsg)
#        from sympy.abc import t
#
#        try:
#            Q, Qp = Bezier.Q[self.deg]
#
#        except KeyError:
#            M = buildBezMatrix(self.deg)
#            Z = np.diag(createPowerBasisVector(self.deg))
#            Q = M.I * Z * M
#
#            shiftedQ = np.vstack(
#                    (np.roll(m.A1, self.deg-i) for i, m in enumerate(Q)))
#            Qp = np.flip(shiftedQ, 0)
#
#            Q = lambdify(t, Matrix(Q))
#            Qp = lambdify(t, Matrix(Qp))
#
#            Bezier.Q[self.deg] = Q, Qp
#
#        c1 = Q(splitPoint)*self.cpts.T
#        c2 = Qp(splitPoint)*self.cpts.T
#
#        return (Bezier(c1.T, tau=self.tau, tf=self.tf),
#                Bezier(c2.T, tau=self.tau, tf=self.tf))


class RationalBezier(BezierParams):
    """Rational Bezier curve for trajectory generation

    """
    def __init__(self, cpts=None, weights=None, tau=None, tf=1.0):
        super().__init__(cpts=cpts, tau=tau, tf=tf)
        self._weights = np.array(weights, ndmin=2)


def bezierCurve(cpts, tau):
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
        typically only be on the range of [0,1] but it should work if it's not
        on that range.
    :type tau: numpy.ndarray
    :return: Numpy array of length tau of the Bezier curve evaluated at each
        value of tau.
    :rtype: numpy.ndarray
    """
    cpts = np.array(cpts)
    tau = np.array(tau, dtype=np.float64)
    tauLen = tau.size
    n = cpts.size-1
    curve = np.empty(tauLen)

    coeffs = buildBezMatrix(n)
    for i, t in enumerate(tau):
        powerBasis = np.power(t, range(n+1))
        curve[i] = np.dot(powerBasis, np.dot(coeffs, cpts.T))

    return curve


@numba.jit
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


@numba.jit(nopython=True)
def diffBez(n, tf=1.0):
    """
    Takes the derivative of the control points for a Bezier curve. The
    resulting control points can be used to construct a Bezier curve that is
    the derivative of the original.

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


@numba.jit
def elevBez(N, R=1):
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
        for j in range(N+1):
            T[j, i] = binom(N, j) * binom(R, i-j) / binom(N+R, i)

    return T


@numba.jit
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


@numba.jit
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


@numba.jit
def bezProductCoefficients(m, n):
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

    coefMat = np.zeros(((m+1)*(n+1), m+n+1))

    for k in range(m+n+1):
        den = binom(m+n, k)
        for j in range(max(0, k-n), min(m, k)+1):
            coefMat[m*j+k, k] = binom(m, j)*binom(n, k-j)/den

    return coefMat
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


def _normSquare(x, Nveh, Ndim, prodM):
    """Compute the control points of the square of the norm of a vector

    normSquare(x, Nveh, Ndim, prodM)

    INPUT: Ndim*Nveh by N matrix x = [x1,...,x_Nveh)], x_i in R^Ndim
    OUTPUT: control points of ||x_i||^2 ... Nveh by N matrix

    Code ported over from Venanzio Cichella's MATLAB norm_square function.
    NOTE: This only works on 1D or 2D matricies. It will fail for 3 or more.
    """
    x = np.array(x)
    if x.ndim == 1:
        x = x[None]

    m, N = x.shape

    xsquare = np.zeros((m, prodM.shape[0]))

    for i in range(m):
        xaug = np.dot(x[i, None].T, x[i, None])
        xnew = xaug.reshape((N**2, 1))
        xsquare[i, :] = np.dot(prodM, xnew).T[0]

    S = np.zeros((Nveh, Nveh*Ndim))

    for i in range(Nveh):
        for j in range(Ndim):
            S[i, Ndim*i+j] = 1

    return np.dot(S, xsquare)

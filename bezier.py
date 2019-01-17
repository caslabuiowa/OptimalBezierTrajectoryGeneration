# -*- coding: utf-8 -*-
"""
Bezier curve library.

Written by Calvin Kielas-Jensen
"""

#TODO
#   * Move the functions to the Bezier class rather than have them be module
#   * Add a function to return minimum distance between two curves.
#   * Precompute the elevation matrix
#   * Convert as many functions as possible over to numba.njit for speed up
#   * Change everything back to np.array since matrix may be removed
#   * Make RationalBezier inherit Bezier (after changing Bezier)

import sys
if sys.version_info[0] != 2 or sys.version_info[1] != 7:
    msg = ('WARNING: This code was developed in Python 2.7\n'
           'Your current version is {}.'.format(sys.version))
    print(msg)

import numpy as np
import numpy.matlib as mat
from scipy.special import binom
from sympy import lambdify, Matrix

class Bezier:
    """
    Bezier curve class.
    
    Allows the user to construct Bezier curves of arbitrary dimension and
    degrees.
    """
    
    # Class level variables
    Q = dict()
    
    def __init__(self, ctrlPts=None, tau=None):
        if ctrlPts is not None:
            self._ctrlPts = mat.mat(ctrlPts)
            self._dim = self.ctrlPts.shape[0]
            self._deg = self.ctrlPts.shape[1] - 1
        else:
            self._dim = None
            self._deg = None            
            
        # Set a default value for tau if it is not passed in
        if tau is None:
            self._tau = np.arange(0, 1.01, 0.01)
        else:
            self._tau = np.array(tau)
    
    @property
    def ctrlPts(self):
        return self._ctrlPts
    
    @ctrlPts.setter
    def ctrlPts(self, value):
#        del self._curve
        self._curve = None
        self._ctrlPts = mat.mat(value)
    
    @property
    def curve(self):
        try:
            return self._curve
        except AttributeError:
            self._curve = np.zeros([self._dim, len(self._tau)])
            for i, pts in enumerate(self._ctrlPts):
                self._curve[i] = bezierCurve(pts, self._tau)
            return self._curve
        
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
    def tau(self):
        return self._tau
    
    @tau.setter
    def tau(self, val):
#        del self._curve
        self._curve = None
        self._tau = val
        
    def diff(self):
        """
        Returns a Bezier object whose control points are the derivative of the
        control points of the original object.
        """
        cpts = self._ctrlPts
        cptsDotElev = mat.zeros(cpts.shape)
        for i in range(self._dim):
            cptsDot = cpts[i,:]*diffBez(self._deg)
            cptsDotElev[i,:] = cptsDot*elevBez(self._deg-1)
        return Bezier(cptsDotElev, self._tau)
    
    def normSquare(self):
        """
        Returns a Bezier object for the norm squared result of the current
        Bezier curve.
        """
        prodM = prodMatrix(self.deg)
        return Bezier(_normSquare(self.ctrlPts, 1, self.dim, prodM))
    
    def elev(self, R=1):
        """
        Elevates the degree of the Bezier curve by R (default is 1) and returns
        a new, higher degree Bezier object.
        """
        elevPts = np.vstack(
                (cpts*elevBez(self.deg, R) for cpts in self._ctrlPts))
        return Bezier(elevPts)
    
    def split(self, splitPoint):
        """
        Splits the Bezier curve at tau = splitPoint.
        
        This method uses the matrix representation of Bezier curves and can be
        found at the following source:
        https://pomax.github.io/bezierinfo/#matrixsplit
        """
        if splitPoint < 0 or splitPoint > 1:
            errorMsg = (
                    'Can only split the curve at a tau '
                    'value between 0 and 1.'
                    )
            raise ValueError(errorMsg)
        from sympy.abc import t
        
        try:
            Q, Qp = Bezier.Q[self.deg]
            
        except KeyError:    
            M = buildBezMatrix(self.deg)
            Z = np.diag(createPowerBasisVector(self.deg))
            Q = M.I * Z * M
            
            shiftedQ = np.vstack(
                    (np.roll(m.A1, self.deg-i) for i, m in enumerate(Q)))
            Qp = np.flip(shiftedQ, 0)
            
            Q = lambdify(t, Matrix(Q))
            Qp = lambdify(t, Matrix(Qp))
            
            Bezier.Q[self.deg] = Q, Qp
            
        c1 = Q(splitPoint)*self.ctrlPts.T
        c2 = Qp(splitPoint)*self.ctrlPts.T
            
        return Bezier(c1.T), Bezier(c2.T)
    
    def minMax(self):
        """
        Finds the minimum and maximum values of the Bezier curve.
        
        Searches along the interval of 0-1 using the degree elevation method.
        Note that this is a quick approximation and not guaranteed to generate
        an exact estimate.
        
        In the future, GJK along with DeCasteljau's algorithm will be used to
        converge to the actual values.
        
        RETURNS:
            (minVal, maxVal) - Tuple of two matrix elements. The first matrix
                is the minimum values, the second matrix is the maximum values.
                The two matrix elements are column vectors whose length is
                equal to the number of dimensions being used.
        """
        elevatedCurve = self.elev(self._deg*10)
        
        minVal = elevatedCurve.ctrlPts.min(axis=1)
        maxVal = elevatedCurve.ctrlPts.max(axis=1)
        
        return (minVal, maxVal)
    
    def add(self, curve):
        """
        Adds two Bezier curves
        """
        if not isinstance(curve, Bezier):
            msg = 'Both curves being added must be of type Bezier.'
            raise TypeError(msg)
            
        cpts = self.ctrlPts + curve.ctrlPts
        
        return Bezier(cpts)
    
    def __add__(self, other):
        return self.add(other)
    
    def sub(self, curve):
        """
        Subtracts two Bezier curves
        """
        if not isinstance(curve, Bezier):
            msg = 'Both curves being subtracted must be of type Bezier.'
            raise TypeError(msg)
            
        cpts = self.ctrlPts - curve.ctrlPts
        
        return Bezier(cpts)
    
    def __sub__(self, other):
        return self.sub(other)
    
    def divide(self, denominator):
        """
        Divides one Bezier curve by another resulting in a rational Bezier
        curve.
        
        RETURNS:
            RationalBezier object resulting in the division of two Bezier
                curves.
        """
        if not isinstance(denominator, Bezier):
            msg = 'The denominator must be a Bezier object, not a %s'.format(
                    type(denominator))
            raise TypeError(msg)
            
        cpts = self.ctrlPts.astype(np.float64) / denominator.ctrlPts
        weights = denominator.ctrlPts
        
        return RationalBezier(cpts, weights)
    
    def __div__(self, other):
        return self.divide(other)
    
    def multiply(self, multiplicand):
        """
        Computes the product of two Bezier curves.
        
        Source: Section 5.1 of "The Bernstein Polynomial Basis: A Centennial
        Retrospective" by Farouki.
        """
        if not isinstance(multiplicand, Bezier):
            msg = 'The multiplicand must be a Bezier object, not a %s'.format(
                    type(multiplicand))
            raise TypeError(msg)
            
        dim = self.dim
        if multiplicand.dim != dim:
            msg = ('The dimension of both Bezier curves must be the same.\n'
                   'The first dimension is {} and the second is {}'.format(dim,
                                           multiplicand.dim))
            raise ValueError(msg)
            
        a = np.array(self.ctrlPts, dtype=np.float64, ndmin=2)
        b = np.array(multiplicand.ctrlPts, dtype=np.float64, ndmin=2)
        m = self.deg
        n = multiplicand.deg
        
        c = np.empty((dim, m+n+1))
        
        for d in range(dim):
            for k in np.arange(0, m+n+1):
                summation = 0
                for j in np.arange(max(0, k-n), min(m, k)+1):
                    summation += binom(m, j)*binom(n, k-j)*a[d,j]*b[d,k-j]
                c[d, k] = summation / binom(m+n, k)
                
        return Bezier(c)
        
    def __mul__(self, other):
        return self.multiply(other)
    
    def __str__(self):
        return "Bezier Curve with Control Points: {}".format(self.ctrlPts)
    
    def __repr__(self):
        return "Bezier({}, {})".format(self._ctrlPts, self._tau)
    
class RationalBezier:
    """
    Rational Bezier curve class
    
    Create a rational Bezier curve. A rational Bezier curve is defined by the
    following equation:
        B(t) = SUM(i=0, n)[ b_{i,n}(t) * P_i * w_i ] 
               ------------------------------------
                  SUM(i=0, n)[ b_{i,n}(t) * w_i ]
    Where b_{i,n} is the ith term for an n degree Bernstein basis polynomial,
    P_i is the ith control point of the Bezier curve in the numerator, and w_i
    is the ith weight. The weight, w_i, is defined by the control points of the
    Bezier curve in the denominator.
    """
    def __init__(self, ctrlPts=None, weights=None, tau=None):
        if ctrlPts is not None and weights is not None:
            # Want ndmin=2 so that we can treat the arrays like 2D matrices
            self._ctrlPts = np.array(ctrlPts, dtype=np.float64, ndmin=2)
            self._weights = np.array(weights, dtype=np.float64, ndmin=2)
            
            if ctrlPts.size != weights.size:
                msg = 'Control points and weights arrays must be the same size.'
                raise ValueError(msg)
            
            # Initialize degree and dimension
            self.dim = self._ctrlPts.shape[0]
            self.deg = self._ctrlPts.shape[1] - 1
        else:
            self.dim = None
            self.deg = None            
            
        # Set a default value for tau if it is not passed in
        if tau is None:
            self._tau = np.arange(0, 1.01, 0.01)
        else:
            self._tau = np.array(tau)
            
    def __repr__(self):
        return "RationalBezier({}, {}, {})".format(self._ctrlPts,
                              self._weights, self._tau)
        
    @property
    def ctrlPts(self):
        return self._ctrlPts
    
    @ctrlPts.setter
    def ctrlPts(self, value):
        self._curve = None
        self._ctrlPts = np.array(value, dtype=np.float64, ndmin=2)
        
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, value):
        self._curve = None
        self._weights = np.array(value, dtype=np.float64, ndmin=2)
        

def bezierCurve(cpts, tau):
    """
    Creates a 1 dimensional Bezier curve using the designated control points
    and values of tau.
    
    All the symbolic matrix math is done using numpy matrices. Need to convert
    to sympy matrix for evaluating the expression otherwise recursion errors
    will occur.
    
    Effectively evaluates the following expression:
        T*M*P
    Where
    T is the power basis vector [1 t t^2 t^3 ... t^N]
    M is the binomial matrix (more information found in buildBezMatrix)
    P is a vector of Bezier weights (i.e. control points)
    
    INPUTS:
        cpts (1 x N+1 numpy.mat) - Single row matrix of N+1 control points for
        a one dimensional Bezier curve.
        
        tau (numpy.array) - Values at which to evaluate Bezier curve. Should
        typically only be on the range of [0,1] but it should work if it's not
        on that range.
        
    RETURNS:
        curve(tau) (numpy.array) - Numpy array of length tau of the Bezier
        curve evaluated at each value of tau.
    """
    from sympy.abc import t
    tau = np.array(tau)

    cpts = mat.mat(cpts)
    n = cpts.size-1
    coeffs = buildBezMatrix(n)
    expr = createPowerBasisVector(n) * coeffs * cpts.T
    curve = lambdify(t, Matrix(expr), 'numpy')
    
    return curve(tau)

def buildBezMatrix(n):
    """
    Builds a matrix of coefficients of the power basis to a Bernstein
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
    """
    bezMatrix = mat.zeros((n+1, n+1))
    
    for k in np.arange(0, n+1):
        for i in np.arange(k, n+1):
            bezMatrix[k, i] = (-1)**(i-k) * binom(n, i) * binom(i, k)
            
    return bezMatrix.T

def createPowerBasisVector(n):
    """
    Creates a simple power vector in the form of:
        [1, t, t^2, t^3, ..., t^n]
    """
    from sympy.abc import t
    
    return np.power(t, range(n+1))

def diffBez(N, tf=1):
    """
    Takes the derivative of the control points for a Bezier curve. The
    resulting control points can be used to construct a Bezier curve that is
    the derivative of the original.
    """
    Dm = N * (np.vstack([mat.zeros(N), mat.identity(N)]) -
        np.vstack([mat.identity(N), mat.zeros(N)]))
    
    return Dm

def elevBez(N, R=1):
    """
    Creates a matrix to elevate a Bezier curve of degree N to degree N+R.
    The elevation is performed as such:
        B_(N)*T = B_(N+1) where * is the dot product.
    """
    T = mat.zeros((N+1, N+R+1))
    for i in range(N+R+1):
        for j in range(N+1):
            T[j,i] = binom(N,j) * binom(R,i-j) / binom(N+R,i)
            
    return T

def prodMatrix(N):
    """
    This function produces a matrix which can be used to compute ||x dot x||^2
    i.e. xaug = x'*x;
    xaug = reshape(xaug',[length(x)^2,1]);
    y = Prod_T*xaug;
    or simply norm_square(x)
    prodM is the coefficient of bezier multiplication.
    
    Code ported over from Venanzio Cichella's MATLAB Prod_Matrix function.
    """
    T = mat.zeros((2*N+1, (N+1)**2))
    
    for j in np.arange(2*N+1):
        for i in np.arange(max([0, j-N]), min(N,j)+1):
            if N >= i and N >= j-i and 2*N >= j and j-i >= 0:
                T[j, N*i+j] = binom(N,i)*binom(N,j-i) / binom(2*N,j)

    return T

def _normSquare(x, Nveh, Ndim, prodM):
    """
    Compute the control points of the square of the norm of a vector
    
    normSquare(x, Nveh, Ndim, prodM)
    
    INPUT: Ndim*Nveh by N matrix x = [x1,...,x_Nveh)], x_i \in R^Ndim
    OUTPUT: control points of ||x_i||^2 ... Nveh by N matrix
    
    Code ported over from Venanzio Cichella's MATLAB norm_square function.
    NOTE: This only works on 1D or 2D matricies. It will fail for 3 or more.
    """
    x = np.array(x)
    if x.ndim == 1:
        x = x[None]
        
    m, N = x.shape
        
    xsquare = np.zeros((m,prodM.shape[0]))
    
    for i in range(m):
        xaug = np.dot(x[i,None].T, x[i,None])
        xnew = xaug.reshape((N**2,1))
        xsquare[i, :] = np.dot(prodM, xnew).T[0]
        
    S = np.zeros((Nveh, Nveh*Ndim))
    
    for i in range(Nveh):
        for j in range(Ndim):
            S[i, Ndim*i+j] = 1
            
    return np.dot(S, xsquare)

def angularRate(bezTraj):
    """
    Finds the angular rate of the 2D Bezier Curve.
    
    The equation for the angular rate is as follows:
        psiDot = (yDdot*xDot - xDdot*yDot) / (xDot^2 + yDot^2)
        Note the second derivative (Ddot) vs the first (Dot)
    
    RETURNS:
        RationalBezier - This function returns a rational Bezier curve because
            we must divide two Bezier curves.
    """
    if bezTraj.dim != 2:
        msg = ('The input curve must be two dimensional,\n'
               'instead it is {} dimensional'.format(bezTraj.dim))
        raise ValueError(msg)
    
    # We add epsilon to the denominator to avoid divide by zero errors
    eps = np.finfo(np.float64).eps
    
    x = Bezier(bezTraj.ctrlPts[0,:])
    xDot = x.diff()
    xDdot = xDot.diff()
    
    y = Bezier(bezTraj.ctrlPts[1,:])
    yDot = y.diff()
    yDdot = yDot.diff()
    
    numerator = yDdot*xDot - xDdot*yDot
    denominator = xDot*xDot + yDot*yDot
    
    cpts = numerator.ctrlPts / (denominator.ctrlPts+eps)
    weights = denominator.ctrlPts
    
    return RationalBezier(cpts, weights)

def angularRateSqr(bezTraj):
    """
    Finds the squared angular rate of the 2D Bezier Curve.
    
    The equation for the angular rate is as follows:
        psiDot = ((yDdot*xDot - xDdot*yDot))^2 / (xDot^2 + yDot^2)^2
        Note the second derivative (Ddot) vs the first (Dot)
    
    RETURNS:
        RationalBezier - This function returns a rational Bezier curve because
            we must divide two Bezier curves.
    """
    if bezTraj.dim != 2:
        msg = ('The input curve must be two dimensional,\n'
               'instead it is {} dimensional'.format(bezTraj.dim))
        raise ValueError(msg)
    
    # We add epsilon to the denominator to avoid divide by zero errors
    eps = np.finfo(np.float64).eps
    
    x = Bezier(bezTraj.ctrlPts[0,:])
    xDot = x.diff()
    xDdot = xDot.diff()
    
    y = Bezier(bezTraj.ctrlPts[1,:])
    yDot = y.diff()
    yDdot = yDot.diff()
    
    numerator = yDdot*xDot - xDdot*yDot
    numerator = numerator*numerator
    denominator = xDot*xDot + yDot*yDot
    denominator = denominator*denominator
    
    cpts = numerator.ctrlPts / (denominator.ctrlPts+eps)
    weights = denominator.ctrlPts
    
    return RationalBezier(cpts, weights)

if __name__ == "__main__":
    """
    Example code
    """
    import matplotlib.pyplot as plt
    plt.close('all')
    
    # Constant values
    NUM_VEH = 2 # Number of vehicles
    DIM = 2 # Dimension of the trajectories (usually 2D or 3D)
    
    # Optimal control points for 2 vehicles have been pre-computed
    vehCpts1 = [ 
            [0., 0., 0.19777111, 0.50000004, 0.80222879, 1., 1.],
            [0., 0.60602932, 1.21951607, 1.86339207, 2.88618266, 3.9393627, 5.]
            ]
    
    vehCpts2 = [
            [1., 1., 0.80222874, 0.50000022, 0.19777105, 0., 0.],
            [0., 1.06063715, 2.11381678, 3.13660759, 3.78048357, 4.3939704, 5.]
            ]
    
    """
    Plotting Trajectories
    """
    # Create the Bezier curves that represent the trajectories
    vehTraj1 = Bezier(vehCpts1)
    vehTraj2 = Bezier(vehCpts2)
    
    # Plot the trajectories
    plt.figure(1)
    plt.plot(vehTraj1.curve[0], vehTraj1.curve[1], 'b-',
             vehTraj1.ctrlPts[0], vehTraj1.ctrlPts[1], 'b.--')
    plt.plot(vehTraj2.curve[0], vehTraj2.curve[1], 'r-',
             vehTraj2.ctrlPts[0], vehTraj2.ctrlPts[1], 'r.--')
    plt.title('Vehicle Trajectories', fontsize=28)
    plt.xlabel('X Position', fontsize=20)
    plt.ylabel('Y Position', fontsize=20)
    
    """
    Plotting Velocities
    """
    # Take the derivative of the trajectories to get velocities
    vehVel1 = vehTraj1.diff().normSquare()
    vehVel2 = vehTraj2.diff().normSquare()
    
    # Plot the velocities
    plt.figure(2)
    plt.plot(vehVel1.tau, vehVel1.curve[0], 'b-',
             np.linspace(0, 1, vehVel1.deg+1),
             np.asarray(vehVel1.ctrlPts).squeeze(), 'b.--')
    plt.plot(vehVel2.tau, vehVel2.curve[0], 'r-',
             np.linspace(0, 1, vehVel2.deg+1),
             np.asarray(vehVel2.ctrlPts).squeeze(), 'r.--')
    plt.title('Vehicle Velocities', fontsize=28)
    plt.xlabel('Tau', fontsize=20)
    plt.ylabel('$Velocity^2$', fontsize=20)
    
    """
    Plotting Accelerations
    """
    # Take the derivative of the velocities to get accelerations
    vehAccel1 = vehVel1.diff().normSquare()
    vehAccel2 = vehVel2.diff().normSquare()
    
    # Plot the velocities
    plt.figure(3)
    plt.plot(vehAccel1.tau, vehAccel1.curve[0], 'b-',
             np.linspace(0, 1, vehAccel1.deg+1),
             np.asarray(vehAccel1.ctrlPts).squeeze(), 'b.--')
    plt.plot(vehAccel2.tau, vehAccel2.curve[0], 'r-',
             np.linspace(0, 1, vehAccel2.deg+1),
             np.asarray(vehAccel2.ctrlPts).squeeze(), 'r.--')
    plt.title('Vehicle Accelerations', fontsize=28)
    plt.xlabel('Tau', fontsize=20)
    plt.ylabel('$Acceleration^2$', fontsize=20)
    
    """
    Plotting Separation
    """
    vehList = [vehTraj1, vehTraj2]
    distVeh = []
    for i in range(NUM_VEH):
            for j in range(i, NUM_VEH):
                if j>i:
                    dv = Bezier(vehList[i].ctrlPts -
                        vehList[j].ctrlPts)
                    distVeh.append(dv.normSquare().elev(10))
                    
    plt.figure(4)
    plt.title('Squared Distances', fontsize=28)
    for dist in distVeh:
        plt.plot(dist._tau, dist.curve.squeeze())
        plt.plot(np.linspace(0,1,dist.deg+1),
                 np.asarray(dist.ctrlPts).squeeze(), '.')
    plt.xlabel('Tau', fontsize=16)
    plt.ylabel('$Distance^2$', fontsize=20)
    
    """
    Approximating Angular Rate
    
    The equation for the angular rate is as follows:
        psiDot = (yDdot*xDot - xDdot*yDot) / (xDot^2 + yDot^2)
        Note the second derivative (Ddot) vs the first (Dot)
    """
    # To approximate the angular rate of both vehicles, we will first elevate
    # the degree of the trajectories. We do this because as the degree gets
    # higher, the control points approach the actual curve. An arbitrary value
    # of 30 is used to elevate the curves. That means that the final degree
    # of each curve will be 30 + original degree
    elevTraj1 = vehTraj1.elev(30)
    elevTraj2 = vehTraj2.elev(30)
    
    # Calculate the angular rates. For optimization, we want to square of the
    # angular rates so that we only need a positive maximum constraint rather
    # than positive maximum and negative minimum constraints
    angularRateSqr1 = angularRateSqr(elevTraj1)
    angularRateSqr2 = angularRateSqr(elevTraj2)
    
    # Plot the approximate angular rate
    plt.figure(5)
    plt.title('Approximate Squared Angular Rates')
    plt.plot(np.linspace(0, 1, angularRateSqr1.deg+1),
             np.asarray(angularRateSqr1.ctrlPts).squeeze(), 'b.-')
    plt.plot(np.linspace(0, 1, angularRateSqr2.deg+1),
             np.asarray(angularRateSqr2.ctrlPts).squeeze(), 'r.-')
    plt.xlabel('Tau', fontsize=16)
    plt.ylabel('$Angular$ $Rate^2$', fontsize=20)
    
    plt.show()

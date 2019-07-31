#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 06:16:46 2019

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import numba
from numba import njit, jit
import numpy as np

import bezier as bez


DEG_ELEV = 10


class BezOptimization:
    def __init__(self,
                 numVeh=1,
                 dimension=1,
                 degree=5,
                 minimizeGoal='Euclidean',
                 maxSep=0.9,
                 minSpeed=0,
                 maxSpeed=1e6,
                 maxAngRate=1e6,
                 initPoints=None,
                 finalPoints=None,
                 initSpeeds=None,
                 finalSpeeds=None,
                 initAngs=None,
                 finalAngs=None,
                 tf=1.0,
                 pointObstacles=None,
                 shapeObstacles=None):

        self.pointObstacles = pointObstacles
        self.shapeObstacles = shapeObstacles

        self._numCols = degree+1
        if initPoints is not None:
            self._numCols -= 2
        if initSpeeds is not None:
            self._numCols -= 2

        self.model = {'numVeh': numVeh,
                      'dim': dimension,
                      'deg': degree,
                      'minGoal': minimizeGoal,
                      'maxSep': maxSep,
                      'minSpeed': minSpeed,
                      'maxSpeed': maxSpeed,
                      'maxAngRate': maxAngRate,
                      'initPoints': np.atleast_2d(initPoints),
                      'finalPoints': np.atleast_2d(finalPoints),
                      'initSpeeds': np.atleast_1d(initSpeeds),
                      'finalSpeeds': np.atleast_1d(finalSpeeds),
                      'initAngs': np.atleast_1d(initAngs),
                      'finalAngs': np.atleast_1d(finalAngs),
                      'tf': tf}

    @property
    def objectiveFunction(self):
        minGoal = self.model['minGoal'].lower()

        objectivesDict = {'euclidean': self.euclideanObjective,
                          'timeopt': lambda x: x[-1],
                          'accel': self.accelObjective,
                          'jerk': self.jerkObjective,
                          }

        try:
            return objectivesDict[minGoal]
        except KeyError:
            err = ('The provided minimize goal, {}, is not a valid goal. '
                   'The available minimize goals are:\n{}'
                   ).format(minGoal, objectivesDict.keys())
            raise ValueError(err)

    @property
    def temporalSeparationConstraints(self):

        if self.pointObstacles is not None:
            numObs = self.model['numVeh'] + len(self.pointObstacles)
            obstacleList = []
            for obstacle in self.pointObstacles:
                for d in range(self.model['dim']):
                    obstacleList.append([obstacle[d]]*(self.model['deg']+1))

            def wrapper(x):
                y = np.vstack((self.reshapeVector(x), obstacleList))
                return _temporalSeparationConstraints(y,
                                                      numObs,
                                                      self.model['dim'],
                                                      self.model['maxSep'])
        else:
            def wrapper(x):
                y = self.reshapeVector(x)
                return _temporalSeparationConstraints(y,
                                                      self.model['numVeh'],
                                                      self.model['dim'],
                                                      self.model['maxSep'])

        return wrapper

    def spatialSeparationConstraints(self, x):
        """
        """
        numVeh = self.model['numVeh']
        dim = self.model['dim']
        maxSep = self.model['maxSep']
        numObs = numVeh + len(self.shapeObstacles)

        distances = []
        vehList = []

        y = self.reshapeVector(x)

        for i in range(numVeh):
            vehList.append(bez.Bezier(y[i*dim:(i+1)*dim, :]))

        for obstacle in self.shapeObstacles:
            vehList.append(obstacle)

        for i in range(numObs):
            for j in range(i, numObs):
                if j > i:
                    distances.append(vehList[i].minDist(vehList[j]))

        return np.array(distances)-maxSep

    @property
    def minSpeedConstraints(self):
        """
        """
        def wrapper(x):
            if self.model['minGoal'].lower() == 'timeopt':
                tf = x[-1]
            else:
                tf = self.model['tf']
            y = self.reshapeVector(x)
            return _minSpeedConstraints(y,
                                        self.model['numVeh'],
                                        self.model['dim'],
                                        tf,
                                        self.model['minSpeed']
                                        )
        return wrapper

    @property
    def maxSpeedConstraints(self):
        """
        """
        def wrapper(x):
            if self.model['minGoal'].lower() == 'timeopt':
                tf = x[-1]
            else:
                tf = self.model['tf']
            y = self.reshapeVector(x)
            return _maxSpeedConstraints(y,
                                        self.model['numVeh'],
                                        self.model['dim'],
                                        tf,
                                        self.model['maxSpeed']
                                        )
        return wrapper

    @property
    def maxAngularRateConstraints(self):
        """
        """
        def wrapper(x):
            if self.model['minGoal'].lower() == 'timeopt':
                tf = x[-1]
            else:
                tf = self.model['tf']
            y = self.reshapeVector(x)
            return _maxAngularRateConstraints(y,
                                              self.model['numVeh'],
                                              self.model['dim'],
                                              tf,
                                              self.model['maxAngRate']
                                              )
        return wrapper

    def generateGuess(self, std=0, seed=None):
        """
        """
        dim = self.model['dim']
        deg = self.model['deg']
        numVeh = self.model['numVeh']
        tf = self.model['tf']
        initPoints = self.model['initPoints']
        finalPoints = self.model['finalPoints']
        initSpeeds = self.model['initSpeeds']
        finalSpeeds = self.model['finalSpeeds']
        initAngs = self.model['initAngs']
        finalAngs = self.model['finalAngs']

        np.random.seed(seed)

        xGuess = []

        for i in range(numVeh):
            for j in range(dim):
                if initSpeeds[0] is None:
                    line = np.linspace(initPoints[i, j],
                                       finalPoints[i, j],
                                       deg+1)
                    line += np.random.randn(deg+1)*std

                else:
                    if dim != 2:
                        err = ('The dimension must be 2 for initial and final '
                               'speeds and angles.')
                        raise ValueError(err)

                    initMag = initSpeeds[i]*tf/deg
                    finalMag = finalSpeeds[i]*tf/deg
                    if j % 2 == 0:
                        initPt = initPoints[i, j] + initMag*np.cos(initAngs[i])
                        finalPt = (finalPoints[i, j] -
                                   finalMag*np.cos(finalAngs[i]))
                    else:
                        initPt = initPoints[i, j] + initMag*np.sin(initAngs[i])
                        finalPt = (finalPoints[i, j] -
                                   finalMag*np.sin(finalAngs[i]))

                    line = np.linspace(initPt, finalPt, deg+1-2)
                    line += np.random.randn(deg+1-2)*std

                xGuess.append(line[1:-1])

        if self.model['minGoal'].lower() == 'timeopt':
            xGuess.append([tf])

        return np.concatenate(xGuess)

    def reshapeVector(self, x):
        """
        """
        dim = self.model['dim']
        deg = self.model['deg']
        numVeh = self.model['numVeh']
        tf = self.model['tf']
        initPoints = self.model['initPoints']
        finalPoints = self.model['finalPoints']
        initSpeeds = self.model['initSpeeds']
        finalSpeeds = self.model['finalSpeeds']
        initAngs = self.model['initAngs']
        finalAngs = self.model['finalAngs']

        numCols = self._numCols
        numRows = dim*self.model['numVeh']

        # If we are optimizing time, grab the last element which is tf
        if self.model['minGoal'].lower() == 'timeopt':
            tf = x[-1]
            x = x[:-1]

        y = np.empty((dim*numVeh, deg+1))
        offset = 0

        if initPoints is not None:
            offset += 1
            for i in range(initPoints.shape[0]):
                y[i*dim:(i+1)*dim, 0] = initPoints[i]
                y[i*dim:(i+1)*dim, -1] = finalPoints[i]

        if initSpeeds[0] is not None:
            offset += 1

            initMag = initSpeeds*tf/deg
            finalMag = finalSpeeds*tf/deg
            y[::2, 1] = initPoints[:, 0] + initMag*np.cos(initAngs)       # X
            y[1::2, 1] = initPoints[:, 1] + initMag*np.sin(initAngs)      # Y
            y[::2, -2] = finalPoints[:, 0] - finalMag*np.cos(finalAngs)   # X
            y[1::2, -2] = finalPoints[:, 1] - finalMag*np.sin(finalAngs)  # Y

        y[:, offset:-offset] = x.reshape((numRows, numCols))

        return y

    def euclideanObjective(self, x):
        """
        """
        return _euclideanObjective(self.reshapeVector(x),
                                   self.model['numVeh'],
                                   self.model['dim'])

    def accelObjective(self, x):
        """
        """
        return _minAccelObjective(self.reshapeVector(x),
                                  self.model['numVeh'],
                                  self.model['dim'],
                                  self.model['tf'])

    def jerkObjective(self, x):
        """
        """
        return _minJerkObjective(self.reshapeVector(x),
                                 self.model['numVeh'],
                                 self.model['dim'],
                                 self.model['tf'])


def _temporalSeparationConstraints(y, nVeh, dim, maxSep):
    """Calculate the separation between vehicles.

    The maximum separation is found by degree elevation.

    NOTE: This only works for 2 dimensions.

    :param x: Optimization vector
    :type x: numpy.ndarray
    :param nVeh: Number of vehicles
    :type nVeh: int
    :param dim: Dimension of the vehicles. Currently only works for 2D
    :type dim: int
    :param maxSep: Maximum separation between vehicles.
    :type maxSep: float
    """
    if nVeh > 1:
        distVeh = []
        vehList = []

        for i in range(nVeh):
            vehList.append(bez.Bezier(y[i*dim:(i+1)*dim, :]))

        for i in range(nVeh-1):
            for j in range(i+1, nVeh):
                dv = vehList[i] - vehList[j]
                distVeh.append(dv.normSquare().elev(DEG_ELEV))
#                distVeh.append(dv.normSquare().min())

        distances = np.concatenate([i.cpts.squeeze() for i in distVeh])
#        distances = np.array(distVeh)

        return (distances - maxSep**2).squeeze()

    else:
        return None


def _minSpeedConstraints(y, nVeh, dim, tf, minSpeed):
    """Creates the minimum velocity constraints.

    Useful in systems such as aircraft who may not fall below a certain speed.

    :param x: Optimization vector
    :type x: numpy.ndarray
    :param nVeh: Number of vehicles
    :type nVeh: int
    :param dim: Dimension of the vehicles. Currently only works for 2D.
    :type dim: int
    :param model: See model description in BezOptimization class description.
    :type model: dict
    :param minSpeed: Minimum speed of the vehicle.
    :type minSpeed: float
    :return: Inequality constraint for the minimum speed.
    :rtype: float
    """
#    y = reshapeVector(x, nVeh, dim, model)
#    if model['type'].lower() == 'timeopt':
#        tf = x[-1]
#    else:
#        tf = model['tf']

    speeds = []

    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf)
        speed = pos.diff()
        speeds.append(speed)

    speedSqr = [curve.normSquare().elev(DEG_ELEV) for curve in speeds]

    speeds = np.concatenate([i.cpts.squeeze() for i in speedSqr])

    return (speeds - minSpeed**2).squeeze()


def _maxSpeedConstraints(y, nVeh, dim, tf, maxSpeed):
    """Creates the maximum velocity constraints.

    Useful for limiting the maximum speed of a vehicle.

    :param x: Optimization vector
    :type x: numpy.ndarray
    :param nVeh: Number of vehicles
    :type nVeh: int
    :param dim: Dimension of the vehicles. Currently only works for 2D
    :type dim: int
    :param model: See model description in BezOptimization class description
    :type model: dict
    :param maxSpeed: Maximum speed of the vehicle.
    :type maxSpeed: float
    :return: Inequality constraint for the maximum speed
    :rtype: float
    """
#    y = reshapeVector(x, nVeh, dim, model)
#    if model['type'].lower() == 'timeopt':
#        tf = x[-1]
#    else:
#        tf = model['tf']

    speeds = []

    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf)
        speed = pos.diff()
        speeds.append(speed)

    speedSqr = [curve.normSquare().elev(DEG_ELEV) for curve in speeds]

    speeds = np.concatenate([i.cpts.squeeze() for i in speedSqr])

    return (maxSpeed**2 - speeds).squeeze()


def _maxAngularRateConstraints(y, nVeh, dim, tf, maxAngRate):
    """Creates the maximum angular rate constraint.

    This is useful for a dubins car model that has a constraint on the maximum
    angular rate. The dimension is assumed to be 2.

    :param x: Optimization vector
    :type x: numpy.ndarray
    :param nVeh: Number of vehicles
    :type nVeh: int
    :param dim: Dimension of the vehicles. Currently only works for 2D
    :type dim: int
    :param model: See model description in BezOptimization class description
    :type model: dict
    :param maxAngRate: Maximum angular rate of the vehicle (in radians).
    :type maxAngRate: float
    :return: Inequality constraint for the maximum angular rate
    :rtype: float
    """
#    y = reshapeVector(x, nVeh, dim, model)
#    if model['type'].lower() == 'timeopt':
#        tf = x[-1]
#    else:
#        tf = model['tf']

    angularRates = []
    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf)
        angRate = _angularRateSqr(pos.elev(DEG_ELEV))
        angularRates.append(angRate)

    angularRateCpts = np.concatenate(
            [i.cpts.squeeze() for i in angularRates])

    return (maxAngRate**2 - angularRateCpts).squeeze()


@njit(cache=True)
def _euclideanObjective(y, nVeh, dim):
    """Sums the Euclidean distance between control points.

    The Euclidean difference between each neighboring pair of control points is
    summed for each vehicle.

    :param y: Optimized vector that has been reshaped using the reshapeVector
        function.
    :type y: numpy.ndarray
    :param nVeh: Number of vehicles
    :type nVeh: int
    :param dim: Dimension of the vehicles. Currently only works for 2D
    :type dim: int
    :return: Sum of the Euclidean distances
    :rtype: float
    """
    summation = 0.0
    temp = np.empty(3)
    length = y.shape[1]
    for veh in range(nVeh):
        for i in range(length-1):
            for j in range(dim):
                temp[j] = y[veh*dim+j, i+1] - y[veh*dim+j, i]

            summation += _norm(temp)  # np.linalg.norm(temp)

    return summation


@njit(cache=True)
def _norm(x):
    """
    """
    summation = 0.
    for val in x:
        summation += val*val

    return np.sqrt(summation)


def _minAccelObjective(y, nVeh, dim, tf):
    """
    """
    curves = []
    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf)
        vel = pos.diff()

        accel = vel.diff()
        curves.append(accel)

    summation = 0.0
    for curve in curves:
        temp = curve.normSquare().elev(DEG_ELEV)
        summation = summation + temp.cpts.sum()

    return summation


@njit(cache=True)
def _minJerkObjective(y, nVeh, dim, tf):
    """
    """
    curves = []
    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf)
        vel = pos.diff()

        jerk = vel.diff().diff()
        curves.append(jerk)

    summation = 0.0
    for curve in curves:
        temp = curve.normSquare().elev(DEG_ELEV)
        summation = summation + temp.cpts.sum()

    return summation


@jit(cache=True)
def _angularRate(bezTraj):
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

    x = bezTraj.x
    xDot = x.diff()
    xDdot = xDot.diff()

    y = bezTraj.y
    yDot = y.diff()
    yDdot = yDot.diff()

    numerator = yDdot*xDot - xDdot*yDot
    denominator = xDot*xDot + yDot*yDot

    cpts = numerator.cpts / (denominator.cpts)
    weights = denominator.cpts

    return bez.RationalBezier(cpts, weights)


#@jit(cache=True)
def _angularRateSqr(bezTraj):
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

    x = bezTraj.x
    xDot = x.diff()
    xDdot = xDot.diff()

    y = bezTraj.y
    yDot = y.diff()
    yDdot = yDot.diff()

    numerator = yDdot*xDot - xDdot*yDot
    numerator = numerator*numerator
    denominator = xDot*xDot + yDot*yDot
    denominator = denominator*denominator

    cpts = numerator.cpts / (denominator.cpts)
    weights = denominator.cpts

    return bez.RationalBezier(cpts, weights)


if __name__ == '__main__':
    numVeh = 2
    dim = 2
    deg = 5
    xLen = numVeh*dim*(deg+1-4)
    params = {'numVeh': numVeh, 'dimension': dim, 'degree': deg,
              'initPoints': np.array([[1, 2], [3, 4]]),
              'finalPoints': np.array([[5, 6], [7, 8]]),
              'initSpeeds': np.array([3, 3]),
              'finalSpeeds': np.array([10, 10]),
              'initAngs': np.array([np.pi/2, np.pi/2]),
              'finalAngs': np.array([0, 0]),
              'pointObstacles': [[1, 2], [3, 4]]
              }

    bezopt = BezOptimization(**params)

    xGuess = bezopt.generateGuess()

    x = np.random.randint(0, 10, xLen)
    cpts1 = bezopt.reshapeVector(x)[:2, :]
    cpts2 = bezopt.reshapeVector(x)[2:, :]

    print('cpts1:\n{},\ncpts2:\n{}'.format(cpts1, cpts2))
    print('X Guess:\n{}'.format(xGuess))

    c1guess = bez.Bezier(bezopt.reshapeVector(xGuess)[:2, :])
    c2guess = bez.Bezier(bezopt.reshapeVector(xGuess)[2:, :])

    plt.close('all')
    ax1 = c1guess.plot()
    c2guess.plot(ax1)
    plt.title('X Guess')

    c1 = bez.Bezier(cpts1)
    c2 = bez.Bezier(cpts2)

    ax2 = c1.plot()
    c2.plot(ax2)
    plt.title('Reshape Vector')
    plt.show()













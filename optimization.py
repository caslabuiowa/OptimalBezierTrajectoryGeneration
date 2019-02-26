#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:17:47 2018

@author: ckielasjensen
"""

import numpy as np
import numba

import bezier as bez


class BezOptimization:
    """Bezier curve optimization class.

    Use this along with the Bezier class to optimize trajectories.

    The model dict is an important parameter used to help the reshapeVector
    produce the desired output. The dict parameters that are supported are:
        type - String representing the type of model. Currently, the
                supported models are: 'dubins', 'uav', and 'general'
        initPoints - Initial points of the vehicles
        finalPoints - Final points of the vehicles
        initSpeeds - Initial speeds of the vehicles
        finalSpeeds - Final speeds of the vehicles
        initAngs - Initial angles (in radians) of the vehicles that follow
            the ROS standards REP103 where 0 rad is East
        finalAngs - Final angles of the vehicles (in radians)
        tf - Final time when the vehicles reach their final points.

    :param nVeh: Number of vehicles
    :type nVeh: int
    :param dimension: Dimension of the vehicles. Currently only works for 2D
    :type dimension: int
    :param degree: Degree of the Bezier curves being used
    :type degree: int
    :param minimizeGoal: Element to be minimized. This string can be one of a
        few different values:
            vel - Minimize the sum of velocities of the vehicle trajectories
            accel - Minimize the sum of accelerations of the vehicle
                trajectories
            jerk - Minimize the sum of jerks of the vehicle trajectories
            euclidean - Minimize the sum of the Euclidean distance between the
                control points of each trajectory.
    :type minimizeGoal: str
    :param maxSep: Maximum separation between vehicles at each point in time.
    :type maxSep: float
    :param minSpeed: Minimum speed of the vehicles
    :type minSpeed: float
    :param maxSpeed: Maximum speed of the vehicles
    :type maxSpeed: float
    :param maxAngRate: Maximum angular rate of the vehicles
    :type maxAngRate: float
    :param modelType: Model type being optimized. The models currently
        supported are: 'dubins', 'uav', and 'general
    :type modelType: str
    :param maxSep: Maximum separation between vehicles.
    :type maxSep: float

    startPoints (np.array) - Initial positions of the vehicles in the
    following format (for lower dimensions, omit the y and/or z values):
        [ [v0x, v0y, v0z],
          [v1x, v1y, v1z],
          ...
          [vNx, vNy, vNz] ]

    endPoints (np.array) - Final positions of the vehicles using the same
    format found in startPoints.

    minGoal (str) - Can be one of the following strings:
        'pos' - Minimize distance traveled
        'vel' - Minimize the velocity
        'accel' - Minimize the acceleration
        'jerk' - Minimize the jerk
    """

    def __init__(self,
                 numVeh=1,
                 dimension=1,
                 degree=5,
                 minimizeGoal='Euclidean',
                 maxSep=0.9,
                 minSpeed=0,
                 maxSpeed=1e6,
                 maxAngRate=1e6,
                 modelType=None,
                 initPoints=None,
                 finalPoints=None,
                 initSpeeds=None,
                 finalSpeeds=None,
                 initAngs=None,
                 finalAngs=None,
                 tf=1.0):

        self.nVeh = numVeh
        self.dim = dimension
        self.deg = degree
        self.minGoal = minimizeGoal
        self.maxSep = maxSep
        self.minSpeed = minSpeed
        self.maxSpeed = maxSpeed
        self.maxAngRate = maxAngRate

        self.model = {'type': modelType,
                      'initPoints': np.atleast_2d(initPoints),
                      'finalPoints': np.atleast_2d(finalPoints),
                      'initSpeeds': initSpeeds,
                      'finalSpeeds': finalSpeeds,
                      'initAngs': initAngs,
                      'finalAngs': finalAngs,
                      'tf': tf}

        self.separationConstraints = lambda x: _separationConstraints(
                                              x,
                                              self.nVeh,
                                              self.dim,
                                              self.model,
                                              self.maxSep)

        self.minSpeedConstraints = lambda x: _minSpeedConstraints(
                                              x,
                                              self.nVeh,
                                              self.dim,
                                              self.model,
                                              self.minSpeed)

        self.maxSpeedConstraints = lambda x: _maxSpeedConstraints(
                                              x,
                                              self.nVeh,
                                              self.dim,
                                              self.model,
                                              self.maxSpeed)

        self.maxAngularRateConstraints = lambda x: _maxAngularRateConstraints(
                                              x,
                                              self.nVeh,
                                              self.dim,
                                              self.model,
                                              self.maxAngRate)

        self.objectiveFunction = lambda x: _objectiveFunction(
                                              x,
                                              self.nVeh,
                                              self.dim,
                                              self.model,
                                              minGoal=self.minGoal)

    def generateGuess(self, random=False, seed=None):
        """
        Generates an initial guess for the optimizer. Set random to true to
        add random noise to the initial guess. You can provide a seed for
        deterministic results.
        """
        if self.dim != 2:
            msg = 'Optimization is currently built only for 2 dimensions.'
            raise NotImplementedError(msg)

        xGuess = []

        for i in range(self.nVeh):
            for j in range(self.dim):
                line = np.linspace(
#                        self.startPoints[i, j],
#                        self.endPoints[i, j], self.deg+1)
                        self.model['initPoints'][i, j],
                        self.model['finalPoints'][i, j], self.deg+1)

                if random:
                    np.random.seed(seed)
                    line = line + np.random.random(len(line))

                xGuess.append(line[1:-1])

        return np.concatenate(xGuess)


def _separationConstraints(x, nVeh, dim, model, maxSep):
    """Calculate the separation between vehicles.

    The maximum separation is found by degree elevation.

    NOTE: This only works for 2 dimensions.

    :param x: Optimization vector
    :type x: numpy.ndarray
    :param nVeh: Number of vehicles
    :type nVeh: int
    :param dim: Dimension of the vehicles. Currently only works for 2D
    :type dim: int
    :param model: See model description in BezOptimization class description
    :type model: dict
    :param maxSep: Maximum separation between vehicles.
    :type maxSep: float
    """
    y = reshapeVector(x, nVeh, dim, model)
    if model['type'].lower() == 'obstacles':
            nVeh += 2
            obs = np.empty((4, y.shape[1]))
            obs[0, :] = 3
            obs[1, :] = 2
            obs[2, :] = 6
            obs[3, :] = 7
            y = np.vstack((y, obs))

    if nVeh > 1:
        tf = model['tf']

        distVeh = []
        vehList = []
        for i in range(nVeh):
            vehList.append(bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf))

        for i in range(nVeh):
            for j in range(i, nVeh):
                if j > i:
                    dv = vehList[i] - vehList[j]
                    distVeh.append(dv.normSquare().elev(10))

        distances = np.concatenate([i.cpts.squeeze() for i in distVeh])

        return (distances - maxSep**2).squeeze()
    else:
        return None


#def _separationConstraints(x, nVeh, dim, model, maxSep):
#    """Calculate the separation between vehicles.
#
#    The maximum separation is found by degree elevation.
#
#    NOTE: This only works for 2 dimensions.
#
#    :param x: Optimization vector
#    :type x: numpy.ndarray
#    :param nVeh: Number of vehicles
#    :type nVeh: int
#    :param dim: Dimension of the vehicles. Currently only works for 2D
#    :type dim: int
#    :param model: See model description in BezOptimization class description
#    :type model: dict
#    :param maxSep: Maximum separation between vehicles.
#    :type maxSep: float
#    """
#    if nVeh > 1:
#        y = reshapeVector(x, nVeh, dim, model)
#        tf = model['tf']
#
#        distVeh = []
#        vehList = []
#        for i in range(nVeh):
#            vehList.append(bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf))
#
#        for i in range(nVeh):
#            for j in range(i, nVeh):
#                if j > i:
#                    dv = vehList[i] - vehList[j]
#                    distVeh.append(dv.normSquare().min())
#
#        return np.array(distVeh) - maxSep**2
#    else:
#        return None


def _minSpeedConstraints(x, nVeh, dim, model, minSpeed):
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
    y = reshapeVector(x, nVeh, dim, model)
    tf = model['tf']

    speeds = []

    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf)
        speed = pos.diff()
        speeds.append(speed)

    speedSqr = [curve.normSquare().elev(50) for curve in speeds]

    speeds = np.concatenate([i.cpts.squeeze() for i in speedSqr])

    return (speeds - minSpeed**2).squeeze()


def _maxSpeedConstraints(x, nVeh, dim, model, maxSpeed):
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
    y = reshapeVector(x, nVeh, dim, model)
    tf = model['tf']

    speeds = []

    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf)
        speed = pos.diff()
        speeds.append(speed)

    speedSqr = [curve.normSquare().elev(10) for curve in speeds]

    speeds = np.concatenate([i.cpts.squeeze() for i in speedSqr])

    return (maxSpeed**2 - speeds).squeeze()


def _maxAngularRateConstraints(x, nVeh, dim, model, maxAngRate):
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
    y = reshapeVector(x, nVeh, dim, model)
    tf = model['tf']

    angularRates = []
    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf)
        angRate = angularRateSqr(pos.elev(10))
        angularRates.append(angRate)

    angularRateCpts = np.concatenate(
            [i.cpts.squeeze() for i in angularRates])

    return (maxAngRate**2 - angularRateCpts).squeeze()


def _objectiveFunction(x, nVeh, dim, model, minGoal):
    """Objective function to be optimized.



    :param x: Optimization vector
    :type x: numpy.ndarray
    :param nVeh: Number of vehicles
    :type nVeh: int
    :param dim: Dimension of the vehicles. Currently only works for 2D
    :type dim: int
    :param model: See model description in BezOptimization class description
    :type model: dict
    :param minGoal: Element to be minimized. This string can be one of a few
        different values:
            vel - Minimize the sum of velocities of the vehicle trajectories
            accel - Minimize the sum of accelerations of the vehicle
                trajectories
            jerk - Minimize the sum of jerks of the vehicle trajectories
            euclidean - Minimize the sum of the Euclidean distance between the
                control points of each trajectory.
    :type minGoal: str
    :return: Cost of the current iteration according to the minGoal
    :rtype: float
    """
    y = reshapeVector(x, nVeh, dim, model)
    tf = model['tf']
    curves = []
    minGoal = minGoal.lower()

    if minGoal == 'euclidean':
        return euclideanObjective(y, nVeh, dim)

    else:
        for i in range(nVeh):
            pos = bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf)
            vel = pos.diff()
            if minGoal == 'accel':
                accel = vel.diff()
                curves.append(accel)
            elif minGoal == 'jerk':
                jerk = vel.diff().diff()
                curves.append(jerk)

        summation = 0.0
        for curve in curves:
            temp = curve.normSquare()
            summation = summation + temp.cpts.sum()

        return summation


@numba.jit(nopython=True)
def euclideanObjective(y, nVeh, dim):
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
    temp = np.zeros(3)
    length = y.shape[1]
    for veh in range(nVeh):
        for i in range(length-1):
            for j in range(dim):
                temp[j] = y[veh*dim+j, i+1] - y[veh*dim+j, i]

            summation += np.linalg.norm(temp)

    return summation


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


def reshapeVector(x, nVeh, dim, model=None):
    """
    Converts the input vector x into a matrix that includes the start and end
    control points of a Bezier curve for each vehicle in each dimension.

    INPUTS:
        x - Vector of points to be optimized. The length of the vector depends
        on the number of vehicles, dimension, and model.

        nVeh - Number of vehicles.

        dim - Dimension of the trajectories (typically 2 or 3)

        model - Dictionary of the model parameters. The dictionary must include
        the following values:
            * type: Name of the model. The model names currently supported are
              "dubins", "generic", and "uav". See below for more information
              regarding each model.
            * initAngs: Vector of initial angles, in radians, for each vehicle
              where each element corresponds to the ith vehicle. The angles
              follow ROS's REP 103. X is East, Y is North, and Z is up.
            * finalAngs: Same as initAngs but the final angles instead of the
              initial angles.

    RETURNS:
        2D numpy array of Bezier curve control points for each vehicle where
        each row corresponds to the dimension of the current vehicle. The array
        will look like this:
            [[v1x0, v1x1, ..., v1xDegree],
             [v1y0, v1y1, ..., v1yDegree],
             [v1z0, v1z1, ..., v1zDegree],
             [v1dim0, v1dim1, ..., v1dimDegree],
             [v2x0, v2x1, ..., v2xDegree],
             ...
             [vnVehx0, vnVehx1, ..., vnVehxDegree]]

    Model Types:
        Dubins: Uses the Dubin's car model for a differential drive vehicle.
        This model type requires the following parameters in the model
        dictionary: initPoints, finalPoints, initSpeeds, finalSpeeds, initAngs,
        finalAngs, tf.
        The input vector x will not include the first two and last two control
        points for each vehicle. The vector should look like the following
        [X02, X03, X04, ..., X0DEG-1,
         Y02, ..., Y0DEG-1,
         X12, ..., X1DEG-1,
         ...
         XN2, ..., XNDEG-1]
        Note that it is DEG-1 and not DEG-2 because the degree of a Bezier
        curve is already 1 less than the total number of control points.

        UAV:

        Generic: The only fixed values for the generic model are the start and
        end points. The input vector x should look like the following
        [X01, X02, X03, ..., X0DEG,
        Y01, ..., Y0DEG,
        Z01, ..., Z0DEG,
        ...
        XN1, ..., XNDEG]

    The input vector is of the following form:
        [initAngle0, finalAngle0, X01, X02, ..., X0DEG, Y01, Y02, ..., Y0DEG,
         initAngle1, finalAngle1, X11, X12, ..., X1DEG, Y11, Y12, ..., Y1DEG,
         ...
         initAngleNVEH, finalAngleNVEH, ... ]
    """
    x = np.array(x)
    numRows = int(nVeh*dim)
    numCols = int(x.size/numRows)
    x = x.reshape((numRows, numCols))

    # Dict params used by all models
    modelType = model['type'].lower()
    initPoints = np.array(model['initPoints'])
    finalPoints = np.array(model['finalPoints'])

    if modelType == 'dubins' or modelType == 'obstacles':
        """
        Dubin's model input vector:
            [X02, X03, X04, ..., X0DEG-1,
             Y02, ..., Y0DEG-1,
             X12, ..., X1DEG-1,
             ...
             XN2, ..., XNDEG-1]
        """
        if dim != 2:
            msg = 'The Dubin''s car model only accepts 2 dimensions.'
            raise ValueError(msg)
        degree = numCols + 4 - 1
        y = np.empty((numRows, degree+1))

        # Dict params used by Dubin's model specifically
        initSpeeds = np.array(model['initSpeeds'])
        finalSpeeds = np.array(model['finalSpeeds'])
        initAngs = np.array(model['initAngs'])
        finalAngs = np.array(model['finalAngs'])
        tf = model['tf']

        initMag = initSpeeds*tf/degree
        finalMag = finalSpeeds*tf/degree

#        print('InitPoints: {}'.format(initPoints))
#        print('InitAngs: {}'.format(initAngs))
#        print('InitMag: {}'.format(initMag))
#        print('y[::2, 1]: {}'.format(y[::2, 1]))

        y[:, 2:-2] = x
        y[::2, 0] = initPoints[:, 0]     # init X
        y[1::2, 0] = initPoints[:, 1]    # init Y
        y[::2, -1] = finalPoints[:, 0]   # final X
        y[1::2, -1] = finalPoints[:, 1]  # final Y
        y[::2, 1] = initPoints[:, 0] + initMag*np.cos(initAngs)       # X
        y[1::2, 1] = initPoints[:, 1] + initMag*np.sin(initAngs)      # Y
        y[::2, -2] = finalPoints[:, 0] - finalMag*np.cos(finalAngs)   # X
        y[1::2, -2] = finalPoints[:, 1] - finalMag*np.sin(finalAngs)  # Y

    elif modelType == 'generic':
        """
        Generic model input vector:
            [X01, X02, X03, ..., X0DEG,
            Y01, ..., Y0DEG,
            Z01, ..., Z0DEG,
            ...
            XN1, ..., XNDEG]
        """
        degree = numCols + 2 - 1
        y = np.empty((numRows, degree+1))

        y[::2, 0] = initPoints[:, 0]
        y[1::2, 0] = initPoints[:, 1]
        y[::2, -1] = finalPoints[:, 0]
        y[1::2, -1] = finalPoints[:, 1]
        y[:, 1:-1] = x

    elif modelType == 'uav':
        pass

    elif modelType == '3d':
        degree = numCols + 2 - 1
        y = np.empty((numRows, degree+1))

        y[::3, 0] = initPoints[:, 0]
        y[1::3, 0] = initPoints[:, 1]
        y[2::3, 0] = initPoints[:, 2]
        y[::3, -1] = finalPoints[:, 0]
        y[1::3, -1] = finalPoints[:, 1]
        y[2::3, -1] = finalPoints[:, 2]
        y[:, 1:-1] = x

#    elif modelType == 'obstacles':
#        """
#        Obstacles model input vector:
#            [X01, X02, X03, ..., X0DEG,
#            Y01, ..., Y0DEG,
#            Z01, ..., Z0DEG,
#            ...
#            XN1, ..., XNDEG]
#        """
#        degree = numCols + 2 - 1
#        y = np.empty((numRows, degree+1))
#
#        y[::2, 0] = initPoints[:, 0]
#        y[1::2, 0] = initPoints[:, 1]
#        y[::2, -1] = finalPoints[:, 0]
#        y[1::2, -1] = finalPoints[:, 1]
#        y[:, 1:-1] = x

    else:
        msg = '{} is not a valid model type.'.format(modelType)
        raise ValueError(msg)

    return y.astype(float)

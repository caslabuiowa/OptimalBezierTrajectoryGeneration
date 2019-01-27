#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:17:47 2018

@author: ckielasjensen
"""

import numpy as np
import matplotlib.pyplot as plt
import numba

import bezier as bez

# TODO

class BezOptimization:
    """
    Bezier curve optimization class. Use this along with the Bezier class to
    optimize trajectories.

    BezOptimization(nVeh, dim, maxSep, deg, startPoints, endPoints, minElem)

    INPUTS:
        nVeh (int) - Number of vehicles.

        dim (int) - Dimension of trajectories (1D, 2D, or 3D).

        maxSep (float) - Maximum separation of the vehicles (no two should ever
        come closer than maxSep).

        deg (int) - Degree of the Bezier curves.

        startPoints (np.array) - Initial positions of the vehicles in the
        following format (for lower dimensions, omit the y and/or z values):
            [ [v0x, v0y, v0z],
              [v1x, v1y, v1z],
              ...
              [vNx, vNy, vNz] ]

        endPoints (np.array) - Final positions of the vehicles using the same
        format found in startPoints.

        minElem (str) - Can be one of the following strings:
            'pos' - Minimize distance traveled
            'vel' - Minimize the velocity
            'accel' - Minimize the acceleration
            'jerk' - Minimize the jerk
    """
    def __init__(self, nVeh, dim, maxSep, deg, startPoints, endPoints,
                 minElem, minVel=0, maxVel=0.22, maxAngRate=2.84,
                 modelType='Dubins',
                 initVels=None, finalVels=None,
                 initAngs=None, finalAngs=None,
                 tf=1):
        self.nVeh = nVeh
        self.dim = dim
        self.maxSep = maxSep
        self.deg = deg
        self.startPoints = startPoints
        self.endPoints = endPoints
        self.minElem = minElem
        self.minVel = minVel
        self.maxVel = maxVel
        self.maxAngRate = maxAngRate
        self.tf = tf
#        self.xGuess = self.generateGuess()

        # If initial and/or final velocities aren't specified, set them to 0
        if initVels is None:
            initVels = [0]*nVeh
        if finalVels is None:
            finalVels = [0]*nVeh
        # Same for initial and final vangles
        if initAngs is None:
            initAngs = [0]*nVeh
        if finalAngs is None:
            finalAngs = [0]*nVeh

        self.model = {'type': modelType,
                      'initPoints': startPoints,
                      'finalPoints': endPoints,
                      'initVels': initVels,
                      'finalVels': finalVels,
                      'initAngs': initAngs,
                      'finalAngs': finalAngs,
                      'tf': tf}

        self.separationConstraints = lambda x: _separationConstraints(x,
                                              self.nVeh,
                                              self.dim,
                                              self.model,
                                              self.maxSep,
                                              tf=tf)

        self.minVelConstraints = lambda x: _minVelConstraints(x,
                                              self.nVeh,
                                              self.dim,
                                              self.model,
                                              self.minVel,
                                              tf=tf)

        self.maxVelConstraints = lambda x: _maxVelConstraints(x,
                                              self.nVeh,
                                              self.dim,
                                              self.model,
                                              self.maxVel,
                                              tf=tf)

        self.maxAngularRateConstraints = lambda x: _maxAngularRateConstraints(
                                              x,
                                              self.nVeh,
                                              self.dim,
                                              self.model,
                                              self.maxAngRate,
                                              tf=tf)

        self.objectiveFunction = lambda x: _objectiveFunction(x,
                                              self.nVeh,
                                              self.dim,
                                              self.model,
                                              tf=tf,
                                              minElem=self.minElem)

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
                        self.startPoints[i,j],
                        self.endPoints[i,j], self.deg+1)

                if random:
                    np.random.seed(seed)
                    line = line + np.random.random(len(line))

                xGuess.append(line[1:-1])

        return np.concatenate(xGuess)

    def plot(self, x):
        _separationConstraints(x,
                              self.nVeh,
                              self.dim,
                              self.model,
                              self.maxSep,
                              tf=self.tf,
                              plot=True)

        _maxAngularRateConstraints(x,
                              self.nVeh,
                              self.dim,
                              self.model,
                              self.maxAngRate,
                              tf=self.tf,
                              plot=True)

        _objectiveFunction(x,
                          self.nVeh,
                          self.dim,
                          self.model,
                          tf=self.tf,
                          plot=True,
                          minElem=self.minElem)

def _separationConstraints(x, nVeh, dim, model, maxSep, tf=1, plot=False):
    """
    Calculate the separation between vehicles.

    NOTE: This only works for 2 dimensions at the moment.
    """
    if nVeh > 1:
#        y = reshapeVector(x, nVeh, dim, startPoints, endPoints, 0, 0)
        y = reshapeVector(x, nVeh, dim, model)

        distVeh = []
        vehList = []
        for i in range(nVeh):
            vehList.append(bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf))

        for i in range(nVeh):
            for j in range(i, nVeh):
                if j>i:
                    dv = bez.Bezier(vehList[i].cpts -
                        vehList[j].cpts, tf=tf)
                    distVeh.append(dv.normSquare().elev(10))

        if plot:
            plt.figure(101)
            plt.title('Squared Distances', fontsize=28)
            for dist in distVeh:
                plt.plot(dist._tau, dist.curve.squeeze())
                plt.plot(np.linspace(0,1,dist.deg+1),
                         np.asarray(dist.cpts).squeeze(), '.')
                plt.plot([0, 1], [maxSep**2, maxSep**2], 'r--')
            plt.xlabel('Tau', fontsize=16)
            plt.ylabel('$Distance^2$', fontsize=20)

        distances = np.concatenate([np.asarray(i.cpts).squeeze()
            for i in distVeh])

        return (distances - maxSep**2).squeeze()
    else:
        return None

def _minVelConstraints(x, nVeh, dim, model, minVel, tf=1, plot=False):
    """
    Creates the minimum velocity constraints.

    Useful in systems such as aircraft who may not fall below a certain speed.
    """
    y = reshapeVector(x, nVeh, dim, model)

    vels = []

    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf)
        vel = pos.diff()
        vels.append(vel)

    velSqr = [curve.normSquare().elev(50) for curve in vels]

    velocities = np.concatenate([np.asarray(i.cpts).squeeze()
        for i in velSqr])

    return (velocities - minVel**2).squeeze()

def _maxVelConstraints(x, nVeh, dim, model, maxVel, tf=1, plot=False):
    """
    Creates the maximum velocity constraints.

    Useful for limiting the maximum speed of a vehicle.
    """
    y = reshapeVector(x, nVeh, dim, model)

    vels = []

    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf)
        vel = pos.diff()
        vels.append(vel)

    velSqr = [curve.normSquare().elev(50) for curve in vels]

    velocities = np.concatenate([np.asarray(i.cpts).squeeze()
        for i in velSqr])

    return (maxVel**2 - velocities).squeeze()

def _maxAngularRateConstraints(x, nVeh, dim, model, maxAngRate, tf=1,
                               plot=False):
    """
    Creates the maximum angular rate constraint.

    This is useful for a dubins car model that has a constraint on the maximum
    angular rate. The dimension is assumed to be 2.
    """
#    y = reshapeVector(x, nVeh, dim, startPoints, endPoints, 0, 0)
    y = reshapeVector(x, nVeh, dim, model)

    angularRates = []
    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf)
        angRate = bez.angularRateSqr(pos.elev(10))
        angularRates.append(angRate)

    angularRateCpts = np.concatenate(
            [i.cpts.squeeze() for i in angularRates])

    if plot:
        plt.figure()
        plt.title('Approximate Squared Angular Rates')
        for veh in angularRates:
            plt.plot(np.linspace(0, 1, veh.deg+1),
                     np.asarray(veh.cpts).squeeze(), '.-')
        plt.xlabel('Tau', fontsize=16)
        plt.ylabel('$Angular$ $Rate^2$', fontsize=20)

    return (maxAngRate**2 - angularRateCpts).squeeze()

def euclideanObjective(x, nVeh, dim, model):
    y = reshapeVector(x, nVeh, dim, model)
    cost = euclideanDistMatrix(y, nVeh, dim)
    return cost

@numba.jit(nopython=True)
def euclideanDistMatrix(y, nVeh, dim):
    summation = 0
    temp = np.zeros(3)
    length = y.shape[1]
    for veh in range(nVeh):
        for i in range(length-1):
            for j in range(dim):
                temp[j] = y[veh*dim+j, i+1] - y[veh*dim+j, i]

            summation += np.linalg.norm(temp)

    return summation

def _objectiveFunction(x, nVeh, dim, model, tf=1, minElem='accel', plot=False):
#    y = reshapeVector(x, nVeh, dim, startPoints, endPoints, 0, 0)
    y = reshapeVector(x, nVeh, dim, model)
    positions = []
    vels = []
    accels = []
    jerks = []

    if minElem.lower() == 'euclidean':
        return euclideanObjective(x, nVeh, dim, model)

    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :], tf=tf)
        vel = pos.diff()
        accel = vel.diff()
        if minElem == 'jerk':
            jerk = accel.diff()
            jerks.append(jerk)

        positions.append(pos)
        vels.append(vel)
        accels.append(accel)

    curves = {'pos':positions, 'vel':vels, 'accel':accels, 'jerk':jerks}
    summation = 0
    for curve in curves[minElem]:
        temp = curve.normSquare()
        summation = summation + temp.cpts.sum()*100

    if plot:
#        plt.figure(100)
#        plt.title('Velocity, Acceleration, and Jerk of Vehicles', fontsize=24)
        for i in range(len(vels)):
            plt.figure(100)
            vel2 = vels[i].normSquare()
            plt.plot(np.arange(0,1.01,0.01), vel2.curve.squeeze(), '-')

            plt.figure(200)
            accel2 = accels[i].normSquare()
            plt.plot(np.arange(0,1.01,0.01), accel2.curve.squeeze(), '--')

            if minElem == 'jerk':
                plt.figure(300)
                jerk2 = jerks[i].normSquare()
                plt.plot(np.arange(0,1.01,0.01), jerk2.curve.squeeze(), '-.')

#        plt.xlim((0,1))
##        plt.ylim(-0.1, 0.5)
#        plt.xlabel('Tau', fontsize=16)
#        plt.ylabel('Velocity, Acceleration, Jerk', fontsize=16)
#        leg = []
#        for i in range(len(vels)):
#            leg.append('vel{}'.format(i))
#            leg.append('accel{}'.format(i))
#            leg.append('jerk{}'.format(i))
#        plt.legend(leg)
        plt.figure(100)
        plt.legend(['vel{}'.format(i) for i in range(len(vels))])
        plt.xlabel('Tau', fontsize=20)
        plt.ylabel('$Velocity^2$ (No Units)', fontsize=20)
        plt.title('Vehicle Velocities Squared', fontsize=28)

        plt.figure(200)
        plt.legend(['accel{}'.format(i) for i in range(len(accels))])
        plt.xlabel('Tau', fontsize=20)
        plt.ylabel('$Acceleration^2$ (No Units)', fontsize=20)
        plt.title('Vehicle Accelerations Squared', fontsize=28)

        plt.figure(300)
        plt.legend(['jerk{}'.format(i) for i in range(len(jerks))])
        plt.xlabel('Tau', fontsize=20)
        plt.ylabel('$Jerk^2$ (No Units)', fontsize=20)
        plt.title('Vehicle Jerks Squared', fontsize=28)

    return summation


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
        dictionary: initPoints, finalPoints, initVels, finalVels, initAngs,
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

    if modelType == 'dubins':
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
        initVels = np.array(model['initVels'])
        finalVels = np.array(model['finalVels'])
        initAngs = np.array(model['initAngs'])
        finalAngs = np.array(model['finalAngs'])
        tf = model['tf']

        initMag = initVels*tf/degree
        finalMag = finalVels*tf/degree

#        print('InitPoints: {}'.format(initPoints))
#        print('InitAngs: {}'.format(initAngs))
#        print('InitMag: {}'.format(initMag))
#        print('y[::2, 1]: {}'.format(y[::2, 1]))

        y[:, 2:-2] = x
        y[::2, 0] = initPoints[:, 0]    # init X
        y[1::2, 0] = initPoints[:, 1]   # init Y
        y[::2, -1] = finalPoints[:, 0]  # final X
        y[1::2, -1] = finalPoints[:, 1] # final Y
        y[::2, 1] = initPoints[:, 0] + initMag*np.cos(initAngs)      # X
        y[1::2, 1] = initPoints[:, 1] + initMag*np.sin(initAngs)     # Y
        y[::2, -2] = finalPoints[:, 0] - finalMag*np.cos(finalAngs)  # X
        y[1::2, -2] = finalPoints[:, 1] - finalMag*np.sin(finalAngs) # Y

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

    else:
        msg = '{} is not a valid model type.'.format(modelType)
        raise ValueError(msg)

    return y.astype(float)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:17:47 2018

@author: ckielasjensen
"""

import numpy as np
import matplotlib.pyplot as plt

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
                 minElem, minVel = 0, maxVel = 1e12):
        self.nVeh = nVeh
        self.dim = dim
        self.maxSep = maxSep
        self.deg = deg
        self.startPoints = startPoints
        self.endPoints = endPoints
        self.minElem = minElem
        self.minVel = minVel
        self.maxVel = maxVel
#        self.xGuess = self.generateGuess()
        
        self.separationConstraints = lambda x: _separationConstraints(x,
                                              self.nVeh,
                                              self.dim,
                                              self.startPoints,
                                              self.endPoints,
                                              self.maxSep)
        
        self.minVelConstraints = lambda x: _minVelConstraints(x, 
                                              self.nVeh,
                                              self.dim,
                                              self.startPoints,
                                              self.endPoints,
                                              self.minVel)
        
        self.maxVelConstraints = lambda x: _maxVelConstraints(x,
                                              self.nVeh,
                                              self.dim,
                                              self.startPoints,
                                              self.endPoints,
                                              self.maxVel)
        
        self.angularRateConstraints = lambda x: _maxAngularRateConstraints(x,
                                              self.nVeh,
                                              self.dim,
                                              self.startPoints,
                                              self.endPoints,
                                              self.maxVel)
        
        self.objectiveFunction = lambda x: _objectiveFunction(x,
                                              self.nVeh,
                                              self.dim,
                                              self.startPoints,
                                              self.endPoints,
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
                              self.startPoints,
                              self.endPoints,
                              self.maxSep,
                              True)
        
        _objectiveFunction(x,
                          self.nVeh,
                          self.dim,
                          self.startPoints,
                          self.endPoints,
                          True,
                          self.minElem)

def _separationConstraints(x, nVeh, dim, startPoints, endPoints, maxSep, 
                           plot=False):
    """
    Calculate the separation between vehicles.
    
    NOTE: This only works for 2 dimensions at the moment.
    """
    if nVeh > 1:
        y = reshapeVector(x, nVeh, dim, startPoints, endPoints, 0, 0)
        
        distVeh = []
        vehList = []
        for i in range(nVeh):
            vehList.append(bez.Bezier(y[i*dim:i*dim+dim, :]))
        
        for i in range(nVeh):
            for j in range(i, nVeh):
                if j>i:
                    dv = bez.Bezier(vehList[i].ctrlPts -
                        vehList[j].ctrlPts)
                    distVeh.append(dv.normSquare().elev(10))
                    
        if plot:
            plt.figure(101)
            plt.title('Squared Distances', fontsize=28)
            for dist in distVeh:
                plt.plot(dist._tau, dist.curve.squeeze())
                plt.plot(np.linspace(0,1,dist.deg+1),
                         np.asarray(dist.ctrlPts).squeeze(), '.')
                plt.plot([0, 1], [maxSep**2, maxSep**2], 'r--')
            plt.xlabel('Tau', fontsize=16)
            plt.ylabel('$Distance^2$', fontsize=20)
            
        distances = np.concatenate([np.asarray(i.ctrlPts).squeeze()
            for i in distVeh])
        return (distances - maxSep**2).squeeze()
    else:
        return None
    
def _minVelConstraints(x, nVeh, dim, startPoints, endPoints, minVel,
                       plot=False):
    """
    Creates the minimum velocity constraints.
    
    Useful in systems such as aircraft who may not fall below a certain speed.
    """
    y = reshapeVector(x, nVeh, dim, startPoints, endPoints, 0, 0)
    
    positions = []
    vels = []
    
    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :])
        vel = pos.diff()
        
        positions.append(pos)
        vels.append(vel)
        
    velSqr = [curve.normSquare().elev(10) for curve in vels]
    
    velocities = np.concatenate([i.ctrlPts.squeeze() for i in velSqr])
    return (velocities - minVel**2).squeeze()

def _maxVelConstraints(x, nVeh, dim, startPoints, endPoints, maxVel,
                       plot=False):
    """
    Creates the maximum velocity constraints.
    
    Useful for limiting the maximum speed of a vehicle.
    """
    y = reshapeVector(x, nVeh, dim, startPoints, endPoints, 0, 0)
    
    positions = []
    vels = []
    
    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :])
        vel = pos.diff()
        
        positions.append(pos)
        vels.append(vel)
        
    velSqr = [curve.normSquare().elev(10) for curve in vels]
    
    velocities = np.concatenate([i.ctrlPts.squeeze() for i in velSqr])
    return (maxVel**2 - velocities).squeeze()

def _maxAngularRateConstraints(x, nVeh, dim, startPoints, endPoints, maxVel,
                       plot=False):
    """
    Creates the maximum angular rate constraint.
    
    This is useful for a dubins car model that has a constraint on the maximum
    angular rate.
    """
    pass
    
def _objectiveFunction(x, nVeh, dim, startPoints, endPoints,
                      plot=False, minElem='accel'):
    y = reshapeVector(x, nVeh, dim, startPoints, endPoints, 0, 0)
    positions = []
    vels = []
    accels = []
    jerks = []
    
    for i in range(nVeh):
        pos = bez.Bezier(y[i*dim:i*dim+dim, :])
        vel = pos.diff()
        accel = vel.diff()
        jerk = accel.diff()
        
        positions.append(pos)
        vels.append(vel)
        accels.append(accel)
        jerks.append(jerk)
        
    curves = {'pos':positions, 'vel':vels, 'accel':accels, 'jerk':jerks}
    summation = 0
    for curve in curves[minElem]:
        temp = curve.normSquare()
        summation = summation + temp.ctrlPts.sum()
    
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
        plt.ylabel('Velocity (No Units)', fontsize=20)
        plt.title('Vehicle Velocities', fontsize=28)
        
        plt.figure(200)
        plt.legend(['accel{}'.format(i) for i in range(len(accels))])
        plt.xlabel('Tau', fontsize=20)
        plt.ylabel('Acceleration (No Units)', fontsize=20)
        plt.title('Vehicle Accelerations', fontsize=28)
        
        plt.figure(300)
        plt.legend(['jerk{}'.format(i) for i in range(len(jerks))])
        plt.xlabel('Tau', fontsize=20)
        plt.ylabel('Jerk (No Units)', fontsize=20)
        plt.title('Vehicle Jerks', fontsize=28)
        
    return summation

def vector2matrix(x, nVeh, dim, startPoints, endPoints):
    """
    Converts the input vector x into a matrix that includes the start and end
    points of each vehicle in each dimension.
    
    The shape of the matrix is as follows:
        [[v1x1, v1x2, ..., v1xDegree+1], 
         [v1y1, v1y2, ..., v1yDegree+1],
         [v1z1, v1z2, ..., v1zDegree+1],
         [v1dim1, v1dim2, ..., v1dimDegree+1],
         [v2x1, v2x2, ..., v2xDegree+1],
         ...
         [vnVehx1, vnVehx2, ..., vnVehxDegree+1]]
    """
    x = np.array(x)
    numRows = int(nVeh*dim)
    # Since start and end points aren't included yet, we +1 instead of -1
    degree = int(x.size/numRows) + 1
    y = np.zeros((numRows, degree+1))
    
    x = x.reshape((numRows, degree-1))
    y[:,0] = startPoints.reshape(numRows)
    y[:,1:-1] = x
    y[:,-1] = endPoints.reshape(numRows)
    
    return y

def reshapeVector(x, nVeh, dim, startPoints, endPoints, initAngle, finalAngle):
    """
    Converts the input vector x into a matrix that includes the start and end
    points of each vehicle in each dimension.
    
    INPUTS:
        x - 
        
        nVeh -
        
        dim -
        
        startPoints -
        
        endPoints -
        
        initAngle - Either float or list of floats. If it is a list of floats,
        length must be = nVeh as the list corresponds to the initial angle of 
        each vehicle. If it is a single value, all angles are assumed to be the
        same for every vehicle.
        
        finalAngle - Same as initAngle but the final angle.
    
    For the angles, the following convension is in effect:
        * The reference line is the y axis
        * Positive angles are counter clockwise
        * The angle is calculated by looking at the angle between the forward
          facing vector of the robot and the y axis.
    
    The input vector is of the following form:
        [initAngle0, finalAngle0, X01, X02, ..., X0DEG, Y01, Y02, ..., Y0DEG,
         initAngle1, finalAngle1, X11, X12, ..., X1DEG, Y11, Y12, ..., Y1DEG,
         ...
         initAngleNVEH, finalAngleNVEH, ... ]
    
    The shape of the matrix is as follows:
        [[v1x1, v1x2, ..., v1xDegree+1], 
         [v1y1, v1y2, ..., v1yDegree+1],
         [v1z1, v1z2, ..., v1zDegree+1],
         [v1dim1, v1dim2, ..., v1dimDegree+1],
         [v2x1, v2x2, ..., v2xDegree+1],
         ...
         [vnVehx1, vnVehx2, ..., vnVehxDegree+1]]
    """
    x = np.array(x)
    numCols = int(x.size/nVeh)
    degree = (numCols-2)/2 + 3
    
    try:
        initAngle[0]
    except TypeError:
        initAngle = [initAngle]*nVeh
    try:
        finalAngle[0]
    except TypeError:
        finalAngle = [finalAngle]*nVeh
    
    init = np.array([])
    final = np.array([])
    for i in range(nVeh):
        d0 = x[i*numCols]
        df = x[i*numCols+1]
        init = np.append(init, startPoints[i,0] + d0*np.sin(initAngle[i]))  # X
        init = np.append(init, startPoints[i,1] + d0*np.cos(initAngle[i]))  # Y
        final = np.append(final, endPoints[i,0] - df*np.sin(finalAngle[i])) # X
        final = np.append(final, endPoints[i,1] - df*np.cos(finalAngle[i])) # Y
    
    y = np.zeros((int(nVeh*dim), int(degree+1)))
    
    x = x.reshape((nVeh, numCols))[:,2:]
    row, col = x.shape
    y[:,0] = startPoints.reshape(nVeh*dim)
    y[:,1] = init
    y[:,2:-2] = x.reshape((int(row*dim), int(col/dim)))
    y[:,-2] = final
    y[:,-1] = endPoints.reshape(nVeh*dim)
    
    return y


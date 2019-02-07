#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:51:52 2018

@author: ckielasjensen
"""

import scipy.optimize as sop
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

import optimization as opt
import bezier as bez

DIM = 2
DEG = 5
NUM_VEH = 2
MAX_SEP = 0.9
MIN_ELEM = 'accel'
MIN_VEL = 5
SEED = 5

# TODO:
#   * Allow 3D point generation.

def generateStructuredPoints(numVeh, dim=2, seed=None):
    """
    Generates start and end points for an arbitrary number of vehicles in an
    arbitrary dimension. For now it only accepts dim=2. 3D dimensions will be
    built into this later.

    This function uses np.random.shuffle to shuffle the structured points.

    INPUTS:
        * numVeh - Number of vehicles (must be 1 or larger).
        * dim - Dimension in which the vehicles are traveling (1, 2, or 3 would
          be the realistic values but 4 or more is also acceptible).
        * seed - Seed for np.random.seed for deterministic behavior.

    RETURNS:
        2 element tuple of (startPoints, endPoints) where startPoints and
        endPoints are organized as such:
            [[veh_0_X, veh_0_Y, veh_0_Z, veh_0_Ndim],
             [veh_1_X, veh_1_Y, veh_1_Z, veh_1_Ndim]]
    """
    startPoints = np.ones((numVeh, dim))*0
    endPoints = np.ones((numVeh, dim))*5

    startPoints[:,0] = np.arange(0, numVeh)
    endPoints[:,0] = np.arange(0, numVeh)

    np.random.seed(seed)
    np.random.shuffle(startPoints)
    np.random.shuffle(endPoints)

    return (startPoints, endPoints)


def iterCount(fn):
    """
    Decorator function that counts the number of times a function is called.
    Access this count through {functionName}.count
    """
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        return fn(*args, **kwargs)
    wrapper.count = 0
    return wrapper

@iterCount
def minCB(x):
    """
    Minimizer callback to display the progress of the minimizer.
    """
    if minCB.count == 1:
        print('{0:4s}\t{1:9s}\t{2:9s}\t{3:9s}\t{4:9s}'.format(
                'Iter', ' X1', ' X2', ' X3', ' X4'))
    print('{0:4d}\t{1: 3.6f}\t{2: 3.6f}\t{3: 3.6f}\t{4: 3.6f}'.format(
            minCB.count, x[0], x[1], x[2], x[3]))

def animateTrajectory(trajectories):
    global ani

    curveLen = len(trajectories[0].curve[0])
    fig, ax = plt.subplots()
    [ax.plot(traj.curve[0], traj.curve[1], '-', lw=3) for traj in trajectories]
    lines = [ax.plot([], [], 'o', markersize=20)[0] for traj in trajectories]

    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    def animate(frame):
        for i, line in enumerate(lines):
            traj = trajectories[i]
            try:
                line.set_data(traj.curve[0][frame],
                              traj.curve[1][frame])
            except IndexError:
                line.set_data(traj.curve[0][curveLen-frame-1],
                              traj.curve[1][curveLen-frame-1])
        return lines

    plt.axis('off')
    ani = animation.FuncAnimation(fig,
                                  animate,
                                  len(trajectories[0].curve[0])*2,
                                  init_func=init,
                                  interval=50,
                                  blit=True,
                                  repeat=True)

    plt.show()

if __name__ == "__main__":
    plt.close('all')

    ###########################################################################
    ### Prepare data for optimization
    ###########################################################################
    startPoints, endPoints = generateStructuredPoints(NUM_VEH, seed=SEED)
    bezOpt = opt.BezOptimization(NUM_VEH, DIM, MAX_SEP, DEG, startPoints,
                                 endPoints, MIN_ELEM, MIN_VEL)
    xGuess = bezOpt.generateGuess(random=True, seed=SEED)
#    xGuess = [0]*6*NUM_VEH*np.random.random(6*NUM_VEH)
    ineqCons = [{'type': 'ineq', 'fun': bezOpt.separationConstraints}]  # ,
                #{'type': 'ineq', 'fun': bezOpt.maxVelConstraints},
                #{'type': 'ineq', 'fun': bezOpt.maxAngularRateConstraints},
                #{'type': 'ineq', 'fun': bezOpt.minVelConstraints}]

    bounds = sop.Bounds([-2000]*(DEG-1)*NUM_VEH*DIM,
                        [2000]*(DEG-1)*NUM_VEH*DIM)

    ###########################################################################
    ### Optimize and time
    ###########################################################################
    startTime = time.time()
    results = sop.minimize(
            bezOpt.objectiveFunction,
            x0=xGuess,
            method='SLSQP',
            constraints=ineqCons,
            options={'ftol':1e-15, 'disp':True},
            bounds=bounds,
            callback=minCB)
    endTime = time.time()

    print('---')
    print(results)
    print('---')
    cpts = opt.reshapeVector(results.x, NUM_VEH, DIM,
                             startPoints, endPoints, 0, 0)
    print(cpts[0])
    print(cpts[1])
    print('---')
    print('Guess:\n{}\n\nActual:\n{}'.format(
            opt.reshapeVector(xGuess, NUM_VEH, DIM,
                              startPoints, endPoints, 0, 0),
            opt.reshapeVector(results.x, NUM_VEH, DIM,
                              startPoints, endPoints, 0, 0)))
    print('---')
    print('Optimization Time: {}s'.format(endTime-startTime))

    ###########################################################################
    ### Plot Results
    ###########################################################################
    plt.figure(1)
    plt.plot([startPoints[:,0], endPoints[:,0]],
             [startPoints[:,1], endPoints[:,1]], '.-')
    plt.title('Start and End Points of Vehicles', fontsize=28)
    plt.xlabel('X Position', fontsize=20)
    plt.ylabel('Y Position', fontsize=20)

    plt.figure(2)
    curves = []
    for i in range(NUM_VEH):
        curves.append(bez.Bezier(cpts[i*DIM:i*DIM+DIM]))
    for curve in curves:
        plt.plot(curve.curve[0], curve.curve[1], '-',
                 curve.cpts[0], curve.cpts[1], '.--')
    plt.title('Vehicle Trajectories', fontsize=28)
    plt.xlabel('X Position', fontsize=20)
    plt.ylabel('Y Position', fontsize=20)

    bezOpt.plot(results.x)

    animateTrajectory(curves)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 08:31:14 2019

@author: ckielasjensen
"""

import scipy.optimize as sop
import numpy as np
from multiprocessing import Pool
# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import collections
import os
import pandas as pd

import optimization as opt
import bezier as bez

DIM = 2
DEG = 10                    # Must be >= 4
MIN_ELEM = 'euclidean'
SEED = None                 # Set to None for true random
OUTER_CIR_R = 250           # (m)
INNER_CIR_R = 50            # (m)
CIRCLE_CENTER = (0, 0)      # (x, y) (m)
FLIGHT_AREA_MIN = -2000     # (m)
FLIGHT_AREA_MAX = 2000      # (m)
MIN_SPEED = 18              # (m/s)
MAX_SPEED = 32              # (m/s)
MAX_ANG_RATE = 0.2          # (rad/s)
CONS_TIME = 150             # Conservative time added to the minimum time (s)
OUTER2INNER_TIME = 10       # Time to fly from outer target circle to inner (s)
NUM_ITER = 50000            # Number of times to iterate throgh random examples
# NUM_ITER must be divisible by 100 (see main for why)

FlightParams = collections.namedtuple('FlightParams', ['initPoints',
                                                       'finalPoints',
                                                       'initSpeeds',
                                                       'finalSpeeds',
                                                       'initAngs',
                                                       'finalAngs',
                                                       'tf'])


def generateRandomInitialValues(seed=None):
    """
    Generates random initial values for a single vehicle flying to a target
    circle within a defined flight area.

    RETURNS:
        FlightParams namedtuple object with the desired initial parameters for
        the single vehicle flight.
    """
    np.random.seed(seed)

    # Generate a random initial point within the flight area but outside of the
    # target circle
    while True:
        initialPoint = np.random.randint(FLIGHT_AREA_MIN,
                                         FLIGHT_AREA_MAX,
                                         size=(1, 2))
        if np.linalg.norm(initialPoint-CIRCLE_CENTER) > OUTER_CIR_R:
            break

    # Place the final point at a random location on the outer target circle
    randAng = np.pi*np.random.rand()
    finalPoint = np.array([np.cos(randAng), np.sin(randAng)], ndmin=2)
    finalPoint = OUTER_CIR_R*finalPoint + CIRCLE_CENTER

    # Determine the initial and final angles
    initialAngle = np.pi*np.random.rand()
    finalAngle = randAng-np.pi

    # Pick a conservative estimate for the final time
    avgSpeed = np.average((MIN_SPEED, MAX_SPEED))
    tf = np.linalg.norm(finalPoint-initialPoint)/avgSpeed + CONS_TIME

    # Determine initial and final speeds
    initialSpeed = np.random.randint(MIN_SPEED, MAX_SPEED)
    finalSpeeds = (OUTER_CIR_R-INNER_CIR_R) / OUTER2INNER_TIME

    return FlightParams(initPoints=initialPoint,
                        finalPoints=finalPoint,
                        initSpeeds=initialSpeed,
                        finalSpeeds=finalSpeeds,
                        initAngs=initialAngle,
                        finalAngs=finalAngle,
                        tf=tf)


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
    try:
        print('{0:4d}\t{1: 3.6f}\t{2: 3.6f}\t{3: 3.6f}\t{4: 3.6f}'.format(
                minCB.count, x[0], x[1], x[2], x[3]))
    except IndexError:
        print(x)


def animateTrajectory(trajectories):
    """
    Animates the trajectories of the vehicles.
    """
    global ani

    if isinstance(trajectories, bez.Bezier):
        trajectories = [trajectories]

    curveLen = len(trajectories[0].curve[0])
    fig, ax = plt.subplots()
    [ax.plot(traj.curve[0], traj.curve[1], '-', lw=3) for traj in trajectories]
    lines = [ax.plot([], [], 'o', markersize=20)[0] for traj in trajectories]

    def init():
        for line in lines:
            line.set_data([], [])
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


def singleVehicleGuess(initParams, deg, addNoise=True, seed=None):
    """
    Generates a guess for a single vehicle. This also assumes that the initial
    and final points, initial and final angles, and initial and final speeds
    are all fixed values.

    INPUTS:
        initParams - FlightParams object containing the flight parameters for
        a single vehicle flight within a defined area flying towards a defined
        target circle.

        deg - Degree of the Bezier curve

        addNoise - If True, add random noise to the initial guess.

        seed - Seed for the random noise. Set to None for nondeterministic
        random values.

    RETURNS:
        Guess vector of the shape [X2, ..., Xn-2, Y2, ..., Yn-2]
    """

    initPoints = np.array(initParams.initPoints)
    finalPoints = np.array(initParams.finalPoints)
    initSpeeds = np.array(initParams.initSpeeds)
    finalSpeeds = np.array(initParams.finalSpeeds)
    initAngs = np.array(initParams.initAngs)
    finalAngs = np.array(initParams.finalAngs)
    Tf = initParams.tf

    initMag = initSpeeds*Tf/deg
    finalMag = finalSpeeds*Tf/deg

    xInit = initPoints[0, 0] + initMag*np.cos(initAngs)      # X
    yInit = initPoints[0, 1] + initMag*np.sin(initAngs)      # Y
    xFinal = finalPoints[0, 0] - finalMag*np.cos(finalAngs)  # X
    yFinal = finalPoints[0, 1] - finalMag*np.sin(finalAngs)  # Y

    xVals = np.linspace(xInit, xFinal, deg-3)
    yVals = np.linspace(yInit, yFinal, deg-3)

    if addNoise:
        np.random.seed(seed)
        xVals += np.random.random(deg-3)
        yVals += np.random.random(deg-3)

    return np.concatenate([xVals, yVals])


def runIteration(_):
    """
    Runs a single iteration of a random flight test.

    RETURNS:
        Results object from the minimize function

        Time object representing the time it took to run the optimizer
    """
    plt.close('all')

    ###########################################################################
    # Prepare data for optimization
    ###########################################################################
    initParams = generateRandomInitialValues(SEED)
    bezOpt = opt.BezOptimization(numVeh=1,
                                 dimension=DIM,
                                 degree=DEG,
                                 minimizeGoal=MIN_ELEM,
                                 minSpeed=MIN_SPEED,
                                 maxSpeed=MAX_SPEED,
                                 maxAngRate=MAX_ANG_RATE,
                                 modelType='dubins',
                                 initPoints=initParams.initPoints,
                                 finalPoints=initParams.finalPoints,
                                 initSpeeds=initParams.initSpeeds,
                                 finalSpeeds=initParams.finalSpeeds,
                                 initAngs=initParams.initAngs,
                                 finalAngs=initParams.finalAngs,
                                 tf=initParams.tf)

    xGuess = singleVehicleGuess(initParams, DEG, SEED)

    ineqCons = [{'type': 'ineq', 'fun': bezOpt.minSpeedConstraints},
                {'type': 'ineq', 'fun': bezOpt.maxSpeedConstraints},
                {'type': 'ineq', 'fun': bezOpt.maxAngularRateConstraints}]

    ###########################################################################
    # Optimize and time
    ###########################################################################
    startTime = time.time()
    results = sop.minimize(
            bezOpt.objectiveFunction,
            x0=xGuess,
            constraints=ineqCons,
            options={  # 'disp': True,
                     'ftol': 1e-9,
#                     'eps': 1e-21,
                     'maxiter': 250
                     },
#            callback=minCB,
            method='SLSQP')
    endTime = time.time()
    minCB.count = 0

    return results, bezOpt.model, endTime-startTime


def showSingleVehicleTest(trajectory):
    """
    Plots out the results from a single vehicle test run.

    INPUTS:
        trajectory - Bezier object of the flight trajectory
    """

    # Trajectory plot
    fig1, ax1 = plt.subplots()
    outerCircle = plt.Circle(CIRCLE_CENTER, radius=OUTER_CIR_R,
                             color='b', fill=False)
    innerCircle = plt.Circle(CIRCLE_CENTER, radius=INNER_CIR_R,
                             color='r', fill=False)
    ax1.add_artist(outerCircle)
    ax1.add_artist(innerCircle)
    trajectory.plot(axisHandle=ax1)
    plt.title('Flight Trajectory')
    plt.xlabel('X Location (m)')
    plt.ylabel('Y Location (m)')
    plt.xlim([FLIGHT_AREA_MIN, FLIGHT_AREA_MAX])
    plt.ylim([FLIGHT_AREA_MIN, FLIGHT_AREA_MAX])

    plt.show()

    animateTrajectory(trajectory)


def pickTrajectory(resultsList, choice=None):
    """
    """
    if choice is None:
        successful = [i for i in resultsList if i[0].success]
        idx = np.random.randint(0, len(successful))
        result = successful[idx]
    else:
        result = resultsList[choice]

    x = result[0].x
    model = result[1]

    if 'initVels' in model:
        model['initSpeed'] = model['initVels']
    if 'finalVels' in model:
        model['finalSpeed'] = model['finalVels']

    y = opt.reshapeVector(x, 1, 2, model)
    trajectory = bez.Bezier(y, tf=model['tf'])

    return trajectory


def appendToDataFrame(resultsList, dataFrame):
    """
    """
    for result in resultsList:
        status = result[0].status
        x = result[0].x
        nfev = result[0].nfev
        time = result[2]

        data = result[1].copy()
        data['status'] = status
        data['x'] = str(x)
        data['nfev'] = nfev
        data['time'] = time
        data['initPoints'] = str(data['initPoints'][0])
        data['finalPoints'] = str(data['finalPoints'][0])

        temp = pd.DataFrame(data=data, index=[0])

        dataFrame = dataFrame.append(temp, ignore_index=True, sort=False)

    return dataFrame


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    dfCols = ['status', 'x', 'nfev', 'time', 'initPoints', 'finalPoints',
              'initSpeeds', 'finalSpeeds', 'initAngs', 'finalAngs',
              'tf']
    df = pd.DataFrame(columns=dfCols)

    print('Running...')

    # We batch process 100 iterations at a time to avoid using up too much
    # memory. This is why NUM_ITER must be divisible by 100
    for i in range(NUM_ITER//100):

        print('Batch number: {}'.format(i+1))

        p = Pool()
        startTime = time.time()
        resultsList = p.map(runIteration, [i for i in range(100)])
        p.close()

        df = appendToDataFrame(resultsList, df)
        fileName = 'IterateSingleVehicle_MP.csv'
        df.to_csv(fileName, mode='a', header=(not os.path.exists(fileName)))

        resultsList = None

    print('Elapsed Time: {}'.format(time.time() - startTime))

    showSingleVehicleTest(pickTrajectory(resultsList))

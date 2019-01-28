#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:17:17 2019

@author: ckielasjensen
"""

import scipy.optimize as sop
import numpy as np
# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import collections
import pickle
import uuid

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
NUM_ITER = 10            # Number of times to iterate throgh random examples

FlightParams = collections.namedtuple('FlightParams', ['initPt',
                                                       'finalPt',
                                                       'initSpeed',
                                                       'finalSpeed',
                                                       'initAng',
                                                       'finalAng',
                                                       'finalTime'])


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
    finalTime = np.linalg.norm(finalPoint-initialPoint)/avgSpeed + CONS_TIME

    # Determine initial and final speeds
    initialSpeed = np.random.randint(MIN_SPEED, MAX_SPEED)
    finalSpeed = (OUTER_CIR_R-INNER_CIR_R) / OUTER2INNER_TIME

    return FlightParams(initPt=initialPoint,
                        finalPt=finalPoint,
                        initSpeed=initialSpeed,
                        finalSpeed=finalSpeed,
                        initAng=initialAngle,
                        finalAng=finalAngle,
                        finalTime=finalTime)


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

    initPt = np.array(initParams.initPt)
    finalPt = np.array(initParams.finalPt)
    initSpeed = np.array(initParams.initSpeed)
    finalSpeed = np.array(initParams.finalSpeed)
    initAng = np.array(initParams.initAng)
    finalAng = np.array(initParams.finalAng)
    Tf = initParams.finalTime

    initMag = initSpeed*Tf/deg
    finalMag = finalSpeed*Tf/deg

    xInit = initPt[0, 0] + initMag*np.cos(initAng)      # X
    yInit = initPt[0, 1] + initMag*np.sin(initAng)      # Y
    xFinal = finalPt[0, 0] - finalMag*np.cos(finalAng)  # X
    yFinal = finalPt[0, 1] - finalMag*np.sin(finalAng)  # Y

    xVals = np.linspace(xInit, xFinal, deg-3)
    yVals = np.linspace(yInit, yFinal, deg-3)

    if addNoise:
        np.random.seed(seed)
        xVals += np.random.random(deg-3)
        yVals += np.random.random(deg-3)

    return np.concatenate([xVals, yVals])


def runIteration():
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
                                 initPoints=initParams.initPt,
                                 finalPoints=initParams.finalPt,
                                 initSpeeds=initParams.initSpeed,
                                 finalSpeeds=initParams.finalSpeed,
                                 initAngs=initParams.initAng,
                                 finalAngs=initParams.finalAng,
                                 tf=initParams.finalTime)

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
            options={'disp': True,
                     'ftol': 1e-9,
#                     'eps': 1e-21,
                     'maxiter': 250
                     },
#            callback=minCB,
            method='SLSQP')
    endTime = time.time()
    minCB.count = 0

    return results, bezOpt, endTime-startTime


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
        model['initSpeeds'] = model['initVels']
    if 'finalVels' in model:
        model['finalSpeeds'] = model['finalVels']

    y = opt.reshapeVector(x, 1, 2, model)
    trajectory = bez.Bezier(y, tf=model['tf'])

    return trajectory


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    resultsList = []

    try:
        for i in range(NUM_ITER):
            result, bezOpt, optTime = runIteration()
            resultsList.append((result, bezOpt.model, optTime))
            if not i % 10:
                print('\n===> Iteration Number {}\n'.format(i))
    finally:
        fname = 'singleVehicleTests'+str(uuid.uuid4())+'.pickle'
        with open(fname, 'wb') as f:
            pickle.dump(resultsList, f)

    showSingleVehicleTest(pickTrajectory(resultsList))

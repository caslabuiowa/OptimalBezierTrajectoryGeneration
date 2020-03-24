#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:10:46 2019

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.optimize as sop
import time

import bezier as bez
from optimization import BezOptimization


def _temporalSeparationConstraints(y, nVeh, dim, maxSep, degElev=10):
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

        for i in range(nVeh):
            for j in range(i+1, nVeh):
                dv = vehList[i] - vehList[j]
                distVeh.append(dv.normSquare().elev(degElev))

        distances = np.concatenate([i.cpts.squeeze() for i in distVeh])

        return (distances - maxSep**2).squeeze()

    else:
        return None


def animateTrajectory(trajectories):
    """Animates the trajectories

    """
    global ani

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
                                  interval=10,
                                  blit=True,
                                  repeat=True)

    plt.show()


if __name__ == '__main__':
    plt.close('all')

    numVeh = 2
    dim = 2
    deg = 10
    maxSep = 1

    bezopt = BezOptimization(numVeh=numVeh,
                             dimension=dim,
                             degree=deg,
                             minimizeGoal='TimeOpt',
                             maxSep=maxSep,
                             maxSpeed=5,
                             maxAngRate=1,
                             initPoints=[
                                     (0, 5),
                                     (3, 0)],
                             finalPoints=[
                                     (8, 4),
                                     (7, 10)],
                             initSpeeds=[1]*numVeh,
                             finalSpeeds=[1]*numVeh,
                             initAngs=[0, np.pi/2],
                             finalAngs=[0, np.pi/2],
                             pointObstacles=[[3, 2], [6, 7]]
                             )

    curves = []
    for elev in [0, 30, 100]:

        sepCons = lambda x: _temporalSeparationConstraints(
                bezopt.reshapeVector(x), numVeh, dim, maxSep, elev)
        xGuess = bezopt.generateGuess(std=0)
        ineqCons = [{'type': 'ineq', 'fun': sepCons},
                    {'type': 'ineq', 'fun': bezopt.maxSpeedConstraints},
                    {'type': 'ineq', 'fun': bezopt.maxAngularRateConstraints},
                    {'type': 'ineq', 'fun': lambda x: x[-1]}]

        # Precompute the cache for degree elevation
        _ = bez.Bezier(bezopt.reshapeVector(xGuess))
        _.elev(elev)
        _ = _*_

        startTime = time.time()
        print('starting')
        results = sop.minimize(
                    bezopt.objectiveFunction,
                    x0=xGuess,
                    method='SLSQP',
                    constraints=ineqCons,
                    options={'maxiter': 250,
                             'disp': True,
                             'iprint': 2}
                    )
        endTime = time.time()

        print('---')
        print('Computation Time: {}'.format(endTime - startTime))
        print('---')

        cpts = bezopt.reshapeVector(results.x)
        for i in range(numVeh):
            curves.append(bez.Bezier(cpts[i*dim:(i+1)*dim]))
        curves.append(bez.Bezier(cpts))

    ###########################################################################
    # Plot Results
    ###########################################################################

    numVeh = bezopt.model['numVeh']
    dim = bezopt.model['dim']
    maxSep = bezopt.model['maxSep']

    fig, ax = plt.subplots()
#    curves = []
#    for i in range(numVeh):
#        curves.append(bez.Bezier(cpts[i*dim:(i+1)*dim]))
    for curve in curves:
        plt.plot(curve.curve[0], curve.curve[1], '-')
#        plt.plot(curve.cpts[0], curve.cpts[1], '.--')

    obstacle1 = plt.Circle(bezopt.pointObstacles[0],
                           radius=maxSep,
                           edgecolor='Black',
                           facecolor='red')
    obstacle2 = plt.Circle(bezopt.pointObstacles[1],
                           radius=maxSep,
                           edgecolor='Black',
                           facecolor='green')
    ax.add_artist(obstacle1)
    ax.add_artist(obstacle2)
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.title('Vehicle Trajectory', fontsize=28)
    plt.xlabel('X Position', fontsize=20)
    plt.ylabel('Y Position', fontsize=20)

    animateTrajectory(curves)

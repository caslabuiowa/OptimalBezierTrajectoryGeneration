#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:11:19 2019

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.optimize as sop
import time

import bezier as bez
from optimization import BezOptimization

CAS_IMG = np.array([
        [  0., 255., 255., 255.,   0., 255., 255., 255.,   0., 255., 255.,
        255.,   0.],
       [  0., 255.,   0.,   0.,   0., 255.,   0., 255.,   0., 255.,   0.,
          0.,   0.],
       [  0., 255.,   0.,   0.,   0., 255.,   0., 255.,   0., 255., 255.,
        255.,   0.],
       [  0., 255.,   0.,   0.,   0., 255., 255., 255.,   0.,   0.,   0.,
        255.,   0.],
       [  0., 255.,   0.,   0.,   0., 255.,   0., 255.,   0.,   0.,   0.,
        255.,   0.],
       [  0., 255., 255., 255.,   0., 255.,   0., 255.,   0., 255., 255.,
        255.,   0.]])


def generatePointsFromImage(img):
    """
    """
    numVeh = img[img > 0].size
    rowLen = np.ceil(np.sqrt(numVeh))

    initPts = np.empty((numVeh, 3))
    finalPts = np.empty((numVeh, 3))

    for i in range(numVeh):
        x = i % rowLen
        y = np.floor(i/rowLen)
        initPts[i] = (x, y, 0)

    finalY, finalX = np.nonzero(np.flipud(img))

    finalY = finalY - finalY.min()
    finalX = finalX - finalX.min()

    initPts[:, 0] += np.round((finalX.max() - initPts[:, 0].max()) / 2)
    initPts[:, 1] += np.round((finalY.max() - initPts[:, 1].max()) / 2)

    finalPts[:, 0] = finalX
    finalPts[:, 1] = finalY
    finalPts[:, 2] = 10

    return (numVeh, initPts, finalPts)


def generate3DGuess(initPts, finalPts, deg):
    """
    """
    length = initPts.shape[0]
    lines = np.empty((length*3, deg-1))
    for i in range(length):
        x = np.linspace(initPts[i, 0], finalPts[i, 0], deg+1)[1:-1]
        y = np.linspace(initPts[i, 1], finalPts[i, 1], deg+1)[1:-1]
        z = np.linspace(initPts[i, 2], finalPts[i, 2], deg+1)[1:-1]
        lines[3*i, :] = x
        lines[3*i + 1, :] = y
        lines[3*i + 2, :] = z

    return lines.reshape((1, -1)).squeeze()


def animate3DTrajectory(trajectories):
    """Animates the trajectories

    """
    global ani

    curveLen = len(trajectories[0].curve[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    [ax.plot(traj.curve[0],
             traj.curve[1],
             traj.curve[2], '-', lw=3) for traj in trajectories]
    lines = [ax.plot([],
                     [],
                     [], 'o', markersize=20)[0] for traj in trajectories]

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    def animate(frame):
        for i, line in enumerate(lines):
            traj = trajectories[i]
            try:
                line.set_data(traj.curve[0][frame],
                              traj.curve[1][frame])
                line.set_3d_properties(traj.curve[2][frame])
            except IndexError:
                line.set_data(traj.curve[0][curveLen-frame-1],
                              traj.curve[1][curveLen-frame-1])
                line.set_3d_properties(traj.curve[2][curveLen-frame-1])
        return lines

    plt.axis('off')
    ani = animation.FuncAnimation(fig,
                                  animate,
                                  len(trajectories[0].curve[0])*2,
                                  init_func=init,
                                  interval=5,
                                  blit=True,
                                  repeat=True)

    plt.show()


if __name__ == '__main__':
#    plt.close('all')

    img = CAS_IMG
    numVeh, initPts, finalPts = generatePointsFromImage(img)

    bezopt = BezOptimization(numVeh=numVeh,
                             dimension=3,
                             degree=5,
                             minimizeGoal='Euclidean',
                             maxSep=0.9,
                             initPoints=initPts,
                             finalPoints=finalPts,
                             )

    xGuess = generate3DGuess(initPts, finalPts, bezopt.model['deg'])

    ineqCons = [{'type': 'ineq', 'fun': bezopt.temporalSeparationConstraints}]

    startTime = time.time()
    results = sop.minimize(
                bezopt.objectiveFunction,
                x0=xGuess,
                method='SLSQP',
                constraints=ineqCons,
                options={'maxiter': 100,
                         'disp': True,
                         'iprint': 2}
                )
    endTime = time.time()

    print('---')
    print('Computation Time: {}'.format(endTime - startTime))
    print('---')

    dim = bezopt.model['dim']
    cpts = bezopt.reshapeVector(results.x)

    curves = []
    for i in range(numVeh):
        curves.append(bez.Bezier(cpts[i*dim:(i+1)*dim]))

    ax = curves[0].plot(showCpts=False)
    for curve in curves[1:]:
        curve.plot(ax, showCpts=False)

    for curve in curves:
        plt.plot([curve.cpts[0, -1]],
                 [curve.cpts[1, -1]],
                 [curve.cpts[2, -1]],
                 'k.', markersize=50)

    animate3DTrajectory(curves)

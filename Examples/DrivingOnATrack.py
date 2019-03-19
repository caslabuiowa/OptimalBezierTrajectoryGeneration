#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:11:04 2019

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sop
import time

import bezier as bez
from optimization import BezOptimization

if __name__ == '__main__':
    track1 = bez.Bezier([[0, 0, 0, 3, 4, 5, 6, 7, 10, 10, 10],
                         [0, 3, 4, 5, 6, 6, 6, 6, 7, 8, 10]])

    track2 = bez.Bezier([[4,  4,  4,  7,  8,  9, 10, 11, 14, 14, 14],
                         [0, 3, 4, 4, 4, 5, 5, 5, 7, 8, 10]])
    tracks = [track1, track2]

    bezopt = BezOptimization(numVeh=1,
                             dimension=2,
                             degree=10,
                             minimizeGoal='TimeOpt',
                             maxSep=0.5,
                             maxSpeed=5,
                             maxAngRate=0.5,
                             initPoints=(2, 1),
                             finalPoints=(12, 9),
                             initSpeeds=1,
                             finalSpeeds=1,
                             initAngs=np.pi/2,
                             finalAngs=np.pi/2,
                             shapeObstacles=tracks
                             )

    xGuess = bezopt.generateGuess()
    xGuess[-1] = 10

    infs = [np.inf]*(bezopt.model['deg']+1-4)*bezopt.model['dim']
    infs.append(1e-3)
    bounds = sop.Bounds(np.array(infs), np.inf)

    ineqCons = [{'type': 'ineq', 'fun': bezopt.maxSpeedConstraints},
                {'type': 'ineq', 'fun': bezopt.maxAngularRateConstraints},
                {'type': 'ineq', 'fun': bezopt.spatialSeparationConstraints}]

    startTime = time.time()
    results = sop.minimize(
                bezopt.objectiveFunction,
                x0=xGuess,
                method='SLSQP',
                constraints=ineqCons,
                bounds=bounds,
                options={'maxiter': 250,
                         'disp': True,
                         'iprint': 2}
                )
    endTime = time.time()

    print('---')
    print('Computation Time: {}'.format(endTime - startTime))
    print('---')

    cpts = bezopt.reshapeVector(results.x)

    ###########################################################################
    # Plot Results
    ###########################################################################

    numVeh = bezopt.model['numVeh']
    dim = bezopt.model['dim']
    maxSep = bezopt.model['maxSep']

    fig, ax = plt.subplots()
    curves = []
    for i in range(numVeh):
        curves.append(bez.Bezier(cpts[i*dim:(i+1)*dim]))
    for curve in curves:
        plt.plot(curve.curve[0], curve.curve[1], '-',
                 curve.cpts[0], curve.cpts[1], '.--')

    track1.plot(ax, showCpts=False, color='black', linewidth=2)
    track2.plot(ax, showCpts=False, color='black', linewidth=2)

    plt.xlim([0, 15])
    plt.ylim([0, 10])
    plt.title('Vehicle Trajectory', fontsize=28)
    plt.xlabel('X Position', fontsize=20)
    plt.ylabel('Y Position', fontsize=20)

    plt.show()

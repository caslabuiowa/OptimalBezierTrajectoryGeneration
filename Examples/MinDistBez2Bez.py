#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:42:17 2019

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import time

import bezier as bez


def plotPoly(poly, ax, linecolor='b'):
    from scipy.spatial import ConvexHull

    pts = poly.copy()
    hull = ConvexHull(pts, qhull_options='QJ')

    ax.plot(pts.T[0], pts.T[1], pts.T[2], 'ko')

    for s in hull.simplices:
        s = np.append(s, s[0])
        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2],
                color=linecolor, linestyle='-')


if __name__ == '__main__':
    plt.close('all')
    plt.rcParams.update({
            'font.size': 20,
            'pdf.fonttype': 42,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'lines.linewidth': 3,
            'lines.markersize': 15
            })

    cpts1 = np.array([(0, 1, 2, 3, 4, 5),
                      (1, 2, 0, 0, 2, 1),
                      (0, 1, 2, 3, 4, 5)])

    cpts2 = np.array([(0, 1, 2, 3, 4, 5),
                      (3, 2, 0, 0, 2, 3),
                      (5, 4, 3, 2, 1, 0)])

    cpts3 = np.array([(0, 1, 2, 3, 4, 5),
                      (0, 1, 2, 3, 4, 5),
                      (0, 0, 0, 0, 0, 0)])

    cpts4 = np.array([(5, 4, 3, 2, 1, 0),
                      (0, 1, 2, 3, 4, 5),
                      (0, 0, 0, 0, 0, 0,)])

    cpts4[1, :] -= 1

    cpts5 = cpts1 - 3

    cpts6 = np.array([(0, 1, 2, 3, 4, 5),
                      (5, 0, 2, 5, 7, 5)])

    cpts7 = np.array([(0, 1, 3, 5, 7, 7, 8, 9, 9),
                      (0, 6, 9, 6, 8, 3, 7, 8, 3)])

    poly1 = np.array([(1, 3, 3),
                      (1, 3, 2),
                      (1, 4, 1),
                      (3, 3, 3),
                      (1, 5, 1)])

    poly2 = np.array([(1, 1, 3),
                      (1, 1, 2),
                      (1, 2, 1),
                      (4, 0, 2),
                      (1, 3, 1)])

    poly3 = np.array([(1, 1, 0),
                      (1, 3, 0),
                      (2, 5, 0),
                      (4, 4, 0)])

    c1 = bez.Bezier(cpts1)
    c2 = bez.Bezier(cpts2)
    c3 = bez.Bezier(cpts3)
    c4 = bez.Bezier(cpts4)
    c5 = bez.Bezier(cpts5)
    c6 = bez.Bezier(cpts6)
    c7 = bez.Bezier(cpts7)

    # ---
    # Bezier to Bezier
    # ---
    d31, t31, t1 = c3.minDist(c1)
    d32, t32, t2 = c3.minDist(c2)
    d34, t34, t4 = c3.minDist(c4)
    d35, t35, t5 = c3.minDist(c5)

    # Plot curves
    ax1 = c1.plot(showCpts=False, color='g')
    c2.plot(ax1, showCpts=False, color='g')
    c3.plot(ax1, showCpts=False, color='b')
    c4.plot(ax1, showCpts=False, color='r')
    c5.plot(ax1, showCpts=False, color='g')
    # Plot minimum distance lines
    plt.plot(np.array((c3(t31)[0], c1(t1)[0])).squeeze(),
             np.array((c3(t31)[1], c1(t1)[1])).squeeze(),
             np.array((c3(t31)[2], c1(t1)[2])).squeeze(), 'k.--')

    plt.plot(np.array((c3(t32)[0], c2(t2)[0])).squeeze(),
             np.array((c3(t32)[1], c2(t2)[1])).squeeze(),
             np.array((c3(t32)[2], c2(t2)[2])).squeeze(), 'k.--')

    plt.plot(np.array((c3(t34)[0], c4(t4)[0])).squeeze(),
             np.array((c3(t34)[1], c4(t4)[1])).squeeze(),
             np.array((c3(t34)[2], c4(t4)[2])).squeeze(), 'k.--')

    plt.plot(np.array((c3(t35)[0], c5(t5)[0])).squeeze(),
             np.array((c3(t35)[1], c5(t5)[1])).squeeze(),
             np.array((c3(t35)[2], c5(t5)[2])).squeeze(), 'k.--')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Minimum Distance Between Curves')

    plt.show()
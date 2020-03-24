#!/usr/bin/env python3
"""
Bezier Package Usage Examples

AUTHOR: Calvin Kielas-Jensen
ORGANIZATION: University of Iowa Cooperative Autonomous Systems Lab

This file is meant to provide simple usage examples for the Bezier library. The
original use case for the Bezier library is to generate trajectories for
autonomous vehicles in an efficient manner. However, additional use cases are
limited only by the imagination of the user. Some additional uses include
computer graphics and video game design.
"""

import numpy as np
import matplotlib.pyplot as plt

import bezier as bez


def plotPoly(poly, ax):
    from scipy.spatial import ConvexHull

    pts = poly.copy()
    hull = ConvexHull(pts, qhull_options='QJ')

    ax.plot(pts.T[0], pts.T[1], pts.T[2], 'ko')

    for s in hull.simplices:
        s = np.append(s, s[0])
        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], 'b-')


if __name__ == '__main__':
    plt.close('all')

    # Control Points for curves
    cpts1 = np.array([(0, 1, 2, 3, 4, 5),   # X
                      (1, 2, 0, 0, 2, 1),   # Y
                      (0, 1, 2, 3, 4, 5)])  # Z

    cpts2 = np.array([(0, 1, 2, 3, 4, 5),
                      (3, 2, 0, 0, 2, 3),
                      (5, 4, 3, 2, 1, 0)])

    cpts3 = np.array([(0, 1, 2, 3, 4, 5),
                      (0, 1, 2, 3, 4, 5),
                      (0, 0, 0, 0, 0, 0)])

    cpts4 = np.array([(5, 4, 3, 2, 1, 0),
                      (0, 1, 2, 3, 4, 5),
                      (0, 0, 0, 0, 0, 0,)])

    poly1 = np.array([(1, 1, 3),
                      (1, 1, 2),
                      (1, 2, 1),
                      (3, 1, 3),
                      (1, 3, 1)])

    poly2 = np.array([(1, 1, 3),
                      (1, 1, 2),
                      (1, 2, 1),
                      (3, -1, 3),
                      (1, 3, 1)])

    # Creating curves from control points
    c1 = bez.Bezier(cpts1)
    c2 = bez.Bezier(cpts2)
    c3 = bez.Bezier(cpts3)
    c4 = bez.Bezier(cpts4)

    # ---
    # Example 1 - minimum distance between curves
    # ---
    dist12, t1, t2 = c1.minDist(c2)
    print(f'The minimum distance between C1 and C2 is {dist12}')

    # Plot the curves and the distance
    # Note that when plotting, the control points of the curve will also be
    # plotted.
    ax1 = c1.plot()
    c2.plot(ax1)
    plt.plot(np.array((c1(t1)[0], c2(t2)[0])).squeeze(),
             np.array((c1(t1)[1], c2(t2)[1])).squeeze(),
             np.array((c1(t1)[2], c2(t2)[2])).squeeze(), 'r-')
    plt.title('Minimum distance between 2 Bezier curves')

    # ---
    # Example 2 - minimum distance between a curve and a polygon
    # ---
    dist1p1, t1p, pt1 = c1.minDist2Poly(poly1)
    print(f'The minimum distance between C1 and Poly1 is {dist1p1}')

    ax2 = c1.plot()
    plotPoly(poly1, ax2)
    plt.plot(np.array((c1(t1p)[0], pt1[0])),
             np.array((c1(t1p)[1], pt1[1])),
             np.array((c1(t1p)[2], pt1[2])))
    plt.title('Minimum distance between a Bezier curve and a polygon')

    # ---
    # Example 3 - collision detection between two curves
    # ---
    collCheck34 = c3.collCheck(c4)
    if collCheck34 == 1:
        print('No collision detected between C3 and C4')
    else:
        print('Collision detected between C3 and C4')

    ax3 = c3.plot()
    c4.plot(ax3)
    plt.title('Collision between two Bezier curves')

    # ---
    # Example 4 - collision detection between a curve and a polygon
    # ---
    collCheck1p2 = c1.collCheck2Poly(poly2)
    if collCheck1p2 == 1:
        print('No collision detected between C1 and Poly2')
    else:
        print('Collision detected between C1 and Poly2')

    ax4 = c1.plot()
    plotPoly(poly2, ax4)
    plt.title('Collision check between a Bezier curve and polygon')

    plt.show()

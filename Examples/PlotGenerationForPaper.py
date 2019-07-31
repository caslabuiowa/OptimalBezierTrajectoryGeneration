#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import time

import bezier as bez


def plotPoly(poly, ax, linecolor='b'):
    from scipy.spatial import ConvexHull

    if poly.shape[1] == 2:
        is2d = True
    else:
        if (poly[:, 2] == 0).all():
            is2d = True
        else:
            is2d = False

    pts = poly.copy()

    if is2d:
#        ax.plot(pts[0], pts[1], 'k.-')
        hull = ConvexHull(pts[:, :2])
    else:
        ax.plot(pts.T[0], pts.T[1], pts.T[2], 'k.-')
        hull = ConvexHull(pts, qhull_options='QJ')

    for s in hull.simplices:
#        s = np.append(s, s[0])
        if is2d:
            ax.plot(pts[s, 0], pts[s, 1], color=linecolor, linestyle='-')
        else:
            ax.plot(pts[s, 0], pts[s, 1], pts[s, 2],
                    color=linecolor, linestyle='-')


if __name__ == '__main__':
    plt.close('all')
    plt.rcParams.update({
            'font.size': 40,
            'pdf.fonttype': 42,
            'xtick.labelsize': 40,
            'ytick.labelsize': 40,
            'lines.linewidth': 4,
            'lines.markersize': 18
            })

    cpts1 = np.array([(0, 1, 3, 0, 2, 1),
                      (3, 3, 5, 6, 5, 8),
                      (0, 0, 0, 0, 0, 0)])

    cpts2 = np.array([(8, 7, 5, 5, 7, 8),
                      (0, 1, 2, 3, 4, 5),
                      (0, 0, 0, 0, 0, 0)])

    cpts3 = np.array([(2, 1, 2, 3, 4, 5),
                      (0, 1, 3, 5, 4, 2),
                      (0, 0, 0, 0, 0, 0)])

    cpts4 = np.array([(8, 7, 6, 5, 4, 3),
                      (2, 4, 5, 8, 3, 1),
                      (0, 0, 0, 0, 0, 0)])

    cpts5 = cpts1 - 3
    cpts5[2, :] = 0

    cpts6 = np.array([(0, 1, 2, 3, 4, 5),
                      (5, 0, 2, 5, 7, 5)])

    cpts7 = np.array([(0, 1, 3, 5, 7, 7, 8, 9, 9),
                      (0, 6, 9, 6, 8, 3, 7, 8, 3)])

    poly1 = np.array([(3, 1, 0),
                      (3, 4, 0),
                      (4, 2, 0)])

    poly2 = np.array([(0, 6, 0),
                      (0, 7, 0),
                      (1, 8, 0),
                      (2, 7, 0),
                      (1, 9, 0)])

    poly3 = np.array([(-3, 2, 0),
                      (-3, 4, 0),
                      (-2, 6, 0),
                      (-1, 2, 0),
                      (-1, 5, 0)])

    c1 = bez.Bezier(cpts1)
    c2 = bez.Bezier(cpts2)
    c3 = bez.Bezier(cpts3)
    c4 = bez.Bezier(cpts4)
    c5 = bez.Bezier(cpts5)
    c6 = bez.Bezier(cpts6)
    c7 = bez.Bezier(cpts7)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ---
    # Bezier to Bezier
    # ---
    d31, t31, t1 = c3.minDist(c1)
    d32, t32, t2 = c3.minDist(c2)
    d34, t34, t4 = c3.minDist(c4)
    d35, t35, t5 = c3.minDist(c5)

    # Plot curves
    c1.plot(ax1, showCpts=False, color='g')
    c2.plot(ax1, showCpts=False, color='g')
    c3.plot(ax1, showCpts=False, color='b')
    c4.plot(ax1, showCpts=False, color='r')
    c5.plot(ax1, showCpts=False, color='g')

    # Plot minimum distance lines
    ax1.plot(np.array((c3(t31)[0], c1(t1)[0])).squeeze(),
             np.array((c3(t31)[1], c1(t1)[1])).squeeze(), 'k.--')

    ax1.plot(np.array((c3(t32)[0], c2(t2)[0])).squeeze(),
             np.array((c3(t32)[1], c2(t2)[1])).squeeze(), 'k.--')

    ax1.plot(np.array((c3(t34)[0], c4(t4)[0])).squeeze(),
             np.array((c3(t34)[1], c4(t4)[1])).squeeze(), 'k.--')

    ax1.plot(np.array((c3(t35)[0], c5(t5)[0])).squeeze(),
             np.array((c3(t35)[1], c5(t5)[1])).squeeze(), 'k.--')

    # ---
    # Bezier to Polyhedron/Polygon
    # ---
    d1p1, t1p1, pt1 = c1.minDist2Poly(poly1)
    d1p2, t1p2, pt2 = c1.minDist2Poly(poly2)
    d1p3, t1p3, pt3 = c1.minDist2Poly(poly3)

    # Plot curve and polyhedrons
    c1.plot(ax2, showCpts=False, color='b')
    plotPoly(poly1, ax2, linecolor='g')
    plotPoly(poly2, ax2, linecolor='r')
    plotPoly(poly3, ax2, linecolor='g')
    # Plot minimum distance lines
    ax2.plot(np.array((c1(t1p1)[0], pt1[0])).squeeze(),
             np.array((c1(t1p1)[1], pt1[1])).squeeze(), 'k.--')

    # Not plotting this since it doesn't exist due to a collision
#    ax2.plot(np.array((c1(t1p2)[0], pt2[0])).squeeze(),
#             np.array((c1(t1p2)[1], pt2[1])).squeeze(), 'k.--')

    ax2.plot(np.array((c1(t1p3)[0], pt3[0])).squeeze(),
             np.array((c1(t1p3)[1], pt3[1])).squeeze(), 'k.--')

    ax1.set_xlabel('(a)')
    ax2.set_xlabel('(b)')
    ax1.set_ylim((-0.5, 9.5))
    ax2.set_ylim((-0.5, 9.5))
    ax1.set_yticks(np.arange(1, 10, 2))
    ax2.set_yticks(np.arange(1, 10, 2))
    ax1.set_xticks(np.arange(-2, 9, 2))
    ax2.set_xticks(np.arange(-3, 4, 2))

    # ---
    # Elevation and extrema
    # ---
    ax3 = c6.plot(showCpts=False, label='Bernstein Polynomial')
    e5h = ax3.plot(c6.cpts[0, :], c6.cpts[1, :], '.--', label='Degree 5')
    e10h = ax3.plot(c6.elev(5).cpts[0, :], c6.elev(5).cpts[1, :], '.--',
                    label='Degree 10')
    plt.legend(loc='upper left')

    e15h = ax3.plot(c6.elev(10).cpts[0, :], c6.elev(10).cpts[1, :], '.--',
                    label='Degree 15')
    e20h = ax3.plot(c6.elev(15).cpts[0, :], c6.elev(15).cpts[1, :], '.--',
                    label='Degree 20')

    start = time.time()
    ymin = c6.min(dim=1)
    stop = time.time()
    print(f'The computation time for evaluating the minimum of C6 is '
          f'{stop-start:0.3e} s.')
    start = time.time()
    ymax = c6.max(dim=1)
    stop = time.time()
    print(f'The computation time for evaluating the maximum of C6 is '
          f'{stop-start:0.3e} s.')
    minmaxh = ax3.plot([-0.5, 5.5], [ymin, ymin], 'r:', label='Min and Max')
    ax3.plot([-0.5, 5.5], [ymax, ymax], 'r:')

    hull = ConvexHull(c6.cpts.T)
    for i, simplex in enumerate(hull.simplices):
        if i == 0:
            hullh = ax3.plot(c6.cpts[0, simplex], c6.cpts[1, simplex], 'k:',
                             label='Convex Hull')
        else:
            ax3.plot(c6.cpts[0, simplex], c6.cpts[1, simplex], 'k:')

    leg = Legend(ax3, [e15h[0], e20h[0], minmaxh[0], hullh[0]],
                 ['Degree 15', 'Degree 20', 'Min and Max', 'Convex Hull'],
                 loc='lower right')
    ax3.add_artist(leg)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim([-0.5, 5.5])
    plt.ylim([-0.5, 8.5])
    ax3.set_yticks(np.arange(0, 9, 2))

    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)

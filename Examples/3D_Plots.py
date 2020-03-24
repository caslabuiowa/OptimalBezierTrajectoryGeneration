#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
            'font.size': 24,
            'pdf.fonttype': 42,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'lines.linewidth': 3,
            'lines.markersize': 18
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
    plt.tight_layout()

    # ---
    # Bezier to Polyhedron/Polygon
    # ---
    d1p1, t1p1, pt1 = c1.minDist2Poly(poly1)
    d1p2, t1p2, pt2 = c1.minDist2Poly(poly2)
    d1p3, t1p3, pt3 = c1.minDist2Poly(poly3)

    # Plot curve and polyhedrons
    ax2 = c1.plot(showCpts=False, color='b')
    plotPoly(poly1, ax2, linecolor='g')
    plotPoly(poly2, ax2, linecolor='r')
    plotPoly(poly3, ax2, linecolor='g')
    # Plot minimum distance lines
    plt.plot(np.array((c1(t1p1)[0], pt1[0])).squeeze(),
             np.array((c1(t1p1)[1], pt1[1])).squeeze(),
             np.array((c1(t1p1)[2], pt1[2])).squeeze(), 'k.--')

    plt.plot(np.array((c1(t1p3)[0], pt3[0])).squeeze(),
             np.array((c1(t1p3)[1], pt3[1])).squeeze(),
             np.array((c1(t1p3)[2], pt3[2])).squeeze(), 'k.--')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Minimum Distance Between Curve and Polyhedrons')
    plt.tight_layout()

    # ---
    # Elevation and extrema
    # ---
    ax3 = c6.plot(showCpts=False, label='Bernstein Polynomial')
    ax3.plot(c6.cpts[0, :], c6.cpts[1, :], '.--', label='Degree 5')
    ax3.plot(c6.elev(5).cpts[0, :], c6.elev(5).cpts[1, :], '.--',
             label='Degree 10')
    ax3.plot(c6.elev(10).cpts[0, :], c6.elev(10).cpts[1, :], '.--',
             label='Degree 15')
    ax3.plot(c6.elev(15).cpts[0, :], c6.elev(15).cpts[1, :], '.--',
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
    ax3.plot([-0.5, 5.5], [ymin, ymin], 'r:', label='Min and Max')
    ax3.plot([-0.5, 5.5], [ymax, ymax], 'r:')

    hull = ConvexHull(c6.cpts.T)
    for i, simplex in enumerate(hull.simplices):
        if i == 0:
            ax3.plot(c6.cpts[0, simplex], c6.cpts[1, simplex], 'k:',
                     label='Convex Hull')
        else:
            ax3.plot(c6.cpts[0, simplex], c6.cpts[1, simplex], 'k:')

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim([-0.5, 5.5])
    plt.ylim([-0.5, 7.5])
    plt.tight_layout()

#    # ---
#    # Convex Hull
#    # ---
#    ax4 = c7.plot(label='Bernstein Polynomial')
#    hull = ConvexHull(c7.cpts.T)
#    for i, simplex in enumerate(hull.simplices):
#        if i == 0:
#            ax4.plot(c7.cpts[0, simplex], c7.cpts[1, simplex], 'k:',
#                     label='Convex Hull')
#        else:
#            ax4.plot(c7.cpts[0, simplex], c7.cpts[1, simplex], 'k:')
#
#    plt.legend()
#    plt.xlabel('X')
#    plt.ylabel('Y')
#    plt.xlim([0, 9])
#    plt.ylim([0, 9])
#    ax4.set_xticks([0, 1, 3, 5, 7, 9])
#    ax4.set_yticks([0, 1, 3, 5, 7, 9])
#    plt.tight_layout()

    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)

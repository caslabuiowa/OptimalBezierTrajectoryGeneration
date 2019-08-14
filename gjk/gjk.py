#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 21:25:46 2018

@author: ckielasjensen
"""

# For spatial separation, simply return a value like 0 or -1 if a collision is
# detected. That way the algorithm won't continuously run. The 0 or -1 will
# work with the minimum spatial separation because it looks for the upper
# bound of minimum values. Check out Algorithm 2 from the paper.

# NOTE: This library uses the numba library to significantly spead up certain
#   operations.

from numba import njit
import numpy as np


def gjk(polygon1, polygon2, method='nearest', *args, **kwargs):
    """
    """
    gjkAlgorithms = {'collision': gjkCollision, 'nearest': gjkNearest}
    algo = gjkAlgorithms[method.lower()]

    return algo(polygon1, polygon2, *args, **kwargs)


@njit(cache=True)
def gjkNearest(polygon1, polygon2, maxIter=10):
    """
    Finds the shortest distance between two polygons using the GJK algorithm.
    It will hit the maximum number of iterations if the polygons overlap.

    INPUTS:
        polygon1 (np.array) - N x 3 numpy array of points where N is the number
        of points in the polygon. For a 2D case, set the 3rd column to zero.

        polygon2 (np.array) - Same as polygon1.

        maxIter (int) - Maximum number of iterations to run the algorithm.
    """
    # Arbitrary initial direction
    direction = np.array((1, 0, 0))  # np.random.random(3)

    A = support(polygon1, direction) - support(polygon2, -direction)
    B = support(polygon1, -direction) - support(polygon2, direction)

    direction, directionMag = closestPointToOrigin(A, B)

    for i in range(maxIter):
        C = support(polygon1, -direction) - support(polygon2, direction)
#        print(('Simplex - A: {}\n'
#               '          B: {}\n'
#               '          C: {}\n'
#               '          DIR: {}\n').format(A, B, C, direction))

        if (C == A).all() or (C == B).all():
            return directionMag

        p1, p1Mag = closestPointToOrigin(A, C)
        p2, p2Mag = closestPointToOrigin(B, C)
        if p1Mag < p2Mag:
            B = C
            direction = p1
            directionMag = p1Mag
        else:
            A = C
            direction = p2
            directionMag = p2Mag

#    print('WARNING: Maximum number of iterations met.')
    return 0.0


def gjkCollision(polygon1, polygon2):
    """
    """
    errorMsg = (
            'The gjkCollision function is currently under development and is '
            'unavailable at this time.'
            )
    raise NotImplementedError(errorMsg)


@njit(cache=True)
def support(shape, direction):
    """
    Returns the point in shape that is furthest in the desired direction.

    NOTE: Shape must contain at least 2 rows.

    INPUTS:
        shape (np.array) - N x 3 numpy array of points where N is the number
        of points in the polygon. For a 2D case, set the 3rd column to zero.

        direction (np.array) - 1 x 3 numpy array specifying the direction using
        standard (x,y,z) coordinates. For the 2D case, set z to 0.
    """
    N = shape.shape[0]
    supportPoint = shape[0]
#    print('SUPPORT - SupportPoint:{}'.format(supportPoint))
    maxDist = dot(supportPoint, direction)

    for i in range(N):
        curDist = dot(shape[i], direction)
#        print('SUPPORT - CurDist:{}'.format(curDist))
        if curDist > maxDist:
            maxDist = curDist
            supportPoint = shape[i]
#            print('SUPPORT - NewPt:{}'.format(supportPoint))

    return supportPoint


@njit(cache=True)
def closestPointToOrigin2(a, b):
    """
    Finds the closest point to the origin on the line AB.
    This will raise an exception if a and b are equal.

    INPUTS:
        a (np.array) - 1x3 array of x, y, and z vector coordinates. For the 2D
        case, simply set z to zero.

        b (np.array) - See a. Must be different from b otherwise it will result
        in a divide by zero exception.
    """
    # Vector from a to b and vector from a to origin
    AB = b - a
    A0 = -a

    closestPoint = AB * (dot(A0, AB) / dot(AB, AB)) + a
    distance = np.sqrt(dot(closestPoint, closestPoint))

    return closestPoint, distance


@njit(cache=True)
def closestPointToOrigin(a, b):
    """
    Finds the closest point to the origin on the line AB.
    This will raise an exception if a and b are equal.

    INPUTS:
        a (np.array) - 1x3 array of x, y, and z vector coordinates. For the 2D
        case, simply set z to zero.

        b (np.array) - See a. Must be different from b otherwise it will result
        in a divide by zero exception.

    RETURNS:
        (np.array) closest point on line segment to the origin.

    SOURCE:
        https://math.stackexchange.com/questions/2193720/find-a-point-on-a-
        line-segment-which-is-the-closest-to-other-point-not-on-the-li
    """
    v = b - a
    u = a
    t = - dot(v, u) / dot(v, v)

    if t >= 0 and t <= 1:
        closestPt = (1-t)*a + t*b
    elif t > 1:
        closestPt = b.astype(np.float64)
    else:
        closestPt = a.astype(np.float64)

    return closestPt, np.sqrt(dot(closestPt, closestPt))


@njit(cache=True)
def dot(a, b):
    """
    Fast implementation of the dot product.

    Assumes a and b are both 3D vectors. For the 2D case, simply set the last
    value to zero for both vectors.

    INPUTS:
        a (np.array) - 1x3 array of x, y, and z vector coordinates. For the 2D
        case, simply set z to zero.

        b (np.array) - See a.

    RETURNS:
        a*b where * is the dot product

    NOTE: On the test computer, using np.dot(a,b) took about 563 ns to compute
    while this function took about 338 ns providing a speed up of about 1.6x.
    """
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@njit(cache=True)
def tripleProduct(a, b, c):
    """
    Fast implementation of the vector triple product.
    Vectors a, b, and c should all be length 3.
    Uses the following identity to compute the triple product:
        a x (b x c) = b(a*c) - c(a*b)

    INPUTS:
        a (np.array) - 1x3 array of x, y, and z vector coordinates. For the 2D
        case, simply set z to zero.

        b (np.array) - See a.

        c (np.array) - See a.

    RETURNS:
        a x (b x c) where x is the cross product

    NOTE: On the test computer, using np.cross(a,np.cross(b,c)) took about 46us
    to compute while this function took about 701 ns to compute providing a
    speed up of about 65x.
    """
    AdotC = dot(a, c)
    AdotB = dot(a, b)
    return np.array((
            b[0]*AdotC - c[0]*AdotB,
            b[1]*AdotC - c[1]*AdotB,
            b[2]*AdotC - c[2]*AdotB
            ))


#@njit(cache=True)
def gjkNew(poly1, poly2, maxIter=128, verbose=False):
    """

    RETURNS:
        flag (int) - The flag will either be -1, 0, or 1:
            * -1: Maximum number of iterations met
            * 0: Collision detected
            * 1: No collision detected, minimum distance available

        info (tuple) - If flag is 1, info will contain 3 values:
            * info[0]: point on poly1 at which the minimum distance occurs
            * info[1]: point on poly2 at which the minimum distance occurs
            * info[3]: minimum distance between the two shapes
            If flag is -1 or 0, info will be an empty tuple.
    """
    simplex = dict()
#    direction = poly1.mean(axis=0) - poly2.mean(axis=0)
    direction = np.array((1, 0, 0), dtype=float)

    for _ in range(maxIter):

        simplex, direction = doSimplex(poly1, poly2, simplex, direction)
        if verbose:
            print(f'Iter: {_}, Direction: {direction}\n'
                  f'  --> Simplex: {simplex}')

        if 'collision' in simplex.keys():
            return 0, ()

        # No collision
        elif simplex['A'].dot(direction) < 0:

            closestPt1, closestPt2, distance = minimumDistance(poly1,
                                                               poly2,
                                                               simplex,
                                                               direction)

            return 1, (closestPt1, closestPt2, distance)

    print('Maximum iterations met')
    return -1, ()


def minimumDistance(poly1, poly2, simplex, direction):
    """
    """
    converged = False
    while True:
        oldSimplex = simplex.copy()
        simplex, direction = doSimplex(poly1, poly2, simplex, direction)

        for point in oldSimplex.values():
#            print()
#            print('Old:')
#            for key in oldSimplex.keys():
#                print(f'  {key}: {oldSimplex[key]}')
#            print('---')
#            print('New:')
#            for key in simplex.keys():
#                print(f'  {key}: {simplex[key]}')
#            print('=======================================================')
            if (simplex['A'] == point).all():
                converged = True
                simplex = oldSimplex.copy()
                break

        if converged:
            break

    if 'C' in simplex.keys():
        A0 = -simplex['A']
        AB = simplex['B'] - simplex['A']
        AC = simplex['C'] - simplex['A']
        ABC = np.cross(AB, AC)

        # Origin closest to A, C or AC
        if np.cross(ABC, AC).dot(A0) >= 0:
            t, distance = weightedOriginToLine(simplex['A'],
                                               simplex['C'])
            poly1pt = (1-t)*simplex['Apts'][0] + t*simplex['Cpts'][0]
            poly2pt = (1-t)*simplex['Apts'][1] + t*simplex['Cpts'][1]

        # Origin closest to A, B, or AB
        elif np.cross(AB, ABC).dot(A0) >= 0:
            t, distance = weightedOriginToLine(simplex['A'],
                                               simplex['B'])
            poly1pt = (1-t)*simplex['Apts'][0] + t*simplex['Bpts'][0]
            poly2pt = (1-t)*simplex['Apts'][1] + t*simplex['Bpts'][1]

        # Origin closest to plane ABC
        else:
            baryTriple, distance = weightedOriginToPlane(simplex['A'],
                                                         simplex['B'],
                                                         simplex['C'])

            A = simplex['A']
            B = simplex['B']
            C = simplex['C']
            A1 = simplex['Apts'][0]
            B1 = simplex['Bpts'][0]
            C1 = simplex['Cpts'][0]
            A2 = simplex['Apts'][1]
            B2 = simplex['Bpts'][1]
            C2 = simplex['Cpts'][1]

            poly1pt = (baryTriple[0]*(A+A2) +
                       baryTriple[1]*(B+B2) +
                       baryTriple[2]*(C+C2))

            poly2pt = (baryTriple[0]*(A1-A) +
                       baryTriple[1]*(B1-B) +
                       baryTriple[2]*(C1-C))

    # 2 point simplex
    elif 'B' in simplex.keys():
        t, distance = weightedOriginToLine(simplex['A'],
                                           simplex['B'])
        poly1pt = (1-t)*simplex['Apts'][0] + t*simplex['Bpts'][0]
        poly2pt = (1-t)*simplex['Apts'][1] + t*simplex['Bpts'][1]

    # 1 point simplex
    elif 'A' in simplex.keys():
        distance = np.linalg.norm(simplex['A'])
        poly1pt = simplex['Apts'][0]
        poly2pt = simplex['Apts'][1]

    else:
        raise ValueError('Simplex should at least be a point to check '
                         'for the minimum distance.')

    return poly1pt, poly2pt, distance


@njit(cache=True)
def closestPointToLine(a, b):
    """
    Finds the closest point to the origin on the line AB.
    This will raise an exception if a and b are equal.

    INPUTS:
        a (np.array) - 1x3 array of x, y, and z vector coordinates. For the 2D
        case, simply set z to zero.

        b (np.array) - See a. Must be different from b otherwise it will result
        in a divide by zero exception.

    RETURNS:
        (np.array) closest point on line segment to the origin.

    SOURCE:
        https://math.stackexchange.com/questions/2193720/find-a-point-on-a-
        line-segment-which-is-the-closest-to-other-point-not-on-the-li
    """
    v = b - a
    u = a
    t = - dot(v, u) / dot(v, v)

    if t >= 0 and t <= 1:
        closestPt = (1-t)*a + t*b
    elif t > 1:
        closestPt = b.astype(np.float64)
    else:
        closestPt = a.astype(np.float64)

    return closestPt, np.sqrt(dot(closestPt, closestPt))


@njit(cache=True)
def weightedOriginToLine(A, B):
    """Finds the closest point to the origin on the line AB and its distance.
    This will print a warning if A and B are equal.

    INPUTS:
        A (np.array) - 1x3 array of x, y, and z vector coordinates. For the 2D
        case, simply set z to zero.

        B (np.array) - See a. Must be different from b otherwise it will result
        in a divide by zero exception.

    RETURNS:
        t (float) - Weight corresponding to where on the line the closest point
        to the origin lies. If t=0, the closest point to the origin is A, if
        t=1, the closest point to the origin is B.

        dist (float) - Distance between the closest point to the origin and the
        origin

    SOURCE:
        https://math.stackexchange.com/questions/2193720/find-a-point-on-a-
        line-segment-which-is-the-closest-to-other-point-not-on-the-li
    """
    if (A == B).all():
#        print('[!] Warning, points are identical.')
        return 0, np.sqrt(dot(A, A))

    v = B - A
    u = A
    t = -dot(v, u) / dot(v, v)

    if t > 1:
        t = 1
    elif t < 0:
        t = 0

    closestPt = (1-t)*A + t*B
    dist = np.sqrt(dot(closestPt, closestPt))

    return t, dist


def weightedOriginToPlane(A, B, C):
    """
    SOURCES:
        * https://math.stackexchange.com/questions/100761/how-do-i-find-the-
          projection-of-a-point-onto-a-plane
        * https://math.stackexchange.com/questions/588871/minimum-distance-
          between-point-and-face
    """
    N = np.cross(B-A, C-A)
    n = N / np.linalg.norm(N)

    a = n[0]
    b = n[1]
    c = n[2]

    d = A[0]
    e = A[1]
    f = A[2]

    # First, we find the closest point on the Minkowski difference
    t = (a*d + b*e + c*f) / (a**2 + b**2 + c**2)
    closestPt = t*n
    dist = np.sqrt(dot(closestPt, closestPt))

    # Then, we find the Barycentric triple so that we can project the point
    # onto each shape.
    PA = A - closestPt
    PB = B - closestPt
    PC = C - closestPt

    areaABC2 = np.linalg.norm(N)
    alpha = np.linalg.norm(np.cross(PB, PC)) / areaABC2
    beta = np.linalg.norm(np.cross(PC, PA)) / areaABC2
    gamma = 1 - alpha - beta

    baryTriple = (alpha, beta, gamma)

    return baryTriple, dist


#@njit(cache=True)
#def closestPointToPlane(A, B, C):
##    print(f'A: {A}, B: {B}, C: {C}')
#    A0 = -A
#    N = np.cross(B-A, C-A)
#    n = N / np.linalg.norm(N)
#
#    distance = A0.dot(n)
#    closestPt = -distance*n
#
#    return closestPt, distance


@njit(cache=True)
def supportPts(poly1, poly2, direction):
    """
    """
    p1 = support(poly1, direction)
    p2 = support(poly2, -direction)
    newPt = p1 - p2

    return newPt, (p1, p2)


#@njit(cache=True)
def doSimplex(poly1, poly2, simplex, direction):
    """
    """
    if 'A' not in simplex.keys():
        simplex, direction = simplex0pt(poly1, poly2, simplex, direction)

    elif 'B' not in simplex.keys():
        simplex, direction = simplex1pt(poly1, poly2, simplex, direction)

    elif 'C' not in simplex.keys():
        simplex, direction = simplex2pt(poly1, poly2, simplex)

    elif 'D' not in simplex.keys():
        simplex, direction = simplex3pt(poly1, poly2, simplex)

    elif 'D' in simplex.keys():
        simplex, direction = simplex4pt(poly1, poly2, simplex)

    else:
        raise ValueError('The simplex should only have 0-4 points.')

    return simplex, direction


#@njit(cache=True)
def simplex0pt(poly1, poly2, simplex, direction):
    """
    """
    simplex['A'], simplex['Apts'] = supportPts(poly1, poly2, direction)

    return simplex, direction


#@njit(cache=True)
def simplex1pt(poly1, poly2, simplex, direction):
    """
    """
    simplex['B'] = simplex['A']
    simplex['Bpts'] = simplex['Apts']
    direction = -direction
    simplex['A'], simplex['Apts'] = supportPts(poly1, poly2, direction)

    return simplex, direction


#@njit(cache=True)
def simplex2pt(poly1, poly2, simplex):
    """
    """
    t, distance = weightedOriginToLine(simplex['A'], simplex['B'])
    closestPt = (1-t)*simplex['A'] + t*simplex['B']
    direction = -closestPt
    simplex['C'] = simplex['A']
    simplex['Cpts'] = simplex['Apts']
    simplex['A'], simplex['Apts'] = supportPts(poly1, poly2, direction)

    return simplex, direction


#@njit(cache=True)
def simplex3pt(poly1, poly2, simplex):
    """
    """
    # Name line vectors for clarity
    A0 = -simplex['A']
    AB = simplex['B'] - simplex['A']
    AC = simplex['C'] - simplex['A']
    ABC = np.cross(AB, AC)

    # If origin is out of simplex and closest to A or AC
    if np.cross(ABC, AC).dot(A0) > 0:
        # If closest to line AC
        if AC.dot(A0) > 0:
            direction = np.cross(np.cross(AC, A0), AC)
            # Keep points A and C, replace B
            simplex['B'] = simplex['A']
            simplex['Bpts'] = simplex['Apts']

        # Otherwise we are closest to A or AB
        else:
            # Star case
            # If closest to AB
            if AB.dot(A0) > 0:
                direction = np.cross(np.cross(AB, A0), AB)
                # Keep points A and B, replace C
                simplex['C'] = simplex['A']
                simplex['Cpts'] = simplex['Apts']

            # Otherwise we are closest to point A
            else:
                direction = simplex['A']
                simplex.clear()

    # If closest to line AB
    elif np.cross(AB, ABC).dot(A0) > 0:
        # Star case
        # If closest to AB
        if AB.dot(A0) > 0:
            direction = np.cross(np.cross(AB, A0), AB)
            # Keep points A and B, replace C
            simplex['C'] = simplex['A']
            simplex['Cpts'] = simplex['Apts']

        # Otherwise we are closest to point A
        else:
            direction = -simplex['A']
            simplex.clear()

    # Otherwise we are on, above, or below the ABC plane
    else:
        # If we are on the ABC plane, there is a collision
        if ABC.dot(A0) == 0:
            simplex['collision'] = True
            direction = np.array((0, 0, 0))

        elif ABC.dot(A0) > 0:
            direction = ABC

            simplex['D'] = simplex['C']
            simplex['Dpts'] = simplex['Cpts']
            simplex['C'] = simplex['B']
            simplex['Cpts'] = simplex['Bpts']
            simplex['B'] = simplex['A']
            simplex['Bpts'] = simplex['Apts']

        else:
            # If the origin is below the simplex, swap the points to make the
            # search direction positive in the normal direction.
            direction = -ABC

            simplex['D'] = simplex['B']
            simplex['Dpts'] = simplex['Bpts']
            simplex['B'] = simplex['A']
            simplex['Bpts'] = simplex['Apts']

    simplex['A'], simplex['Apts'] = supportPts(poly1, poly2, direction)

    return simplex, direction


#@njit(cache=True)
def simplex4pt(poly1, poly2, simplex):
    """
    """
    A0 = -simplex['A']

    AB = simplex['B'] - simplex['A']
    AC = simplex['C'] - simplex['A']
    AD = simplex['D'] - simplex['A']

    ABC = np.cross(AB, AC)
    ACD = np.cross(AC, AD)
    ADB = np.cross(AD, AB)

    if ABC.dot(A0) > 0:
        simplex.pop('D')
        simplex, direction = simplex3pt(poly1, poly2, simplex)

    elif ACD.dot(A0) > 0:
        simplex['B'] = simplex['C']
        simplex['Bpts'] = simplex['Cpts']
        simplex['C'] = simplex.pop('D')
        simplex['Cpts'] = simplex.pop('Dpts')
        simplex, direction = simplex3pt(poly1, poly2, simplex)

    elif ADB.dot(A0) > 0:
        simplex['C'] = simplex['B']
        simplex['Cpts'] = simplex['Bpts']
        simplex['B'] = simplex.pop('D')
        simplex['Bpts'] = simplex.pop('Dpts')
        simplex, direction = simplex3pt(poly1, poly2, simplex)

    else:
        simplex['collision'] = True
        direction = np.array((0, 0, 0))

    return simplex, direction


# Test code
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial import ConvexHull

    poly1 = np.array([
            (4, 11, 0),
            (4, 5, 0),
            (9, 9, 0)
            ], dtype=float)

    poly2 = np.array([
            (5, 6, 0),
            (10, 2, 0),
            (13, 1, 0),
            (12, 3, 0),
            (15, 6, 0)
            ], dtype=float)

    poly3 = np.array([
            (4, 11, -1),
            (4, 5, -1),
            (9, 9, -1),
            (7, 8, 3)
            ], dtype=float)

    poly4 = np.array([
            (4, 11, 3),
            (4, 5, 3),
            (9, 9, 3),
            (7, 8, -1)
            ], dtype=float)

    poly5 = np.array([
            (4, 11, -3),
            (4, 5, -3),
            (9, 9, -3),
            (7, 8, -1)
            ], dtype=float)

    poly6 = np.array([
            (4, 11, 0),
            (4, 5, 1),
            (9, 9, 2),
            (7, 8, 3)
            ], dtype=float)

    poly7 = np.array([
            (-1, -1, 0),
            (1, 1, 0),
            (1, -1, 0),
            (-1, 1, 0)
            ], dtype=float)

    poly8 = np.array([
            (-1, -1, -3),
            (1, 1, -3),
            (1, -1, -3),
            (-1, 1, -3),
            (0, 0, -1)
            ], dtype=float)

    plt.close('all')

    verbose = False
    print(gjkNew(poly1, poly2, verbose=verbose))
    print(gjkNew(poly1, poly3, verbose=verbose))
    print(gjkNew(poly1, poly4, verbose=verbose))
    print(gjkNew(poly1, poly5, verbose=verbose))
    print(gjkNew(poly5, poly6, verbose=verbose))
    print(gjkNew(poly7, poly8, verbose=verbose))

    polyTest = [poly5, poly6]

    flag, info = gjkNew(polyTest[0], polyTest[1])
    if flag > 0:
        polyPts = info[0:2]
    else:
        print('Warning, no minimum distance computed')
        polyPts = ([0]*3, [0]*3)
    polyPts = np.array(polyPts)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for poly in polyTest:
        pts = poly.copy()
        hull = ConvexHull(pts, qhull_options='QJ')

        ax.plot(pts.T[0], pts.T[1], pts.T[2], 'ko')

        for s in hull.simplices:
            s = np.append(s, s[0])
            ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], 'b-')

    plt.plot(polyPts[:, 0], polyPts[:, 1], polyPts[:, 2], 'r*-')

    plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 21:25:46 2018

@author: ckielasjensen
"""

# NOTE: This library uses the numba library to significantly spead up certain
#   operations.

#from scipy.spatial import ConvexHull
import numpy as np
from numba import njit

def gjk(polygon1, polygon2, method='nearest', *args, **kwargs):
    """
    """
    gjkAlgorithms = {'collision':gjkCollision, 'nearest':gjkNearest}
    algo = gjkAlgorithms[method.lower()]
    
    return algo(polygon1, polygon2, *args, **kwargs)

@njit
def gjkNearest(polygon1, polygon2, maxIter=1000):
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
    direction = np.array((1,0,0)) #np.random.random(3)
    
    A = support(polygon1, direction) - support(polygon2, -direction)
    B = support(polygon1, -direction) - support(polygon2, direction)
    
    direction, directionMag = closestPointToOrigin(A, B)
    
    for i in range(maxIter):
        C = support(polygon1, -direction) - support(polygon2, direction)
#        print(('Simplex - A: {}\n'
#               '          B: {}\n'
#               '          C: {}\n'
#               '          DIR: {}\n').format(A, B, C, direction))
        
        if (C==A).all() or (C==B).all():
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
            
    print('WARNING: Maximum number of iterations met.')
    return None

def gjkCollision(polygon1, polygon2):
    """
    """
    errorMsg = (
            'The gjkCollision function is currently under development and is '
            'unavailable at this time.'
            )
    raise NotImplementedError(errorMsg)

@njit
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

@njit
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
    distance = np.sqrt(dot(closestPoint,closestPoint))

    return closestPoint, distance

@njit
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
    t = - dot(v,u) / dot(v,v)

    if t >= 0 and t <= 1:
        closestPt = (1-t)*a + t*b
    elif t > 1:
        closestPt = b.astype(np.float64)
    else:
        closestPt = a.astype(np.float64)
        
    return closestPt, np.sqrt(dot(closestPt, closestPt))

@njit
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

@njit
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
    
# Test code
if __name__=="__main__":
    poly1 = np.array([
            (4, 11, 0),
            (4, 5, 0),
            (9, 9, 0)
            ])
        
    poly2 = np.array([
            (8, 6, 0),
            (10, 2, 0),
            (13, 1, 0),
            (15, 6, 0)
            ])
        
    retVal = gjk(poly1, poly2)
    
    print(retVal)
    
    poly3 = np.random.random((10,3))
    poly4 = np.random.random((10,3))
    
    print(gjk(poly3, poly4))
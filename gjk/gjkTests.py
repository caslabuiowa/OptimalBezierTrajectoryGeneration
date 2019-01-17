#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:49:56 2018

This test file uses examples found at:
    www.dyn4j.org/2010/04/gjk-distance-closest-points/

@author: ckielasjensen
"""

import numpy as np
import gjk

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
    
retVal = gjk.gjk(poly1, poly2)

for i in retVal:
    print(i)
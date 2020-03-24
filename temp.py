#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:59:40 2019

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import numpy as np

import bezier as bez

NPTS = 1001

plt.close('all')

cpts1 = np.array([(0, 1, 2, 3, 4, 5),
                  (3, 4, 2, 6, 7, 8)])

cpts2 = np.array([(0, 1, 2, 3, 4, 5),
                  (7, 8, 5, 9, 2, 3)])

c1 = bez.Bezier(cpts1, tau=np.linspace(0, 1, NPTS))
c2 = bez.Bezier(cpts2, tau=np.linspace(0, 1, NPTS))

normsquare = (c1-c2).normSquare()
thing = ((c1-c2)*(c1-c2)).cpts[0, :] + ((c1-c2)*(c1-c2)).cpts[1, :]

ax = normsquare.plot()

x = np.linspace(0, 1, NPTS)
y = (c1.curve-c2.curve)[0, :]**2 + (c1.curve-c2.curve)[1, :]**2

ax.plot(x, y)

plt.show()

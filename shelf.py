#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 23:09:48 2018

@author: heller
"""

import numpy as np
import sympy as sp


def gains_2d(order):
    max_rE = np.max([sp.chebyshevt_root(order + 1, i)
                     for i in range(order + 1)])
    return [sp.chebyshevt(n, max_rE) for n in range(order+1)]


def gains_3d(order):
    x = sp.symbols('x')
    if order < 5:
        max_rE = np.max(sp.roots(sp.legendre_poly(order+1, x)).keys())
    else:
        max_rE = np.max(sp.nroots(sp.legendre_poly(order+1, x)))
    return [sp.legendre(n, max_rE) for n in range(order+1)]

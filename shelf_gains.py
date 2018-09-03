#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 17:34:41 2018

@author: heller
"""

import numpy as np
import mpmath as mp

from mpmath import legendre, chebyt
from mpmath import taylor, polyroots, nprint

mp.dps = 50

def legendre_roots(order):
    return polyroots(taylor(lambda x: legendre(order, x), 0, order)[::-1])

def chebyt_roots(order):
    return polyroots(taylor(lambda x: chebyt(order, x), 0, order)[::-1])




def shelf_gains_3d(order):
    rE_max = np.max(legendre_roots(order + 1))
    return [legendre(n, rE_max) for n in range(order + 1)]


def shelf_gains_2d(order):
    rE_max = np.max(chebyt_roots(order + 1))
    return [chebyt(n, rE_max) for n in range(order + 1)]

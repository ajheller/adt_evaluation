#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 23:09:48 2018

@author: heller
"""

# This file is part of the Ambisonic Decoder Toolbox (ADT)
# Copyright (C) 2018-19  Aaron J. Heller <heller@ai.sri.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from __future__ import division, print_function

import numpy as np
import sympy as sp


# max rE gains
#  from Heller, et al. LAC 2012
def max_rE_gains_2d(order, numeric=True):
    max_rE = np.max([sp.chebyshevt_root(order + 1, i)
                     for i in range(order + 1)])
    return [sp.chebyshevt(n, max_rE) for n in range(order+1)]


def max_rE_gains_3d(order, numeric=True):
    """max rE for a given order is the largest root of the order+1 Legendre
    polynomial"""

    x = sp.symbols('x')
    lp = sp.legendre_poly(order+1, x)

    # there are more efficient methods to find the roots of the Legendre
    # polynomials, but this is good enough for our purposes
    # See discussion at:
    #   https://math.stackexchange.com/questions/12160/roots-of-legendre-polynomial
    if order < 5 and not numeric:
        roots = sp.roots(lp)
    else:
        roots = sp.nroots(lp)

    # the roots can be in the keys of a dictionary or in a list,
    # this works for either one
    max_rE = np.max([*roots])

    return [sp.legendre(n, max_rE) for n in range(order+1)]


# cardioid gains
#  from Moreau Table 3.5, page 69
def cardioid_gains_2d(ambisonic_order):
    l = ambisonic_order
    return [sp.factorial(l)**2 / (sp.factorial(l+m) * sp.factorial(l-m))
            for m in range(l+1)]


def cardioid_gains_3d(ambisonic_order):
    l = ambisonic_order
    return [(sp.factorial(l) * sp.factorial(l+1)) /
            (sp.factorial(l+m+1) * sp.factorial(l-m))
            for m in range(l+1)]

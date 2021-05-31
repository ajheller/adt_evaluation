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



import numpy as np
import scipy.special as spec
import sympy as sp
from scipy import interpolate as interp

# max rE gains
#  from Heller, et al. LAC 2012

def max_rE_2d(ambisonic_order: int) -> float:
    """Maximum achievable |rE| with a uniform 2-D speaker array."""
    roots, _ = spec.roots_chebyt(ambisonic_order + 1)
    return roots.max()


def max_rE_gamma_2d(sh_l):
    max_rE = max_rE_2d(np.max(sh_l))
    return np.array([np.polyval(spec.chebyt(deg), max_rE)
                     for deg in sh_l])


def max_rE_3d(ambisonic_order: int) -> float:
    """Maximum achievable |rE| with a uniform 3-D speaker array."""
    roots, _ = spec.roots_legendre(ambisonic_order + 1)
    return roots.max()


def max_rE_gamma_3d(sh_l):
    max_rE = max_rE_3d(np.max(sh_l))
    return np.array([np.polyval(spec.legendre(deg), max_rE)
                     for deg in sh_l])


def max_rE_gains_2d(order, numeric=True):
    """Deprecated."""
    max_rE = np.max([sp.chebyshevt_root(order + 1, i)
                     for i in range(order + 1)])
    return [sp.chebyshevt(n, max_rE) for n in range(order + 1)]


def max_rE_gains_3d(order, numeric=True):
    """max rE for a given order is the largest root of the order+1 Legendre
    polynomial"""

    x = sp.symbols('x')
    lp = sp.legendre_poly(order + 1, x)

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

    return [sp.legendre(n, max_rE) for n in range(order + 1)]


# inverses of max_rE_nd
def rE_to_ambisonic_order_function(dims, max_order=50):
    x = np.arange(max_order)
    if dims == 2:
        y = [max_rE_2d(o) for o in np.arange(max_order)]
    elif dims == 3:
        y = [max_rE_3d(o) for o in np.arange(max_order)]
    else:
        raise ValueError("dims should be 2 or 3")

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    fn = interp.interp1d(y, x,
                         'quadratic',
                         bounds_error=False,
                         fill_value=(0.0, max_order))
    return fn


rE_to_ambisonic_order_3d = rE_to_ambisonic_order_function(3)
rE_to_ambisonic_order_2d = rE_to_ambisonic_order_function(2)


# cardioid gains, aka in-phase gains
#  from Moreau Table 3.5, page 69
# we use sympy to do exact arithmetic here
# TODO: port factorial_quotient from MATLAB ADT to get rid of SymPy
def cardioid_gains_2d(ambisonic_order):
    l = ambisonic_order
    return [sp.factorial(l) ** 2 / (sp.factorial(l + m) * sp.factorial(l - m))
            for m in range(l + 1)]


def cardioid_gamma_2d(sh_l):
    l = np.max(sh_l)
    return [sp.factorial(l) ** 2 / (sp.factorial(l + m) * sp.factorial(l - m))
            for m in sh_l]


def cardioid_gains_3d(ambisonic_order):
    l = ambisonic_order
    return [(sp.factorial(l) * sp.factorial(l + 1)) /
            (sp.factorial(l + m + 1) * sp.factorial(l - m))
            for m in range(l + 1)]


def cardioid_gamma_3d(sh_l):
    l = np.max(sh_l)
    return [(sp.factorial(l) * sp.factorial(l + 1)) /
            (sp.factorial(l + m + 1) * sp.factorial(l - m))
            for m in sh_l]


# function to match LF and HF perceptual gains
#  note that gammas here is the set for all the channels
def gamma0(gammas, matching_type='rms', n_spkrs=None):
    E_gain = np.sum(gammas**2)
    if matching_type in ('energy', 1):
        g2 = n_spkrs / E_gain
    elif matching_type in ('rms', 2):
        g2 = len(gammas) / E_gain
    elif matching_type in ('amp', 3):
        g2 = 1
    else:
        raise ValueError(f"Unknown matching_type = {matching_type}")
    return np.sqrt(g2)


# full-featured API
def gamma(sh_l,
          decoder_type: str = 'max_rE',
          decoder_3d: bool = True,
          return_matrix: bool = False) -> np.ndarray:
    #
    # fill in defaults
    try:
        iter(sh_l)  # is sh_l iterable?
    except TypeError:
        sh_l = range(sh_l + 1)

    decoder_type = decoder_type.upper()

    #
    if decoder_type in ('MAX_RE', 'HF'):
        if decoder_3d:
            ret = max_rE_gamma_3d(sh_l)
        else:
            ret = max_rE_gamma_2d(sh_l)

    elif decoder_type in ('CARDIOID', 'IN_PHASE', 'LARGE_AREA'):
        if decoder_3d:
            ret = cardioid_gamma_3d(sh_l)
        else:
            ret = cardioid_gamma_2d(sh_l)

    elif decoder_type in ('VELOCITY', 'MATCHING', 'BASIC', 'LF'):
        ret = np.ones_like(sh_l)

    else:
        raise ValueError(f'Unknown decoder type: {decoder_type}')

    if return_matrix:
        ret = np.diag(list(map(float, ret)))

    return ret

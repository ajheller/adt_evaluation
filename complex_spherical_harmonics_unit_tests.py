#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:45:50 2019

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


from __future__ import print_function
from __future__ import division

from scipy import special, pi, sin, cos, sqrt
from scipy.integrate import dblquad

import numpy as np
from numpy import conj, real, imag, floor, array, meshgrid


# quadpack doesn't take complex integrands, so we gotta do it twice. Sheesh!!!
def complex_dblquad(func, a, b, gfun, hfun):

    def real_func(x, y):
        return real(func(x, y))

    def imag_func(x, y):
        return imag(func(x, y))

    real_integral, real_error = dblquad(real_func, a, b, gfun, hfun)
    imag_integral, imag_error = dblquad(imag_func, a, b, gfun, hfun)
    return (complex(real_integral, imag_integral),
            complex(real_error, imag_error))


##
def check_sph_ortho_c(l1, m1, l2, m2):
    return complex_dblquad(
        # arguments to lambda need to be reversed from args to dblquad (wtf?)
        lambda phi, theta:
            # scipy sph_harm takes order/degree  (wtf?)
            special.sph_harm(m1, l1, theta, phi) *
            conj(special.sph_harm(m2, l2, theta, phi)) * sin(phi),
        0, 2*pi,  # range of theta
        gfun=lambda x: 0, hfun=lambda y: pi  # range of phi
        )


def test_ortho(l1, m1, l2, m2, z, e=1e-10):
    if l1 == l2 and m1 == m2:
        # should be 1
        test = abs(z-1) < abs(e)
    else:
        # should be 0
        test = abs(z) < abs(e)
    return test


def ortho_test_complex(max_degree):
    for l1 in range(0, max_degree+1):
        for m1 in range(0, l1+1):
            for l2 in range(0, max_degree+1):
                for m2 in range(0, l2+1):
                    z, e = check_sph_ortho_c(l1, m1, l2, m2)
                    print(l1, m1, l2, m2, abs(z), abs(e),
                          "Pass = ", test_ortho(l1, m1, l2, m2, z))

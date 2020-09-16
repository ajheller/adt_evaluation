#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 09:21:47 2014

@author: heller
"""

# This file is part of the Ambisonic Decoder Toolbox (ADT)
# Copyright (C) 2014-19  Aaron J. Heller <heller@ai.sri.com>
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

## BEWARE: there are many conventions in use for the spherical harmonics


def real_sph_harm(l, m, theta, phi, phi_is_elevation=False, cs_phase=False):
    """
    Compute real spherical harmonics, Y_lm.

    Parameters
    ----------
        l: array-like
           spherical harmonic degree, 0 <= l
        m: array-like
           spherical harmonic order, -l <= m <= l
        theta: array-like
           azimuth, counter-clockwise angle about the +z axis from the +x axis
        phi: array-like
           zenith angle, angle from the +z axis
        phi_is_elevation: boolean
           interpret phi as elevation, angle from x-y plane
        cs_phase: boolean
           include condon-shortly phase (default=False)

    basically this translates from the complex spherical harmonics to real

    """

    if phi_is_elevation:
        phi = pi/2 - phi

    # args to sph_harm are order, degree, azimuth, zenith_angle
    abs_m = np.abs(m)
    Y = special.sph_harm(abs_m, l, theta, phi)

    # At thist point Y could be a scalar or a Numpy ndarray. In the Pythonic
    # EAFP style, we should assume it's an array and handle the scalar case in
    # a TypeError exception.
    # However, we want to make sure that the unit tests use the same code as as
    # production calls, so we test Y to see if it scalar and make it a
    # 1x1 ndarray and later return the value of element [0,0]

    Y_scalar = np.isscalar(Y)
    if Y_scalar:
        Y = np.array(((Y,),))
        Y_scalar = True

    if not cs_phase:
        # the textbook definition
        #  Y *= (-1)**abs_m
        # equivalent but faster
        Y[abs_m % 2 != 0] *= -1

    Y_real = np.real(Y)
    Y_real[m > 0] *= sqrt(2)
    Y_real[m < 0] = sqrt(2) * np.imag(Y[m < 0])

    if Y_scalar:
        return Y_real[0, 0]
    else:
        return Y_real


def lm_broadcast(l, m, theta, phi,
                 transpose=False,
                 return_ml=False):
    """Produce 2D array-like objects where each row is constant l,m pair and
    each column is a constant theta, phi pair.
    """
    if transpose:
        a_m, a_theta = meshgrid(m, theta)
        a_l, a_phi  = meshgrid(l, phi)
    else:
        a_theta, a_m = meshgrid(theta, m)
        a_phi, a_l = meshgrid(phi, l)

    if return_ml:
        return a_m, a_l, a_theta, a_phi
    else:
        return a_l, a_m, a_theta, a_phi


def real_sph_harm_transform(l, m, az, el, cs_phase=False):
    return real_sph_harm(*lm_broadcast(l, m, theta=az, phi=el),
                         phi_is_elevation=True, cs_phase=cs_phase)


# ---------------------- Unit tests --------------------------- #


# dblquad is the quadpack function for 2D integration
#  are we calling it correctly?
def dblquad_test():
    # should be 1/3 * 4*pi = 4.1887902047863905
    #  note this is the random energy efficiency of a figure-8 microphone
    qq, ee = dblquad(
        lambda y, x: cos(y)**2 * sin(y),
        0, 2*pi,
        lambda x: 0, lambda y: pi)

    return qq, ee


def lm_generator(max_degree: int = 3, pred = lambda l, m: True):
    return ((l, m)
            for l in range(max_degree+1)
            for m in range(-l, l+1)
            if pred(l, m))


def real_sph_harm_inner_product(l1, m1, l2, m2):
    return dblquad(
        # arguments to lambda need to be reversed from args to dblquad (wtf?)
        lambda phi, theta:
            real_sph_harm(l1, m1, theta, phi) *
            real_sph_harm(l2, m2, theta, phi) *
            sin(phi),
        0, 2*pi,  # range of theta
        lambda x: 0, lambda y: pi  # range of phi
        )


def ortho_test_real(max_degree=3):
    all_ok = True
    for l1 in range(0, max_degree+1):
        for m1 in range(-l1, l1+1):
            for l2 in range(0, max_degree+1):
                for m2 in range(-l2, l2+1):
                    z, e = real_sph_harm_inner_product(l1, m1, l2, m2)
                    ok = check_inner_product(l1, m1, l2, m2, z)
                    all_ok &= ok
                    print(l1, m1, l2, m2, abs(z), abs(e), "Pass=", ok)
    return all_ok

def check_inner_product(l1, m1, l2, m2, z, e=1e-10):
    if l1 == l2 and m1 == m2:
        # should be 1
        test = abs(z-1) < abs(e)
    else:
        # should be 0
        test = abs(z) < abs(e)
    return test


#
# same tests but using spherical designs to do the quadrature

import spherical_grids as sg


def real_sph_harm_inner_product_discrete(l1, m1, l2, m2,
                                         grid=sg.t_design5200()):
    if True:
        Y1 = real_sph_harm_transform(l1, m1, grid.az, grid.el)
        Y2 = real_sph_harm_transform(l2, m2, grid.az, grid.el)
        s = (grid.w * Y1 @ Y2.T)[0, 0]
    else:
        # take advantage of broadcasting
        Y = real_sph_harm_transform((l1, l2), (m1, m2), grid.az, grid.el)
        s = grid.w * Y[0, :] @ Y[1, :].T
    return s


def rsh_transform_unit_test(max_degree=3, grid=sg.t_design5200()):
    l, m = zip(*lm_generator(max_degree))
    Y = real_sph_harm_transform(l, m, grid.az, grid.el)
    # compute the grammian
    G = grid.w * Y @ Y.T.conj()
    # should be 1 on the diagonal, 0 off.
    max_error = np.max(np.abs(G - np.identity(G.shape[0])))
    return max_error < 1e-8, max_error, G


# -------------- validate condon-shortley phase --------------- #
#  TODO: work in progress

# at the equator (pi/2 in scipy implementaion)
#   zero crossing for sin (negative m) components at zero should be +
#   zero corssing for cos (positive m) components at -pi/2 should be +
def check_condon_shortley_phase_real(l, m, delta=1e-4):
    if not m > 0:
        return 0
    if m < 0:
        theta = 0
        phi = pi/2
    elif m > 0:
        theta = -pi/2
        phi = pi/2

    ok = (real_sph_harm(l, m, theta+delta, phi) -
          real_sph_harm(l, m, theta-delta, phi)) > 0
    return ok


def plot_cs_phase(max_degree=3):
    import matplotlib.pyplot as plt
    az_range = np.linspace(-pi/8, pi/8, 50)
    el_range = np.zeros_like(az_range)
    lm = [(l, m)
          for l in range(max_degree+1)
          for m in range(-l, l+1)
          if m != 0]
    l, m = zip(*lm)

    Yf = real_sph_harm_transform(l, m, az_range, el_range, cs_phase=False).T
    Yt = real_sph_harm_transform(l, m, az_range, el_range, cs_phase=True).T

    plt.plot(az_range, Yf)
    plt.legend(lm, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    plt.plot(az_range, Yt)
    plt.legend(lm, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    return Yf, Yt, lm


def plot_associated_legendre(max_degree, polar_plot=False):
    import matplotlib.pyplot as plt
    el_range = np.linspace(-pi/2, pi/2, 100)
    lm = list(lm_generator(max_degree, lambda l, m: m == 0))
    for l, m in lm:
        #alp = special.lpmv(np.abs(m), l, np.sin(el_range))
        alp = real_sph_harm(l, m, 0, el_range)
        if polar_plot:
            plt.polar(el_range[alp >= 0], alp[alp >= 0])
            plt.polar(el_range[alp < 0]-pi, -alp[alp < 0], '-.')
        else:
            plt.plot(el_range, alp)
    plt.legend(lm)
    plt.show()
    return alp

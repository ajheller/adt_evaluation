#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:29:51 2018

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


from __future__ import division
import numpy as np
from numpy import pi

from collections import namedtuple

#import cPickle as pickle
import pickle


# these follow the MATLAB convention for spherical coordinates


def cart2sph(x, y, z):
    r_xy = np.hypot(x, y)
    r = np.hypot(r_xy, z)
    el = np.arctan2(z, r_xy)
    az = np.arctan2(y, x)
    return az, el, r


def sph2cart(az, el, r=1):
    z = r * np.sin(el) * np.ones_like(az)
    r_cos_el = r * np.cos(el)
    x = r_cos_el * np.cos(az)
    y = r_cos_el * np.sin(az)
    return x, y, z

# these follow the physics convention of zenith angle, azimuth

def sphz2cart(zen, az, r=1):
    return sph2cart(az, pi/2-zen, r)

Grid = namedtuple('Grid', ('ux', 'uy', 'uz', 'u', 'az', 'el', 'w', 'shape'))


def az_el(resolution=180):
    u = np.linspace(-pi, pi, (2 * resolution) + 1)
    v = np.linspace(-pi/2, pi/2, resolution + 1)
    el, az = np.meshgrid(v, u)

    ux, uy, uz = sph2cart(az, el)

    # quadrature weights (sum to 4pi, area of a unit sphere)
    du = np.abs(u[1]-u[0])
    dv = np.abs(v[1]-v[0])
    w = np.cos(el) * du * dv
    # the last row overlaps the first so set it's weights to zero
    w[-1, :] = 0

    return Grid(ux, uy, uz,
                np.vstack([x.ravel() for x in (ux, uy, uz)]),
                az, el,
                w, el.shape)

    #return x, y, z, az, el, w


def az_el_unit_test(resolution=1000):
    g = az_el(resolution)
    sw = np.sum((g.ux**2 + g.uy**2 + g.uz**2) * g.w) / (4 * pi)
    sx = np.sum(g.ux**2 * g.w) / (1/3) / (4 * pi)
    return sw, sx  # both should be 1


# http://neilsloane.com/sphdesigns/
def load_t_design_cart(file, four_pi=True):
    "load spherical t-designs from a file of unit vectors, x, y, z"
    with open(file, 'r') as f:
        t = np.fromfile(f, dtype=np.float64, sep=" ").reshape(-1, 3)
    ux = t[:, 0]
    uy = t[:, 1]
    uz = t[:, 2]
    az, el, _ = cart2sph(ux, uy, uz)  # r is 1
    w = np.ones(ux.shape) / ux.shape[0]
    if four_pi:
        w *= 4 * pi

    return Grid(ux, uy, uz,
                np.vstack([q.ravel() for q in (ux, uy, uz)]),
                az, el,
                w, w.shape)


# https://www-user.tu-chemnitz.de/~potts/workgroup/graef/computations/pointsS2.php.en
def load_t_design_sph(file, four_pi=True):
    "load spherical t-designs from a file of azimuths and zenith angles"
    with open(file, 'r') as f:
        t = np.fromfile(f, dtype=np.float64, sep=" ").reshape(-1, 2)
    az = t[:, 0]
    # convert from zenith angle to elevation
    el = pi/2 - t[:, 1]
    ux, uy, uz = sph2cart(az, el)
    w = np.ones(ux.shape) / ux.shape[0]
    if four_pi:
        w *= 4 * pi

    return Grid(ux, uy, uz,
                np.vstack((ux, uy, uz)),
                az, el,
                w, w.shape)


# short cuts for the two spherical designs I use most frequently
def t_design240(four_pi=True):
    return load_t_design_cart("data/des.3.240.21.txt", four_pi)


def t_design5200(four_pi=True):
    return load_t_design_sph("data/Design_5200_100_random.dat.txt", four_pi)

# also http://web.maths.unsw.edu.au/~rsw/Sphere/EffSphDes/ss.html


def spherical_cap(T, u, angle):
    """return boolean array of points in T within angle of unit vector u

    Inputs:
        T - a spherical grid object
        u - unit vector for center of cap
        a - angular extent of cap
    Outputs:
        boolean array of points in the cap
        index of point closest to u
        error of point
    """

    p = np.dot(u, T.u).squeeze()

    c_max = np.argmax(p)

    a_err = u - T.u[:, c_max]

    c = p > np.arccos(angle)

    return c, c_max, a_err, 2*np.arcsin(np.linalg.norm(a_err)/2)

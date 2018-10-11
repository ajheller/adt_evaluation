#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:29:51 2018

@author: heller
"""

from __future__ import division
import numpy as np
from numpy import pi

from collections import namedtuple

import cPickle as pickle


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
def t_design(four_pi=True):
    with open("data/des.3.240.21.txt", 'r') as f:
        t = np.fromfile(f, dtype=np.float64, sep=" ").reshape(-1, 3)
    x = t[:, 0]
    y = t[:, 1]
    z = t[:, 2]
    az, el, w = cart2sph(x, y, z)
    w /= x.shape[0]
    if four_pi:
        w *= 4 * pi
    #return x, y, z, az, el, w
    return Grid(x, y, z,
                np.vstack([q.ravel() for q in (x, y, z)]),
                az, el,
                w, w.shape)


# https://www-user.tu-chemnitz.de/~potts/workgroup/graef/computations/pointsS2.php.en
def t_design5200(four_pi=True):
    with open("data/Design_5200_100_random.dat.txt", 'r') as f:
        t = np.fromfile(f, dtype=np.float64, sep=" ").reshape(-1, 2)
    az = t[:, 0]
    # convert from zenith angle to elevation
    el = pi/2 - t[:, 1]
    x, y, z = sph2cart(az, el)
    w = np.ones(x.shape) / x.shape[0]
    if four_pi:
        w *= 4 * pi

    return Grid(x, y, z,
                np.vstack((x, y, z)),
                az, el,
                w, w.shape)

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

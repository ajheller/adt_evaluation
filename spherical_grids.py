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

import warnings
from collections import namedtuple

import numpy as np
from numpy import pi

# import cPickle as pickle

"""
In NumPy and MATLAB, grids create a N-dimensional coordinate space used for
evaluating functions over the space. These functions


Note: there are many different conventions for spherical coordinates and almost
all writeups found on the web use zenith angle (angle from the Z-axis) for the
vertical angle.

Ambisonics follows the MATLAB convention using elevation angle (angle from X-Y
plane) for the vertical angle.  See discussions here:
    http://mathworld.wolfram.com/SphericalCoordinates.html
and
    http://pcfarina.eng.unipr.it/Aurora/HOA_explicit_formulas.htm
"""


# these follow the MATLAB convention for spherical coordinates
def cart2sph(x, y, z):
    """Convert from Cartesian to spherical coordinates, using MATLAB convention
    for spherical coordinates.

    Parameters
    ----------
        x, y, z: array-like
           Cartesian coordinates

    Returns
    -------
        az, el, r: nd-array
            azimuth, elevation (radians)
            radius (input units)

    """
    r_xy = np.hypot(x, y)
    r = np.hypot(r_xy, z)
    el = np.arctan2(z, r_xy)
    az = np.arctan2(y, x)
    return az, el, r


def sph2cart(az, el, r=1):
    """Convert from spherical to Cartesian coordinates, using MATLAB convention
    for spherical coordinates.

    Parameters
    ----------
        az, e, r: ndarray
            azimuth, elevation (radians)
            radius (input units), optional (default 1)

    Returns
    -------
        x, y, z: ndarray
           Cartesian coordinates (in input units)

    Notes
    -----
        The function is vectorized and return valuew will be the same shape as
        the inputs.

    """
    z = r * np.sin(el) * np.ones_like(az)
    r_cos_el = r * np.cos(el)
    x = r_cos_el * np.cos(az)
    y = r_cos_el * np.sin(az)
    return x, y, z


# these follow the physics convention of zenith angle, azimuth
def sphz2cart(zen, az, r=1):
    "Spherical to cartesian using Physics conventxion, e.g. Arfkin"
    return sph2cart(az, pi / 2 - zen, r)


def cart2sphz(x, y, z):
    """Cartesian to spherical using Physics convention

    Parameters
    ----------
        x, y, z: ndarray
           Cartesian coordinates (in input units)


    Returns
    -------
        zentih angle: ndarray
            angle from +z-axis in radians

        azimuth: ndarray
            angle from the x-axis in the x-y plane in radians

        radius: ndarray
           distance from the origin in input units
    """
    az, el, r = cart2sph(x, y, z)
    return (pi / 2 - el), az, r


#  Grid
Grid = namedtuple('Grid', ('ux', 'uy', 'uz', 'u', 'az', 'el', 'w', 'shape'))
Grid.__doc__ = """ \
Immutable collection of points in S^2, the surface of the 3-D unit sphere

    Fields
    ------
        ux, uy, uz: ndarray
            x, y, z components of the unit vectors
        u: 3xN ndarray
            the unit vectors as columns of an array
        az, el: ndarray
            azimuth and elevations of the unit vectors (radians)
        w: ndarray
            quadrature weights of the points
        shape: tuple
            shape of the grid

    Note
    ----
        The factory functions must fill in all of the fields
"""


#  geodetic
def az_el(resolution=180):
    """
    return a grid with equal sampling in azimuth and elevation
    """
    u = np.linspace(-pi, pi, (2 * resolution) + 1)
    v = np.linspace(-pi / 2, pi / 2, resolution + 1)
    el, az = np.meshgrid(v, u)

    ux, uy, uz = sph2cart(az, el)

    # quadrature weights (sum to 4pi, area of a unit sphere)
    du = np.abs(u[1] - u[0])
    dv = np.abs(v[1] - v[0])
    w = np.cos(el) * du * dv
    # the last row overlaps the first so set it's weights to zero
    w[-1, :] = 0

    return Grid(ux, uy, uz,
                np.vstack([x.ravel() for x in (ux, uy, uz)]),
                az, el,
                w, el.shape)


def az_el_unit_test(resolution=1000):
    g = az_el(resolution)
    sw = np.sum((g.ux ** 2 + g.uy ** 2 + g.uz ** 2) * g.w) / (4 * pi)
    sx = np.sum(g.ux ** 2 * g.w) / (1 / 3) / (4 * pi)
    return sw, sx  # both should be 1


#           -------------- Spherical Designs --------------
#  Spherical design: a finite set of N points on the d-dimensional unit
#  d-sphere S^d such that the average value of any polynomial f of degree t or
#  less on the set equals the average value of f on the whole sphere (that is,
#  the integral of f over S^d divided by the area or measure of S^d).
#
#  https://en.wikipedia.org/wiki/Spherical_design
#  http://mathworld.wolfram.com/SphericalDesign.html


# http://neilsloane.com/sphdesigns/
def load_t_design_cart(file, four_pi=True):
    "load spherical t-designs from a file of unit vectors, x, y, z"
    with open(file, 'r') as f:
        t = np.fromfile(f, dtype=np.float64, sep=" ").reshape(-1, 3)
    ux = t[:, 0]
    uy = t[:, 1]
    uz = t[:, 2]
    az, el, _r = cart2sph(ux, uy, uz)  # _r so linters don't flag as unused
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
    el = pi / 2 - t[:, 1]
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


# --------------- Spherical Cap -------------------
#   http://mathworld.wolfram.com/SphericalCap.html
#   https://en.wikipedia.org/wiki/Spherical_cap

axis_names = dict(x=(1.0, 0.0, 0.0),
                  y=(0.0, 1.0, 0.0),
                  z=(0.0, 0.0, 1.0))


def spherical_cap(T, u, angle, min_angle=0):
    """return boolean array of points in T within angle of unit vector u

    Parameters
    ----------

        T: collection of unit vectors
            a spherical grid object

        u: vector-like or string
            unit vector for center of cap or axis name

        angle: float
            angular extent of cap in radians

        min_angle: float
            start of cap in radians to create a spherical frustrum

    Returns
    -------
        boolean array of points in the cap

        index of point closest to u

        error of point

    Example
    -------

    References
    ----------
        http://mathworld.wolfram.com/SphericalCap.html

        https://en.wikipedia.org/wiki/Spherical_cap
    """

    # convert axis name to unit vector
    try:
        u = axis_names[u.lower()]
    except AttributeError or KeyError:
        pass

    # check that u is a unit vector
    norm = np.sqrt(np.dot(u, u))
    if not np.isclose(norm, 1):
        warnings.warn("u is not a unit vector, normalizing")
        u = u / norm

    # retrieve unit vectors
    try:
        Tu = T.u
    except AttributeError as ae:
        Tu = T

    # if Tu is not not compatible with u, try the transpose, still can fail in
    # np.dot()
    if len(u) != len(Tu):
        Tu = Tu.transpose()

    # compute the cap
    p = np.dot(u, Tu).squeeze()
    c = (p >= np.cos(angle)) & (p <= np.cos(min_angle))

    # find the index of the element of T closest to u and compute the error
    c_max = np.argmax(p)
    a_err = u - Tu[:, c_max]

    return c, c_max, a_err, 2 * np.arcsin(np.linalg.norm(a_err) / 2)

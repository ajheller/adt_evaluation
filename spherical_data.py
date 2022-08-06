#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 17:13:07 2020

@author: heller
"""
# This file is part of the Ambisonic Decoder Toolbox (ADT)
# Copyright (C) 2018-20  Aaron J. Heller <heller@ai.sri.com>
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

import warnings
from dataclasses import dataclass, field

import numpy as np

from scipy import interpolate

# cached_property is only in 3.8+
#  backport available at https://pypi.org/project/backports.cached-property/
try:
    from functools import cached_property
except ImportError:
    try:
        from backports.cached_property import cached_property
    except ModuleNotFoundError as ie:
        print("run 'pip install backports.cached-property' and try again")
        raise ie

from numpy import pi
from numpy import pi as π


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
        az, el, r: ndarray
            azimuth, elevation (radians)
            radius (input units), optional (default 1)

    Returns
    -------
        x, y, z: ndarray
           Cartesian coordinates (in input units)

    Notes
    -----
        The function is vectorized and return values will be the same shape as
        the inputs.

    """
    z = r * np.sin(el) * np.ones_like(az)
    r_cos_el = r * np.cos(el)
    x = r_cos_el * np.cos(az)
    y = r_cos_el * np.sin(az)
    return x, y, z


# these follow the physics convention of zenith angle, azimuth
def sphz2cart(zen, az, r=1):
    """Spherical to cartesian using Physics conventxion, e.g. Arfkin."""
    return sph2cart(az, pi / 2 - zen, r)


def cart2sphz(x, y, z):
    """Cartesian to spherical using Physics convention.

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


# --------------- Spherical Cap -------------------
#   http://mathworld.wolfram.com/SphericalCap.html
#   https://en.wikipedia.org/wiki/Spherical_cap


axis_names = dict(x=(1.0, 0.0, 0.0), y=(0.0, 1.0, 0.0), z=(0.0, 0.0, 1.0))


def spherical_cap(T, u, angle, min_angle=0):
    """Return boolean array of points in T within 'angle' of unit vector 'u'.

    Parameters
    ----------
        T: collection of unit vectors
            a spherical grid object

        u: vector-like or string
            unit vector for center of cap or axis name

        angle: float
            angular extent of cap in radians

        min_angle: float
            start of cap in radians to create a spherical frustum

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
    #
    # convert axis name to unit vector
    try:
        u = axis_names[u.lower()]
    except AttributeError or KeyError:
        pass

    # u needs to be a unit vector, normalize if not
    norm = np.sqrt(np.dot(u, u))
    if not np.isclose(norm, 1):
        warnings.warn("u is not a unit vector, normalizing")
        u = u / norm

    # retrieve unit vectors
    try:
        Tu = T.u
    except AttributeError:
        Tu = T

    # if Tu is not compatible with u, try its transpose
    # this still can fail in np.dot()
    if len(u) != len(Tu):
        Tu = Tu.transpose()

    # compute the cap
    p = np.dot(u, Tu).squeeze()
    c = (p >= np.cos(angle)) & (p <= np.cos(min_angle))

    # find the index of the element of T closest to u and compute the error
    c_max = np.argmax(p)
    a_err = u - Tu[:, c_max]

    return c, c_max, a_err, 2 * np.arcsin(np.linalg.norm(a_err) / 2)


def spherical_interp(T, u, angle, r, **kwargs):
    """
    Make a surface of revolution from the curve r(angle) about the axis u.
    """
    try:
        u = axis_names[u.lower()]
    except AttributeError or KeyError:
        pass

    # u needs to be an array of unit vectors, normalize if not
    norm = np.sqrt(np.dot(u, u))
    if not np.isclose(norm, 1):
        warnings.warn("u is not a unit vector, normalizing")
        u = u / norm

    # retrieve unit vectors
    try:
        Tu = T.u
    except AttributeError:
        Tu = T

    # if Tu is not compatible with u, try its transpose
    # this still can fail in np.dot()
    if len(u) != len(Tu):
        Tu = Tu.transpose()

    # the interpolator
    f = interpolate.interp1d(angle, r, **kwargs)

    # angle with axis of each point in the t-design
    p = np.arccos(np.dot(u, Tu).squeeze())

    return f(p)


#
# ---- the class definition ----


@dataclass
class SphericalData:
    """A class to hold spherical data and provide conversions."""

    x: np.ndarray = field(default_factory=lambda: np.array(None))
    y: np.ndarray = field(default_factory=lambda: np.array(None))
    z: np.ndarray = field(default_factory=lambda: np.array(None))
    name: str = "data"

    _primary_attrs = ["x", "y", "z", "name"]

    # def __init__(self, name="data"):
    #     self.xyz = None
    #     self.name = name

    # TODO: How much do we want to keep users from shooting themselves in the
    #  foot? The current implementation raises an AttributeError if the user
    #  tries to set any attributes other than those explicitly called out in
    #  _primary attrs.
    def __setattr__(self, name, value):
        if name in self._primary_attrs:
            super().__setattr__(name, value)
            # print(f'clearing caches: {name} <-- {value}')
            self._clear_cached_properties()
        elif name in ("xyz", "cart"):
            return self.set_from_cart(*value)
        else:
            raise AttributeError

    def __str__(self):
        return f"{__class__.__name__}({self.name})"

    def _clear_cached_properties(self):
        # need a persistant copy because we're deleting keys in the loop
        keys = list(self.__dict__.keys())
        for key in keys:
            if key not in self._primary_attrs:
                # print(f"   clearing: {key}")
                delattr(self, key)

    def set_from_cart(self, x, y, z):
        if not (x.shape == y.shape == z.shape):
            raise ValueError("x, y, z not the same shape.")
        else:
            self.x = x
            self.y = y
            self.z = z
            self._clear_cached_properties()
            return self

    def set_from_sph(self, theta, phi, rho=1, phi_is_zenith=False):
        if phi_is_zenith:
            phi = pi / 2 - phi
        return self.set_from_cart(*sph2cart(theta, phi, rho))

    def set_from_aer(self, az, el, r=1):
        self.set_from_sph(az, el, r, phi_is_zenith=False)
        return self

    # TODO: implement this
    def set_from_cyl(self, theta, rho, z):
        raise NotImplementedError
        return self

    @property
    def cart(self):
        """Return x, y, and z components in a tuple."""
        return self.x, self.y, self.z

    @cached_property
    def xyz(self):
        """Return x, y, z as an iterable container of vectors."""
        return np.c_[self.x.ravel(), self.y.ravel(), self.z.ravel()]

    @cached_property
    def u(self):
        """unit vectors as an iterable container."""
        return (
            self.xyz
            /
            # this makes a column vector without copying
            np.linalg.norm(self.xyz, axis=1)[None].T
        )

    @cached_property
    def sph(self):
        """azimuth, elevation, radius as a tuple."""
        return cart2sph(self.x, self.y, self.z)

    @cached_property
    def sphz(self):
        """azimuth, zenith angle, radius as a tuple."""
        return cart2sphz(self.x, self.y, self.z)

    @property
    def x0(self):
        """Return x flattened."""
        return self.x.ravel()

    @property
    def y0(self):
        """Return y flattened."""
        return self.y.ravel()

    @property
    def z0(self):
        """Return z flattened."""
        return self.z.ravel()

    @property
    def az(self):
        return self.sph[0]

    @property
    def el(self):
        return self.sph[1]

    @property
    def r(self):
        return self.sph[2]

    @property
    def shape(self):
        return self.x.shape

    def unravel(self, x):
        return x.reshape(self.shape)

    def angle_from_axis(self, u=(0, 0, 1)):
        """
        Return an array with angles from a the specified unit vector.

        Parameters
        ----------
        u : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # allow magic axis names
        try:
            u = axis_names[u.lower()]
        except AttributeError or KeyError:
            pass

        # make sure u is a unit vector
        u /= np.linalg.norm(u)
        # need clip due to round-off error
        return np.arccos(np.clip(np.dot(self.u, u), -1, 1))

    def cap(self, u, angle, **kwargs):
        return spherical_cap(self.u, u, angle, **kwargs)

    def interp_el(self, u, phi, r, *args, **kwargs):
        f = interpolate.interp1d(phi, r, *args, **kwargs)
        w = f(self.angle_from_axis(u))
        return w


def from_cart(*args, sd_class=SphericalData, **kwargs):
    return sd_class().set_from_cart(*args, **kwargs)


def from_sph(*args, sd_class=SphericalData, **kwargs):
    return sd_class().set_from_sph(*args, **kwargs)


# %%
def unit_test():
    import spherical_grids

    sg240 = spherical_grids.t_design240()
    q = from_sph(sg240.az, sg240.el)
    u = q.u
    v = q.az
    w = q.sph
    q.set_from_cart(*np.asarray(((1, 0, 2), (0, 1, 0), (0, 0, 1))).T)
    # cache should be cleared
    if q.u == u:
        print("FAIL: caches were not cleared!!")
    return q, (u, v, w)

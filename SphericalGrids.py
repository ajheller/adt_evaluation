#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 17:13:07 2020

@author: heller
"""

import numpy as np
from numpy import pi
from dataclasses import dataclass, field

import SphericalData as sd


@dataclass
class SphericalGrid(sd.SphericalData):
    """A class to support regular sampling in S^2."""

    # quadrature weight
    w: np.ndarray = field(default_factory=lambda: np.array(None))

    _primary_attrs = ['x', 'y', 'z', 'w', 'name']

#           -------------- Spherical Designs --------------
#  Spherical design: a finite set of N points on the d-dimensional unit
#  d-sphere S^d such that the average value of any polynomial f of degree t or
#  less on the set equals the average value of f on the whole sphere (that is,
#  the integral of f over S^d divided by the area or measure of S^d).
#
#  https://en.wikipedia.org/wiki/Spherical_design
#  http://mathworld.wolfram.com/SphericalDesign.html


# factory functions
# http://neilsloane.com/sphdesigns/
def load_t_design_cart(file, four_pi=True):
    "load spherical t-designs from a file of unit vectors, x, y, z"
    with open(file, 'r') as f:
        t = np.fromfile(f, dtype=np.float64, sep=" ").reshape(-1, 3)
    w = np.ones_like(t[:, 0]) / t[:, 0].size
    if four_pi:
        w *= 4 * pi

    grid = sd.from_cart(*(t.T), sd_class=SphericalGrid)
    grid.w = w
    grid.name = 'T-design'

    return grid


# https://www-user.tu-chemnitz.de/~potts/workgroup/graef/computations/pointsS2.php.en
def load_t_design_sph(file, four_pi=True):
    "load spherical t-designs from a file of azimuths and zenith angles"
    with open(file, 'r') as f:
        t = np.fromfile(f, dtype=np.float64, sep=" ").reshape(-1, 2)

    w = np.ones_like(t[:, 0]) / t[:, 0].size
    if four_pi:
        w *= 4 * pi

    grid = sd.from_sph(*(t.T), phi_is_zenith=True,
                                sd_class=SphericalGrid)
    grid.w = w
    grid.name = 'T-design'

    return grid


# short cuts for the two spherical designs I use most frequently
def t_design240(four_pi=True):
    """Return Sloane's 240 3-Design"""
    return load_t_design_cart("data/des.3.240.21.txt", four_pi)


def t_design5200(four_pi=True):
    return load_t_design_sph("data/Design_5200_100_random.dat.txt", four_pi)



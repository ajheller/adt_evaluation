#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 17:13:07 2020

@author: heller
"""

import numpy as np
from numpy import pi
from dataclasses import dataclass, field

import SphericalData as SphD


@dataclass
class SphericalGrid(SphD.SphericalData):
    """A class to support regular sampling in S^2."""

    # quadrature weights, dΩ
    w: np.ndarray = field(default_factory=lambda: np.array(0, dtype=np.float))

    _primary_attrs = ['x', 'y', 'z', 'w', 'name']

    @property
    def dΩ(self):
        return self.w

    def surface_integral(self, f, coord_type='cart'):
        if 'cart' == coord_type:
            args = (self.x, self.y, self.z)
        elif 'sph' == coord_type:
            args = (self.az, self.el)
        else:
            raise ValueError(f'Unknown type: {coord_type}')
        return np.sum(f(*args) * self.dΩ)

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
def load_t_design_cart(file, four_pi=False):
    """load spherical t-designs from a file of unit vectors, x, y, z."""
    with open(file, 'r') as f:
        t = np.fromfile(f, dtype=np.float64, sep=" ").reshape(-1, 3)
    w = np.ones_like(t[:, 0]) / t[:, 0].size
    if four_pi:
        w *= 4 * pi
    return SphericalGrid(*t.T, w=w,
                         name='T-design')


# https://www-user.tu-chemnitz.de/~potts/workgroup/graef/computations/pointsS2.php.en
def load_t_design_sphz(file, four_pi=False):
    """Load spherical t-designs from a file of azimuths and zenith angles."""
    with open(file, 'r') as f:
        t = np.fromfile(f, dtype=np.float64, sep=" ").reshape(-1, 2)
    w = np.ones_like(t[:, 0]) / t[:, 0].size
    if four_pi:
        w *= 4 * pi
    return SphericalGrid(*SphD.sphz2cart(t[:, 1], t[:, 0]), w=w,
                         name='T-design-sph')


# short cuts for the two spherical designs I use most frequently
def t_design240(*args, **kwargs) -> SphericalGrid:
    """Return Sloane's 240 3-Design"""
    return load_t_design_cart("data/des.3.240.21.txt",
                              *args, **kwargs)


def t_design5200(*args, **kwargs) -> SphericalGrid:
    return load_t_design_sphz('data/Design_5200_100_random.dat.txt',
                              *args, **kwargs)


# %% unit tests
def unit_tests():
    for t in (t_design240(),
              t_design5200()):
        s = [t.surface_integral(f)
             for f in (lambda x, y, z: 1,      # = 1
                       lambda x, y, z: x**2,   # = 1/3
                       lambda x, y, z: x * y,  # = 0
                       lambda x, y, z: x * z)  # = 0
             ]
        pass_all = np.all(np.isclose(s, (1, 1/3, 0, 0)))
        print(f"{'Pass' if pass_all else 'FAIL'}: {s} {t.name}")

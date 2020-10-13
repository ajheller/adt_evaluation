#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:01:24 2020

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

import numpy as np
from numpy import pi as π
from dataclasses import dataclass, field

import SphericalData as SphD


@dataclass
class SpeakerArray(SphD.SphericalData):
    """A class to represent loudspeaker arrays."""

    is_real: np.array = field(default_factory=lambda: np.array(0,
                                                               dtype=np.bool))
    ids: list = field(default_factory=lambda: [])

    _primary_attrs = ['x', 'y', 'z', 'is_real', 'name', 'ids']

    def append(self, other, is_real=None):
        # TODO: should x, y, z be delegated to the base class?
        self.x = np.append(self.x, other.x)
        self.y = np.append(self.y, other.y)
        self.z = np.append(self.z, other.z)
        self.ids += other.ids
        if is_real is None:
            self.is_real = np.append(self.is_real, other.is_real)
        else:
            # override the is_real field with the optional value
            self.is_real = np.append(self.is_real,
                                     np.full_like(self.is_real, is_real,
                                                  shape=other.shape))
        return self


# TODO: should there be ignore and no-op codes?
# unit codes and conversion factors to meters and radians
to_base = {'R': 1,      # Radians
           'D': π/180,  # Degrees
           'G': π/200,  # Gradians (grad, grade, gons, metric degrees)
           'M': 1,                    # Meters
           'C': 1/100,                # Centimeters
           'I': 2.54/100,           # Inches
           'F': 12 * 2.54/100,        # Feet
           'L': 660 * 12 * 2.54/100,  # furLongs
           'S': 67 * 2.54/100}        # Smoots


def convert_units(quantity, from_unit, to_unit=None):
    """Convert a quantity between two units. No dimensional analysis."""
    q = quantity * to_base[from_unit.upper()]
    if to_unit:
        q /= to_base[to_unit]
    return q


# canonical ordering of coordinates
# coords that made sense: 'XYZ', 'AER', 'ANR', 'ARZ', but...
# we need a unique location for each; horizontal, vertical, radial
# gives: 'XZY', 'AER', 'ANR', 'AZR'
to_canonical = {'X': 0, 'Y': 2, 'Z': 1,
                'A': 0, 'E': 1, 'R': 2,  # Azimuth, Elevation, Radius
                'N': 1}  # zeNith angle (can't use Z)


"""
% TODO: copied from the MATLAB ADT.

%AMBI_MAT2SPKR_ARRAY convert matrix of speaker coordinates to SPKR_ARRAY struct
%   A is nx3 matrix of speaker coordinates
%     if A is a string, read coordinates from a file
%
%   coord_code identifies each coordinate (default: AER)
%     A, E, R = azimuth, elevation, radius
%     N = zeNith angle, angle fron North pole (can't use Z)
%     X, Y, Z = cartesian coordinates in Ambisonic convention:
%                  +/- X = front/back
%                  +/- Y = left/right
%                  +/- Z = up/down
%     U, V, W = cartesian coordinates, but with sign negated
%               (see SPKR_ARRAY_Boardroom for example)
%     can be in any order and mixed, e.g. ARZ for cylindrical.
%
%   unit_code identifies units for each coordinate (default: DDM)
%     D = Degrees
%     R = Radians
%     G = Gradians (aka - gon, grad, grade, metric degree)
%     C = Centimeters
%     M = Meters
%     F = Feet
%     I = Inches
%     S = Smoots
%     L = furLongs
%
%   name optional name of speaker array, defaults to filename
%
"""

def from_array(a, coord_code='AER', unit_code='DDM',
               array_name=None, ids='S', is_real=True):
    # make sure it's an Nx3 numpy array
    a = np.asarray(a).reshape(-1, 3)
    num_spkrs = len(a)

    if np.isscalar(ids):
        ids = [ids+'%02d' % i for i in range(num_spkrs)]
    elif len(ids) != num_spkrs:
        raise ValueError("len(ids) != num_spkrs")

    if np.isscalar(is_real):
        is_real = np.full(num_spkrs, is_real, dtype=np.bool)
    elif len(is_real) != num_spkrs:
        raise ValueError("len(is_real) != num_spkrs")

    if array_name is None:
        array_name = 'Amb' + str(num_spkrs)

    # convert the columns to base units -- meter, radians
    for (col, code) in enumerate(unit_code):
        a[:, col] *= to_base[code[0].upper()]

    # coordinate untangling
    aa = np.zeros_like(a)
    ac = [' '] * 3
    for col, code in enumerate(coord_code):
        c = code[0].upper()
        to_col = to_canonical[c]
        aa[:, to_col] = a[:, col]
        ac[to_col] = c
    ac = ''.join(ac)
    print("ac:", ac)

    # make the SA object
    s = SpeakerArray(ids=ids, is_real=is_real, name=array_name)
    # TODO: is there a slicker way to do this?
    if ac == 'XZY':
        s.set_from_cart(*[aa[:, to_canonical[c]] for c in 'XYZ'])
    elif ac == 'AER':
        s.set_from_aer(*[aa[:, to_canonical[c]] for c in 'AER'])
    elif ac == 'ANR':
        s.set_from_sph(*[aa[:, to_canonical[c]] for c in 'ANR'])
    elif ac == 'AZR':
        s.set_from_cyl(*[aa[:, to_canonical[c]] for c in 'ARZ'])
    else:
        raise NotImplementedError(f'Sorry, {ac} not implemented.')

    return s


# convenience function that takes three vectors (or scalars) of coordinates
def from_vectors(c0, c1, c2, *args, **kwargs):
    if np.isscalar(c1):
        c1 = np.full_like(c0, c1, dtype=np.float)
    if np.isscalar(c2):
        c2 = np.full_like(c0, c2, dtype=np.float)

    if len(c0) == len(c1) == len(c2):
        return from_array(np.column_stack((c0, c1, c2)).astype(np.float),
                          *args, **kwargs)
    else:
        raise ValueError("c0, c1, c2 must be the same length, "
                         f"but were {list(map(len, (c0, c1, c2)))}.")


#
def unit_test():
    """Run tests for this module."""
    from matplotlib import pyplot as plt
    import example_speaker_arrays as esa
    s = esa.stage2017()
    plt.scatter(s.x, s.y, c=s.z, marker='o')
    plt.grid()
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.scatter(s.az*180/π, s.el*180/π, c='white', marker='o')
    for x, y, r, t in zip(s.az*180/π, s.el*180/π, s.r, s.ids):
        plt.text(x, y, t,
                 bbox=dict(facecolor='lightblue', alpha=0.4),
                 horizontalalignment='center',
                 verticalalignment='center')
    plt.grid()
    # plt.colorbar()
    plt.show()

    return s

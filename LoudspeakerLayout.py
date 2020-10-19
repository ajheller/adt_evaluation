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

import json
from dataclasses import dataclass, field
from operator import itemgetter
from collections.abc import Sequence  # for type declarations

import numpy as np
from numpy import pi as π

import SphericalData as SphD


@dataclass
class LoudspeakerLayout(SphD.SphericalData):
    """A class to represent loudspeaker arrays."""

    description: str = ""
    is_real: np.array = field(default_factory=lambda: np.array(0,
                                                               dtype=np.bool))
    ids: list = field(default_factory=lambda: [])

    _primary_attrs = ['x', 'y', 'z', 'is_real', 'name', 'ids', 'description']

    def __add__(self, other):
        """Append two layouts."""
        return append_layouts(self, other)

    def set_is_real(self, is_real=True):
        """Set the is_real attribute of self, broadcasting if necessary."""
        try:
            # assume it is a sequence
            if len(is_real) == len(self.is_real):
                self.is_real = is_real
            else:
                raise ValueError("len(is_real) != len(self.is_real)")
        except TypeError:
            # assume it is a scalar
            self.is_real = np.full(self.shape, is_real, dtype=np.bool_)
        return self

    def set_imaginary(self):
        """Set the is_real attribute of the array to False."""
        self.set_is_real(False)
        return self

    def to_json(self, channels=None, gains=None):
        """Return a JSON dictionary in IEM plugin format."""
        #
        if channels is None:
            channels = range(1, len(self.ids) + 1)  # 1-based
        if gains is None:
            gains = np.ones_like(self.az)

        # IEM format documented here:
        #  https://plugins.iem.at/docs/configurationfiles/#the-loudspeakerlayout-object

        records = zip(self.az*180/π, self.el*180/π, self.r,
                      map(bool, ~self.is_real),  # json needs bool not bool_
                      channels, gains,
                      self.ids)
        columns = ('Azimuth', 'Elevation', 'Radius',
                   'IsImaginary', 'Channel', 'Gain',
                   'id')
        ls_dict = [dict(zip(columns, rec)) for rec in records]
        return dict({"LoudspeakerLayout":
                     {"Name": self.name,
                      "Description": self.description,
                      "Loudspeakers": ls_dict}})

    def to_iem_file(self, file, **kwargs):
        """Write LoudspeakerLayout to IEM JSON file."""
        with open(file, 'w') as f:
            json.dump(obj=self.to_json(**kwargs), indent=4,
                      fp=f)


def append_layouts(l1, l2,
                   name=None, description=None):
    """Append two layouts."""
    #
    if name is None:
        name = l1.name
    if description is None:
        description = l1.description

    xyz = np.append(l1.xyz, l2.xyz, axis=0)
    ids = np.append(l1.ids, l2.ids, axis=0)
    is_real = np.append(l1.is_real, l2.is_real, axis=0)

    l3 = LoudspeakerLayout(*xyz.T, name=name, description=description,
                           ids=ids, is_real=is_real)
    return l3


#
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


def from_array(a: Sequence,
               coord_code: Sequence = 'AER',
               unit_code: Sequence = 'DDM',
               name: str = None,
               description=None,
               ids='S', is_real=True) -> LoudspeakerLayout:
    """

    :type unit_code: object
    """
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

    if name is None:
        name = 'Amb' + str(num_spkrs)

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
    s = LoudspeakerLayout(ids=ids, is_real=is_real,
                          name=name, description=description)
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
def from_vectors(c0, c1, c2, **kwargs) -> LoudspeakerLayout:
    if np.isscalar(c1):
        c1 = np.full_like(c0, c1, dtype=np.float)
    if np.isscalar(c2):
        c2 = np.full_like(c0, c2, dtype=np.float)

    if len(c0) == len(c1) == len(c2):
        return from_array(np.column_stack((c0, c1, c2)).astype(np.float),
                          **kwargs)
    else:
        raise ValueError("c0, c1, c2 must be the same length, "
                         f"but were {list(map(len, (c0, c1, c2)))}.")


def from_iem_file(file):
    """Load a layout from an IEM-format file."""
    obj = json.load(open(file, 'r'))
    lsl_dict = obj["LoudspeakerLayout"]

    name = lsl_dict['Name']
    description = lsl_dict['Description']
    az, el, r, is_imaginary, channel, gain, ids = \
        zip(*[itemgetter(*('Azimuth', 'Elevation', 'Radius',
                           'IsImaginary', 'Channel', 'Gain', 'id'))(ls)
              for ls in lsl_dict["Loudspeakers"]])

    lsl = from_vectors(az, el, r,
                       coord_code='AER', unit_code='DDM',
                       ids=ids, is_real=~np.array(is_imaginary),
                       name=name, description=description)

    return lsl, obj


def unit_test_iem():
    import example_speaker_arrays as esa
    s = esa.stage2017()
    # json dump, read
    s.to_iem_file("test_iem_stage.json")
    lsl = from_iem_file("test_iem_stage.json")

    return lsl


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

    lsl = unit_test_iem()

    return s, lsl

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:28:40 2020

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

from pathlib import Path

import numpy as np
from numpy import pi as π

# adt modules
import LoudspeakerLayout as LSL

# data directory
_data_dir = Path(__file__).parent/"data"


# a single imaginary speakers for AllRAD
def nadir(r=1):
    """Imaginary speaker at the nadir (south pole)."""
    return LSL.from_array((0, 0, -r), coord_code='XYZ', unit_code='MMM',
                          array_name="imaginary speaker at nadir",
                          ids=["*IN"], is_real=False)


def zenith(r=1):
    """Imaginary speaker at the zenith (north pole)."""
    return LSL.from_array((0, 0, r), coord_code='XYZ', unit_code='MMM',
                          array_name="imaginary speaker at zenith",
                          ids=["*IA"], is_real=False)


def polygon(n, radius=1, unit='M', center_spkr=False, *args, **kwargs):
    """Construct regular polygon arrays."""
    az = np.linspace(0 if center_spkr else π/n, 2*π, n, endpoint=False)
    return LSL.from_vectors(az, 0, radius,
                            unit_code='RR'+unit,
                            coord_code='AER',
                            **kwargs)


def stage2017():
    """CCRMA Stage array."""
    return LSL.from_array(
        (
         # == towers 8:
         # theoretical angles, have to be calibrated
         27,     3.9,   216,
         -27,    3.9,   216,
         63,     8,     162,
         -63,    8,     162,
         117,    8,     162,
         -117,   8,     162,
         153,    3.9,   216,
         -153,   3.9,   216,

         # == upper 8
         23,     29,  171,
         -23,    29,  171,
         90,     58,  109,
         -90,    58,  109,
         157,    31,  167,
         -157,   31,  167,
         0,      70,  108,
         180,    70,  108,

         # == ring of 12 (rails)
         9,      4,   237,
         -9,     4,   237,
         45,     6,   187,
         -45,    6,   187,
         81,     8,   131,
         -81,    8,   131,
         99,     8,   130,
         -99,    8,   130,
         135,    6,   185,
         -135,   6,   185,
         171,    4,   238,
         -171,   4,   238,

         # == ring of 14 (lower trusses)
         14,     18,  243,
         -14,    18,  243,
         39,     22,  200,
         -39,    22,  200,
         60,     30,  154,
         -60,    30,  154,
         90,     34,  139,
         -90,    34,  139,
         122,    30,  153,
         -122,   30,  153,
         144,    22,  201,
         -144,   22,  201,
         166,    19,  243,
         -166,   19,  243,

         # == ring of 6 (upper trusses)
         0,      31,  180,
         39,     47,  128,
         -39,    47,  128,
         146,    47,  129,
         -146,   47,  129,
         180,    33,  180,

         # == lower ring of 8 in towers
         27,     -10,   216,
         -27,    -10,   216,
         63,     -14,   162,
         -63,    -14,   162,
         117,    -14,   162,
         -117,   -14,   162,
         153,    -10,   216,
         -153,   -10,   216,
         ),
        coord_code='AER',
        unit_code='DDI',
        array_name='stage',
        ids=(
          'S01', 'S02', 'S03', 'S04',
          'S05', 'S06', 'S07', 'S08',
          'S09', 'S10', 'S11', 'S12',
          'S13', 'S14', 'S15', 'S16',
          'D17', 'D18', 'D19', 'D20', 'D21', 'D22',
          'D23', 'D24', 'D25', 'D26', 'D27', 'D28',
          'D29', 'D30', 'D31', 'D32', 'D33', 'D34', 'D35',
          'D36', 'D37', 'D38', 'D39', 'D40', 'D41', 'D42',
          'D43', 'D44', 'D45', 'D46', 'D47', 'D48',
          'L01', 'L02', 'L03', 'L04',
          'L05', 'L06', 'L07', 'L08',
            )
        )


# TODO: generalize this to load speaker arrays from spreadsheets
def iem_cube():
    """Return the Cube array at IEM."""

    a = np.genfromtxt(_data_dir/"LScoordinates.csv",
                      skip_header=1, names=True, delimiter=',',
                      deletechars='')

    # get column_names from a.dtype
    all_column_names = a.dtype.names
    print(all_column_names)
    column_names = ('x_[m]', 'y_[m]', 'z_[m]')

    # TODO: partse this from the column names
    column_coords = [s[0] for s in column_names]
    column_units = ['m', 'm', 'm']
    column_values = [a[i] for i in column_names]

    s = LSL.from_vectors(*column_values,
                         array_name="IEM_Cube",
                         coord_code=column_coords,
                         unit_code=column_units)
    return s

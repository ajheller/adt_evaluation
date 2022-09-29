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
import loudspeaker_layout as lsl

# data directory
_data_dir = Path(__file__).parent / "data"


# a single imaginary speakers for AllRAD
def nadir(radius=1.0, is_imaginary=True):
    """Speaker at the nadir (south pole)."""
    return lsl.from_array(
        (0, 0, -radius),
        coord_code="XYZ",
        unit_code="MMM",
        name="imaginary speaker at nadir",
        ids=["*IN"],
        is_real=not is_imaginary,
    )


def zenith(radius=1.0, is_imaginary=True):
    """Speaker at the zenith (north pole)."""
    return lsl.from_array(
        (0, 0, radius),
        coord_code="XYZ",
        unit_code="MMM",
        name="imaginary speaker at zenith",
        ids=["*IZ"],
        is_real=not is_imaginary,
    )


def polygon(
    n, *, elevation=0, radius=1.0, unit="M", offset=0.0, center_spkr=False, **kwargs
):
    """Construct regular polygon arrays."""
    az = np.linspace(0, 2 * π, n, endpoint=False) + offset
    if not center_spkr:
        az += π / n
    return lsl.from_vectors(
        az, elevation, radius, unit_code="RR" + unit, coord_code="AER", **kwargs
    )


def home_dome(add_imaginary=True):
    """Nando's home array."""
    a = np.array(
        [
            # fmt: off
         # a regular octagon (LR) at lower level
         [  22.5,  0.0, 1.6],
         [ -22.5,  0.0, 1.6],
         [  67.5,  0.0, 1.6],
         [ -67.5,  0.0, 1.6],
         [ 112.5,  0.0, 1.6],
         [-112.5,  0.0, 1.6],
         [ 157.5,  0.0, 1.6],
         [-157.5,  0.0, 1.6],
         # a regular pentagon at upper level, rotated
         # +-72 + 12, +-144 + 12 (L R C SL SR)
         [  84.0, 45.0, 1.35],
         [ -60.0, 45.0, 1.35],
         [  12.0, 45.0, 1.35],
         [ 156.0, 45.0, 1.35],
         [-132.0, 45.0, 1.35],
            # fmt: on
        ]
    )

    layout = lsl.from_array(
        a=a,
        coord_code="AER",
        unit_code="DDM",
        name="HomeDome",
        ids=("1", "2", "3", "4", "5", "6", "7", "8", "UL", "UR", "UC", "USL", "USR"),
        description="Nando's home array, 8+5",
    )
    if add_imaginary:
        layout += nadir(radius=1.6, is_imaginary=True)
        # lsl += zenith(radius=1.6, is_imaginary=True)

    return layout


# alias for backward compatibility
nando_dome = home_dome


def emb_dome(config=1, add_imaginary=True):
    """EMB's home array."""
    # Eric's array is an octagon at ear level and a square 30-deg elevation
    # speakers lie on a 2-meter sphere
    layout = lsl.append_layouts(
        polygon(8, elevation=0, radius=2, unit="M", center_spkr=False, ids="M"),
        polygon(4, elevation=π / 6, radius=2, unit="M", center_spkr=True, ids="U"),
        name="EMB",
        description="EMB's home array, 8+4",
    )
    if add_imaginary:
        layout += nadir(radius=2)
    return layout


def emb_cmap484(add_imaginary=True):
    """EMB's home array."""
    # Eric's array is an octagon at ear level and a square 30-deg elevation
    # speakers lie on a 2-meter sphere
    layout = lsl.append_layouts(
        polygon(8, elevation=0, radius=2, unit="M", center_spkr=False, ids="M"),
        polygon(4, elevation=π / 6, radius=2, unit="M", center_spkr=True, ids="U"),
        name="CMAP-4c8s4c",
        description="EMB's home array, 8+4+4",
    )

    layout += polygon(
        4, elevation=-π / 6, radius=2, unit="M", center_spkr=True, ids="L"
    )
    if add_imaginary:
        layout += nadir(radius=2)
        layout += zenith(radius=2)
    return layout


def emb_cmap888(add_imaginary=True):
    """EMB's home array."""
    # Eric's array is an octagon at ear level and a square 30-deg elevation
    # speakers lie on a 2-meter sphere
    layout = lsl.append_layouts(
        polygon(8, elevation=0, radius=2, unit="M", center_spkr=False, ids="M"),
        polygon(8, elevation=π / 6, radius=2, unit="M", center_spkr=True, ids="U"),
        name="CMAP-8c8s8c",
        description="EMB's home array, 8+8+8",
    )

    layout += polygon(
        8, elevation=-π / 6, radius=2, unit="M", center_spkr=True, ids="L"
    )
    if add_imaginary:
        layout += nadir(radius=2)
        layout += zenith(radius=2)
    return layout


def emb_cmap686(add_imaginary=True, bottom_center=False):
    """EMB's home array."""
    # Eric's array is an octagon at ear level and a square 30-deg elevation
    # speakers lie on a 2-meter sphere

    if bottom_center:
        name = "CMAP-6c8s6c"
    else:
        name = "CMAP-6c8s6s"

    layout = lsl.append_layouts(
        polygon(8, elevation=0, radius=2, unit="M", center_spkr=False, ids="M"),
        polygon(6, elevation=π / 6, radius=2, unit="M", center_spkr=True, ids="U"),
        name=name,
        description="EMB's home array, 8+6+6",
    )

    layout += polygon(
        6, elevation=-π / 6, radius=2, unit="M", center_spkr=bottom_center, ids="L"
    )
    if add_imaginary:
        layout += nadir(radius=2)
        layout += zenith(radius=2)
    return layout


def amb_10_8_4(add_imaginary=True):
    layout = lsl.append_layouts(
        polygon(10, elevation=0, radius=2, unit="M", center_spkr=False, ids="M"),
        polygon(8, elevation=π / 6, radius=2, unit="M", center_spkr=True, ids="U"),
        name="Dome 10+8+4",
        description="Dome-10+8+4",
    )

    layout += polygon(
        4, elevation=π / 3, radius=2, unit="M", center_spkr=False, ids="L"
    )
    if add_imaginary:
        layout += nadir(radius=2)
        layout += zenith(radius=2)
    return layout


def stage2017(add_imaginary=True):
    """CCRMA Stage array."""
    layout = lsl.from_array(
        (
            # == towers 8:
            # theoretical angles, have to be calibrated
            27,
            3.9,
            216,
            -27,
            3.9,
            216,
            63,
            8,
            162,
            -63,
            8,
            162,
            117,
            8,
            162,
            -117,
            8,
            162,
            153,
            3.9,
            216,
            -153,
            3.9,
            216,
            # == upper 8
            23,
            29,
            171,
            -23,
            29,
            171,
            90,
            58,
            109,
            -90,
            58,
            109,
            157,
            31,
            167,
            -157,
            31,
            167,
            0,
            70,
            108,
            180,
            70,
            108,
            # == ring of 12 (rails)
            9,
            4,
            237,
            -9,
            4,
            237,
            45,
            6,
            187,
            -45,
            6,
            187,
            81,
            8,
            131,
            -81,
            8,
            131,
            99,
            8,
            130,
            -99,
            8,
            130,
            135,
            6,
            185,
            -135,
            6,
            185,
            171,
            4,
            238,
            -171,
            4,
            238,
            # == ring of 14 (lower trusses)
            14,
            18,
            243,
            -14,
            18,
            243,
            39,
            22,
            200,
            -39,
            22,
            200,
            60,
            30,
            154,
            -60,
            30,
            154,
            90,
            34,
            139,
            -90,
            34,
            139,
            122,
            30,
            153,
            -122,
            30,
            153,
            144,
            22,
            201,
            -144,
            22,
            201,
            166,
            19,
            243,
            -166,
            19,
            243,
            # == ring of 6 (upper trusses)
            0,
            31,
            180,
            39,
            47,
            128,
            -39,
            47,
            128,
            146,
            47,
            129,
            -146,
            47,
            129,
            180,
            33,
            180,
            # == lower ring of 8 in towers
            27,
            -10,
            216,
            -27,
            -10,
            216,
            63,
            -14,
            162,
            -63,
            -14,
            162,
            117,
            -14,
            162,
            -117,
            -14,
            162,
            153,
            -10,
            216,
            -153,
            -10,
            216,
        ),
        coord_code="AER",
        unit_code="DDI",
        name="CCRMA-Stage",
        ids=(
            "S01",
            "S02",
            "S03",
            "S04",
            "S05",
            "S06",
            "S07",
            "S08",
            "S09",
            "S10",
            "S11",
            "S12",
            "S13",
            "S14",
            "S15",
            "S16",
            "D17",
            "D18",
            "D19",
            "D20",
            "D21",
            "D22",
            "D23",
            "D24",
            "D25",
            "D26",
            "D27",
            "D28",
            "D29",
            "D30",
            "D31",
            "D32",
            "D33",
            "D34",
            "D35",
            "D36",
            "D37",
            "D38",
            "D39",
            "D40",
            "D41",
            "D42",
            "D43",
            "D44",
            "D45",
            "D46",
            "D47",
            "D48",
            "L01",
            "L02",
            "L03",
            "L04",
            "L05",
            "L06",
            "L07",
            "L08",
        ),
    )
    if add_imaginary:
        layout += nadir(radius=1.6, is_imaginary=True)
    return layout


# TODO: generalize this to load speaker arrays from spreadsheets
def iem_cube():
    """Return the Cube array at IEM."""

    a = np.genfromtxt(
        _data_dir / "LScoordinates.csv",
        skip_header=1,
        names=True,
        delimiter=",",
        deletechars="",
    )

    # get column_names from a.dtype
    all_column_names = a.dtype.names
    print(all_column_names)
    column_names = ("x_[m]", "y_[m]", "z_[m]")

    # TODO: parse this from the column names
    column_coords = [s[0] for s in column_names]
    column_units = ["m", "m", "m"]
    column_values = [a[i] for i in column_names]

    s = lsl.from_vectors(
        *column_values,
        name="IEM_Cube",
        coord_code=column_coords,
        unit_code=column_units
    )
    return s


def envelop():
    """Return the array at Envelop SF."""
    # conveted from the MATLAB written by Andrew Kimpel
    # coordinates (feet)
    #  here y is front/back, x is left/right, z is up/down
    #  conversion to Ambisonics convention is handled in call to
    #  ambi_spkr_array
    x1 = -5.30  # x L-R
    y1 = 12.87  # y F-B
    x2 = -8.74  # x L-R
    y2 = +4.62  # y L-R
    x3 = -3.33  # y T
    y3 = +7.29  # x T
    z1 = +5.00  # z M-U|D
    z2 = +7.81  # z T

    a = (
        "L1.M",
        [-x1, y1, 0],
        "R1.M",
        [+x1, y1, 0],
        "L2.M",
        [-x2, y2, 0],
        "R2.M",
        [+x2, y2, 0],
        "L3.M",
        [-x2, -y2, 0],
        "R3.M",
        [+x2, -y2, 0],
        "L4.M",
        [-x1, -y1, 0],
        "R4.M",
        [+x1, -y1, 0],
        "L1.U",
        [-x1, y1, z1],
        "R1.U",
        [+x1, y1, z1],
        "L2.U",
        [-x2, y2, z1],
        "R2.U",
        [+x2, y2, z1],
        "L3.U",
        [-x2, -y2, z1],
        "R3.U",
        [+x2, -y2, z1],
        "L4.U",
        [-x1, -y1, z1],
        "R4.U",
        [+x1, -y1, z1],
        "L1.D",
        [-x1, y1, -z1],
        "R1.D",
        [+x1, y1, -z1],
        "L2.D",
        [-x2, y2, -z1],
        "R2.D",
        [+x2, y2, -z1],
        "L3.D",
        [-x2, -y2, -z1],
        "R3.D",
        [+x2, -y2, -z1],
        "L4.D",
        [-x1, -y1, -z1],
        "R4.D",
        [+x1, -y1, -z1],
        "F.T",
        [0, +y3, z2],
        "B.T",
        [0, -y3, z2],
        "L.T",
        [-x3, 0, z2],
        "R.T",
        [+x3, 0, z2],
    )
    ids = a[::2]
    yxz = a[1::2]

    return lsl.from_array(
        yxz, coord_code="YXZ", unit_code="FFF", ids=ids, name="EnvelopSF"
    )


def uniform240(name="Uniform240"):
    """Return 240-speaker uniform layout."""
    #
    import spherical_grids as sg

    g = sg.t_design240()

    s = lsl.from_array(g.xyz, name=name, coord_code="XYZ", unit_code="MMM")
    return s

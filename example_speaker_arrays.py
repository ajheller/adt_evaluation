#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:28:40 2020

@author: heller
"""

# system modules
from pathlib import Path
import numpy as np

# adt modules
import SpeakerArray as sa

# this directory
_data_dir = Path(__file__).parent/"data"

def nadir():
    return \
    sa.from_array((0, 0, -1), coord_code='XYZ', unit_code='MMM',
                  array_name="imaginary speaker at nadir",
                  ids=["*I0"], is_real=False)
def stage2017():
    return \
    sa.from_array((
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


def iem_cube():
    d = np.loadtxt(_data_dir/"LScoordinates.csv", delimiter=',', skiprows=2)
    s = sa.from_array(d[:,1:], array_name="IEM_Cube",
                      coord_code='XYZ', unit_code="MMM")
    return s

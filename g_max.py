#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 20:20:45 2018

@author: heller
"""

from __future__ import division, print_function

import os.path as path
import numpy as np

# plotly is not available via conda
# use pip install plotly to install
import plotly
import plotly.graph_objs as go
from plotly import tools as tls

import matplotlib.pyplot as plt


import real_spherical_harmonics as rsh
import acn_order as acn

import spherical_grids as grids
from spherical_grids import cart2sph, sph2cart

import adt_scmd


Sgg = np.zeros(np.shape(az))

for i in range(28):
    gi = np.reshape(g[i,:], np.shape(az))
    ggi = np.abs(gi - np.max(gi)) < 1e-2
    Sgg[ggi] = gi[ggi]

print("max gain for entire array = ", np.max(g), 20*np.log10(np.max(g)))

plot_rX(Sgg, "max gain for each loudspeaker (dB)")
plt.show()
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 15:38:30 2018

@author: heller
"""

import numpy as np

# plotly is not available via conda
# use pip install plotly to install
import plotly
import plotly.graph_objs as go
from plotly import tools as tls

import matplotlib.pyplot as plt


import real_spherical_harmonics as rsh
import acn_order as acn

import grids
from grids import cart2sph, sph2cart

import adt_scmd


flat_trace = go.Surface(x=az,
                        y=el,
                        z=np.zeros(np.shape(az)),
                        cmin=0.5,
                        cmax=1,
                        surfacecolor=np.reshape(rEr, np.shape(az)),
                        colorscale='Jet')

flat_layout = go.Layout(
        scene=dict(
                aspectratio=dict(x=1, y=1, z=0.1),
                zaxis=dict(range=(-0.1, 0.1))))

plotly.offline.plot({'data': [flat_trace], 'layout': flat_layout},
                    filename='flat.html')
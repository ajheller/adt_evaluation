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

# this assumes you've aready run rVrE.py and some variables are left in the workspace
#  (I know this is bad, but I'm just doing some experiments)

flat_trace = go.Surface(name='rE',
                        x=az * 180/np.pi,
                        y=el * 180/np.pi,
                        z=np.zeros(np.shape(az)),
                        cmin=0.7,
                        cmax=1,
                        surfacecolor=np.reshape(rEr, np.shape(az)),
                        colorscale='Jet',
                        lighting=dict(ambient=1.0),
                        hoverinfo='text',
                        text=np.vectorize(
                                lambda a, e, c:
                                    "az: %.1f<br>el: %.1f<br> r<sub>E</sub>: %.2f" %
                                    (a, e, c))
                            (az * 180/np.pi, el * 180/np.pi,
                             np.reshape(rEr, np.shape(az))))

speakers = go.Scatter3d(name='Speakers (actual locations)',
                     x=spkr_az * 180/np.pi,
                     y=spkr_el * 180/np.pi,
                     z=-0.01 + np.zeros(np.shape(spkr_az)),
                     mode='markers',
                     marker=dict(color='black'),
                     hoverinfo='text',
                     visible=True,
                     text=np.vectorize(
                             lambda a, e, r, c:
                                 "%s<br>az: %.1f<br>el: %.1f<br> r: %.1f"
                                 % (c, a, e, r))
                             (spkr_az * 180/np.pi, spkr_el * 180/np.pi,
                              spkr_rr, spkr_id))
camera = dict(
    up=dict(x=1, y=0, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=-2))

flat_layout = go.Layout(
        showlegend=True,
        legend=dict(orientation="h"),
        scene=dict(
                xaxis=dict(title='azimuth (degrees)'),
                yaxis=dict(title='elevation (degrees)'),
                camera=camera,
                aspectratio=dict(x=2, y=1, z=0.1),
                zaxis=dict(range=(-0.1, 0.1))))


plotly.offline.plot({'data': [flat_trace, speakers], 'layout': flat_layout},
                    filename='flat.html')
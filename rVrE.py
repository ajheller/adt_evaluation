#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:26:45 2018

@author: heller
"""

import numpy as np

# plotly is not available via conda
# use pip install plotly to install
import plotly
import plotly.graph_objs as go
from plotly import tools

import matplotlib.pyplot as plt


import real_spherical_harmonics as rsh

import grids

x, y, z, az, el, w = grids.az_el()

ylm4= rsh.real_sph_harm_acn(4, az, el + np.pi/2)

c = ylm4
ca = np.abs(ylm4)

plt.imshow(c)

data = [go.Surface(x=ca*x, y=ca*y, z=ca*z,
                   surfacecolor=c,
                   colorscale='Jet')]

layout = go.Layout(title="Ylm")

plotly.offline.plot({'data': data, 'layout': layout},
                    filename="tmp.html")

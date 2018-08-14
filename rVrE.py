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
import acn_order as acn

import grids

x, y, z, az, el, w = grids.az_el()


# sample spherial harmonics at grid points
ambisonic_order = 3
max_acn = acn.acn(ambisonic_order, ambisonic_order)

az0 = az.ravel()  # 1-D view of az, like (:) in MATLAB
el0 = el.ravel()

test_dirs_Y = np.array([rsh.real_sph_harm_acn(i, az0, el0+np.pi/2)
                        for i in range(max_acn+1)])

M = np.genfromtxt("test_data/" +
                  "CCRMA_Listening_Room_3h3p_allrad_5200_rE_max_2_band-rEmax.csv",
                  dtype=np.float64,
                  delimiter=',')

g = np.matmul(M, test_dirs_Y)

g2 = g * g.conjugate() # if g's might be complex

# pressure
P = np.sum(g, 0)


# real_sph_harm expects zenith angle, so add pi/2
#ylm4= rsh.real_sph_harm_acn(4, az, el + np.pi/2)

c = np.reshape(test_dirs_Y[12,:], np.shape(az))
ca = np.abs(c)

plt.imshow(c)

data = [go.Surface(x=ca*x, y=ca*y, z=ca*z,
                   surfacecolor=c,
                   colorscale='Jet')]

layout = go.Layout(title="Ylm")

plotly.offline.plot({'data': data, 'layout': layout},
                    filename="tmp.html")

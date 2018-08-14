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

x, y, z, az, el, w = grids.az_el(resolution=72)


# sample spherial harmonics at grid points
ambisonic_order = 3
max_acn = acn.acn(ambisonic_order, ambisonic_order)

az0 = az.ravel()  # 1-D view of az, like (:) in MATLAB
el0 = el.ravel()

test_dirs_Y = np.array([rsh.real_sph_harm_acn(i, az0, el0+np.pi/2)
                        for i in range(max_acn+1)])

M = np.genfromtxt("/Users/heller/Documents/adt/decoders/" +
                  "CCRMA_Listening_Room_3h3p_allrad_5200_rE_max_2_band-rEmax.csv",
                  dtype=np.float64,
                  delimiter=',')

Su = np.genfromtxt("/Users/heller/Documents/adt/decoders/" +
                   "ccrma_listening_room.csv",
                   dtype=np.float64,
                   delimiter=',')

g = np.matmul(M, test_dirs_Y)

g2 = np.real(g * g.conjugate())  # if g's might be complex

# pressure
P = np.sum(g, 0)

rVxyz = np.real(np.matmul(Su, g) / np.array([P, P, P]))
rVr = np.linalg.norm(rVxyz, 2, 0)
rVu = rVxyz / np.array([rVr, rVr, rVr])

E = np.sum(g2, 0)

rExyz = np.matmul(Su, g2) / np.array([E, E, E])
rEr = np.linalg.norm(rExyz, 2, 0)
rEu = rExyz / np.array([rEr, rEr, rEr])


if False:
    xyz = rVxyz
    r = rVr
else:
    xyz = rExyz
    r = rEr

c = np.reshape(r, np.shape(az))
ca = np.abs(c)

plt.imshow(np.reshape(rVr, np.shape(az)).transpose(),
           extent=(-180, 180, 90, -90))
ax = plt.gca()
ax.xaxis.set_ticks(np.linspace(-180, 180, 9))
ax.yaxis.set_ticks(np.linspace(90, -90, 5))
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Elevation (degrees)')
plt.title('magnitude of rE vs. test direction')
plt.colorbar()
plt.show()

data = [go.Surface(x=np.reshape(xyz[0, :], np.shape(az)),
                   y=np.reshape(xyz[1, :], np.shape(az)),
                   z=np.reshape(xyz[2, :], np.shape(az)),
                   cmin=0.5,
                   cmax=1,
                   surfacecolor=c,
                   colorscale='Jet',
                   contours=dict(z=dict(show=True),
                                 y=dict(show=True),
                                 x=dict(show=True)))]

layout = go.Layout(title="Ylm")

plotly.offline.plot({'data': data, 'layout': layout},
                    filename="tmp.html")

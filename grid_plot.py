#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:58:19 2018

@author: heller
"""

# This file is part of the Ambisonic Decoder Toolbox (ADT)
# Copyright (C) 2018-19  Aaron J. Heller <heller@ai.sri.com>
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

__debug = True

__colormap = 'jet'


def toDeg(radians):
    return radians * 180/np.pi

drE = 180/np.pi * np.arccos(np.sum(rEu * xyz0, 0));

t1 = go.Scatter(
        name="rE dir",
        visible=True,
        mode='markers',
        marker=dict(
                size=5,
                color = drE, #set color equal to a variable
                colorscale='Jet',
                cmin=0,
                cmax=10,
                showscale=True),

        x = toDeg(rEaz),
        y = toDeg(rEel))

t2 = go.Scatter(
        name="Source dir",
        visible=True,
        mode='markers',
        marker=dict(
                size=2,
                color='black'),

        x = toDeg(az0),
        y = toDeg(el0))

sh = np.shape(az)
xx = toDeg(np.unwrap(np.reshape(rEaz, sh), 3*np.pi, 0))
yy = toDeg(np.reshape(rEel, sh))

tt = [go.Scatter(x=xx[i,:], y=yy[i,:], mode='lines') for i in range(0,sh[0],2)]
#tt = tt + [go.Scatter(x=xx[:,1], y=yy[:,i], mode='lines') for i in range(0,sh[1],2)]

s1 = go.Scatter(
        name='Speakers',
        x=toDeg(spkr_az),
        y=toDeg(spkr_el),
        mode='markers',
        marker=dict(color='blue',
                    size=20),
        hoverinfo='text',
        visible=True,
        text=np.vectorize(
                lambda a, e, r, c:
                "<B>%s</B><br>az: %.1f<br>el: %.1f<br> r: %.1f" % (c, a, e, r))
                (toDeg(spkr_az), toDeg(spkr_el), spkr_rr, spkr_id))

l1 = go.Layout(
        title=name,
        showlegend=True,
        legend=dict(orientation="h"),
        xaxis=dict(range=(185, -185), title='azimuth (degrees)'),
        yaxis=dict(range=(-95, 95), title='elevation (degrees)'),
        # aspectratio=dict(x=2, y=1)
        )

#f1 = go.Figure(data=[s1,t1,t2], layout=l1)
f1 = go.Figure(data=tt, layout=l1)

plotly.offline.plot(f1)


def plot_dir_error_grid(rEaz, rEel, az, el, scmd):
    rEaz = np.reshape(rEaz, np.shape(az))
    rEel = np.reshape(rEel, np.shape(el))

    # unwrap rEaz
    # fixup = np.logical_and((rEaz - az) > np.pi, az > -np.pi*0.9)
    fixup = np.logical_and((rEaz - az) > np.pi, True)
    rEaz[fixup] = rEaz[fixup] - (2 * np.pi)
    # fixup = np.logical_and((rEaz - az) < -np.pi, az < np.pi*0.9)
    fixup = np.logical_and((rEaz - az) < -np.pi, True)
    rEaz[fixup] = rEaz[fixup] + (2 * np.pi)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.plot(rEaz, rEel)
    ax.hold(True)
    ax.plot(rEaz.transpose(), rEel.transpose())
    ax.plot(scmd['S']['az'], scmd['S']['el'], '*k')
    ax.hold(False)

    plotly_fig = tls.mpl_to_plotly(fig)
    plotly.offline.plot(plotly_fig)

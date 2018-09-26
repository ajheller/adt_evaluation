#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 22:06:32 2018

@author: heller
"""
from __future__ import division, print_function
import numpy as np

# plotly is not available via conda
# use 'pip install plotly' to install
import plotly
import plotly.graph_objs as go
from plotly import tools as tls


def sph2cart(az, el, r=1):
    z = np.sin(el)
    r_cos_el = r * np.cos(el)
    x = r_cos_el * np.cos(az)
    y = r_cos_el * np.sin(az)
    return x, y, z


def plot_sph(az_grid, el_grid, r_grid, c_grid, name):

    x, y, z = sph2cart(az_grid, el_grid, r_grid)

    data = [go.Surface(name='rE',
                       x=x, y=y, z=z,
                       surfacecolor=c_grid,
                       #cmin=0.7,
                       #cmax=np.ceil(np.max(rEr)*10)/10,
                       colorscale='Portland',
                       hoverinfo='text',
                       text=np.vectorize(lambda u, v, c: "r: %.2f<br>a: %.1f<br>e: %.1f"
                                         % (c, u, v))(az_grid*180/np.pi,
                                                      el_grid*180/np.pi,
                                                      r_grid),

                       contours=dict(z=dict(show=True),
                                     y=dict(show=True),
                                     x=dict(show=True)))
            ]

    layout = go.Layout(title=name,
                       showlegend=True,
                       legend=dict(orientation="h"),
                       scene=dict(aspectratio=dict(x=1, y=1, z=1)))

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='antenna.html')

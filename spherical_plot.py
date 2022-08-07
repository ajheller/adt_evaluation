#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 22:06:32 2018

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

    data = [
        go.Surface(
            name="rE",
            x=x,
            y=y,
            z=z,
            surfacecolor=c_grid,
            # cmin=0.7,
            # cmax=np.ceil(np.max(rEr)*10)/10,
            colorscale="Portland",
            hoverinfo="text",
            text=np.vectorize(
                lambda u, v, c: "r: %.2f<br>a: %.1f<br>e: %.1f" % (c, u, v)
            )(az_grid * 180 / np.pi, el_grid * 180 / np.pi, r_grid),
            contours=dict(z=dict(show=True), y=dict(show=True), x=dict(show=True)),
        )
    ]

    layout = go.Layout(
        title=name,
        showlegend=True,
        legend=dict(orientation="h"),
        scene=dict(aspectratio=dict(x=1, y=1, z=1)),
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename="antenna.html")

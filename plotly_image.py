#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 15:38:30 2018

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


# this produces the 2-D plots of rE etc.

import os
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

# this assumes you've aready run rVrE.py and some variables are left in the
# workspace (I know this is bad, but I'm just doing some experiments)
#   az, el,

from rVrE import T, C, S, \
                 P, E, rEr, rEu, rE_min, xyz0, \
                 spkr_az, spkr_el, spkr_rr, spkr_id, \
                 name




def plotly_image(T, X, Xmin=-np.inf, Xmax=+np.inf,
                 name="",
                 hovertext_format="az: %.1f<br>el: %.1f<br>X: %.2f",
                 visible=True,
                 showlegend=True,
                 colorscale='Jet'):

    XX = np.reshape(X, T.shape)

    # https://plot.ly/python/reference/#heatmap
    trace = go.Heatmap(
            name=name,
            x=T.az[:, 0] * 180/np.pi,
            y=T.el[0, :] * 180/np.pi,
            z=np.clip(XX, Xmin, Xmax).transpose(),
            visible=visible,
            colorscale=colorscale,
            hoverinfo='text',
            text=(np.vectorize(
                lambda a, e, c:
                hovertext_format % (a, e, c))
                (T.az * 180/np.pi,
                 T.el * 180/np.pi,
                 XX)).transpose())
    return trace


# ---  rE

rE_trace = \
    plotly_image(T, rEr,
                 Xmin=rE_min, Xmax=1.0,
                 name='Magnitude of r<sub>E</sub> vs. Test Direction',
                 hovertext_format="az: %.1f<br>el: %.1f<br>r<sub>E</sub>: %.2f",
                 visible=True,
                 showlegend=True)

Angular_spread_trace = \
    plotly_image(T, np.arccos(rEr) * 180.0/np.pi,
                 # Xmin=0.0,
                 Xmax=45.0,
                 name='Angular Spread (deg) vs. Test Direction',
                 hovertext_format="az: %.1f<br>el: %.1f<br>AS: %.2f",
                 visible=False,
                 showlegend=True)

Derr_trace = \
    plotly_image(T, 180/np.pi * np.arccos(np.sum(rEu * xyz0, 0)),
                 Xmin=0.0, Xmax=15.0,
                 name='r<sub>E</sub> Direction Error (deg) vs. Test Direction',
                 hovertext_format="az: %.1f<br>el: %.1f<br>Derr: %.2f",
                 visible=False,
                 showlegend=True)

E_trace = \
    plotly_image(T, 10*np.log10(E),
                 name="Energy gain (dB) vs. Test Direction",
                 hovertext_format="az: %.1f<br>el: %.1f<br>E: %.2f dB",
                 visible=False,
                 showlegend=True)


def speaker_hovertext(az, el, rr, id):
    def ht(az, el, rr, id):
        return ("<B>%s</B><br>az: %.1f<br>el: %.1f<br> r: %.1f"
                % (id, az, el, rr))
    return np.vectorize(ht, otypes=[str])(az, el, rr, id)


speakers = go.Scatter(
        name='Speakers',
        x=spkr_az * 180/np.pi,
        y=spkr_el * 180/np.pi,
        mode='markers',
        marker=dict(color='black', size=10),
        hoverinfo='text',
        visible=True,
        showlegend=True,
        text=speaker_hovertext(spkr_az * 180/np.pi,
                               spkr_el * 180/np.pi,
                               spkr_rr, spkr_id)
        )

camera = dict(
    up=dict(x=1, y=0, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=-2))

updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=list([
            dict(label='Magnitude rE',
                 method='update',
                 args=[{'visible': [True, False, False, False, True]},
                       {'title': name + "-%dH%dV" % (C['h_order'], C['v_order'])
                        + "<br>" + rE_trace['name']
                          # , 'annotations': high_annotations
                        }]),
            dict(label='Angular Spread',
                 method='update',
                 args=[{'visible': [False, True, False, False, True]},
                       {'title': name + "-%dH%dV" % (C['h_order'], C['v_order'])
                        + "<br>" + Angular_spread_trace['name']
                          # , 'annotations': high_annotations
                        }]),
            dict(label='Direction Error',
                 method='update',
                 args=[{'visible': [False, False, True, False, True]},
                       {'title': name + "-%dH%dV" % (C['h_order'], C['v_order'])
                        + "<br>" + Derr_trace['name']
                        # , 'annotations': low_annotations
                        }]),
            dict(label='Energy Gain',
                 method='update',
                 args=[{'visible': [False, False, False, True, True]},
                       {'title': name + "-%dH%dV" % (C['h_order'], C['v_order'])
                        + "<br>" + E_trace['name']
                        # , 'annotations': low_annotations
                        }])
                    ]))])

layout = go.Layout(
        title=name + u"-%dH%dV" % (C['h_order'], C['v_order'])
            + u"<br>" + rE_trace['name'],
        showlegend=True,
        hovermode='closest',
        legend=dict(orientation="h"),
        xaxis=dict(title='azimuth (degrees)', range=(185, -185)),
        yaxis=dict(title='elevation (degrees)', scaleanchor='x', scaleratio=1),
        scene=dict(aspectratio=dict(x=2, y=1)),
        updatemenus=updatemenus
        )

# fig = tls.make_subplots(rows=2, cols=1)

fig = go.Figure(data=[rE_trace, Angular_spread_trace, Derr_trace, E_trace,
                      speakers],
                layout=layout)

out_dir = "plotly"

if __name__ == '__main__':

    plotly.offline.plot(fig,
                        filename=os.path.join(out_dir, S['name']
                            + "-" + ("%dH%dV" % (C['h_order'], C['v_order']))
                            + "_rE.html"))

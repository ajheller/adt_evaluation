#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 15:38:30 2018

@author: heller
"""
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
from grids import cart2sph, sph2cart

import adt_scmd

# this assumes you've aready run rVrE.py and some variables are left in the
# workspace (I know this is bad, but I'm just doing some experiments)
#   az, el,

rX = rEr; cmin=0.7; cmax=1
rX2 = 180/np.pi * np.arccos(np.sum(rEu * xyz0, 0)); cmin2=0; cmax2=10;

if False:
    flat_trace = go.Surface(
        name='rE',
        x=az * 180/np.pi,
        y=el * 180/np.pi,
        z=np.zeros(np.shape(az)),
        cmin=cmin,
        cmax=cmax,
        surfacecolor=np.reshape(rX, np.shape(az)),
        colorscale='Jet',
        lighting=dict(ambient=1.0),
        hoverinfo='text',
        text=np.vectorize(
                lambda a, e, c:
                "az: %.1f<br>el: %.1f<br>dir diff: %.2f" % (a, e, c))
                (az * 180/np.pi,
                 el * 180/np.pi,
                 np.reshape(rX, np.shape(az))))

# https://plot.ly/python/reference/#heatmap
rE_trace = go.Heatmap(
        name='rE magnitude vs. test direction',
        x=az[:,0] * 180/np.pi,
        y=el[0,:] * 180/np.pi,
        z=np.clip(np.reshape(rX, np.shape(az)).transpose(), cmin, cmax),
        visible=True,
        showlegend=True,
        #cmin=cmin,
        #cmax=cmax,
        colorscale='Jet',
        #lighting=dict(ambient=1.0),
        hoverinfo='text',
        text=(np.vectorize(
                lambda a, e, c:
                "az: %.1f<br>el: %.1f<br>r<sub>E</sub>: %.2f" % (a, e, c))
                (az * 180/np.pi,
                 el * 180/np.pi,
                 np.reshape(rX, np.shape(az)))).transpose())

dd_trace = go.Heatmap(
        name='rE Direction Error vs. test direction',
        x=az[:,0] * 180/np.pi,
        y=el[0,:] * 180/np.pi,
        z=np.clip(np.reshape(rX2, np.shape(az)).transpose(), cmin2, cmax2),
        visible=False,
        showlegend=True,
        #cmin=cmin,
        #cmax=cmax,
        colorscale='Jet',
        #lighting=dict(ambient=1.0),
        hoverinfo='text',
        text=(np.vectorize(
                lambda a, e, c:
                "az: %.1f<br>el: %.1f<br>Derr</sub>: %.2f" % (a, e, c))
                (az * 180/np.pi,
                 el * 180/np.pi,
                 np.reshape(rX2, np.shape(az)))).transpose())

speakers = go.Scatter(
        name='Speakers',
        x=spkr_az * 180/np.pi,
        y=spkr_el * 180/np.pi,
        mode='markers',
        marker=dict(color='black', size=10),
        hoverinfo='text',
        visible=True,
        showlegend=True,
        text=np.vectorize(
                lambda a, e, r, c:
                "<B>%s</B><br>az: %.1f<br>el: %.1f<br> r: %.1f" % (c, a, e, r))
                (spkr_az * 180/np.pi, spkr_el * 180/np.pi, spkr_rr, spkr_id))

camera = dict(
    up=dict(x=1, y=0, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=-2))

updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=list([
            dict(label = 'rE',
                 method = 'update',
                 args = [{'visible': [True, False, True]},
                         {'title': name + "-%dH%dV"%(C['h_order'],C['v_order'])
                          + "<br>" + rE_trace['name']
                          #, 'annotations': high_annotations
                          }]),
            dict(label = 'Error',
                 method = 'update',
                 args = [{'visible': [False, True, True]},
                         {'title': name + "-%dH%dV"%(C['h_order'],C['v_order'])
                         + "<br>" + dd_trace['name']
                          #, 'annotations': low_annotations
                          }])
    ]))])

flat_layout = go.Layout(
        title="name" + u"-%dH%dV"%(C['h_order'],C['v_order'])
                + u"<br>" + rE_trace['name'],
        showlegend=True,
        hovermode='closest',
        legend=dict(orientation="h"),
        xaxis=dict(title='azimuth (degrees)', range=(185, -185)),
        yaxis=dict(title='elevation (degrees)', scaleanchor='x', scaleratio=1),
        scene=dict(aspectratio=dict(x=2, y=1)),
        updatemenus=updatemenus
        )

#fig = tls.make_subplots(rows=2, cols=1)

fig = go.Figure(data=[rE_trace, dd_trace, speakers],
                layout=flat_layout)

out_dir = "plotly"

plotly.offline.plot(fig,
                    filename= os.path.join(out_dir, S['name']
                    + "-" + ("%dH%dV"%(C['h_order'], C['v_order']))
                    + "_rE.html"))

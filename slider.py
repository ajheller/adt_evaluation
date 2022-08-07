#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 18:37:50 2018

@author: heller
"""

import plotly
import plotly.graph_objs as go
from plotly import tools as tls

import plotly.plotly as py
import numpy as np


# from
#  https://plot.ly/python/sliders/

data = [
    dict(
        visible=False,
        line=dict(color="#00CED1", width=6),
        name="𝜈 = " + str(step),
        x=np.arange(0, 10, 0.01),
        y=np.sin(step * np.arange(0, 10, 0.01)),
    )
    for step in np.arange(0, 5, 0.1)
]
data[10]["visible"] = True

# py.iplot(data, filename='Single Sine Wave')


steps = []
for i in range(len(data)):
    step = dict(
        method="restyle",
        args=["visible", [False] * len(data)],
    )
    step["args"][1][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [
    dict(active=10, currentvalue={"prefix": "Frequency: "}, pad={"t": 50}, steps=steps)
]

layout = dict(sliders=sliders)

fig = dict(data=data, layout=layout)

plotly.offline.plot(fig, filename="plotly/Sine_Wave_Slider.html")

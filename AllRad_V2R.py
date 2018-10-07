#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:49:44 2018

@author: heller
"""

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

from scipy.spatial import Delaunay

import adt_scmd
import spherical_grids as sg
import ray_triangle_intersection as rti

__debug = True

__colormap = 'jet'

smcd_dir = "examples"
# smcd_dir = "/Users/heller/Documents/adt/examples/"

example = 0  # <<<<<---------- change this to change datasets
interior_view = False

scmd_file = ("SCMD_env_asym_tri_oct_4ceil.json",
             "SCMD_brh_spring2017.json")[example]

Su, C, M, D, scmd = adt_scmd.load(path.join(smcd_dir, scmd_file))

print "\n\nread: %s\n" % path.join(smcd_dir, scmd_file)

# assume Su is 3xL array of unit vectors to loudspeakers

#  Vx, Vy, Vz, Vaz, Vel, Vw = sg.t_design()
V = sg.t_design5200()
Vxyz = np.zeros(V.u.shape)

#  https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.html
tri = Delaunay(Su.transpose())
H = tri.convex_hull

o = np.array([0, 0, 0])
Vtri = np.zeros(len(V.uz), dtype=np.integer)
V2R = np.zeros((Su.shape[1], V.u.shape[1]))

for i in range(len(V.uz)):  # iterate over the virtual loudspeakers
    d = V.u[:, i]
    for j in range(len(H)):
        p0 = Su[:, H[j, 0]]
        p1 = Su[:, H[j, 1]]
        p2 = Su[:, H[j, 2]]
        flag, bu, bv, bt = rti.ray_triangle_intersection(o, d, p0, p1, p2)
        bw = 1 - bu - bv

        if flag and bt > 0:
            # virtual speaker i intersects face j
            Vtri[i] = j
            # coordinates of intersection for plotting
            Vxyz[:, i] = bw*p0 + bu*p1 + bv*p2
            # fill in gains, normalize for energy
            b = np.array([bw, bu, bv])

            V2R[H[j, :], i] = b / np.linalg.norm(b)

            break

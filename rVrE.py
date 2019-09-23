#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:26:45 2018

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
from numpy import pi  # cause np.pi is messy

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

__debug = True

__colormap = 'jet'


#
def plot_dir_error_grid(rEaz, rEel, az, el, scmd):
    rEaz = np.reshape(rEaz, az.shape)
    rEel = np.reshape(rEel, el.shape)

    # unwrap rEaz
    # fixup = np.logical_and((rEaz - az) > pi, az > -pi*0.9)
    fixup = np.logical_and((rEaz - az) > pi, True)
    rEaz[fixup] = rEaz[fixup] - (2 * pi)
    # fixup = np.logical_and((rEaz - az) < -pi, az < pi*0.9)
    fixup = np.logical_and((rEaz - az) < -pi, True)
    rEaz[fixup] = rEaz[fixup] + (2 * pi)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.plot(rEaz, rEel)
    ax.hold(True)
    ax.plot(rEaz.transpose(), rEel.transpose())
    ax.plot(S['az'], S['el'], '*k')
    ax.hold(False)

    plotly_fig = tls.mpl_to_plotly(fig)
    plotly.offline.plot(plotly_fig)


def plot_rX(rX, title, clim=None):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    plt.imshow(np.fliplr(np.flipud(np.reshape(rX, T.shape).transpose())),
               extent=(180, -180, -90, 90),
               cmap=__colormap)
    if clim:
        plt.clim(clim)
    ax.xaxis.set_ticks(np.linspace(180, -180, 9))
    ax.yaxis.set_ticks(np.linspace(-90, 90, 5))
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Elevation (degrees)')
    plt.title(title)
    plt.colorbar()

    if False:
        # plotly does not work with MPL images (yet)
        plotly.offline.plot_mpl(fig)
    else:
        plt.show()


def plot_dir_diff(u1, u2, title='Diff (degrees)', clim=None):
    # we assume u1 and u2 are unit vectors
    angle_diff = np.arccos(np.sum(u1 * u2, 0))
    plot_rX(angle_diff * 180/pi, title, clim)


def ravel(M):
    return M.reshape(M, -1)


def unravel(M):
    return np.reshape(M, T.shape)


# ---- start of main ----

# load the ADT results
smcd_dir = "examples"
# smcd_dir = "/Users/heller/Documents/adt/examples/"

example = 3  # <<<<<---------- change this to change datasets
interior_view = True

scmd_file = ("SCMD_env_asym_tri_oct_4ceil.json",
             "SCMD_brh_spring2017.json",
             "SCMD_stage2017.json",
             "SCMD_run_dec_itu.json")[example]

Su, C, M, D, scmd = adt_scmd.load(path.join(smcd_dir, scmd_file))

print "\n\nread: %s\n" % path.join(smcd_dir, scmd_file)

# add a speaker at the top and bottom
if True:
    Su = np.column_stack((Su, np.array([[0, 0], [0, 0], [+1, -1]])))

tri = Delaunay(Su.transpose())


# x, y, z, az, el, w = grids.az_el(resolution=72)
T = grids.az_el(resolution=72)

# I use 0 to indicate a flattened version of a grid
az0 = T.az.ravel()  # 1-D view of az, like (:) in MATLAB
el0 = T.el.ravel()
x0 = T.ux.ravel()
y0 = T.uy.ravel()
z0 = T.uz.ravel()
w0 = T.w.ravel()
xyz0 = T.u


test_dirs_Y = np.array(
        [np.sqrt(4 * pi) * n * rsh.real_sph_harm(m, l, az0, pi/2 - el0)
         for l, m, n in zip(C['sh_l'], C['sh_m'], C['norm'])])

if False:
    #  plot the first few SH's test plot_rX and the SH's themselvesss
    for i in range(4):
        plot_rX(np.reshape(test_dirs_Y[i, :], T.shape),
                'ACN%d %s' % (i, acn.acn2fuma_name(i)))

#
# Note:
#   NumPy matmul == MATLAB *
#   NumPy * == MATLAB .*


# pressure gains, g,  from each test direction to each speaker
if True:
    # M is the basic solution
    gamma = np.array(D['hf_gains'])
    #  apply gains to tranform to max rE matrix
    Mhf = np.matmul(M, np.diag(gamma[C['sh_l']]))
    g = np.matmul(Mhf, test_dirs_Y)
else:
    g = np.matmul(M, test_dirs_Y)

# Energy gain from each test direction to each speaker
g2 = np.real(g * g.conjugate())  # if g's might be complex

# pressure & rV
P = np.sum(g, 0)
rVxyz = np.real(np.matmul(Su, g) / np.array([P, P, P]))
rVaz, rVel, rVr = cart2sph(rVxyz[0, :], rVxyz[1, :], rVxyz[2, :])
rVu = rVxyz / np.array([rVr, rVr, rVr])

# energy & rE
E = np.sum(g2, 0)
rExyz = np.matmul(Su, g2) / np.array([E, E, E])
rEaz, rEel, rEr = cart2sph(rExyz[0, :], rExyz[1, :], rExyz[2, :])
rEu = rExyz / np.array([rEr, rEr, rEr])

# decoder gains
decoder_gain = np.sqrt(np.sum(E * w0) / (4*pi))

print "decoder diffuse gain = %f, (%f db)\n" \
        % (decoder_gain, 20 * np.log10(decoder_gain))
print "decoder peak gain = %f\n" % np.max(g)

# matplotlib plots
if False:
    plot_rX(20*np.log10(E/np.mean(E)), 'Energy gain (dB) vs. test direction',
            clim=(-6, 6))

    plot_rX(rVr, 'magnitude of rV vs. test direction', clim=(0.5, 1))
    plot_rX(rEr, 'magnitude of rE vs. test direction', clim=(0.5, 1))

    plot_dir_diff(rVu, rEu, 'rV rE direction diff (degrees)', clim=(0, 10))

    plot_dir_diff(rVu, xyz0, 'rV direction error (degrees)', clim=(0, 10))
    plot_dir_diff(rEu, xyz0, 'rE direction error (degrees)', clim=(0, 10))


if False:
    xyz = rVxyz
    r = rVr
else:
    xyz = rExyz
    r = rEr


c = np.reshape(r, T.shape)
ca = np.abs(c)

S = scmd['S']
spkr_r = 1.25  # np.squeeze(S['r'])
spkr_rr = np.squeeze(S['r'])
spkr_az = np.squeeze(S['az'])
spkr_el = np.squeeze(S['el'])
spkr_id = np.squeeze(S['id'])
spkr_x = spkr_rr * np.squeeze(S['x'])
spkr_y = spkr_rr * np.squeeze(S['y'])
spkr_z = spkr_rr * np.squeeze(S['z'])
spkr_floor = np.min(spkr_z) - 0.5  # 1/2-meter below lowest spkr

spkr_ux = 0.9 * np.min(spkr_rr)*Su[0, :]
spkr_uy = 0.9 * np.min(spkr_rr)*Su[1, :]
spkr_uz = 0.9 * np.min(spkr_rr)*Su[2, :]

max_rr = np.max(spkr_rr)
plt_range = (-max_rr, max_rr)

# matplotlib named colors
#   https://matplotlib.org/examples/color/named_colors.html


#  https://plot.ly/python/reference/#scatter3d
#    NaN ends the line
spkr_stands = go.Scatter3d(
        name="Speaker Stands",
        x=np.array([[spkr_x[i], spkr_x[i], 0, np.NaN]
                    for i in range(len(spkr_x))]).ravel(),
        y=np.array([[spkr_y[i], spkr_y[i], 0, np.NaN]
                    for i in range(len(spkr_y))]).ravel(),
        z=np.array([[spkr_z[i], spkr_floor, spkr_floor, np.NaN]
                    for i in range(len(spkr_z))]).ravel(),
        mode='lines',
        hoverinfo='none',
        line=dict(color='black'),
        visible=True,
        connectgaps=False)

#  https://plot.ly/python/reference/#scatter3d
#    NaN ends the line
if False:
    # draw speaker vectors to unit sphere
    spkr_vector = go.Scatter3d(
        name="Speaker Vector",
        x=np.array([[spkr_x[i], spkr_ux[i], np.NaN]
                    for i in range(len(spkr_x))]).ravel(),
        y=np.array([[spkr_y[i], spkr_uy[i], np.NaN]
                    for i in range(len(spkr_y))]).ravel(),
        z=np.array([[spkr_z[i], spkr_uz[i], np.NaN]
                    for i in range(len(spkr_z))]).ravel(),
        mode='lines',
        hoverinfo='none',
        line=dict(color='blue', dash='dot'),
        visible=True,
        connectgaps=False)
else:
    # draw speaker vectors to the origin
    spkr_vector = go.Scatter3d(
        name="Speaker Vector",
        x=np.array([[spkr_x[i], 0, np.NaN]
                    for i in range(len(spkr_x))]).ravel(),
        y=np.array([[spkr_y[i], 0, np.NaN]
                    for i in range(len(spkr_y))]).ravel(),
        z=np.array([[spkr_z[i], 0, np.NaN]
                    for i in range(len(spkr_z))]).ravel(),
        mode='lines',
        hoverinfo='none',
        line=dict(color='blue', dash='dot'),
        visible=True,
        connectgaps=False)


#  https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.html
#  tri = Delaunay(Su.transpose())
cve_x = []
cve_y = []
cve_z = []
for face in tri.convex_hull:
    for i in [0, 1, 2, 0]:
        cve_x.append(spkr_ux[face[i]])
        cve_y.append(spkr_uy[face[i]])
        cve_z.append(spkr_uz[face[i]])
    cve_x.append(None)
    cve_y.append(None)
    cve_z.append(None)


#  https://plot.ly/python/reference/#scatter3d
cvedges = go.Scatter3d(
        name="Convex Hull edges",
        x=cve_x,
        y=cve_y,
        z=cve_z,
        mode='lines+markers',
        hoverinfo='none',
        line=dict(color='white', width=6),
        visible=True,
        connectgaps=False)

# rE
# Plotly does not support legend entries for Surface or Mesh (sigh)
#  https://community.plot.ly/t/how-to-name-axis-and-show-legend-in-mesh3d-and-surface-3d-plots/1819
rE_plot = go.Surface(
        name='Energy Model Localization Vector (r<sub>E</sub>)',
        x=0.9 * np.min(spkr_rr) * np.reshape(xyz[0, :], T.shape),
        y=0.9 * np.min(spkr_rr) * np.reshape(xyz[1, :], T.shape),
        z=0.9 * np.min(spkr_rr) * np.reshape(xyz[2, :], T.shape),
        cmin=0.7,
        cmax=np.ceil(np.max(rEr)*10)/10,
        surfacecolor=c,
        colorscale='Portland',
        hoverinfo='text',
        visible=False,
        opacity=1.0,
        text=np.vectorize(lambda u, v, c: "rE: %.2f<br>a: %.1f<br>e: %.1f"
                          % (c, u, v))
             (T.az*180/pi, T.el*180/pi, np.reshape(r, T.shape)),
        contours=dict(z=dict(show=True),
                      y=dict(show=True),
                      x=dict(show=True)))

# the speakers
# https://plot.ly/python/alpha-shapes/
# https://plot.ly/python/reference/#mesh3d
spkr_cv_hull = go.Mesh3d(
        name='Speakers (unit sphere)',
        alphahull=0,  # 0 to compute convex hull
        x=spkr_ux,
        y=spkr_uy,
        z=spkr_uz,
        hoverinfo='text',
        visible=True,
        opacity=0.7,
        color='#1f77b4',
        # markers=dict(color='orange', size=15),
        # plot_edges=True,
        # vertexcolor='red',
        showlegend=True,
        # flatshading=True,
        text=np.squeeze(S['id']))

#  https://plot.ly/python/reference/#scatter3d
spkr_locs = go.Scatter3d(
        name='Loudspeakers',
        x=spkr_x,
        y=spkr_y,
        z=spkr_z,
        mode='markers',
        marker=dict(color='orange', size=10, line=dict(color='gray', width=4)),
        hoverinfo='text',
        visible=True,
        text=np.vectorize(
                lambda a, e, r, c:
                     "<b>%s</b><br>az: %.1f&deg;<br>el: %.1f&deg;<br> r: %.1f m"
                     % (c, a, e, r))
                     (spkr_az * 180/pi,
                      spkr_el * 180/pi,
                      spkr_rr,
                      spkr_id))

data = [
        rE_plot,
        spkr_cv_hull,
        cvedges,
        spkr_locs,
        spkr_stands,
        spkr_vector
        ]

name = "Loudspeaker array: " + S['name'] \
        + "<br>Decoder: AllRAD (%dH%dV)" % (C['h_order'], C['v_order']) \
        # + "<br>Energy-Model Localization Vector (r<sub>E</sub>)"


updatemenus = list([
        dict(type="buttons",
             active=-1,
             buttons=list([
                dict(label='Convex Hull',
                     method='update',
                     args=[{'visible': [False, True, True,
                                        True, True, True]},
                           {'title': name + "<br>-----<br>" +
                            spkr_cv_hull['name']
                            # , 'annotations': low_annotations
                            }]),
                dict(label='rE vector',
                     method='update',
                     args=[{'visible': [True, False, False,
                                        True, True, True]},
                           {'title': name + "<br>-----<br>" +
                            rE_plot['name']
                            # , 'annotations': high_annotations
                            }]),
                dict(label='None',
                     method='update',
                     args=[{'visible': [False, False, False,
                                        True, True, True]},
                           {'title': name
                            # , 'annotations': high_annotations
                            }])
                    ]))])


#  https://plot.ly/python/user-guide/#layout
layout = go.Layout(
        title=name + "<br>-----<br>" + spkr_cv_hull['name'],
        showlegend=True,
        legend=dict(orientation="h"),
        updatemenus=updatemenus,
        scene=dict(
                aspectratio=dict(x=1, y=1, z=1),

                xaxis=dict(title='front/back', range=plt_range,
                           showbackground=True,
                           backgroundcolor='rgb(230, 230,230)'),
                yaxis=dict(title='left/right', range=plt_range,
                           showbackground=True,
                           backgroundcolor='rgb(230, 230,230)'),
                zaxis=dict(title='up/down', range=plt_range,
                           showbackground=True,
                           backgroundcolor='rgb(230, 230,230)'),

                annotations=[dict(showarrow=False,
                                  xanchor='center',
                                  font=dict(color="black", size=16),
                                  x=xx, y=yy, z=zz, text=tt)
                             for xx, yy, zz, tt in
                             ((max_rr, 0, 0, 'front'),
                              (-max_rr, 0, 0, 'back'),
                              (0, max_rr, 0, 'left'),
                              (0, -max_rr, 0, 'right'),
                              (0, 0, max_rr, 'top'),
                              (0, 0, -max_rr, 'bottom'))]))

if False:
    layout['scene']['camera'] = \
        dict(up=dict(x=0, y=0, z=1),
             center=dict(x=0, y=0, z=0),
             eye=(dict(x=0, y=0, z=0.5) if interior_view
                  else
                  dict(x=1.25, y=1.25, z=1.25)))

#  https://plot.ly/python/user-guide/#figure
fig = go.Figure(data=data, layout=layout)

if __name__ == '__main__':
    #  https://plot.ly/python/getting-started/#initialization-for-offline-plotting
    if True:
        plotly.offline.plot(fig,
                            filename='plotly/%s-speaker-array-rE.html' % S['name'],
                            include_plotlyjs=True,
                            output_type='file')
    else:
        div = plotly.offline.plot(fig,
                                  filename='plotly/%s-speaker-array.html' % S['name'],
                                  include_plotlyjs=False,
                                  output_type='div')

        with open("plotly/div" + S['name'] + ".html", 'w') as f:
            f.write(div)


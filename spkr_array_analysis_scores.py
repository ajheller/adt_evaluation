#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 15:39:29 2019

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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from numpy import pi
from scipy.spatial import Delaunay

# plotly is not available via conda
# use pip install plotly to install
import plotly


def plot_tri(tri, name="foo", cmap_name="jet"):
    """
    Plot convex hull, with face scores.

    Parameters
    ----------
    tri : TYPE
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is 'foo'.
    cmap : TYPE, optional
        DESCRIPTION. The default is 'jet'.

    Returns
    -------
    None.

    """
    scores = face_scores(tri)

    # set up color map
    cmap = plt.get_cmap(cmap_name)
    cNorm = mpl.colors.Normalize(vmin=0, vmax=pi / 2)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)

    #
    spkr_cv_hull = go.Mesh3d(
        name=name + " Speakers (unit sphere)",
        # alphahull=0,  # 0 to compute convex hull
        x=tri.points[:, 0],
        y=tri.points[:, 1],
        z=tri.points[:, 2],
        i=tri.convex_hull[:, 0],
        j=tri.convex_hull[:, 1],
        k=tri.convex_hull[:, 2],
        hoverinfo="text",
        visible=True,
        opacity=1.0,
        color="#1f77b4",
        facecolor=scalarMap.to_rgba(scores),
        # autocolorscale=True,
        colorscale="Jet",
        showscale=True,
        cmin=0,
        cmax=np.pi,
        # markers=dict(color='orange', size=15),
        flatshading=True,
        # intensity=scores,
        # text=np.squeeze(S['id'] + ["Imaginary1", "Imaginary2"])
    )

    face_end = np.array(((None, None, None),))
    face_indices = np.array((0, 1, 2, 0))
    cvh_edges = np.concatenate(
        [
            np.append(tri.points[f[face_indices]], face_end, axis=0)
            for f in tri.convex_hull
        ],
        axis=0,
    )

    cvedges = go.Scatter3d(
        name="Convex Hull edges",
        x=cvh_edges[:, 0],
        y=cvh_edges[:, 1],
        z=cvh_edges[:, 2],
        mode="lines+markers",
        hoverinfo="none",
        line=dict(color="yellow", width=8),
        # line=dict(color=np.random.random(100), width=10),
        visible=True,
        connectgaps=False,
    )

    # set up figure
    fig = go.Figure(
        data=[spkr_cv_hull, cvedges],
        layout=go.Layout(
            title="Face scores: " + spkr_cv_hull["name"],
            showlegend=True,
        ),
    )

    plotly.offline.plot(
        fig,
        filename="plotly/%s-speaker-array-rE.html" % name,
        include_plotlyjs=True,
        output_type="file",
    )

    return tri


#
def face_scores(tri):
    """
    Score each face in a Delunay triangulation.

    Parameters
    ----------
    tri : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    """
    This function computes the score of each face in a Delaunay
    triangulation, where a face's score is defined as the maximum
    absolute difference between any two of its angles. The indices
    of the output array correspond to the indices of the triples
    representing each face in tri.convex_hull (not the indices
    stored inside tri.convex_hull to refer to the points.)
    """

    # helper function to compute angle between two vectors (dot product divided
    # by product of magnitudes, and then take the arc cosine)
    def angle(v1, v2, axis=0):
        return np.arccos(
            np.sum(v1 * v2, axis=axis)
            / (np.linalg.norm(v1, axis=axis) * np.linalg.norm(v2, axis=axis))
        )

    # 3x3 matrix used for computing consecutive differences (used in computing
    # edges as well as scores)

    consec_diffs = np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
    # represent faces as sets of points
    faces = tri.points[tri.convex_hull]
    # compute edges (tip to tail; vectors representing edges must sum to 0)
    edges = np.matmul(consec_diffs, faces)

    # compute angles between edges (negate one of them so that all edges are
    # tail to tail)
    angles = np.zeros_like(edges[:, 0])
    angles[:, 0] = angle(edges[:, 0], -edges[:, 1], axis=1)
    angles[:, 1] = angle(edges[:, 0], -edges[:, 2], axis=1)
    angles[:, 2] = angle(edges[:, 1], -edges[:, 2], axis=1)

    # compute score as maximum absolute difference between any two angles
    scores = np.max(abs(np.matmul(angles, consec_diffs.T)), axis=1)
    return scores


def delunay_edges(tri, unique=True):
    points = tri.points
    vis, nis = tri.vertex_neighbor_vertices

    e = [nis[vis[i] : vis[i + 1]] for i, p in enumerate(points)]

    if unique:
        u = [vi[vi > i] for i, vi in enumerate(e)]

    return u if unique else e


def edge_scores(tri):
    points = tri.points
    vis, nis = tri.vertex_neighbor_vertices
    if False:
        for i in range(len(vis) - 1):
            for j in nis[vis[i] : vis[i + 1]]:
                print(i, j, np.dot(points[i], points[j]))
    else:
        s = [
            (nis[vis[i] : vis[i + 1]], np.dot(points[nis[vis[i] : vis[i + 1]]], p))
            for i, p in enumerate(points)
        ]
    return s


#
#  -------- examples ----------------

#  ccrma stage
stage = np.array(
    [
        [+8.88943201e-01, +4.52939184e-01, +6.80152907e-02],
        [+8.88943201e-01, -4.52939184e-01, +6.80152907e-02],
        [+4.49572295e-01, +8.82335310e-01, +1.39173101e-01],
        [+4.49572295e-01, -8.82335310e-01, +1.39173101e-01],
        [-4.49572295e-01, +8.82335310e-01, +1.39173101e-01],
        [-4.49572295e-01, -8.82335310e-01, +1.39173101e-01],
        [-8.88943201e-01, +4.52939184e-01, +6.80152907e-02],
        [-8.88943201e-01, -4.52939184e-01, +6.80152907e-02],
        [+8.05091685e-01, +3.41741145e-01, +4.84809620e-01],
        [+8.05091685e-01, -3.41741145e-01, +4.84809620e-01],
        [+3.24481965e-17, +5.29919264e-01, +8.48048096e-01],
        [+3.24481965e-17, -5.29919264e-01, +8.48048096e-01],
        [-7.89026661e-01, +3.34921947e-01, +5.15038075e-01],
        [-7.89026661e-01, -3.34921947e-01, +5.15038075e-01],
        [+3.42020143e-01, +0.00000000e00, +9.39692621e-01],
        [-3.42020143e-01, +4.18853874e-17, +9.39692621e-01],
        [+9.85282381e-01, +1.56053398e-01, +6.97564737e-02],
        [+9.85282381e-01, -1.56053398e-01, +6.97564737e-02],
        [+7.03233176e-01, +7.03233176e-01, +1.04528463e-01],
        [+7.03233176e-01, -7.03233176e-01, +1.04528463e-01],
        [+1.54912056e-01, +9.78076226e-01, +1.39173101e-01],
        [+1.54912056e-01, -9.78076226e-01, +1.39173101e-01],
        [-1.54912056e-01, +9.78076226e-01, +1.39173101e-01],
        [-1.54912056e-01, -9.78076226e-01, +1.39173101e-01],
        [-7.03233176e-01, +7.03233176e-01, +1.04528463e-01],
        [-7.03233176e-01, -7.03233176e-01, +1.04528463e-01],
        [-9.85282381e-01, +1.56053398e-01, +6.97564737e-02],
        [-9.85282381e-01, -1.56053398e-01, +6.97564737e-02],
        [+9.22806073e-01, +2.30081395e-01, +3.09016994e-01],
        [+9.22806073e-01, -2.30081395e-01, +3.09016994e-01],
        [+7.20557188e-01, +5.83495706e-01, +3.74606593e-01],
        [+7.20557188e-01, -5.83495706e-01, +3.74606593e-01],
        [+4.33012702e-01, +7.50000000e-01, +5.00000000e-01],
        [+4.33012702e-01, -7.50000000e-01, +5.00000000e-01],
        [+5.07639105e-17, +8.29037573e-01, +5.59192903e-01],
        [+5.07639105e-17, -8.29037573e-01, +5.59192903e-01],
        [-4.58923545e-01, +7.34431195e-01, +5.00000000e-01],
        [-4.58923545e-01, -7.34431195e-01, +5.00000000e-01],
        [-7.50107495e-01, +5.44984996e-01, +3.74606593e-01],
        [-7.50107495e-01, -5.44984996e-01, +3.74606593e-01],
        [-9.17432633e-01, +2.28741646e-01, +3.25568154e-01],
        [-9.17432633e-01, -2.28741646e-01, +3.25568154e-01],
        [+8.57167301e-01, +0.00000000e00, +5.15038075e-01],
        [+5.30012271e-01, +4.29195475e-01, +7.31353702e-01],
        [+5.30012271e-01, -4.29195475e-01, +7.31353702e-01],
        [-5.65402265e-01, +3.81368643e-01, +7.31353702e-01],
        [-5.65402265e-01, -3.81368643e-01, +7.31353702e-01],
        [-8.38670568e-01, +1.02707523e-16, +5.44639035e-01],
        [+8.77470133e-01, +4.47093364e-01, -1.73648178e-01],
        [+8.77470133e-01, -4.47093364e-01, -1.73648178e-01],
        [+4.40505042e-01, +8.64539823e-01, -2.41921896e-01],
        [+4.40505042e-01, -8.64539823e-01, -2.41921896e-01],
        [-4.40505042e-01, +8.64539823e-01, -2.41921896e-01],
        [-4.40505042e-01, -8.64539823e-01, -2.41921896e-01],
        [-8.77470133e-01, +4.47093364e-01, -1.73648178e-01],
        [-8.77470133e-01, -4.47093364e-01, -1.73648178e-01],
    ]
)

#
#  Steve Katz's home array
skatz = np.array(
    [
        [+9.93307380e-01, +0.00000000e00, +1.15500858e-01],
        [+8.16422993e-01, +5.69597437e-01, -9.49329061e-02],
        [+8.16422993e-01, -5.69597437e-01, -9.49329061e-02],
        [+5.32623641e-01, +8.25566644e-01, +1.86418275e-01],
        [+5.69571772e-01, -7.97400481e-01, +1.99350120e-01],
        [+5.97285351e-17, +9.75441002e-01, +2.20260871e-01],
        [+5.94040954e-17, -9.70142500e-01, +2.42535625e-01],
        [-5.60334907e-01, +8.07924750e-01, +1.82434621e-01],
        [-5.97395105e-01, -7.78002927e-01, +1.94500732e-01],
        [+4.98595351e-01, +6.23244189e-01, +6.02469383e-01],
        [+4.98595351e-01, -6.23244189e-01, +6.02469383e-01],
        [-4.32236723e-01, +6.48355085e-01, +6.26743248e-01],
        [-4.32236723e-01, -6.48355085e-01, +6.26743248e-01],
    ]
)

#
#  Envelop array, as built
envelop = np.array(
    [
        [+9.24663218e-01, +3.80785941e-01, +0.00000000e00],
        [+9.24663218e-01, -3.80785941e-01, +0.00000000e00],
        [+4.67329915e-01, +8.84083000e-01, +0.00000000e00],
        [+4.67329915e-01, -8.84083000e-01, +0.00000000e00],
        [-4.67329915e-01, +8.84083000e-01, +0.00000000e00],
        [-4.67329915e-01, -8.84083000e-01, +0.00000000e00],
        [-9.24663218e-01, +3.80785941e-01, +0.00000000e00],
        [-9.24663218e-01, -3.80785941e-01, +0.00000000e00],
        [+8.70216769e-01, +3.58364326e-01, +3.38079553e-01],
        [+8.70216769e-01, -3.58364326e-01, +3.38079553e-01],
        [+4.17025940e-01, +7.88919202e-01, +4.51326775e-01],
        [+4.17025940e-01, -7.88919202e-01, +4.51326775e-01],
        [-4.17025940e-01, +7.88919202e-01, +4.51326775e-01],
        [-4.17025940e-01, -7.88919202e-01, +4.51326775e-01],
        [-8.70216769e-01, +3.58364326e-01, +3.38079553e-01],
        [-8.70216769e-01, -3.58364326e-01, +3.38079553e-01],
        [+8.70216769e-01, +3.58364326e-01, -3.38079553e-01],
        [+8.70216769e-01, -3.58364326e-01, -3.38079553e-01],
        [+4.17025940e-01, +7.88919202e-01, -4.51326775e-01],
        [+4.17025940e-01, -7.88919202e-01, -4.51326775e-01],
        [-4.17025940e-01, +7.88919202e-01, -4.51326775e-01],
        [-4.17025940e-01, -7.88919202e-01, -4.51326775e-01],
        [-8.70216769e-01, +3.58364326e-01, -3.38079553e-01],
        [-8.70216769e-01, -3.58364326e-01, -3.38079553e-01],
        # [0, 0, 1], [0, 0, -1],
    ]
)

tetra = np.array([[1, 1, 1], [-1, 1, -1], [-1, -1, 1], [1, -1, -1]]) / np.linalg.norm(
    [1, 1, 1]
)

#  ---

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html

# CCRMA stage
stage_tri = Delaunay(stage)

# Steve Katz's home
skatz_tri = Delaunay(skatz)

# envelop
envelop_tri = Delaunay(envelop)
tetra_tri = Delaunay(tetra)

if __name__ == "__main__":
    print("CCRMA stage:\n%s\n" % face_scores(stage_tri))
    plot_tri(stage_tri, "ccrma stage")

    print("Steve Katz's home:\n%s\n" % face_scores(skatz_tri))
    plot_tri(skatz_tri, "Steve Katz")

    print("Envelop:\n%s\n" % face_scores(envelop_tri))
    plot_tri(envelop_tri, "Envelop")

    print("Tetra:\n%s\n" % face_scores(tetra_tri))
    plot_tri(tetra_tri, "tetra", "viridis")

# ###############################################################3
# previous code
#
"""
# interesting attributes
points = tri.points
faces_indices = tri.convex_hull
neighbors = tri.neighbors
ii, pi = tri.vertex_neighbor_vertices


for i in range(len(faces)):
    face = faces[i]
    angles = np.zeros(3)
    angles[0] = np.arccos(np.dot(points[face[0]], points[face[1]]))
    angles[1] = np.arccos(np.dot(points[face[0]], points[face[2]]))
    angles[2] = np.arccos(np.dot(points[face[1]], points[face[2]]))
    score = np.std(angles)
    print("Face %d: score=%.16f" % (i, score/0.1208050699725919))


consec_diffs = np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
# represent faces as sets of points
faces = points[faces_indices]
# compute edges (tip to tail; vectors representing edges must sum to 0)
edges = np.zeros_like(faces)
edges[:, 0] = faces[:, 1]-faces[:, 0]
edges[:, 1] = faces[:, 2]-faces[:, 1]
edges[:, 2] = faces[:, 0]-faces[:, 2]
# compute angles between edges (negate one of them so that all edges are tail to tail)
angles = np.zeros_like(edges[:, 0])
angles[:, 0] = np.arccos(dircos(edges[:, 0], -edges[:, 1], axis=1))
angles[:, 1] = np.arccos(dircos(edges[:, 0], -edges[:, 2], axis=1))
angles[:, 2] = np.arccos(dircos(edges[:, 1], -edges[:, 2], axis=1))
# compute scores as maximum difference between any two angles and print results
# I have to use np.tile because np.diff does not "wrap around"
scores = np.max(np.abs(np.diff(np.tile(angles, 2), axis=1)), axis=1)
print(scores)
"""

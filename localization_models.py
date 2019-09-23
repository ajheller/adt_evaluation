#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 17:29:18 2018

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
from numpy import pi

import spherical_grids as sg
import real_spherical_harmonics as rsh
import basic_decoders as bd

import matplotlib.pyplot as plt


def compute_rVrE_fast(M, Su, Y_test_dirs):
    "this is a fast interface for repeated calls inside an optimizer"

    # pressure & rV
    g = np.matmul(M, Y_test_dirs)
    P = np.sum(g, 0)
    rVxyz = np.real(np.matmul(Su, g) / np.array([P, P, P]))

    # energy & rE
    g2 = np.real(g * g.conjugate())  # the g's might be complex
    E = np.sum(g2, 0)
    rExyz = np.matmul(Su, g2) / np.array([E, E, E])

    return P, rVxyz, E, rExyz


def xyz2aeru(xyz):
    az, el, r = sg.cart2sph(xyz[0, :], xyz[1, :], xyz[2, :])
    u = xyz / np.array((r, r, r))
    return az, el, r, u


def compute_rVrE(l, m, M, Su, test_dirs=sg.az_el()):

    Y_test_dirs = rsh.real_sph_harm_transform(l, m,
                                              test_dirs.az.ravel(),
                                              test_dirs.el.ravel())

    P, rVxyz, E, rExyz = compute_rVrE_fast(M, Su, Y_test_dirs)

    rVaz, rVel, rVr, rVu = xyz2aeru(rVxyz)
    rEaz, rEel, rEr, rEu = xyz2aeru(rExyz)

    return rVr.reshape(test_dirs.shape), rEr.reshape(test_dirs.shape)


def test(order=3, decoder=1, ss=True):
    l, m = zip(*[(l, m) for l in range(order+1) for m in range(-l, l+1)])

    if ss:
        s_az = (pi/4, 3*pi/4, -3*pi/4, -pi/4, 0, 0)
        s_el = (0, 0, 0, 0, pi/2, -pi/2)
    else:
        s = sg.t_design240()
        s_az = s.az
        s_el = s.el

    if decoder == 1:
        M = bd.allrad(l, m, s_az, s_el)
    elif decoder == 2:
        M = bd.allrad2(l, m, s_az, s_el)
    elif decoder == 3:
        M = bd.inversion(l, m, s_az, s_el)
    else:
        raise ValueError("Unknown decoder type: %d" % decoder)

    rVr, rEr, = compute_rVrE(l, m, M,
                             np.array(sg.sph2cart(s_az, s_el)))

    plot_rX(rVr, "rVr", (0.5, 1))
    plot_rX(rEr, "rEr", (0.5, 1))

    return rVr, rEr


def plot_rX(rX, title, clim=None):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    plt.imshow(np.fliplr(np.flipud(rX.transpose())),
               extent=(180, -180, -90, 90),
               cmap='jet')
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

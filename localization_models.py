#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 17:29:18 2018

@author: heller
"""

from __future__ import division, print_function
import numpy as np
from numpy import pi

import spherical_grids as sg
import real_spherical_harmonics as rsh
import basic_decoders as bd

import matplotlib.pyplot as plt


def compute_rVrE(l, m, M, Su):

    test_dirs = sg.az_el()
    Y_test_dirs = rsh.real_sph_harm_transform(l, m,
                                              test_dirs.az.ravel(),
                                              test_dirs.el.ravel())
    g = np.matmul(M, Y_test_dirs)
    g2 = np.real(g * g.conjugate())  # if g's might be complex

    # pressure & rV
    P = np.sum(g, 0)
    rVxyz = np.real(np.matmul(Su, g) / np.array([P, P, P]))
    rVaz, rVel, rVr = sg.cart2sph(rVxyz[0, :], rVxyz[1, :], rVxyz[2, :])
    rVu = rVxyz / np.array([rVr, rVr, rVr])

    # energy & rE
    E = np.sum(g2, 0)
    rExyz = np.matmul(Su, g2) / np.array([E, E, E])
    rEaz, rEel, rEr = sg.cart2sph(rExyz[0, :], rExyz[1, :], rExyz[2, :])
    rEu = rExyz / np.array([rEr, rEr, rEr])

    return [np.reshape(rX, test_dirs.shape) for rX in [rVr, rEr]]


def test(order=3, decoder=1, ss=True):
    l, m = zip(*[(l,m) for l in range(order+1) for m in range(-l, l+1)])

    if ss:
        s_az = (pi/4, 3*pi/4, -3*pi/4, -pi/4, 0, 0)
        s_el = (0, 0, 0, 0, pi/2, -pi/2)
    else:
        s = sg.t_design()
        s_az = s.az
        s_el = s.el

    if decoder == 1:
        M = bd.allrad(l, m, s_az, s_el)
    else:
        M = bd.inversion(l, m, s_az, s_el)

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

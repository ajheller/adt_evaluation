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
from numpy import pi as π

import spherical_grids as sg
import real_spherical_harmonics as rsh
# import basic_decoders as bd
import shelf

import matplotlib.pyplot as plt


def compute_rVrE_fast(M, Su, Y_test_dirs):
    """
    Compute rV and rE via fast interface for use in optimizers.

    Parameters
    ----------
    M : TYPE
        DESCRIPTION.
    Su : TYPE
        DESCRIPTION.
    Y_test_dirs : TYPE
        DESCRIPTION.

    Returns
    -------
    P : TYPE
        DESCRIPTION.
    rVxyz : TYPE
        DESCRIPTION.
    E : TYPE
        DESCRIPTION.
    rExyz : TYPE
        DESCRIPTION.

    """
    #
    # pressure & rV
    g = np.matmul(M, Y_test_dirs)
    P = np.sum(g, 0)
    rVxyz = np.real(np.matmul(Su, g) / P)

    # energy & rE
    g2 = np.real(g * g.conjugate())  # the g's might be complex
    E = np.sum(g2, 0)
    rExyz = np.matmul(Su, g2) / E

    return P, rVxyz, E, rExyz


def xyz2aeru(xyz):
    """Cartesian to az, el, radius, unit_vector."""
    az, el, r = sg.cart2sph(*xyz)
    u = xyz / r  # np.array((r, r, r))
    return az, el, r, u


def compute_rVrE(l, m, M, Su, test_dirs=sg.az_el()):
    """Compute rV and rE, single call interface."""
    Y_test_dirs = rsh.real_sph_harm_transform(l, m,
                                              test_dirs.az.ravel(),
                                              test_dirs.el.ravel())

    P, rVxyz, E, rExyz = compute_rVrE_fast(M, Su, Y_test_dirs)

    rVaz, rVel, rVr, rVu = xyz2aeru(rVxyz)
    rEaz, rEel, rEr, rEu = xyz2aeru(rExyz)

    return rVr.reshape(test_dirs.shape), rEr.reshape(test_dirs.shape)


def plot_rX(rX, title, clim=None, cmap='jet', show=True):
    """
    Plot rV or rE magnitude.

    Parameters
    ----------
    rX : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    clim : TYPE, optional
        DESCRIPTION. The default is None.
    cmap : TYPE, optional
        DESCRIPTION. The default is 'jet'.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    plt.imshow(np.fliplr(np.flipud(rX.transpose())),
               extent=(180, -180, -90, 90),
               cmap=cmap)
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
        pass  # plotly.offline.plot_mpl(fig)
    else:
        if show:
            plt.show()

    return fig


def plot_loudspeakers(Su, **plot_args):
    # unit vector to az-el
    S_az, S_el, *_ = sg.cart2sph(*Su)
    plt.scatter(S_az * 180/π, S_el * 180/π, **plot_args)


def plot_performance(M, Su, degree,
                     title="",
                     plot_spkrs=True):

    out_figs = []

    l, m = zip(*rsh.lm_generator(degree))
    test_dirs = sg.az_el()
    Y_test_dirs = rsh.real_sph_harm_transform(l, m,
                                              test_dirs.az.ravel(),
                                              test_dirs.el.ravel())

    P, rVxyz, E, rExyz = compute_rVrE_fast(M, Su, Y_test_dirs)

    rVaz, rVel, rVr, rVu = xyz2aeru(rVxyz)
    rEaz, rEel, rEr, rEu = xyz2aeru(rExyz)

    # the arg to arccos can get epsilon larger than 1 due to round off,
    # which produces NaNs, so clip to [-1, 1]
    rE_dir_err = np.arccos(np.clip(np.sum(rEu * test_dirs.u, axis=0),
                                   -1, 1)) * 180/π

    # magnitude of rE
    if True:
        fig = plot_rX(rEr.reshape(test_dirs.shape),
                      title=(f'{title}, order={degree}\n' +
                             'magnitude of rE vs. test direction'),
                      clim=(0.5, 1))
        out_figs.append(fig)

        # contour plot of rE
        fig = plt.figure(figsize=(10, 5))
        plt.contourf(rEr.reshape(test_dirs.shape).T,
                     np.convolve((0.5, 0.5),
                                 [0] + [shelf.max_rE_3d(o)
                                        for o in range(0, degree+2)] + [1, 1],
                                 'valid'),
                     extent=(180, -180, -90, 90),
                     cmap='jet')
        plt.xlabel("azimuth (degrees)")
        plt.ylabel("elevation (degrees)")
        plt.colorbar()

        if plot_spkrs:
            plot_loudspeakers(Su, c='w', marker='D')

        plt.title(f"{title}, order={degree}\n" +
                  "magnitude of rE vs. test direction")
        out_figs.append(fig)
        plt.show()

    # E vs td
    if True:
        E_dB = 10*np.log10(E.reshape(test_dirs.shape))
        E_dB_ceil = np.ceil(E_dB.max())
        fig = plot_rX(E_dB,
                      title=(f'{title}, order={degree}\n' +
                             'E (dB) vs. test_direction'),
                      clim=(E_dB_ceil-20, E_dB_ceil)
                      )
        out_figs.append(fig)

    # direction error
    if True:
        plot_rX(rE_dir_err.reshape(test_dirs.shape),
                title='%s, order=%d\ndir error' % (title, degree),
                clim=(0, 20))

        fig = plt.figure(figsize=(10, 5))
        plt.contourf(rE_dir_err.reshape(test_dirs.shape).T,
                     list(range(0, 15, 2)),
                     extent=(180, -180, -90, 90),
                     cmap='jet')
        plt.xlabel("azimuth (degrees)")
        plt.ylabel("elevation (degrees)")
        plt.colorbar()

        # overlay loudspeaker positions
        if plot_spkrs:
            plot_loudspeakers(Su, c='w', marker='D')

        plt.title('%s, order=%d\ndirection error' % (title, degree))
        out_figs.append(fig)
        plt.show()

    return out_figs


def plot_matrix(M, title=""):
    fig = plt.figure()
    plt.matshow(20*np.log10(np.abs(M)), cmap='jet')
    plt.colorbar()
    plt.clim((-60, 0))
    plt.title("%s\nMatrix element gains (dB)" % title)
    plt.xlabel("Program channels (ACN order)")
    plt.ylabel("Loudspeakers")
    plt.show()
    return fig


if __name__ == "__main__":
    import basic_decoders as bd

    def test(order=3, decoder=1, ss=False):
        """
        Basic Decoders unit tests.

        Parameters
        ----------
        order : TYPE, optional
            DESCRIPTION. The default is 3.
        decoder : TYPE, optional
            DESCRIPTION. The default is 1.
        ss : TYPE, optional
            DESCRIPTION. The default is True.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        rVr : TYPE
            DESCRIPTION.
        rEr : TYPE
            DESCRIPTION.

        """
        l, m = zip(*[(l, m) for l in range(order+1) for m in range(-l, l+1)])

        if ss:
            s_az = (π/4, 3*π/4, -3*π/4, -π/4, 0, 0)
            s_el = (0, 0, 0, 0, π/2, -π/2)
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

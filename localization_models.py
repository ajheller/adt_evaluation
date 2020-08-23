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
# import basic_decoders as bd

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
    rVxyz = np.real(np.matmul(Su, g) / P) # np.array([P, P, P]))

    # energy & rE
    g2 = np.real(g * g.conjugate())  # the g's might be complex
    E = np.sum(g2, 0)
    rExyz = np.matmul(Su, g2) / E  #np.array([E, E, E])

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


def plot_rX(rX, title, clim=None, cmap='jet'):
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
        pass # plotly.offline.plot_mpl(fig)
    else:
        plt.show()

    return fig


def plot_performance(M, Su, degree, title=""):
    l, m = zip(*rsh.lm_generator(degree))
    test_dirs = sg.az_el()
    Y_test_dirs = rsh.real_sph_harm_transform(l, m,
                                              test_dirs.az.ravel(),
                                              test_dirs.el.ravel())

    P, rVxyz, E, rExyz = compute_rVrE_fast(M, Su, Y_test_dirs)

    rVaz, rVel, rVr, rVu = xyz2aeru(rVxyz)
    rEaz, rEel, rEr, rEu = xyz2aeru(rExyz)

    rE_dir_err = np.arccos(np.sum(rEu * test_dirs.u, axis=0)) * 180/np.pi

    plot_rX(rEr.reshape(test_dirs.shape),
            title='%s, order=%d\nrE vs. test direction' % (title, degree),
            clim=(0.5, 1))
    plot_rX(10*np.log10(E.reshape(test_dirs.shape)),
            title='%s, order=%d\nE (dB) vs. test_direction' % (title, degree),
            clim=(-6, 6)
            )
    plot_rX(rE_dir_err.reshape(test_dirs.shape),
            title='%s, order=%d\ndir error' % (title, degree),
            clim=(0, 20))




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

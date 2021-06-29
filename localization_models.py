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


import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as π

import real_spherical_harmonics as rsh
import shelf
import spherical_grids as sg


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


def compute_rVrE(sh_l, sh_m, M, Su, *, test_dirs=sg.az_el()):
    """Compute rV and rE, single call interface."""
    Y_test_dirs = rsh.real_sph_harm_transform(sh_l, sh_m,
                                              test_dirs.az.ravel(),
                                              test_dirs.el.ravel())

    P, rVxyz, E, rExyz = compute_rVrE_fast(M, Su, Y_test_dirs)

    rVaz, rVel, rVr, rVu = xyz2aeru(rVxyz)
    rEaz, rEel, rEr, rEu = xyz2aeru(rExyz)

    return rVr.reshape(test_dirs.shape), rEr.reshape(test_dirs.shape)


def compute_rVrE_dict(sh_l, sh_m, M, Su, *, test_dirs=None,
                      return_dict=True):
    """Compute rV and rE, single call interface."""
    #
    if test_dirs is None:
        test_dirs = sg.az_el()
    Y_test_dirs = rsh.real_sph_harm_transform(sh_l, sh_m,
                                              test_dirs.az.ravel(),
                                              test_dirs.el.ravel())

    P, rVxyz, E, rExyz = compute_rVrE_fast(M, Su, Y_test_dirs)

    # FIXME: I think this works, but needs to be tested, same for rExyz
    # FIXME: ... for now just reshape the scalar quantities
    # rVxyz = rVxyz.reshape(3, *test_dirs.shape)
    # rExyz = rExyz.reshape(3, *test_dirs.shape)

    P = P.reshape(test_dirs.shape)
    E = E.reshape(test_dirs.shape)

    rVaz, rVel, rVr, rVu = xyz2aeru(rVxyz)
    rEaz, rEel, rEr, rEu = xyz2aeru(rExyz)

    rVaz = rVaz.reshape(test_dirs.shape)
    rVel = rVel.reshape(test_dirs.shape)
    rEaz = rEaz.reshape(test_dirs.shape)
    rEel = rEel.reshape(test_dirs.shape)

    return dict(P=P, rVxyz=rVxyz,
                rVaz=rVaz, rVel=rVel, rVr=rVr, rVu=rVu,
                E=E, rExyz=rExyz,
                rEaz=rEaz, rEel=rEel, rEr=rEr, rEu=rEu,
                M=M, sh_l=sh_l, sh_m=sh_m, test_dirs=test_dirs)


def plot_az_el_grid(sh_l, sh_m, M, Su, el_lim=-π/4, title=None, show=True):
    p = compute_rVrE_dict(sh_l, sh_m, M, Su)
    az = p['rEaz']
    el = p['rEel']
    el = np.unwrap(el, axis=0)
    az = np.unwrap(az, axis=1)

    el = np.unwrap(el, axis=1)
    az = np.unwrap(az, axis=0)

    while np.mean(az) > np.pi:
        az -= 2*np.pi
    while np.mean(az) < -np.pi:
        az += 2*np.pi

    taz = p['test_dirs'].az  # not used currently
    tel = p['test_dirs'].el

    # magic incantation to flip the x-axis of the plot
    plt.gca().invert_xaxis()

    # plot lines of constant elevation
    for iy in range(0, 180, 10):
        if tel[0, iy] > el_lim:
            plt.plot(az[1:-1, iy]*180/np.pi,
                     el[1:-1, iy]*180/np.pi,
                     zorder=1000)

    # plot lines of constant azimuth
    for ix in range(0, 361, 10):
        el_plot_range = (tel[0, :] > el_lim) & (tel[0, :] < π/2)
        # NOTE: use the following line to get plots identical to AES150 paper
        # el_plot_range = slice(85, -1)
        # plt.plot(az[ix, 85:-1]*180/np.pi, el[ix, 85:-1]*180/np.pi)
        plt.plot(az[ix, el_plot_range]*180/np.pi,
                 el[ix, el_plot_range]*180/np.pi,
                 zorder=1000)

    if True:
        line_color = "xkcd:light gray"
        # plot lines of constant azimuth
        for iy in range(0, 180, 10):
            if tel[0, iy] > el_lim:
                plt.plot(taz[1:-1, iy]*180/np.pi,
                         tel[1:-1, iy]*180/np.pi,
                         color=line_color,
                         zorder=0)

        # plot lines of constant azimuth
        for ix in range(0, 361, 10):
            el_plot_range = (tel[0, :] > el_lim) & (tel[0, :] < π/2)
            # NOTE: use the following to get plots identical to AES150 paper
            # el_plot_range = slice(85, -1)
            # plt.plot(az[ix, 85:-1]*180/np.pi, el[ix, 85:-1]*180/np.pi)
            plt.plot(taz[ix, el_plot_range]*180/np.pi,
                     tel[ix, el_plot_range]*180/np.pi,
                     color=line_color,
                     zorder=0)

    else:
        plt.grid()
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def diffuse_field_gain(M):
    g_spkr = np.sum(M*M, axis=1)
    g_total = np.sum(g_spkr)
    return g_spkr, g_total


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


def plot_loudspeakers(Su: np.ndarray, **plot_args) -> None:
    """Overlay loudspeaker positions on an existing figure."""
    #
    # unit vector to az-el
    S_az, S_el, *_ = sg.cart2sph(*Su)
    # a white diamond with a black dot in the center
    plt.scatter(S_az * 180 / π, S_el * 180 / π, c='w', marker='D', **plot_args)
    plt.scatter(S_az * 180 / π, S_el * 180 / π, c='k', marker='.', **plot_args)


def plot_performance(M, Su, sh_l, sh_m,  # /,  # / instroduced in 3.8
                     mask_matrix=None,
                     title="",
                     plot_spkrs=True,
                     test_dirs=None,
                     el_lim=-π/4):
    """Compute and plot basic performance metrics of a decoder matrix."""
    # fill in defaults
    if test_dirs is None:
        test_dirs = sg.az_el()

    ambisonic_order = np.max(sh_l)  # FIXME: is this correct for mixed orders?

    # we want to return a list of all the figures to make the HTML report.
    # accumulate them in out_figs
    out_figs = []

    Y_test_dirs = rsh.real_sph_harm_transform(sh_l, sh_m,
                                              test_dirs.az.ravel(),
                                              test_dirs.el.ravel())
    if mask_matrix is not None:
        Y_test_dirs = mask_matrix @ Y_test_dirs

    P, rVxyz, E, rExyz = compute_rVrE_fast(M, Su, Y_test_dirs)

    rVaz, rVel, rVr, rVu = xyz2aeru(rVxyz)
    rEaz, rEel, rEr, rEu = xyz2aeru(rExyz)

    # the arg to arccos can get epsilon larger than 1 due to round off,
    # which produces NaNs, so clip to [-1, 1]
    rE_dir_err = np.arccos(np.clip(np.sum(rEu * test_dirs.u.T, axis=0),
                                   -1, 1)) * 180 / π

    # magnitude of rE
    if True:
        fig = plot_rX(rEr.reshape(test_dirs.shape),
                      title=(f'{title}\n' +
                             'magnitude of rE vs. test direction'),
                      clim=(0.5, 1),
                      show=False)
        out_figs.append(fig)
        plt.show()

    # plot of ambisonic order
    if True:
        fig = plot_rX(
            (shelf.rE_to_ambisonic_order_3d(
                rEr.reshape(test_dirs.shape)) - ambisonic_order).round(),
            title=(f'{title}\n'
                   f'relative ambisonic order '
                   f'({ambisonic_order}) vs. test direction'),
            clim=(-3, +3),
            cmap='Spectral_r',
            show=False)

        if plot_spkrs:
            plot_loudspeakers(Su)

        out_figs.append(fig)
        plt.show()

    # E vs td
    if True:
        E_dB = 10 * np.log10(E.reshape(test_dirs.shape))
        E_dB_ceil = np.ceil(E_dB.max())
        fig = plot_rX(E_dB,
                      title=(f'{title}\n' +
                             'E (dB) vs. test_direction'),
                      clim=(E_dB_ceil - 20, E_dB_ceil),
                      show=False,
                      )
        out_figs.append(fig)
        plt.show()

    # direction error
    if True:
        fig = plot_rX(rE_dir_err.reshape(test_dirs.shape),
                      title=f'{title}\n' +
                            'direction error (deg)',
                      clim=(0, 20),
                      show=False)
        out_figs.append(fig)
        plt.show()

        fig = plot_rX(((rE_dir_err.reshape(test_dirs.shape) / 3).round()) * 3,
                      title=f'{title}\n' +
                            'direction error (deg)',
                      clim=(0, 20),
                      show=False)

        # overlay loudspeaker positions
        if plot_spkrs:
            plot_loudspeakers(Su)

        out_figs.append(fig)
        plt.show()

    if True:
        fig = plt.figure(figsize=(10, 4))
        plot_az_el_grid(sh_l, sh_m, M, Su,
                        title=f"{title}\nrE directions",
                        el_lim=el_lim,
                        show=False)
        if plot_spkrs:
            plot_loudspeakers(Su, zorder=1500)

        out_figs.append(fig)
        plt.show()

    if True:
        fig = plot_rX(rVr.reshape(test_dirs.shape),
                      title=(f'{title}\n' +
                             'magnitude of rV vs. test direction'),
                      #clim=(0.5, 1),
                      clim=(0.5, 1.5),
                      show=False)
        out_figs.append(fig)
        plt.show()

    if True:
        fig = plot_matrix(M, title=f"{title}")
        out_figs.append(fig)

    return out_figs


def plot_performance_LF(M_lf, M_hf, Su, sh_l, sh_m, el_lim=-π/4,
                        title=""):

    T = sg.az_el()
    Y_test_dirs = rsh.real_sph_harm_transform(sh_l, sh_m, T.az, T.el)

    P, rVxyz, _, _ = compute_rVrE_fast(M_lf, Su, Y_test_dirs)
    rVaz, rVel, rVr, rVu = xyz2aeru(rVxyz)

    print("mean rV", np.mean(rVr))

    _, _, E, rExyz = compute_rVrE_fast(M_hf, Su, Y_test_dirs)
    rEaz, rEel, rEr, rEu = xyz2aeru(rExyz)

    out_figs = []

    fig = plot_rX(rVr.reshape(T.shape),
                  title=f"{title}\nMagnitude of rV vs. test direction",
                  clim=(0.7, 1.1),
                  show=False)
    out_figs.append(fig)
    plt.show()

    ev_dot = np.sum(rEu * rVu, axis=0)
    dir_diff = np.arccos(np.clip(ev_dot, -1, 1)) * 180/np.pi
    print("mean rV/rE direction error", np.mean(dir_diff))
    fig = plot_rX(dir_diff.reshape(T.shape),
                  title=f"{title}\nrE vs. rV direction difference (degrees)",
                  clim=(0, 20),
                  show=False)
    out_figs.append(fig)
    plt.show()

    return out_figs


def plot_matrix(M, title="", min_dB=-60):
    """Display the matrix as an image."""
    fig = plt.figure()
    M_clipped = np.clip(np.abs(M), 10**(min_dB/20), np.inf)
    plt.matshow(20 * np.log10(M_clipped.T), fignum=0, cmap='jet')
    plt.colorbar()
    plt.clim((min_dB, 0))
    plt.title("%s\nMatrix element gains (dB)" % title)
    plt.ylabel("Program channels (ACN order)")
    plt.xlabel("Loudspeakers")
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
        sh_l, sh_m = zip(*rsh.lm_generator(order))

        if ss:
            s_az = (π / 4, 3 * π / 4, -3 * π / 4, -π / 4, 0, 0)
            s_el = (0, 0, 0, 0, π / 2, -π / 2)
        else:
            s = sg.t_design240()
            s_az = s.az
            s_el = s.el

        if decoder == 1:
            M = bd.allrad(sh_l, sh_m, s_az, s_el)
        elif decoder == 2:
            M = bd.allrad2(sh_l, sh_m, s_az, s_el)
        elif decoder == 3:
            M = bd.inversion(sh_l, sh_m, s_az, s_el)
        else:
            raise ValueError("Unknown decoder type: %d" % decoder)

        rVr, rEr, = compute_rVrE(sh_l, sh_m, M,
                                 np.array(sg.sph2cart(s_az, s_el)))

        plot_rX(rVr, "rVr", (0.5, 1))
        plot_rX(rEr, "rEr", (0.5, 1))

        return rVr, rEr

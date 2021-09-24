#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:57:11 2018

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



import numpy as np
from scipy.spatial import Delaunay  # for AllRAD decoder

import ray_triangle_intersection as rti
import real_spherical_harmonics as rsh
import spherical_grids as sg
from localization_models import compute_rVrE, plot_rX


def channel_spec(degree, order, norm=1, cs_phase=None):
    def cs(l, m, n=1): return l, m, n

    return [cs(l, m) for l in range(degree + 1) for m in range(-l, l + 1)]


def projection(degree, order,
               speakers_azimuth, speakers_elevation):
    """Compute basic decoder matrix by projection method.

    Parameters
    ----------
        degree: int or array-like collection of ints
            degree (l) of each spherical harmonic.
        order: int or array-like collection of ints
            order (m) of each spherical harmonic.
        speakers_azimuth, speakers_elevation: array-like collection of floats:
            speaker azimuths and elevations in radians.

    Returns
    -------
        numpy.ndarray
            Basic decoder matrix

    Note
    ----
        Optimal for platonic solids, more generally spherical designs only.
        This is here mostly for comparison to other methods.
    """
    if np.isscalar(degree):
        sh_l, sh_m, *_ = zip(*channel_spec(degree, order))
    else:
        sh_l, sh_m = degree, order

    M = rsh.real_sph_harm_transform(sh_l, sh_m,
                                    np.array(speakers_azimuth).ravel(),
                                    np.array(speakers_elevation).ravel())
    return M.transpose()


def inversion(degree, order,
              speakers_azimuth, speakers_elevation):
    """
    Compute basic decoder matrix by inversion method (aka, mode matching).

    Parameters
    ----------
        degree: (array-like)
            degree (l) of each spherical harmonic.
        order: (array-like)
            order (m) of each sperical harmonic.
        speakers_azimuth: (array-like)
            speaker azimuths in radians.
        speakers_elevation: (array-like)
            speaker elevations in radians.

    Returns
    -------
        Basic decoder matrix

    Note
    ----
        Optimal for uniform arrays only.

    """
    M_proj = projection(degree, order, speakers_azimuth, speakers_elevation)
    M = np.linalg.pinv(M_proj.transpose())

    return M


def constant_energy_inversion(degree, order,
                              speakers_azimuth, speakers_elevation,
                              alpha=1):
    """
    Compute basic decoder matrix by Energy-Limited Inversion.

    Parameters
    ----------
    degree: (array-like)
        degree (l) of each spherical harmonic.
    order: (array-like)
        order (m) of each sperical harmonic.
    speakers_azimuth: (array-like)
        speaker azimuths in radians.
    speakers_elevation: (array-like)
        speaker elevations in radians.
    alpha : float, optional
        mixing coefficient between inversion (α=0) and constant energy (α=1).
        The default is 1. Intermediate values not implemented yet!

    Returns
    -------
    M : nd-array
        Basic decoder matrix

    """
    M_proj = projection(degree, order, speakers_azimuth, speakers_elevation)
    U, S, V = np.linalg.svd(M_proj, full_matrices=False, compute_uv=True)
    print("Singular values = ", S)

    # TODO do something clever with the singular values here
    #  for constant energy set all the singular values to one
    #  for mode matching PINV, invert non-zero entried of S
    if alpha == 0:
        # this is pseudinverse (pinv)
        Sinv = 1 / S
        Sinv[np.isclose(S, 0, atol=1e-5)] = 0
    elif alpha == 1:
        Sinv = np.ones_like(S)
    else:
        raise NotADirectoryError("Sorry, mixing not implemented yet!")

    M = np.matmul(V.T, np.diag(Sinv), U.T)

    return M


def allrad(degree, order,
           speakers_azimuth, speakers_elevation,
           speaker_is_real=None,
           v_az=None, v_el=None,
           vbap_norm=True,
           allrad2=False):
    """
    Compute basic decoder matrix by the AllRAD method.

    Parameters
    ----------
    degree : TYPE
        DESCRIPTION.
    order : TYPE
        DESCRIPTION.
    speakers_azimuth : TYPE
        DESCRIPTION.
    speakers_elevation : TYPE
        DESCRIPTION.
    speaker_is_real : TYPE, optional
        DESCRIPTION. The default is None.
    v_az : TYPE, optional
        DESCRIPTION. The default is None.
    v_el : TYPE, optional
        DESCRIPTION. The default is None.
    vbap_norm : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    M : TYPE
        DESCRIPTION.

    """
    # defaults
    if v_az is None:
        td = sg.t_design5200()
        v_az = td.az
        v_el = td.el

    Su = np.array(sg.sph2cart(speakers_azimuth, speakers_elevation))
    Vu = np.array(sg.sph2cart(v_az, v_el))

    V2R, Vtri, Vxyz = allrad_v2rp(Su, Vu, vbap_norm=vbap_norm)

    Mv = inversion(degree, order, v_az, v_el)
    M = np.matmul(V2R, Mv)
    
    if allrad2:
      M_sign = np.sign(M)
      M = M_sign * np.sqrt(V2R**2 @ Mv**2)

    if speaker_is_real is not None:
        # get rid of rows corresponding to imaginary speakers
        M = M[speaker_is_real]

    return M


def allrad2(degree, order,
            spkrs_az, spkrs_el,
            v_az=None, v_el=None,
            vbap_norm=True):
    """
    Compute decoder by AllRAD2 method.  Not implemented.

    Parameters
    ----------
    degree : TYPE
        DESCRIPTION.
    order : TYPE
        DESCRIPTION.
    spkrs_az : TYPE
        DESCRIPTION.
    spkrs_el : TYPE
        DESCRIPTION.
    v_az : TYPE, optional
        DESCRIPTION. The default is None.
    v_el : TYPE, optional
        DESCRIPTION. The default is None.
    vbap_norm : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    M_allrad : TYPE
        DESCRIPTION.

    """
    raise NotImplementedError("Sorry... AllRAD2 not implemented yet.")
    # defaults
    if v_az is None:
        td = sg.t_design5200()
        v_az = td.az
        v_el = td.el

    # TODO understand and implement allrad2 :) :)
    M_allrad = allrad(degree, order,
                      spkrs_az, spkrs_el,
                      v_az=v_az, v_el=v_el,
                      vbap_norm=vbap_norm)

    # TODO rest of AllRAD2 method

    return M_allrad


def allrad_v2rp(Su, Vu, vbap_norm=True):
    """
    Compute gain matrix for virtual to real speaker array. Fast method.

    Parameters
    ----------
    Su : TYPE
        DESCRIPTION.
    Vu : TYPE
        DESCRIPTION.
    vbap_norm : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    V2R : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    tri : TYPE
        DESCRIPTION.

    """
    origin = np.array([0, 0, 0])

    n_real_speakers = Su.shape[1]
    n_virtual_speakers = Vu.shape[1]

    tri = Delaunay(Su.transpose())
    H = tri.convex_hull

    # is the origin (listening position) inside the convex hu??
    if tri.find_simplex(origin) < 0:
        raise ValueError(f"The origin {origin} is not contained in the array.")

    p0 = tri.points[H[:, 0], :]
    p1 = tri.points[H[:, 1], :]
    p2 = tri.points[H[:, 2], :]

    a = []
    Hr = np.arange(len(H))

    V2R = np.zeros((n_real_speakers, n_virtual_speakers))
    for i in range(n_virtual_speakers):
        flag, u, v, t = rti.ray_triangle_intersection_p1(origin, Vu[:, i],
                                                         p0, p1, p2)
        valid = flag & (t > 0)  # np.logical_and(flag, t > 0)
        if np.sum(valid) == 1:
            face = Hr[valid][0]
            ur = u[valid][0]
            vr = v[valid][0]
            tr = t[valid][0]
            a.append((face, ur, vr, tr))

            b = np.array([1 - ur - vr, ur, vr])
            if vbap_norm:
                b = b / np.linalg.norm(b)

            V2R[H[face, :], i] = b
        else:
            if np.sum(valid) > 1:
                print("multiple intersections: " + str(i))
            else:
                print("no intersections: " + str(i))

    return V2R, a, tri


def allrad_v2r(Su, Vu):
    """
    Compute gain matrix for virtual to real speaker array.

    Parameters
    ----------
    Su : TYPE
        DESCRIPTION.
    Vu : TYPE
        DESCRIPTION.

    Returns
    -------
    V2R : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    Vxyz : TYPE
        DESCRIPTION.

    """
    tri = Delaunay(Su.transpose())
    H = tri.convex_hull

    o = np.array([0, 0, 0])
    # which face does vertex i intersect?
    Vtri = np.zeros(Vu.shape[1], dtype=np.integer)
    # the coordinates of the intersection
    Vxyz = np.zeros(Vu.shape)

    # the discretization matrix
    V2R = np.zeros((Su.shape[1], Vu.shape[1]))
    a = []
    for i in range(Vu.shape[1]):  # iterate over the virtual loudspeakers
        d = Vu[:, i]
        for j in range(len(H)):
            p0 = Su[:, H[j, 0]]
            p1 = Su[:, H[j, 1]]
            p2 = Su[:, H[j, 2]]
            flag, bu, bv, bt = rti.ray_triangle_intersection(o, d, p0, p1, p2)
            bw = 1 - bu - bv

            if flag and bt > 0:
                a.append((j, bu, bv, bt))
                # virtual speaker i intersects face j
                Vtri[i] = j
                # coordinates of intersection for plotting
                Vxyz[:, i] = bw * p0 + bu * p1 + bv * p2
                # fill in gains, normalize for energy
                b = np.array([bw, bu, bv])

                V2R[H[j, :], i] = b / np.linalg.norm(b)

                break
    return V2R, a, tri  # Vtri, Vxyz


# unit tests
def unit_test():
    import loudspeaker_layout as lsl

    # horitontal square array
    s = lsl.from_vectors((45, 135, -135, -45), 0, 1,
                         coord_code=('az', 'el', 'radius'),
                         unit_code=('degrees', 'degrees', 'meters'))

    l = (0, 1, 1)
    m = (0, 1, -1)

    M_pinv = inversion(l, m, s.az, s.el)
    M_proj = projection(l, m, s.az, s.el)

    M_check = np.matmul(M_proj.T, M_pinv)

    max_error = np.max(np.abs(M_check - np.identity(M_check.shape[0])))

    if np.isclose(max_error, 0):
        print("Pass:", max_error)
    else:
        print("FAIL:", max_error)

    return M_pinv, M_proj, M_check, max_error


def unit_test2(order=3, case=1, debug=True):
    """
    Run basic decoders unit tests.

    Parameters
    ----------
    order : TYPE, optional
        DESCRIPTION. The default is 3.
    case : TYPE, optional
        DESCRIPTION. The default is 0.
    debug : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    M_pinv : TYPE
        DESCRIPTION.
    M_proj : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    M_allrad : TYPE
        DESCRIPTION.
    M_allrad2 : TYPE
        DESCRIPTION.

    """
    import example_speaker_arrays as esa
    import loudspeaker_layout as LSL
    import spherical_data as sd
    import spherical_grids as sg

    if case == 0:
        # octagon
        s = LSL.from_vectors((45, 135, -135, -45, 0, 0),
                             (0, 0, 0, 0, 90, -90), 1,
                             coord_code='AER', unit_code='DDM')
        order = max(order, 1)
    elif case == 1:
        s = sg.t_design240()
    elif case == 2:
        s = esa.iem_cube().append(esa.nadir())
    else:
        print('unknown case')
        return None

    if debug:
        print(s.az)
        print(s.el)
        print(s.r)

    sh_l, sh_m = zip(*[(l, m)
                       for l in range(order + 1)
                       for m in range(-l, l + 1)])

    M_pinv = inversion(sh_l, sh_m, s.az, s.el)
    # fuzz to zero
    M_pinv[np.isclose(0, M_pinv, atol=1e-10, rtol=1e-10)] = 0

    M_proj = projection(sh_l, sh_m, s.az, s.el)

    p = np.allclose(np.matmul(M_proj.transpose(), M_pinv), np.eye(len(sh_l)))

    M_allrad = allrad(sh_l, sh_m, s.az, s.el)

    M_allrad2 = allrad2(sh_l, sh_m, s.az, s.el)

    rV, rE = compute_rVrE(sh_l, sh_m, M_allrad,
                          np.array(sd.sph2cart(s.az, s.el)))

    plot_rX(rV, 'rVr', [0.5, 1])
    plot_rX(rE, 'rEr', [0.5, 1])

    return M_pinv, M_proj, p, M_allrad, M_allrad2


# v2rp, ap, trip = allrad_v2rp(Su, Vu)

if __name__ == '__main__':
    M_pinv, M_proj, p, M_allrad, M_allrad2 = unit_test2(case=2, debug=False)

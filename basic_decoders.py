#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:57:11 2018

@author: heller
"""

from __future__ import division, print_function
import numpy as np
from numpy import pi

from scipy.spatial import Delaunay  # for AllRAD decoder

import real_spherical_harmonics as rsh

import spherical_grids as sg
import ray_triangle_intersection as rti


def channel_spec(degree, order, norm=1, cs_phase=None):
    def cs(l, m, n=1): return (l, m, n)
    return [cs(l, m) for l in range(degree+1) for m in range(-l, l+1)]


def projection(degree, order,
               spkrs_az, spkrs_el):
    """Decoder by projection method.

    Parameters
    ----------
        degree: int or array-like collection of ints
            degree (l) of each spherical harmonic.
        order: int or array-like collectin of ints
            order (m) of each sperical harmonic.
        spkrs_az, spkrs_el: array-like collection of floats:
            speaker azimuths and elevations in radians.

    Returns
    -------
        numpy.ndarray
            Basic decoder matrix

    Note
    ----
        Optimal for platonic solids, and more generally, spherical designs only.
        This is here mostly for comparison to other method.
    """

    if np.isscalar(degree):
        degree, order = zip(*channel_spec(degree, order))

    M = rsh.real_sph_harm_transform(degree, order,
                                    np.array(spkrs_az).ravel(),
                                    np.array(spkrs_el).ravel())
    return M


def inversion(degree, order,
              spkrs_az, spkrs_el):
    """Decoder by inversion method (aka, mode matching).

    Args:
        degree (array-like): degree (l) of each spherical harmonic.
        order (array-like): order (m) of each sperical harmonic.
        spkrs_az (array-like): speaker azimuths in radians.
        spkrs_el (array-like): speaker elevations in radians.

    Returns:
        Basic decoder matrix

    Note:
        Optimal for uniform arrays only.

    """

    Y_spkrs = rsh.real_sph_harm_transform(degree, order,
                                          np.array(spkrs_az).ravel(),
                                          np.array(spkrs_el).ravel())
    M = np.linalg.pinv(Y_spkrs)

    return M


def allrad(degree, order,
           spkrs_az, spkrs_el,
           v_az=None, v_el=None,
           vbap_norm=True):
    # defaults
    if v_az is None:
        td = sg.t_design5200()
        v_az = td.az
        v_el = td.el

    V2R, Vtri, Vxyz = allrad_v2rp(np.array(sg.sph2cart(spkrs_az, spkrs_el)),
                                  np.array(sg.sph2cart(v_az, v_el)),
                                  vbap_norm=vbap_norm)

    Mv = inversion(degree, order, v_az, v_el)
    M = np.matmul(V2R, Mv)

    return M


def allrad_v2rp(Su, Vu, vbap_norm=True):
    tri = Delaunay(Su.transpose())
    H = tri.convex_hull

    p0 = tri.points[H[:, 0], :]
    p1 = tri.points[H[:, 1], :]
    p2 = tri.points[H[:, 2], :]

    origin = np.array([0, 0, 0])
    a = []
    Hr = np.arange(len(H))

    V2R = np.zeros((Su.shape[1], Vu.shape[1]))
    for i in range(5200):
        flag, u, v, t = rti.ray_triangle_intersection_p1(origin, Vu[:, i],
                                                         p0, p1, p2)
        valid = np.logical_and(flag, t > 0)
        face = Hr[valid][0]
        ur = u[valid][0]
        vr = v[valid][0]
        tr = t[valid][0]
        a.append((face, ur, vr, tr))

        b = np.array([1 - ur - vr, ur, vr])
        if vbap_norm:
            b = b / np.linalg.norm(b)

        V2R[H[face, :], i] = b

    return V2R, a, tri


def allrad_v2r(Su, Vu):
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
                Vxyz[:, i] = bw*p0 + bu*p1 + bv*p2
                # fill in gains, normalize for energy
                b = np.array([bw, bu, bv])

                V2R[H[j, :], i] = b / np.linalg.norm(b)

                break
    return V2R, a, tri #Vtri, Vxyz


# unit tests
def unit_test():
    s_az = (pi/4, 3*pi/4, -3*pi/4, -pi/4)
    s_el = (0, 0, 0, 0)

    l = (0, 1, 1)
    m = (0, 1, -1)

    M_pinv = inversion(l, m, s_az, s_el)
    M_proj = projection(l, m, s_az, s_el)

    return M_pinv, M_proj, np.matmul(M_proj, M_pinv)


def unit_test2(order=3):
    if True:
        s_az = (pi/4, 3*pi/4, -3*pi/4, -pi/4, 0, 0)
        s_el = (0, 0, 0, 0, pi/2, -pi/2)
    else:
        s = sg.t_design()
        s_az = s.az
        s_el = s.el

    if order == 1:
        l = (0, 1, 1, 1)
        m = (0, 1, -1, 0)
    elif order == 2:
        l = (0, 1, 1, 1, 2, 2, 2, 2, 2)
        m = (0, 1, -1, 0, -2, -1, 0, 1, 2)
    elif order == 3:
        l = (0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3)
        m = (0, 1, -1, 0, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3)

    M_pinv = inversion(l, m, s_az, s_el)
    # fuzz to zero
    M_pinv[np.isclose(0, M_pinv, atol=1e-10, rtol=1e-10)] = 0

    M_proj = projection(l, m, s_az, s_el)

    p = np.allclose(np.matmul(M_proj, M_pinv), np.eye(len(l)))

    M_allrad = allrad(l, m, s_az, s_el)

    return M_pinv, M_proj, p, M_allrad


#v2rp, ap, trip = allrad_v2rp(Su, Vu)
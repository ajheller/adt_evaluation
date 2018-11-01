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


def projection(degree, order,
               spkrs_az, spkrs_el):

    M = rsh.real_sph_harm_transform(degree, order,
                                    np.array(spkrs_az).ravel(),
                                    np.array(spkrs_el).ravel())
    return M


def inversion(degree, order,
              spkrs_az, spkrs_el):
    Y_spkrs = rsh.real_sph_harm_transform(degree, order,
                                          np.array(spkrs_az).ravel(),
                                          np.array(spkrs_el).ravel())
    M = np.linalg.pinv(Y_spkrs)

    return M


def allrad(degree, order,
           spkrs_az, spkrs_el,
           v_az=None, v_el=None):
    # defaults
    if v_az is None:
        td = sg.t_design5200()
        v_az = td.az
        v_el = td.el

    V2R, Vtri, Vxyz = allrad_v2r(np.array(sg.sph2cart(spkrs_az, spkrs_el)),
                                 np.array(sg.sph2cart(v_az, v_el)))

    Mv = inversion(degree, order, v_az, v_el)
    M = np.matmul(V2R, Mv)

    return M


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

    for i in range(Vu.shape[1]):  # iterate over the virtual loudspeakers
        d = Vu[:, i]
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
    return V2R, Vtri, Vxyz


# unit tests
def unit_test():
    s_az = (pi/4, 3*pi/4, -3*pi/4, -pi/4)
    s_el = (0, 0, 0, 0)

    l = (0, 1, 1)
    m = (0, 1, -1)

    #l = (0,)
    #m = (0,)

    M_pinv = inversion(l, m, s_az, s_el)
    M_proj = projection(l, m, s_az, s_el)

    return M_pinv, M_proj, np.matmul(M_proj, M_pinv )


def unit_test2(order=3):
    if False:
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

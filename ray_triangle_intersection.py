#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:42:39 2018

@author: heller
"""

from __future__ import division

import numpy as np


def ray_triangle_intersection(o, d, p0, p1, p2, epsilon=1e-5):

    e1 = p1 - p0
    e2 = p2 - p0
    q = np.cross(d, e2)
    a = np.dot(e1, q)

    if np.abs(a) < epsilon:
        #  vector is parallel to the plane
        flag, u, v, t = 0, 0, 0, 0
        return flag, u, v, t

    f = 1/a
    s = o - p0
    u = f * np.dot(s, q)

    if u < 0.0:
        #  the intersectin is outside of the triangle
        flag, u, v, t = 0, 0, 0, 0
        return flag, u, v, t

    r = np.cross(s, e1)
    v = f * np.dot(d, r)

    if v < 0 or u + v > 1:
        #  the intersection is outside of the triangle
        return 0, 0, 0, 0

    t = f * np.dot(e2, r)
    flag = 1

    return flag, u, v, t


def dot1(a, b):
    "dot product of vectors in a and b"
    return np.sum(a * b, 1)


# this version is about 20x faster than the preceeding one
def ray_triangle_intersection_p(o, d, p0, p1, p2, epsilon=1e-5):
    # allocate results vectors
    flag = np.full(len(p0), False, dtype=bool)
    u = np.zeros(flag.shape)
    v = np.zeros(flag.shape)
    t = np.zeros(flag.shape)

    # temp
    f = np.zeros(flag.shape)
    r = np.zeros(p0.shape)
    s = np.zeros(p0.shape)

    #
    # Moller and Trumbore
    #   https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
    #   the basic idea is to carry the conditional ahead in flag
    e1 = p1 - p0
    e2 = p2 - p0
    # the scalar triple product, the volume of the parrallelpiped e1, e2, d
    q = np.cross(d, e2)
    a = dot1(e1, q)

    # test if e1, e2, and d are coplanar
    flag = np.abs(a) > epsilon
    f[flag] = 1 / a[flag]
    s[flag, :] = o - p0[flag, :]
    u[flag] = f[flag] * dot1(s[flag, :], q[flag, :])

    # if u < 0.0:
    flag[flag] = u[flag] >= 0.0
    r[flag] = np.cross(s[flag], e1[flag])
    v[flag] = f[flag] * dot1(d, r[flag])

    #  if v < 0 or u + v > 1:
    flag[flag] = v[flag] >= 0
    flag[flag] = u[flag] + v[flag] <= 1

    t[flag] = f[flag] * dot1(e2[flag], r[flag])

    return flag, u, v, t







def test():
    v0 = np.array([10, 0, 0])
    v1 = np.array([0, 10, 0])
    v2 = np.array([0, 0, 10])
    origin = np.array([10, 10, 10])
    direction = np.array([-0.3, -0.5, -0.7])

    flag, u, v, t = ray_triangle_intersection(
            origin, direction, v0, v1, v2)

    return flag, u, v, t

def testp():
    v0 = np.array([[10, 0, 0] for i in range(4)])
    v1 = np.array([[0, 10, 0] for i in range(4)])
    v2 = np.array([[0, 0, 10] for i in range(4)])
    origin = np.array([10, 10, 10])
    direction = np.array([-0.3, -0.5, -0.7])

    flag, u, v, t = ray_triangle_intersection_p(
            origin, direction, v0, v1, v2)

    return flag, u, v, t



def test2(i):
    flag, u, v, t = ray_triangle_intersection_p(origin, Vu[:,i], p0, p1, p2)
    return flag, u, v, t

if False:
    print test()
    print
    print testp()


def test3():
    # assemble the list of face vertices
    p0 = tri.points[H[:, 0], :]
    p1 = tri.points[H[:, 1], :]
    p2 = tri.points[H[:, 2], :]

    origin = np.array([0, 0, 0])
    a = []
    Hr = np.arange(len(H))
    for i in range(5200):
        flag, u, v, t = ray_triangle_intersection_p(origin, Vu[:, i],
                                                    p0, p1, p2)
        valid = np.logical_and(flag, t > 0)
        face = Hr[valid][0]
        ur = u[valid][0]
        vr = u[valid][0]
        tr = t[valid][0]
        a.append((face, ur, vr, tr))
    return a

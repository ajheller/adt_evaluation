#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:29:51 2018

@author: heller
"""

import numpy as np


# these follow the MATLAB convention for spherical coordinates

def cart2sph(x, y, z):
    r_xy = np.hypot(x, y)
    r = np.hypot(r_xy, z)
    el = np.arctan2(z, r_xy)
    az = np.arctan2(y, x)
    return az, el, r


def sph2cart(az, el, r=1):
    z = np.sin(el)
    r_cos_el = r * np.cos(el)
    x = r_cos_el * np.cos(az)
    y = r_cos_el * np.sin(az)
    return x, y, z


def az_el(resolution=180):
    u = np.linspace(-np.pi, np.pi, (2 * resolution) + 1)
    v = np.linspace(-np.pi/2, np.pi/2, resolution + 1)
    el, az = np.meshgrid(v, u)

    x, y, z = sph2cart(az, el)
    w = np.cos(el)  # quadrature weight

    return x, y, z, az, el, w

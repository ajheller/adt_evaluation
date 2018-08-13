#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:29:51 2018

@author: heller
"""

import numpy as np


def az_el(resolution=180):
    u = np.linspace(-np.pi, np.pi, (2 * resolution) + 1)
    v = np.linspace(-np.pi/2, np.pi/2, resolution + 1)

    el, az = np.meshgrid(v, u)
    x = np.cos(az) * np.cos(el)
    y = np.sin(az) * np.cos(el)
    z = np.sin(el)
    w = np.cos(el)  # quadrature weight

    return x, y, z, az, el, w

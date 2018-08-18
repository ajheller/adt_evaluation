#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:46:44 2018

@author: heller
"""

import numpy as np
import json


def load(path):
    with open(path, 'r') as f:
        scmd = json.load(f)

    S = scmd['S']
    C = scmd['C']
    M = scmd['M']

    try:
        M_hf = np.array(M['hf'])
    except KeyError:
        M_hf = np.array(M)

    Su = np.array([S['x'],S['y'], S['z']])[:, :, 0]

    return Su, C, M_hf

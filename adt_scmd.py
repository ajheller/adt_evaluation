#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:46:44 2018

@author: heller
"""

import json

import numpy as np


def load(path):
    with open(path, "r") as f:
        scmd = json.load(f)

    S = scmd["S"]
    C = scmd["C"]
    M = scmd["M"]
    D = scmd["D"]

    try:
        M_hf = np.array(M["hf"])
    except TypeError:
        M_hf = np.array(M)

    Su = np.squeeze([S["x"], S["y"], S["z"]])

    return Su, C, M_hf, D, scmd

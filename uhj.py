#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:38:56 2022

@author: heller
"""

import numpy as np
import scipy.signal as sig


def uhj_encode(B):
    M = [
        [0.9396926, 0.1855740, 0, 0],
        [-0.3420201j, 0.5098604j, 0.6554516, 0],
        [-0.1432j, 0.6512j, -np.sqrt(1 / 2), 0],
        [0, 0, 0, 0.9772],
    ]

    M = np.array([[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ np.array(
        [
            [0.9396926, 0.1855740, 0, 0],
            [-0.3420201j, 0.5098604j, 0.6554516, 0],
            [-0.1432j, 0.6512j, -np.sqrt(1 / 2), 0],
            [0, 0, 0, 0.9772],
        ]
    )

    print(M)

    B = sig.hilbert(B)
    U = M @ B.T
    return U.T.real


def uhj_decode(U):
    L = U[:, 0]
    R = U[:, 1]

    S = sig.hilbert(L + R)
    D = sig.hilbert(L - R)

    Ew = 0.982 * S + 0.164j * D
    Ex = 0.419 * S - 0.828j * D
    Ey = 0.763 * D + 0.385j * S

    return Ew, Ex, Ey

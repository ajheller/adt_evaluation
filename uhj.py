#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:38:56 2022

@author: heller
"""

import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav
from pathlib import Path


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

    L = U[:, 0] / 2
    R = U[:, 1] / 2

    S = sig.hilbert(L + R).astype(np.complex64)
    D = sig.hilbert(L - R).astype(np.complex64)

    # Ew = 0.982 * S + 0.164j * D
    # Ex = 0.419 * S - 0.828j * D
    # Ey = 0.763 * D + 0.385j * S

    E = np.column_stack(
        (
            np.real(0.982 * S + 0.164j * D),
            np.real(0.419 * S - 0.828j * D),
            np.real(0.763 * D + 0.385j * S),
        )
    )

    return E


def uhj_decode_file(file, file_out=None):
    if file_out is None:
        p = Path(file)
        file_out = p.parent / (p.stem + "-E" + p.suffix)

    Fs, U = wav.read(file)
    print(Fs, U.shape)
    E = uhj_decode(U)
    wav.write(file_out, Fs, np.int16(E))

    return E


def impulse_response_decode():
    U = np.zeros((4096, 2))
    U[2048, 0] = 1.0
    E = uhj_decode(U)
    return E

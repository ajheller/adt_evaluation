#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 12:07:00 2022

@author: heller
"""

from pathlib import Path

# Google translate text-to-speech service
from gtts import gTTS  # pip install gtts

import os
import numpy as np

# from matplotlib import pyplot as plt
from scipy.io import wavfile as wav

from functools import lru_cache

# adt modules
import program_channels as pc
import real_spherical_harmonics as rsh

_here = Path(__file__).parent
_outdir = _here / "test-files" / "robojo"


_fs = 48000


@lru_cache(maxsize=None)
def text2speech(text, language="en", fs=_fs):
    myobj = gTTS(text=text, lang=language, slow=False)
    myobj.save("tmp.mp3")
    os.system(f"sox tmp.mp3 tmp.wav rate {fs}")
    wfs, w = wav.read("tmp.wav")

    os.remove("tmp.mp3")
    os.remove("tmp.wav")
    return w


def robojo(
    C,
    text,
    azd,
    eld,
    path=None,
    dt=4,
    fs=_fs,
    elevation=None,
    language="en",
    outdir=_outdir,
):

    os.makedirs(_outdir, exist_ok=True)

    if elevation is None:
        elevation = eld[0]

    if path is None:
        path = f"robojo-{C.id_string(slugify=True)}-el{elevation:+03d}-{language}.wav"

    ho, vo, sh_l, sh_m, id_str = pc.ambisonic_channels(C)
    normalization = C.normalization

    gains = rsh.real_sph_harm_transform(
        sh_l, sh_m, np.array(azd) * np.pi / 180, np.array(eld) * np.pi / 180
    )
    x = np.zeros((gains.shape[0], len(text), dt * fs), dtype=np.int16)

    for i, t in enumerate(text):
        tw = text2speech(t, fs=fs, language=language)
        for j, gain in enumerate(gains[:, i] * normalization * np.sqrt(4 * np.pi)):
            xx = np.floor((tw * gain) + np.random.random(len(tw)))
            # check for clipping
            nclipp = np.sum((np.max(xx) >= np.iinfo(x.dtype).max))
            nclipn = np.sum((np.min(xx) <= np.iinfo(x.dtype).min))
            print(f"Clipped sample count {nclipp} pos; {nclipn} neg")

            x[j, i, : len(tw)] = xx

    y = x.reshape(gains.shape[0], -1)
    wav.write(outdir / path, fs, y.T)
    return path


def test(o, e):
    try:
        C = pc.ChannelsAmbiX(o)
        # C = pc.ChannelsFuMa(o)
        # C = pc.ChannelsN3D(o)
    except ValueError as ve:
        print(ve)
    else:
        s = (
            (0, e, "front"),
            (-45, e, "front, right"),
            (-90, e, "right"),
            (-135, e, "back, right"),
            (-180, e, "back"),
            (135, e, "back, left"),
            (90, e, "left"),
            (45, e, "front, left"),
            (0, 90, "top"),
            (0, -90, "bottom"),
        )

        azd, eld, text = zip(*s)
        return robojo(C, text, azd, eld)


def test2(o, es):
    return [test(o, e) for e in es]


def test3(os=(1, 3, 5, 7), es=(-45, 0, 45)):
    return [test2(o, es) for o in os]

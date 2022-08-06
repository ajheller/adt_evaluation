#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 18:06:58 2022

@author: heller
"""

import program_channels as pc
import example_speaker_arrays as esa
import basic_decoders as bd
import write_faust_decoder as wfd


# reproducing G. Senna did

S = esa.polygon(8)

C = pc.ChannelsN3D(1, 1)
M = bd.inversion(C.sh_l, C.sh_m, S.az, S.el)

wfd.write_faust_decoder_dual_band(
    "decoder-oct-1H1V-N3D.dsp",
    "decoder-oct-1H1V-N3D",
    M,
    C.sh_l,
    S.r,
    C.channel_mask,
    decoder_3d=True,
)

# %%

# array is actually 2D, so this is a better decoder
C = pc.ChannelsN3D(1, 0)
M = bd.inversion(C.sh_l, C.sh_m, S.az, S.el)

wfd.write_faust_decoder_dual_band(
    "decoder-oct-1H0V-N3D.dsp",
    "decoder-oct-1H0V-N3D",
    M,
    C.sh_l,
    S.r,
    C.channel_mask,
    decoder_3d=False,
)

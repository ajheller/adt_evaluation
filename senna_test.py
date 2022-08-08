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

import slugify

# %% reproducing what G. Senna did

S = esa.polygon(8)
S.plot3D()
C = pc.ChannelsN3D(1, 1)
M = bd.inversion(C.sh_l, C.sh_m, S.az, S.el)

name = slugify.slugify(f"{S.name}_{C.id_string()}")

wfd.write_faust_decoder_dual_band(
    f"{name}.dsp",
    name,
    M,
    C.sh_l,
    S.r,
    C.channel_mask,
    is_3d=C.v_order > 0,
)

# %%  array is actually 2D, so this is a better decoder

S = esa.polygon(8)
C = pc.ChannelsN3D(1, 0)
M = bd.inversion(C.sh_l, C.sh_m, S.az, S.el)

name = slugify.slugify(f"{S.name}_{C.id_string()}")

wfd.write_faust_decoder_dual_band(
    f"{name}.dsp",
    name,
    M,
    C.sh_l,
    S.r,
    C.channel_mask,
    is_3d=C.v_order > 0,
)

# %% four speaker array

S = esa.polygon(4)
S.plot3D()
C = pc.ChannelsN3D(1, 0)
M = bd.inversion(C.sh_l, C.sh_m, S.az, S.el)

name = slugify.slugify(f"{S.name}_{C.id_string()}")

wfd.write_faust_decoder_dual_band(
    f"{name}.dsp",
    name,
    M,
    C.sh_l,
    S.r,
    C.channel_mask,
    is_3d=C.v_order > 0,
)

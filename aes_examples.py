#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:55:28 2021

@author: heller
"""
from numpy import pi as π

import example_speaker_arrays as esa
import optimize_dome as od
import program_channels as pc

# %%
S_stage = esa.stage2017(add_imaginary=True)

C31 = pc.ChannelsAmbiX(3, 1)
C33 = pc.ChannelsAmbiX(3, 3)

# %%

# good directionality result
res_31 = od.optimize_dome(
    S_stage,
    ambisonic_order=C31,
    eval_order=C31,
    sparseness_penalty=1.0,
    el_lim=-π / 4,
)
# compare with
res_33 = od.optimize_dome(
    S_stage,
    ambisonic_order=C33,
    eval_order=C31,
    sparseness_penalty=1.0,
    el_lim=-π / 4,
)
# %%

S_nd = esa.nando_dome(add_imaginary=True)

res_31 = od.optimize_dome(
    S_nd,
    ambisonic_order=C31,
    eval_order=C31,
    sparseness_penalty=0,
    el_lim=-π / 4,
)

res_33 = od.optimize_dome(
    S_nd,
    ambisonic_order=C33,
    eval_order=C31,
    sparseness_penalty=0,
    el_lim=-π / 4,
)

# %%

S_nd = esa.nando_dome(add_imaginary=True)

C21 = pc.ChannelsAmbiX(2, 1)

res_21 = od.optimize_dome(
    S_nd,
    ambisonic_order=C21,
    eval_order=C21,
    sparseness_penalty=0,
    el_lim=-π / 4,
)

res_23 = od.optimize_dome(
    S_nd,
    ambisonic_order=C33,
    eval_order=C21,
    sparseness_penalty=0,
    el_lim=-π / 4,
)

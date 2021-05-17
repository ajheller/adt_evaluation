#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:55:28 2021

@author: heller
"""
from numpy import pi as π
import optimize_dome as od
import example_speaker_arrays as esa
import localization_models as lm
import program_channels as pc

# %%
S_stage = esa.stage2017(add_imaginary=True)

# good directionality result
res_31 = od.optimize_dome(S_stage,
                       ambisonic_order=(3,1),
                       eval_order=(3,1),
                       sparseness_penalty=1.0,
                       el_lim=-π/4,
                       )
# compare with
res_33 = od.optimize_dome(S_stage,
                       ambisonic_order=(3,3),
                       eval_order=(3,1),
                       sparseness_penalty=1.0,
                       el_lim=-π/4,
                       )
# %%
S_stage = esa.stage2017(add_imaginary=True)

order = (5, 5)
el_lim = -π/4
M_hf, res_hf = od.optimize_dome(S_stage,
                                ambisonic_order=order,
                                sparseness_penalty=1.0,
                                el_lim=el_lim)

S_stage_real = esa.stage2017(add_imaginary=False)
M_lf, res_lf = od.optimize_dome_LF(M_hf, S_stage_real,
                                   ambisonic_order=order,
                                   el_lim=el_lim)

order_h, order_v, sh_l, sh_m = pc.ambisonic_channels(order)
lm.plot_performance_LF(M_lf, M_hf, S_stage_real.u.T, sh_l, sh_m)


# %%

S_nd = esa.nando_dome(add_imaginary=True)

res_31 = od.optimize_dome(S_nd,
                       ambisonic_order=(3,1),
                       eval_order=(3,1),
                       sparseness_penalty=0,
                       el_lim=-π/4,
                       )


res_33 = od.optimize_dome(S_nd,
                       ambisonic_order=(3,3),
                       eval_order=(3,1),
                       sparseness_penalty=0,
                       el_lim=-π/4,
                       )


# %%

S_nd = esa.nando_dome(add_imaginary=True)

res_21 = od.optimize_dome(S_nd,
                       ambisonic_order=(2,1),
                       eval_order=(2,1),
                       sparseness_penalty=0,
                       el_lim=-π/4,
                       )


res_23 = od.optimize_dome(S_nd,
                       ambisonic_order=(3,3),
                       eval_order=(2,1),
                       sparseness_penalty=0,
                       el_lim=-π/4,
                       )

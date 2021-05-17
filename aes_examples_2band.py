#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 21:38:12 2021

@author: heller
"""
from numpy import pi as π
import optimize_dome as od
import example_speaker_arrays as esa
import localization_models as lm
import program_channels as pc
import basic_decoders as bd

# %%
S_stage = esa.stage2017(add_imaginary=True)

order = (5, 5)
order_h, order_v, sh_l, sh_m = pc.ambisonic_channels(order)

title=f"{S_stage.name}: AllRAD {order_h}H{order_v}V"

M = bd.allrad(sh_l, sh_m,
              S_stage.az, S_stage.el,
              speaker_is_real=S_stage.is_real)

lm.plot_performance(M, S_stage.u[S_stage.is_real].T, sh_l, sh_m,
                    title=title)

lm.plot_matrix(M, title=title)

df_gain_spk, df_gain_tot = lm.diffuse_field_gain(M)

print(df_gain_spk, df_gain_tot)
# %%

el_lim = -π/4
M_hf, res_hf = od.optimize_dome(S_stage,
                                ambisonic_order=order,
                                sparseness_penalty=1.0,
                                el_lim=el_lim)
df_gain_spk, df_gain_tot = lm.diffuse_field_gain(M_hf)

print(df_gain_spk, df_gain_tot)
# %%
S_stage_real = esa.stage2017(add_imaginary=False)
M_lf, res_lf = od.optimize_dome_LF(M_hf, S_stage_real,
                                   ambisonic_order=order,
                                   el_lim=el_lim)


lm.plot_performance_LF(M_lf, M_hf, S_stage_real.u.T, sh_l, sh_m)

# %%
print("HF", lm.diffuse_field_gain(M_hf))
print("LF", lm.diffuse_field_gain(M_lf))

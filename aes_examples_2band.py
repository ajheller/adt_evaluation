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
S = esa.stage2017(add_imaginary=True)

C = pc.ChannelsAmbiX(6, 6)
order_h, order_v, sh_l, sh_m, id_string = pc.ambisonic_channels(C)

title=f"{S.name}: AllRAD {C.id_string()}"

M = bd.allrad(sh_l, sh_m,
              S.az, S.el,
              speaker_is_real=S.is_real)

lm.plot_performance(M, S.u[S.is_real].T, sh_l, sh_m,
                    title=title)

lm.plot_matrix(M, title=title)

df_gain_spk, df_gain_tot = lm.diffuse_field_gain(M)


# %%

el_lim = -π/4
M_hf, res_hf = od.optimize_dome(S,
                                ambisonic_order=C,
                                sparseness_penalty=.50,
                                el_lim=el_lim)

M_hf, res_hf = od.optimize_dome(S,
                                ambisonic_order=C,
                                sparseness_penalty=.50,
                                el_lim=el_lim,
                                do_report=True)

# %% sparseness penalty = 1.0 (best horizontal performance)
M_hf, res_hf = od.optimize_dome(S,
                                ambisonic_order=C,
                                sparseness_penalty=1.0,
                                el_lim=el_lim,
                                do_report=True)

# %% sparseness penalty = 0.0 (speakers are turned off?)
M_hf, res_hf = od.optimize_dome(S,
                                ambisonic_order=C,
                                sparseness_penalty=0.0,
                                el_lim=el_lim,
                                do_report=True)

# %%
S_real = esa.stage2017(add_imaginary=False)
M_lf, res_lf = od.optimize_dome_LF(M_hf, S_real,
                                   ambisonic_order=C,
                                   el_lim=el_lim)

# %%
lm.plot_performance_LF(M_lf, M_hf, S_real.u.T, sh_l, sh_m,
                       title=f"{S_real.name}, {C.id_string()}")

# %%
print("AllRAD", lm.diffuse_field_gain(M))
print("HF", lm.diffuse_field_gain(M_hf))
print("LF", lm.diffuse_field_gain(M_lf))

# %%
import write_faust_decoder as wfd
wfd.write_faust_decoder_vienna('amb.dsp', 'amb',
                               M_lf, M_hf,
                               sh_l, S_real.r)

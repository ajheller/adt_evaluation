#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 14:18:50 2021

@author: heller
"""
import numpy as np

import loudspeaker_layout as lsl
import program_channels as pc
import localization_models as lm
import shelf
import optimize_dome as od
import basic_decoders as bd

import example_speaker_arrays as esa

output_file = 'local/envelop.json'

S = esa.envelop()
spkr_plot_fig = S.plot()
spkr_plan_fig = S.plot_plan()
spkr_azel_fig = S.plot_azel()

# obj
# dict_keys(['Name', 'Description', 'Decoder', 'LoudspeakerLayout'])

# lsl
# dict_keys(['Name', 'Loudspeakers'])

# decoder
# dict_keys(['Name', 'Description', 'ExpectedInputNormalization',
#            'Weights', 'WeightsAlreadyApplied', 'Matrix', 'Routing'])
# %%

C = pc.ChannelsN3D(3, 3)
el_lim = -np.pi/3

# %%

M_sad = bd.projection(C.sh_l, C.sh_m, S.az, S.el)

if True:
    M_sad = M_sad @ shelf.gamma(C.sh_l, decoder_type='cardioid',
                                decoder_3d=True,
                                return_matrix=True)

sad_figs = lm.plot_performance(M_sad, S.u.T, C.sh_l, C.sh_m, el_lim=el_lim,
                               title=f"{S.name}: SAD {C.id_string()}")

# %%

M_allrad = bd.allrad(C.sh_l, C.sh_m, S.az, S.el)
if True:
    M_allrad = M_allrad @ shelf.gamma(C.sh_l, decoder_type='max_rE',
                                      decoder_3d=True,
                                      return_matrix=True)


allrad_figs = lm.plot_performance(M_allrad, S.u.T, C.sh_l, C.sh_m,
                                  el_lim=el_lim,
                                  title=f"{S.name}: AllRAD {C.id_string()}")

# %%

for sp in (0, 1, 0.5):
    M_opt, M_opt_res = od.optimize_dome2(M_allrad, C.sh_l, C.sh_m, S.u.T,
                                         el_lim,
                                         sparseness_penalty=sp)

    opt_figs = lm.plot_performance(M_opt, S.u.T, C.sh_l, C.sh_m, el_lim=el_lim,
                                   title=f"{S.name}: Opt (sp={sp}) {C.id_string()}")

# %%

title_opt_lf = f"{S.name}: Opt  LF/HF {C.id_string()}"

M_hf = M_opt
M_lf, res_lf = od.optimize_dome_LF(M_hf, S,
                                   ambisonic_order=C,
                                   el_lim=el_lim)

opt_LF_figs = lm.plot_performance_LF(M_lf, M_hf, S.u.T, C.sh_l, C.sh_m,
                                     title=title_opt_lf)

# %%  Are diffuse field gains the same?
print(f"\n\n{title_opt_lf}\nDiffuse field gain of each loudspeaker (dB)")
print("HF", lm.diffuse_field_gain(M_hf)[1])
print("LF", lm.diffuse_field_gain(M_lf)[1])
# %%

all_figs = ([None, ] + sad_figs + [None, None],
            [None, ] + allrad_figs + [None, None],
            [None, ] + opt_figs + opt_LF_figs)

import reports
reports.html_report(zip(*all_figs),
                    name=f"{S.name} {C.id_string()}")

# %%

import write_faust_decoder as wfd
import slugify
import json

dec_name = f"{slugify.slugify(S.name)}-{C.h_order}H{C.v_order}V-N3D"

wfd.write_faust_decoder_vienna(dec_name+"-Vienna.dsp",
                               dec_name+"-Vienna",
                               M_lf, M_hf,
                               C.sh_l, S.r,
                               input_mask=C.channel_mask)

# %%
if False:
    iem_dict = {}
    iem_dict['Decoder']['Matrix'] = M_hf.tolist()
    iem_dict['Decoder']['Matrix_LF'] = M_lf.tolist()
    iem_dict['Decoder']['Name'] += 'Optimzed'
    iem_dict['Decoder']['Description'] += ' Optimzed by PyADT'
    iem_dict['Decoder']['optimization'] = 'Optimized by PyADT'

    iem_dict['Name'] += 'Optimzed'
    iem_dict['Description'] += ' Optimzed by PyADT'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(iem_dict, f)

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

input_file = 'local/uqam-j-3435-basic-n3d.json'
output_file = 'local/uqam-j-3435-basic-n3d-opt.json'

S, iem_dict = lsl.from_iem_file(input_file, return_json=True)
S.plot()
S.plot_plan()

# obj
# dict_keys(['Name', 'Description', 'Decoder', 'LoudspeakerLayout'])

# lsl
# dict_keys(['Name', 'Loudspeakers'])

# decoder
# dict_keys(['Name', 'Description', 'ExpectedInputNormalization',
#            'Weights', 'WeightsAlreadyApplied', 'Matrix', 'Routing'])
# %%

iem_decoder = iem_dict['Decoder']

if iem_decoder['ExpectedInputNormalization'] != 'n3d':
    raise ValueError('normalization is not n3d')

M_allrad = np.asarray(iem_decoder['Matrix'])

n_spkr, n_chan = M_allrad.shape

# guess order from number of input channels
h_order = v_order = np.sqrt(n_chan) - 1
el_lim = -np.pi/3

C = pc.ChannelsN3D(h_order, v_order)

if not iem_decoder['WeightsAlreadyApplied']:
    M_allrad = M_allrad @ shelf.gamma(C.sh_l, decoder_type='max_rE',
                                      decoder_3d=True,
                                      return_matrix=True)


iem_figs = lm.plot_performance(M_allrad, S.u.T, C.sh_l, C.sh_m, el_lim=el_lim,
                               title=f"IEM AllRAD {C.id_string()}")

# %%

M_opt, M_opt_res = od.optimize_dome2(M_allrad, C.sh_l, C.sh_m, S.u.T, el_lim)

opt_figs = lm.plot_performance(M_opt, S.u.T, C.sh_l, C.sh_m, el_lim=el_lim,
                               title=f"Opt IEM AllRAD {C.id_string()}")

# %%

title_opt_lf = f"{S.name}: Optimized LF/HF AllRAD {C.id_string()}"

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
import write_faust_decoder as wfd
import slugify
import json

dec_name = f"{slugify.slugify(S.name)}-{h_order}H{v_order}V-N3D"

wfd.write_faust_decoder_vienna(dec_name+"-Vienna.dsp",
                               dec_name+"-Vienna",
                               M_lf, M_hf,
                               C.sh_l, S.r,
                               input_mask=C.channel_mask)

iem_dict['Decoder']['Matrix'] = M_hf.tolist()
iem_dict['Decoder']['Matrix_LF'] = M_lf.tolist()
iem_dict['Decoder']['Name'] += 'Optimzed'
iem_dict['Decoder']['Description'] += ' Optimzed by PyADT'
iem_dict['Decoder']['optimization'] = 'Optimized by PyADT'


iem_dict['Name'] += 'Optimzed'
iem_dict['Description'] += ' Optimzed by PyADT'

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(iem_dict, f)

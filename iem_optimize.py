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

n_chan, n_spkr = M_allrad.shape

# guess order from number of input channels
h_order = v_order = np.sqrt(n_chan - 1)
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

#M_lf_opt, M_lf_opt_res = od.optimize_dome_LF(M_opt, S, )

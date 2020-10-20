#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:05:28 2020

@author: heller
"""

import numpy as np
import pickle

import example_speaker_arrays as esa
import basic_decoders as bd
import optimize_decoder_matrix as odm
import program_channels as pg
import localization_models as lm

_pickle_file = 'Mixed-Order-maxrE.pkl'

def generate(file=_pickle_file):
    s = esa.uniform240()

    M = dict()

    for order_h in range(1, 16):
        for order_v in range(1, order_h+1):
            for mos in ('HV', 'HP'):
                key = (order_h, order_v, mos)
                print(key)
                c = pg.ChannelsAmbiX(order_h, order_v, mixed_order_scheme=mos)

                M_pinv = bd.inversion(c.sh_l, c.sh_m, s.az, s.el)
                M[(*key, 'pinv')] = M_pinv

                M_allrad = bd.allrad(c.sh_l, c.sh_m, s.az, s.el)
                M[(*key, 'allrad')] = M_pinv

                M_opt, res = odm.optimize(M_pinv, s.u.T, c.sh_l, c.sh_m,
                                          raise_error_on_failure=False)

                if res.status == 0:
                    M[(*key, 'opt')] = M_opt
                else:
                    M[(*key, 'opt')] = res

                # lm.plot_performance(M_pinv, s.u.T, *c.sh(), title=key)

        pickle.dump(M, open(file, 'wb'))


def review(file=_pickle_file):
    a = pickle.load(open(file, 'rb'))
    for k, v in a.items():
        print(v['res'])


if __name__ == '__main__':
    generate()
    review()
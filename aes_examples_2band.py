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

import io
import reports

# %%
S = esa.stage2017(add_imaginary=True)
S.plot()

S_real = esa.stage2017(add_imaginary=False)

C = pc.ChannelsAmbiX(6, 6)
order_h, order_v, sh_l, sh_m, id_string = pc.ambisonic_channels(C)

# %%

title_allrad = f"{S.name}: AllRAD {C.id_string()}"

if False:
    M = bd.allrad(sh_l, sh_m,
                  S.az, S.el,
                  speaker_is_real=S.is_real)

    lm.plot_performance(M, S.u[S.is_real].T, sh_l, sh_m,
                        title=title_allrad)

    lm.plot_matrix(M, title=title_allrad)

    print("ALLRad", lm.diffuse_field_gain(M))
# %%

el_lim = -π / 4

M_hf, res_hf = od.optimize_dome(S,
                                ambisonic_order=C,
                                sparseness_penalty=0.5,
                                el_lim=el_lim,
                                do_report="sp-0.5")

# %% sparseness penalty = 1.0 (best horizontal performance)
M_hf, res_hf = od.optimize_dome(S,
                                ambisonic_order=C,
                                sparseness_penalty=1.0,
                                el_lim=el_lim,
                                do_report="sp-1.0")

# %% sparseness penalty = 0.0 (speakers are turned off?)
M_hf, res_hf = od.optimize_dome(S,
                                ambisonic_order=C,
                                sparseness_penalty=0.0,
                                el_lim=el_lim,
                                do_report="sp-0.0")

# %%
M_lf, res_lf = od.optimize_dome_LF(M_hf, S_real,
                                   ambisonic_order=C,
                                   el_lim=el_lim)

# %%

title_lf = f"Array:{S_real.name}, Signals: {C.id_string()}"
lm.plot_performance_LF(M_lf, M_hf, S_real.u.T, sh_l, sh_m,
                       title=title_lf)


def write_plot_performance_LF(
        M_lf, M_hf, S_real, sh_l, sh_m, title):
    """Write reports for LF performance plots."""
    figs = []
    figs.append(lm.plot_performance_LF(M_lf, M_hf, S_real.u.T, sh_l, sh_m,
                                       title=title))
    with io.StringIO() as f:
        print(f"LF optimization report\n",
              file=f)
        report = f.getvalue()
        print(report)
    spkr_array_name = S_real.name
    reports.html_report(zip(*figs),
                        text=report,
                        directory=spkr_array_name,
                        name=f"{spkr_array_name}-{id_string}-LF")


write_plot_performance_LF(M_lf, M_hf, S_real, sh_l, sh_m, title_lf)

# %%
print("HF", lm.diffuse_field_gain(M_hf))
print("LF", lm.diffuse_field_gain(M_lf))

# %%
import write_faust_decoder as wfd

wfd.write_faust_decoder_vienna('amb.dsp', 'amb',
                               M_lf, M_hf,
                               sh_l, S_real.r)

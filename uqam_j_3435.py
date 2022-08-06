#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 21:38:12 2021

@author: heller
"""
import numpy as np
from numpy import pi as π
import optimize_dome as od
import example_speaker_arrays as esa
import localization_models as lm
import program_channels as pc
import basic_decoders as bd
import write_faust_decoder as wfd
import shelf
import io
import reports
import loudspeaker_layout as lsl
import slugify

figs = []

S_real = lsl.from_iem_file('uqam-j-3435.json')
S = S_real
# add virtual speaker at nadir
S += lsl.from_array((0, 0, -1.5), coord_code='XYZ', unit_code='MMM',
                    name="imaginary speaker at nadir", ids=["*IN"],
                    is_real=False)
# add virtual speaker at zenith
S += lsl.from_array((0, 0, 2), coord_code='XYZ', unit_code='MMM',
                    name="imaginary speaker at zenith", ids=["*IZ"],
                    is_real=False)

# print(type(S), S, dir(S))
S.plot()
S.plot_azel()


# C = pc.ChannelsAmbiX(3, 2)
C = pc.ChannelsN3D(3, 3)
order_h, order_v, sh_l, sh_m, id_string = pc.ambisonic_channels(C)

# %%  AllRAD

title_allrad = f"{S.name}: AllRAD {C.id_string()}"

M_allrad = bd.allrad(sh_l, sh_m,
                     S.az, S.el,
                     speaker_is_real=S.is_real)

gamma = shelf.gamma(sh_l, decoder_type='max_rE', decoder_3d=True,
                    return_matrix=True)
M_allrad_hf = M_allrad @ gamma

# %%

figs.append(lm.plot_performance(M_allrad_hf, S.u[S.is_real].T, sh_l, sh_m,
                                title=title_allrad))

#figs.append(lm.plot_matrix(M_allrad_hf, title=title))

df_gain_spk, df_gain_tot = lm.diffuse_field_gain(M_allrad_hf)
print(f"""
{title_allrad}\n
Diffuse field gain of each loudspeaker (dB)
{(10*np.log10(df_gain_spk))}
Diffuse field gain of array {10*np.log10(df_gain_tot)}
""")

# %%  Optimized AllRAD -> M_hf

# optimize allrad design at high frequencies

title_opt = f"{S.name}: Optimized HF AllRAD {C.id_string()}"

el_lim = -π/3

M_hf, res_hf = od.optimize_dome2(M_allrad_hf,
                                 sh_l, sh_m, S_real.u.T,
                                 el_lim=el_lim)

lm.plot_performance(M_hf, S_real.u.T, sh_l, sh_m, title=title_opt)


df_gain_spk, df_gain_tot = lm.diffuse_field_gain(M_hf)
print(f"""
{title_opt}\n
Diffuse field gain of each loudspeaker (dB)
{df_gain_spk}
Diffuse field gain of array {df_gain_tot}
""")

# %%  Optimized LF for above -> M_lf

# optimize allrad design at low frequencies


title_opt_lf = f"{S_real.name}: Optimized LF/HF AllRAD {C.id_string()}"

M_lf, res_lf = od.optimize_dome_LF(M_hf, S_real,
                                   ambisonic_order=C,
                                   el_lim=el_lim)

figs.append(lm.plot_performance_LF(M_lf, M_hf, S_real.u.T, sh_l, sh_m,
                                   title=title_opt_lf))


def write_plot_performance_LF(
        M_lf, M_hf, S_real, sh_l, sh_m, title):
    """Write reports for LF performance plots."""
    figs = []
    figs.append(lm.plot_performance_LF(M_lf, M_hf, S_real.u.T, sh_l, sh_m,
                                       title=title_opt_lf))
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


write_plot_performance_LF(M_lf, M_hf, S_real, sh_l, sh_m, title_opt_lf)

# %%  Are diffuse field gains the same?
print(f"\n\n{title_opt}\nDiffuse field gain of each loudspeaker (dB)")
print("HF", lm.diffuse_field_gain(M_hf))
print("LF", lm.diffuse_field_gain(M_lf))


# %% what happens if we just use inverse gammas to make the LF
#  -- really ugly, not sure why sooo ugly

title_inv_gammas=f"{S_real.name}: Inverse gammas, LF/HF AllRAD {C.id_string()}"
gamma = shelf.gamma(sh_l, decoder_type='max_rE', decoder_3d=True,
                    return_matrix=True)

figs.append(lm.plot_performance_LF(M_hf @ np.linalg.pinv(gamma),
                                   M_hf,
                                   S_real.u.T, sh_l, sh_m,
                                   title=title_inv_gammas))

# %%

dec_name = f"{slugify.slugify(S.name)}-{order_h}H{order_v}V-N3D"

wfd.write_faust_decoder_vienna(dec_name+"-Vienna.dsp",
                               dec_name+"-Vienna",
                               M_lf, M_hf,
                               sh_l, S_real.r,
                               input_mask=C.channel_mask)

wfd.write_faust_decoder_dual_band(dec_name+"-Optimized.dsp",
                                  dec_name+"-Optimized",
                                  M_hf,
                                  sh_l, S_real.r,
                                  input_mask=C.channel_mask)

wfd.write_faust_decoder_dual_band(dec_name+"-AllRAD.dsp",
                                  dec_name+"-AllRAD",
                                  M_allrad,
                                  sh_l, S_real.r,
                                  input_mask=C.channel_mask)

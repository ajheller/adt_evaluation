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

figs = []

# %%
S = esa.nando_dome(add_imaginary=True)
S_real = esa.nando_dome(add_imaginary=False)
S.plot()


C = pc.ChannelsAmbiX(3, 2)
order_h, order_v, sh_l, sh_m, id_string = pc.ambisonic_channels(C)

# %% example gamma calculations

gamma = shelf.gamma(sh_l, decoder_type='max_rE', decoder_3d=True,
                    return_matrix=False)
print(C.id_string(), 'sh_l =', sh_l, '\n', gamma)

gamma0 = shelf.gamma0(gamma, matching_type='rms')
print(C.id_string(), gamma0)

# for 1H1P gamma0 should be +3 dB (from Gerzon, Practical Periphony)
C_1H1P = pc.ChannelsFuMa(1,1)
sh_l_11 = C_1H1P.sh_l

gamma_11 = shelf.gamma(sh_l_11, decoder_type='max_rE', decoder_3d=True,
                    return_matrix=False)
print(C_1H1P.id_string(), 'sh_l =', sh_l_11, '\n', gamma_11)

gamma0_11 = shelf.gamma0(gamma_11, matching_type='rms')
print(C_1H1P.id_string(), gamma0_11, 20*np.log10(gamma0_11))




# %%  AllRAD

title=f"{S.name}: AllRAD {C.id_string()}"

M_allrad = bd.allrad(sh_l, sh_m,
                     S.az, S.el,
                     speaker_is_real=S.is_real)

gamma = shelf.gamma(sh_l, decoder_type='max_rE', decoder_3d=True,
                    return_matrix=True)
M_allrad_hf = M_allrad @ gamma

# %%

figs.append(lm.plot_performance(M_allrad_hf, S.u[S.is_real].T, sh_l, sh_m,
                                title=title))

figs.append(lm.plot_matrix(M_allrad_hf, title=title))

df_gain_spk, df_gain_tot = lm.diffuse_field_gain(M_allrad_hf)
print(f"""
{title}\n
Diffuse field gain of each loudspeaker (dB)
{df_gain_spk}
Diffuse field gain of array {df_gain_tot}
""")

# %%  Optimized AllRAD -> M_hf

# optimize allrad design at high frequencies

title=f"{S.name}: Optimized HF AllRAD {C.id_string()}"

el_lim = -π/4

M_hf, res_hf = od.optimize_dome(S,
                                ambisonic_order=C,
                                sparseness_penalty=0.0,
                                el_lim=el_lim,
                                do_report=True)

df_gain_spk, df_gain_tot = lm.diffuse_field_gain(M_hf)
print(f"""
{title}\n
Diffuse field gain of each loudspeaker (dB)
{df_gain_spk}
Diffuse field gain of array {df_gain_tot}
""")

# %%  Optimized LF for above -> M_lf

# optimize allrad design at low frequencies


title_opt=f"{S_real.name}: Optimized LF/HF AllRAD {C.id_string()}"

M_lf, res_lf = od.optimize_dome_LF(M_hf, S_real,
                                   ambisonic_order=C,
                                   el_lim=el_lim)

figs.append(lm.plot_performance_LF(M_lf, M_hf, S_real.u.T, sh_l, sh_m,
                                   title=title_opt))

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


write_plot_performance_LF(M_lf, M_hf, S_real, sh_l, sh_m, title_opt)

# %%  Are diffuse field gains the same?
print(f"\n\n{title}\nDiffuse field gain of each loudspeaker (dB)")
print("HF", lm.diffuse_field_gain(M_hf))
print("LF", lm.diffuse_field_gain(M_lf))



# %% what happens if we just use inverse gammas to make the LF
#  -- really ugly, not sure why sooo ugly

gamma = shelf.gamma(sh_l, decoder_type='max_rE', decoder_3d=True,
                    return_matrix=True)

figs.append(lm.plot_performance_LF(M_hf @ np.linalg.pinv(gamma),
                                   M_hf,
                                   S_real.u.T, sh_l, sh_m,
                                   title=title_opt))


# %%

wfd.write_faust_decoder_vienna('SAH_ambdecH_ACN_N3D_VO3H2V.dsp',
                               'SAH_ambdecH_ACN_N3D_VO3H2V',
                               M_lf, M_hf,
                               sh_l, S_real.r)

wfd.write_faust_decoder_dual_band('SAH_ambdecH_ACN_N3D_O3H2V.dsp',
                                  'SAH_ambdecH_ACN_N3D_O3H2V',
                                  M_hf,
                                  sh_l, S_real.r)

wfd.write_faust_decoder_dual_band('SAH_ambdecH_ACN_N3D_A3H2V.dsp', 'SAH_ambdecH_ACN_N3D_A3H2V',
                                  M_allrad,
                                  sh_l, S_real.r)

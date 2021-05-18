#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 10:59:36 2021

@author: heller
"""
import io

import jax.numpy as np  # jax overloads numpy
from numpy import pi as π  # I get tired of typing np.pi

import program_channels as pc
import spherical_grids as sg
import shelf
import basic_decoders as bd
import localization_models as lm
import reports

import example_speaker_arrays as esa
import LoudspeakerLayout as ll
from optimize_decoder_matrix import optimize, optimize_LF


# TODO: this is a copy of stage_test that will morph into a more general
# TODO: function
# TODO: need to clean up the handling of imaginary speakers

def optimize_dome(S,  # the speaker array
                  ambisonic_order=3,
                  el_lim=-π/8,
                  tikhonov_lambda=1e-3,
                  sparseness_penalty=1,
                  do_report=False,
                  rE_goal='auto',
                  eval_order=None,
                  random_start=False
                  ):
    """Test optimizer with CCRMA Stage array."""
    #
    #
    order_h, order_v, sh_l, sh_m, id_string = pc.ambisonic_channels(ambisonic_order)
    order = max(order_h, order_v)  # FIXME
    is_3D = order_v > 0

    if eval_order is None:
        eval_order = ambisonic_order
        eval_order_given = False
    else:
        eval_order_given = True

    eval_order_h, eval_order_v, eval_sh_l, eval_sh_m, eval_id_string = \
        pc.ambisonic_channels(eval_order)

    mask_matrix = pc.mask_matrix(eval_sh_l, eval_sh_m, sh_l, sh_m)
    print(mask_matrix)

    # if True:
    #     S = esa.stage2017() + esa.nadir()#stage()
    #     spkr_array_name = S.name
    # else:
    #     # hack to enter Eric's array
    #     S = emb()
    #     spkr_array_name = 'EMB'

    spkr_array_name = S.name



    print('speaker array = ', spkr_array_name)

    S_u = np.array(sg.sph2cart(S.az, S.el, 1))

    gamma = shelf.gamma(sh_l, decoder_type='max_rE', decoder_3d=is_3D,
                        return_matrix=True)

    figs = []
    if not random_start:
        M_start = 'AllRAD'

        M_allrad = bd.allrad(sh_l, sh_m, S.az, S.el,
                             speaker_is_real=S.is_real)

        # remove imaginary speaker from S_u and Sr
        S_u = S_u[:, S.is_real] #S.Real.values]
        Sr = S[S.is_real] #S.Real.values]

        M_allrad_hf = M_allrad @ gamma

        # performance plots
        plot_title = "AllRAD, "
        if eval_order_given:
            plot_title += f"Design: {id_string}, Test: {eval_id_string}"
        else:
            plot_title += f"Signal set={id_string}"

        figs.append(
            lm.plot_performance(M_allrad_hf, S_u, sh_l, sh_m,
                                mask_matrix = mask_matrix,
                                title=plot_title))

        lm.plot_matrix(M_allrad_hf, title=plot_title)

        print(f"\n\n{plot_title}\nDiffuse field gain of each loudspeaker (dB)")
        for n, g in zip(Sr.ids,
                        10 * np.log10(np.sum(M_allrad ** 2, axis=1))):
            print(f"{n:3}:{g:8.2f} |{'=' * int(60 + g)}")

    else:
        M_start = 'Random'
        # let optimizer dream up a decoder on its own
        M_allrad = None
        # more mess from imaginary speakers
        S_u = S_u[:, S.is_real]
        Sr = S[S.is_real]

    # M_allrad = None

    # Objective for E
    T = sg.t_design5200()
    cap, *_ = sg.spherical_cap(T.u,
                               (0, 0, 1),  # apex
                               π/2 - el_lim)
    E0 = np.where(cap, 1.0, 0.1)  # inside, outside

    # np.array([0.1, 1.0])[cap.astype(np.int8)]

    # Objective for rE order+2 inside the cap, order-2 outside
    rE_goal = np.where(cap,
                       shelf.max_rE_3d(order+2), # inside the cap
                       shelf.max_rE_3d(max(order-2, 1)) # outside the cap
                       )

    #np.array([shelf.max_rE_3d(max(order-2, 1)),
    #                    shelf.max_rE_3d(order+2)])[cap.astype(np.int8)]

    M_opt, res = optimize(M_allrad, S_u, sh_l, sh_m, E_goal=E0,
                          iprint=50, tikhonov_lambda=tikhonov_lambda,
                          sparseness_penalty=sparseness_penalty,
                          rE_goal=rE_goal)

    plot_title = f"Optimized {M_start}, "
    if eval_order_given:
        plot_title += f"Design: {id_string}, Test: {eval_id_string}"
    else:
        plot_title += f"Signal set={id_string}"

    figs.append(
        lm.plot_performance(M_opt, S_u, sh_l, sh_m,
                            mask_matrix = mask_matrix,
                            title=plot_title
                            ))

    lm.plot_matrix(M_opt, title=plot_title)

    with io.StringIO() as f:
        print(f"ambisonic_order = {order}\n" +
              f"el_lim = {el_lim * 180 / π}\n" +
              f"tikhonov_lambda = {tikhonov_lambda}\n" +
              f"sparseness_penalty = {sparseness_penalty}\n",
              file=f)

        off = np.isclose(np.sum(M_opt ** 2, axis=1), 0, rtol=1e-6)  # 60dB down
        print("Using:\n", Sr.ids[~off.copy()], file=f)
        print("Turned off:\n", Sr.ids[off.copy()], file=f)

        print("\n\nDiffuse field gain of each loudspeaker (dB)", file=f)
        for n, g in zip(Sr.ids,
                        10 * np.log10(np.sum(M_opt ** 2, axis=1))):
            print(f"{n:3}:{g:8.2f} |{'=' * int(60 + g)}", file=f)
        report = f.getvalue()
        print(report)

    if do_report:
        reports.html_report(zip(*figs),
                            text=report,
                            directory=spkr_array_name,
                            name=f"{spkr_array_name}-order-{order}")

    return M_opt, dict(M_allrad=M_allrad, off=off, res=res)


def optimize_dome_LF(M_hf,
                     S,
                     ambisonic_order=3,
                     el_lim=-π/8):

    order_h, order_v, sh_l, sh_m, id_string = pc.ambisonic_channels(ambisonic_order)

    # the test directions
    T = sg.t_design5200()
    cap = sg.spherical_cap(T.u, (0, 0, 1), 5*np.pi/6)[0]
    W = np.where(cap, 0.1, 1)

    M_lf, res = optimize_LF(M_hf, S.u.T, sh_l, sh_m, W)

    return M_lf, res



def stage_test(ambisonic_order=3, **kwargs):
    S = esa.stage2017()
    return optimize_dome(S, ambisonic_order, **kwargs)


# unit test
if __name__ == '__main__':
    # unit_test()
    try:
        for d in range(1, 8):
            stage_test(d, do_report=True)
    except KeyboardInterrupt:
        print("bye!")

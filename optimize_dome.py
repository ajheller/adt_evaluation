#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 10:59:36 2021

@author: heller
"""
import io
import datetime

import jax.numpy as np  # jax overloads numpy
from numpy import pi as π  # I get tired of typing np.pi

import basic_decoders as bd
import example_speaker_arrays as esa
import localization_models as lm
import loudspeaker_layout as ll
import program_channels as pc
import reports
import shelf
import spherical_grids as sg
from optimize_decoder_matrix import optimize, optimize_LF


#
def optimize_dome2(M_init, sh_l, sh_m, spkr_u, el_lim, sparseness_penalty=0):

    tikhonov_lambda = 1e-3

    # Objective for E
    T = sg.t_design5200()
    cap, *_ = sg.spherical_cap(T.u, (0, 0, 1), π / 2 - el_lim)  # apex
    E0 = np.where(cap, 1.0, 0.1)  # inside, outside

    # np.array([0.1, 1.0])[cap.astype(np.int8)]

    # Objective for rE order+2 inside the cap, order-2 outside
    order = max(sh_l)
    rE_goal = np.where(
        cap,
        shelf.max_rE_3d(order + 2),  # inside the cap
        shelf.max_rE_3d(max(order - 2, 1)),  # outside the cap
    )
    # rE_W = T.interp_el(np.array((-90, 0, +90))*np.pi/180,
    #                    (0.5, 1, 0.5),
    #                    'linear')
    rE_W = 1

    M_opt, res = optimize(
        M_init,
        spkr_u,
        sh_l,
        sh_m,
        E_goal=E0,
        iprint=50,
        tikhonov_lambda=tikhonov_lambda,
        sparseness_penalty=sparseness_penalty,
        rE_goal=rE_goal,
        rE_W=rE_W,
    )

    return M_opt, res


# TODO: need to clean up the handling of imaginary speakers
def optimize_dome(
    S,  # the speaker array
    ambisonic_order=3,
    el_lim=-π / 8,
    tikhonov_lambda=1e-3,
    sparseness_penalty=1,
    do_report=False,
    rE_goal="auto",
    eval_order=None,
    eval_el_lim=None,
    random_start=False,
    quiet=False,
):
    """Optimize a dome array."""
    #
    #
    if np.isscalar(el_lim):
        el_lim = np.array((el_lim, np.inf))

    if eval_el_lim is None:
        eval_el_lim = el_lim

    (order_h, order_v, sh_l, sh_m, id_string) = pc.ambisonic_channels(ambisonic_order)
    order = max(order_h, order_v)  # FIXME
    is_3D = order_v > 0

    if eval_order is None:
        eval_order = ambisonic_order
        eval_order_given = False
    else:
        eval_order_given = True

    (
        eval_order_h,
        eval_order_v,
        eval_sh_l,
        eval_sh_m,
        eval_id_string,
    ) = pc.ambisonic_channels(eval_order)

    mask_matrix = pc.mask_matrix(eval_sh_l, eval_sh_m, sh_l, sh_m)
    # print(mask_matrix)

    spkr_array_name = S.name
    print("speaker array = ", spkr_array_name)

    S_u = np.array(sg.sph2cart(S.az, S.el, 1))

    gamma = shelf.gamma(
        sh_l, decoder_type="max_rE", decoder_3d=is_3D, return_matrix=True
    )

    figs = []
    if not random_start:
        M_start = "AllRAD"

        M_allrad = bd.allrad(sh_l, sh_m, S.az, S.el, speaker_is_real=S.is_real)

        # remove imaginary speaker from S_u and Sr
        S_u = S_u[:, S.is_real]
        Sr = S[S.is_real]

        M_allrad_hf = M_allrad @ gamma

        # performance plots
        plot_title = f"{spkr_array_name} - AllRAD - "
        if eval_order_given:
            plot_title += f"Design: {id_string}, Test: {eval_id_string}"
        else:
            plot_title += f"Signal set={id_string}"

        figs.append(
            lm.plot_performance(
                M_allrad_hf,
                S_u,
                sh_l,
                sh_m,
                mask_matrix=mask_matrix,
                title=plot_title,
                # el_lim=(-np.pi / 2, +np.pi / 2),
                el_lim=eval_el_lim,
                quiet=quiet,
            )
        )

        print(f"\n\n{plot_title}\nDiffuse field gain of each loudspeaker (dB)")
        for n, g in zip(Sr.ids, 10 * np.log10(np.sum(M_allrad**2, axis=1))):
            print(f"{n:3}:{g:8.2f} |{'=' * int(60 + g)}")

    else:
        M_start = "Random"
        # let optimizer dream up a decoder on its own
        M_allrad = None
        # more mess from imaginary speakers
        S_u = S_u[:, S.is_real]
        Sr = S[S.is_real]

    # M_allrad = None

    # Objective for E
    T = sg.t_design5200()
    cap, *_ = sg.spherical_cap(
        T.u,
        (0, 0, 1),
        angle=π / 2 - el_lim[0],
        min_angle=π / 2 - el_lim[1],
    )  # apex
    E0 = np.where(cap, 1.0, 0.1)  # inside, outside

    # np.array([0.1, 1.0])[cap.astype(np.int8)]

    # Objective for rE order+2 inside the cap, order-2 outside
    rE_goal = np.where(
        cap,
        shelf.max_rE_3d(order + 2),  # inside the cap
        shelf.max_rE_3d(max(order - 2, 1)),  # outside the cap
    )

    # rE_W = T.interp_el(np.array((-90, 0, +90))*np.pi/180,
    #                    (0.5, 1, 0.5),
    #                    'linear')
    rE_W = 1

    M_opt, res = optimize(
        M_allrad,
        S_u,
        sh_l,
        sh_m,
        E_goal=E0,
        iprint=50,
        tikhonov_lambda=tikhonov_lambda,
        sparseness_penalty=sparseness_penalty,
        rE_goal=rE_goal,
        rE_W=rE_W,
        rV_W=0.01,
    )

    # plot_title = f"Optimized {M_start}, "
    plot_title = f"{spkr_array_name} - Optimized - "
    if eval_order_given:
        plot_title += f"Design: {id_string}, Test: {eval_id_string}"
    else:
        plot_title += f"Signal set={id_string}"

    figs.append(
        lm.plot_performance(
            M_opt,
            S_u,
            sh_l,
            sh_m,
            mask_matrix=mask_matrix,
            title=plot_title,
            el_lim=eval_el_lim,
            quiet=quiet,
        )
    )

    with io.StringIO() as f:
        print(
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            f"ambisonic_order = {order}\n"
            f"el_lim = {el_lim[0] * 180 / π:0.4f} .. {el_lim[1] * 180 / π:0.4f}\n"
            f"tikhonov_lambda = {tikhonov_lambda}\n"
            f"sparseness_penalty = {sparseness_penalty}\n",
            file=f,
        )

        off = np.isclose(np.sum(M_opt**2, axis=1), 0, rtol=1e-6)  # 60dB down
        print("Using:\n", Sr.ids[~off.copy()], file=f)
        print("Turned off:\n", Sr.ids[off.copy()], file=f)

        print("\n\nDiffuse field gain of each loudspeaker (dB)", file=f)
        for n, g in zip(Sr.ids, 10 * np.log10(np.sum(M_opt**2, axis=1))):
            print(f"{n:3}:{g:8.2f} |{'=' * int(60 + g)}", file=f)
        report = f.getvalue()
        print(report)

    if do_report not in (None, False):
        report_name = f"{spkr_array_name}-{id_string}"
        if do_report is not True:
            report_name += f"-{do_report}"

        reports.html_report(
            zip(*figs), text=report, directory=spkr_array_name, name=report_name
        )

    return M_opt, dict(M_allrad=M_allrad, off=off, res=res)


def optimize_dome_LF(M_hf, S, ambisonic_order=3, el_lim=-π / 8):

    (order_h, order_v, sh_l, sh_m, id_string) = pc.ambisonic_channels(ambisonic_order)

    # the test directions
    T = sg.t_design5200()
    cap = sg.spherical_cap(T.u, (0, 0, 1), 5 * np.pi / 6)[0]
    W = np.where(cap, 0.1, 1)

    M_lf, res = optimize_LF(M_hf, S.u.T, sh_l, sh_m, W)

    return M_lf, res


#
def stage_test(ambisonic_order=3, **kwargs):
    S = esa.stage2017()
    return optimize_dome(S, ambisonic_order, **kwargs)


# unit test
if __name__ == "__main__":
    # unit_test()
    try:
        for d in range(1, 8):
            stage_test(d, do_report=True)
    except KeyboardInterrupt:
        print("bye!")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 18:40:20 2019

@author: heller
"""
# This file is part of the Ambisonic Decoder Toolbox (ADT)
# Copyright (C) 2018-20  Aaron J. Heller <heller@ai.sri.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import getpass
import platform
from datetime import datetime

import shelf


def array2faust_vector(a, prefix=None, suffix=None):
    """
    Return a string in FAUST syntax with the contents of A.

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    prefix : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    s : TYPE
        DESCRIPTION.

    """
    if prefix is None:
        prefix = ""
    if suffix is None:
        suffix = ""

    # make sure 'a' is a ndarray
    a = np.asarray(a)

    if prefix is None:
        s = ""
    else:
        s = prefix

    s += np.array2string(
        a,
        separator=", ",
        suppress_small=True,
        max_line_width=np.inf,
        sign="+",
        threshold=np.inf,
    ).translate(str.maketrans("[]", "()"))
    if suffix is not None:
        s += suffix

    return s


# "{m:14.10f,}".format(m=m)


def matrix2faust(m, prefix="", fid=None):
    """
    Write the matrix m to fid in FAUST syntax.

    Parameters
    ----------
    m : TYPE
        DESCRIPTION.
    prefix : TYPE, optional
        DESCRIPTION. The default is ''.
    fid : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    s : TYPE
        DESCRIPTION.

    """
    s = "".join(
        [
            array2faust_vector(v, prefix=(prefix % i), suffix=";\n")
            for i, v in enumerate(m)
        ]
    )
    if fid is not None:
        fid.write(s)

    return s


def bool2faust(a):
    return "1" if a else "0"


def faust_decoder_description(
    path,
    description,
    array_name,
    order_h,
    order_v,
    coeff_order="acn",
    coeff_scale="N3D",
    input_scale="N3D",
    mixed_order_scheme="HV",
    input_channel_order=None,
    output_speaker_order=None,
):

    run_by = getpass.getuser()
    on_node = platform.node() + " (" + platform.platform() + ")"
    at_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    s = f"""
// Faust Decoder Configuration File
// Written by Ambisonic Decoder Toolbox, version 8.0
// run by {run_by} on {on_node}
// at {at_time}

//------- decoder information -------
// decoder file = {path}
// description = {description}
// speaker array name = {array_name}
// horizontal order   = {order_h}
// vertical order     = {order_v}
// coefficient order  = {coeff_order}
// coefficient scale  = {coeff_scale}
// input scale        = {input_scale}
// mixed-order scheme = {mixed_order_scheme}
// input channel order: {input_channel_order}
// output speaker order: {output_speaker_order}
//-------
"""
    return s


def faust_decoder_configuration(
    name,
    nbands,
    decoder_type,
    decoder_order,
    channel_order,
    nspkrs,
    rspkrs,
    input_mask,
    *,
    gamma0=None,
    gamma1=None,
    xover_freq=380,
    lfhf_ratio_dB=0,
    input_full_set=False,
    delay_comp=True,
    level_comp=True,
    nfc_output=True,
    nfc_input=False,
    output_gain_muting=True,
):
    """
    Write the config for the ADT's decoder in FAUST.

    Parameters
    ----------
    name : TYPE
        DESCRIPTION.
    nbands : TYPE
        DESCRIPTION.
    decoder_type : TYPE
        DESCRIPTION.
    xover_freq : TYPE
        DESCRIPTION.
    lfhf_ratio_dB : TYPE
        DESCRIPTION.
    decoder_order : TYPE
        DESCRIPTION.
    co : TYPE
        DESCRIPTION.
    input_full_set : TYPE
        DESCRIPTION.
    delay_comp : TYPE
        DESCRIPTION.
    level_comp : TYPE
        DESCRIPTION.
    nfc_output : TYPE
        DESCRIPTION.
    nfc_input : TYPE
        DESCRIPTION.
    output_gain_muting : TYPE
        DESCRIPTION.
    nspkrs : TYPE
        DESCRIPTION.
    rspkrs : TYPE
        DESCRIPTION.
    gamma0 : TYPE
        DESCRIPTION.
    gamma1 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #

    # error checking
    if len(rspkrs) != nspkrs:
        raise ValueError("len(rspkrs) != nspkrs")

    if gamma0 is None:
        gamma0 = np.ones(np.max(channel_order))

    radius_str = array2faust_vector(rspkrs, prefix="rs = ", suffix=";\n")
    gamma_str = array2faust_vector(gamma0, prefix="gamma(0) = ", suffix=";\n")
    if gamma1 is not None:
        gamma_str += array2faust_vector(gamma1, prefix="gamma(1) = ", suffix=";\n")
    input_mask_str = ",".join(["_" if m else "!" for m in input_mask])

    s = f"""
// start decoder configuration
declare name	"{name}";

// bands
nbands = {nbands};

// decoder type
decoder_type = {decoder_type};

// crossover frequency in Hz
xover_freq = hslider("xover [unit:Hz]",{xover_freq},200,800,20): dezipper;

// lfhf_balance
lfhf_ratio = hslider("lf/hf [unit:dB]", {lfhf_ratio_dB}, -3, +3, 0.1) :
             mu.db2linear : dezipper;


// decoder order
decoder_order = {decoder_order};

// ambisonic order of each input component
co = {tuple(channel_order)};

// use full or reduced input set
input_full_set = {int(input_full_set)};

// mask for full ambisonic set to channels in use
input_mask(0) = bus(nc);
input_mask(1) = ({input_mask_str});
//FIXME: input_mask(1) = ????

// delay compensation
delay_comp = {int(delay_comp)};

// level compensation
level_comp = {int(level_comp)};

// nfc on input or output
nfc_output = {int(nfc_output)};
nfc_input  = {int(nfc_input)};

// enable output gain and muting controls
output_gain_muting = {int(output_gain_muting)};

// number of speakers
ns = {nspkrs};

// radius for each speaker in meters
{radius_str}

// per order gains, 0 for LF, 1 for HF.
//  Used to implement shelf filters, or to modify velocity matrix
//  for max_rE decoding, and so forth.  See Appendix A of BLaH6.
{gamma_str}

"""
    return s


def gamma2faust(*gammas, comment=True):
    o = []
    if comment:
        o.append(
            """
// per order gains, 0 for LF, 1 for HF.
//  Used to implement shelf filters, or to modify velocity matrix
//  for max_rE decoding, and so forth.  See Appendix A of BLaH6."""
        )

    for (i, g) in enumerate(gammas):
        o.append(array2faust_vector(g, prefix=f"gamma({i}) = ", suffix=";"))
    return "\n".join(o)


def append_implementation(f):
    with open("ambi-decoder_preamble2.dsp", "r") as adp:
        f.write(adp.read())


def write_faust_decoder(path, name, decoder_matrix, sh_l, r, input_mask):
    with open(path, "w") as f:
        f.write(
            faust_decoder_configuration(
                name,
                nbands=1,
                decoder_type=1,
                decoder_order=np.max(sh_l),
                channel_order=sh_l,
                nspkrs=len(r),
                rspkrs=r,
                input_mask=input_mask,
            )
        )
        f.write(matrix2faust(decoder_matrix, prefix="s(%03d, 0) = "))
        append_implementation(f)
        return f.name


def write_faust_decoder_dual_band(path, name, M, sh_l, r, input_mask, **gamma_kw):
    if M.shape != (len(r), len(sh_l)):
        raise ValueError(
            "M.shape != (len(r), len(sh_l))" f"{M.shape} {(len(r), len(sh_l))}"
        )

    # FIXME: this block of code should move to a function in shelf.py
    if True:
        order = np.max(sh_l)
        gamma_hf = shelf.gamma(range(order + 1), **gamma_kw)
        gamma_lf = np.ones_like(gamma_hf)

        # split the gain between LF and HF
        g0 = shelf.gamma0(shelf.gamma(sh_l, **gamma_kw), n_spkrs=len(r))
        sqrt_g0 = np.sqrt(g0)
        print(gamma_lf, gamma_hf, g0)
        gamma_lf /= sqrt_g0
        gamma_hf *= sqrt_g0

    with open(path, "w") as f:
        f.write(faust_decoder_description(path, name, name, None, None))
        f.write(
            faust_decoder_configuration(
                name,
                nbands=2,
                decoder_type=2,
                decoder_order=np.max(sh_l),
                channel_order=sh_l,
                nspkrs=len(r),
                rspkrs=r,
                input_mask=input_mask,
                gamma0=gamma_lf,
                gamma1=gamma_hf,
            )
        )
        f.write(matrix2faust(M, prefix="s(%03d, 0) = "))
        append_implementation(f)
        return f.name


def write_faust_decoder_vienna(path, name, M_lf, M_hf, sh_l, r, input_mask):
    if M_lf.shape != M_hf.shape:
        raise ValueError("M_lf.shape != M_hf.shape" f" {M_lf.shape}, {M_hf.shape}")
    if M_lf.shape != (len(r), len(sh_l)):
        raise ValueError(
            "M_lf.shape != (len(r), len(sh_l))" f"{M_lf.shape} {(len(r), len(sh_l))}"
        )

    with open(path, "w") as f:
        f.write(faust_decoder_description(path, name, name, None, None))
        f.write(
            faust_decoder_configuration(
                name,
                nbands=2,
                decoder_type=3,
                decoder_order=np.max(sh_l),
                channel_order=sh_l,
                nspkrs=len(r),
                rspkrs=r,
                input_mask=input_mask,
                gamma0=np.ones(np.max(sh_l) + 1),
                gamma1=np.ones(np.max(sh_l) + 1),
            )
        )
        f.write(matrix2faust(M_lf, prefix="s(%03d, 0) = "))
        f.write(matrix2faust(M_hf, prefix="s(%03d, 1) = "))
        append_implementation(f)
        return f.name


# %% scripts to compile
import subprocess as sub


def compile_dsp(path, command="faust2sndfile"):
    sub.call([command, path])

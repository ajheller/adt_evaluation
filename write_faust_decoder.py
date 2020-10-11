#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 18:40:20 2019

@author: heller
"""

import numpy as np
import getpass
import platform
import time


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
        prefix = ''
    if suffix is None:
        suffix = ''

    # make sure 'a' is a ndarray
    a = np.asarray(a)

    if prefix is None:
        s = ''
    else:
        s = prefix

    s += np.array2string(a,
                         separator=', ', suppress_small=True,
                         max_line_width=np.inf, sign='+',
                         threshold=np.inf,
                         ).translate(str.maketrans('[]', '()'))
    if suffix is not None:
        s += suffix

    return s

# "{m:14.10f,}".format(m=m)


def matrix2faust(m, prefix='', fid=None):
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
    s = "\n".join([array2faust_vector(v, prefix=(prefix % i), suffix=';\n')
                   for i, v in enumerate(m)])
    if fid is not None:
        fid.write(s)

    return s


def bool2faust(a):
    return '1' if a else '0'


def faust_decoder_description(path,
                              description,
                              array_name,
                              order_h, order_v,
                              coeff_order='acn',
                              coeff_scale='N3D',
                              input_scale='N3D',
                              mixed_order_scheme='HV',
                              input_channel_order=None,
                              output_speaker_order=None
                              ):

    run_by = getpass.getuser()
    on_node = platform.node() + " (" + platform.platform() + ")"
    at_time = time.time()  # make this human readable!

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


def faust_decoder_configuration(name, nbands=2, decoder_type=2,
                                xover_freq=380, lfhf_ratio_dB=0,
                                decoder_order=3,
                                co=(0,
                                    1, 1, 1,
                                    2, 2, 2, 2, 2,
                                    3, 3, 3, 3, 3, 3, 3, 3),
                                input_full_set=False,
                                delay_comp=True, level_comp=True,
                                nfc_output=True, nfc_input=False,
                                output_gain_muting=True,
                                nspkrs=24,
                                rspkrs=24*(1,),
                                gamma0=(1, 1, 1, 1),
                                gamma1=(1, 1, 1, 1)):
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

    radius_str = array2faust_vector(rspkrs, prefix='rs = ', suffix=';\n')
    gamma_str = array2faust_vector(gamma0, prefix="gamma(0) = ", suffix=';\n')
    if gamma1:
        gamma_str += array2faust_vector(gamma1, prefix="gamma(1) = ",
                                        suffix=';\n')

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
co = {co};

// use full or reduced input set
input_full_set = {int(input_full_set)};

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


def write_faust_decoder(path, name, decoder_matrix, sh_l, r):
    with open(path, 'w') as f:
        f.write(
            faust_decoder_configuration(name, nbands=1, decoder_type=1,
                                        decoder_order=np.max(sh_l), co=sh_l,
                                        nspkrs=len(r), rspkrs=r))
        f.write(matrix2faust(decoder_matrix, prefix="s(%03d, 0) = "))
        # append the implementation
        f.write(open("ambi-decoder_preamble2.dsp", 'r').read())

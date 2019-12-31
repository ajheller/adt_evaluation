#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 18:40:20 2019

@author: heller
"""

import numpy as np
# import string as str


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

    # make sure a is an ndarray
    a = np.array(a, copy=False)

    if prefix is None:
        s = ''
    else:
        s = prefix

    s += np.array2string(a,
                         separator=', ', suppress_small=True,
                         max_line_width=np.inf, sign='+'
                         ).translate(str.maketrans('[]', '()'))
    if suffix is not None:
        s += suffix

    return s

# "{m:14.10f,}".format(m=m)


def matrix2faust(m, prefix='', fid=None):
    """
    Write matrix m to fid in FAUST syntax.

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
    s = "\n".join([array2faust_vector(v, prefix=(prefix % i))
                   for i, v in enumerate(m)])
    if fid is not None:
        fid.write(s)

    return s

def bool2faust(a):
    return '1' if a else '0'


"""
// Faust Decoder Configuration File
// Written by Ambisonic Decoder Toolbox, version 8.0
// run by heller on Crean-2.local (MACI64) at 25-Dec-2019 12:13:15

//------- decoder information -------
// decoder file = /Users/heller/Documents/adt/decoders/S_Katz_914_3h1p_allrad_5200_rE_max_2_band.dsp
// description = S_Katz_914_3h1p_allrad_5200_rE_max_2_band
// speaker array name = S_Katz_914
// horizontal order   = 3
// vertical order     = 1
// coefficient order  = acn
// coefficient scale  = SN3D
// input scale        = SN3D
// mixed-order scheme = HP
// input channel order: W Y Z X V U Q P
// output speaker order: 3 1 2 5 6 9 10 7 8 11 12 13 14
//-------
"""


def write_faust_decoder_configuration(name, nbands=2, decoder_type=2,
                                      xover_freq=380, lfhf_ratio_dB=0,
                                      decoder_order=3,
                                      co=(0,1,1,1,2,2,2,2,3,3,3,3,3,3,3,3,3),
                                      input_full_set=False,
                                      delay_comp=True, level_comp=True,
                                      nfc_output=True, nfc_input=False,
                                      output_gain_muting=True,
                                      nspkrs=24, rspkrs=24*(1,),
                                      gamma0=(1,1,1,1),
                                      gamma1=(1,1,1,1)):
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
input_full_set = {bool2faust(input_full_set)};

// delay compensation
delay_comp = {bool2faust(delay_comp)};

// level compensation
level_comp = {bool2faust(level_comp)};

// nfc on input or output
nfc_output = {bool2faust(nfc_output)};
nfc_input  = {bool2faust(nfc_input)};

// enable output gain and muting controls
output_gain_muting = {bool2faust(output_gain_muting)};

// number of speakers
ns = {nspkrs};

// radius for each speaker in meters
rs = (         2.199,         2.676,         2.676,         1.908,         1.784,         1.614,         1.466,         1.949,         1.828,         2.445,         2.445,         2.351,         2.351);

// per order gains, 0 for LF, 1 for HF.
//  Used to implement shelf filters, or to modify velocity matrix
//  for max_rE decoding, and so forth.  See Appendix A of BLaH6.
gamma(0) = (             1,             1,             1,             1);
gamma(1) = (   1.386698221,   1.194136191,  0.8491219424,  0.4225921018);
"""

    return s

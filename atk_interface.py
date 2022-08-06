#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:56:34 2021

@author: heller
"""

import yaml
import json

from adt_utils import timestamp


def write_atk_yml(
    filename,
    M,
    spkr_az,
    spkr_el,
    atk_signal_set="HOA3",
    ordering_type="ACN",
    normalization_type="N3D",
    decoder_type="energy",
    decoder_name="ADTOptimized",
    lf_hf_match_type="rms",
):
    o = dict(
        type="decoder",
        kind=decoder_name,
        ordering=ordering_type,
        normalisation=normalization_type,
        beam=decoder_type,
        match=lf_hf_match_type,
        set=atk_signal_set,
        directions=[[float(x), float(y)] for x, y in zip(spkr_az, spkr_el)],
        matrix=M.tolist(),
    )
    ts = timestamp(join="\n# ")  # no \'s inside {} in f-strings
    with open(filename, "w") as f:
        if True:
            f.write("# written by ADT\n" f"# {ts}" "\n\n")
            yaml.dump(o, f, allow_unicode=True, sort_keys=False)
        else:
            json.dump(o, f, indent=0)


"""
type : 'decoder'

kind : 'ADTpinv'

ordering : 'ACN'

normalisation : 'N3D'

beam : 'energy'

match : 'amp'

alpha : 0

set : 'HOA3'
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:35:22 2020

@author: heller
"""

from dataclasses import dataclass, field
from loudspeaker_layout import LoudspeakerLayout
from program_channels import Channels
import basic_decoders as bd
import write_faust_decoder as wfd

import optimize_dome as od

import numpy as np

from attr import attrs, attrib


@attrs
class Decoder:
    """A class to bundle everything for making decoders under a single API

    Returns:
        _type_: _description_
    """

    C = attrib()
    S = attrib()
    M_basic = attrib()

    name = attrib()
    description = attrib()

    def __init__(
        self,
        C,
        S,
        bands=2,
        f_xover=380,  # Hz
        nfc=True,
        name=None,
        description=None,
    ):
        self.C = C
        self.S = S
        self.bands = bands
        self.f_xover = f_xover
        self.nfc = nfc

    def __str__(self):
        pass

    def description(self):
        return self._description or self.C.id_string()

    def long_name(self, suffix="") -> str:
        return f"{self.name}-{self.description}{suffix}"

    def compute_inversion(self):
        M = bd.inversion(self.C.sh_l, self.C.sh_m, self.S.az, self.S.el)
        self.M_lf = M
        return M

    def compute_allrad(self):
        M = bd.allrad(self.C.sh_l, self.C.sh_m, self.S.az, self.S.el)
        self.M_lf = M
        return M

    def write_ambdec(filename=None):
        pass

    def write_faust(filename=None):
        pass


# A quick and dirty function to make decoders for dome arrays


def make_decoder(C, S, el_lim, eval_el_lim, quiet=False):
    M_hf, result_dict_hf = od.optimize_dome(
        S,
        ambisonic_order=C,
        # eval_order=C,
        sparseness_penalty=1,
        el_lim=el_lim,
        do_report=True,
        random_start=False,
        eval_el_lim=eval_el_lim,
        quiet=quiet,
    )
    M_lf, result_dict_lf = od.optimize_dome_LF(
        M_hf,
        S.real_only(),
        ambisonic_order=C,
        el_lim=el_lim,
    )
    return M_hf, M_lf, result_dict_hf, result_dict_lf

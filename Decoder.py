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

import numpy as np

from attr import attrs, attrib


@attrs
class Decoder:
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

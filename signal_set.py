#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 18:17:51 2021

@author: heller
"""

import numpy as np


class SignalSet:
    def __init__(self):
        self.name = ""
        self.h_order = 0
        self.v_order = 0
        self.sh_l = []
        self.sh_m = []
        self.sh_norm = []
        self.channel_names = []
        self.channel_mask = []

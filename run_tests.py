#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:36:40 2020

@author: heller
"""

import spherical_data
import spherical_grids
import loudspeaker_layout
import basic_decoders
import optimize_decoder_matrix

spherical_data.unit_test()

spherical_grids.unit_tests()

loudspeaker_layout.unit_test()

basic_decoders.unit_test()
basic_decoders.unit_test2()

optimize_decoder_matrix.unit_test()

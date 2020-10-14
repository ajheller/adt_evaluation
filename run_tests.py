#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:36:40 2020

@author: heller
"""

import SphericalData
import SphericalGrids
import LoudspeakerLayout
import basic_decoders
import optimize_decoder_matrix

SphericalData.unit_test()

SphericalGrids.unit_tests()

LoudspeakerLayout.unit_test()

basic_decoders.unit_test()
basic_decoders.unit_test2()

optimize_decoder_matrix.unit_test()

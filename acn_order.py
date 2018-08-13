# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:01:38 2015

@author: heller
"""

# ### acn stuff

from numpy import floor, sqrt


def acn(degree, index):
    "degree is l, index is m"
    if abs(index) > degree:
        raise ValueError("abs(index) should be <= degree")

    return degree**2 + degree + index


def acn2lm(channel_number):
    if channel_number < 0:
        raise ValueError("channel_number should be non-negative integer")

    degree = floor(sqrt(channel_number)).astype(int)
    index = channel_number - degree**2 - degree
    return (degree, index)

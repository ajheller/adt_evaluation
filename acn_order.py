# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:01:38 2015

@author: heller
"""
# This file is part of the Ambisonic Decoder Toolbox (ADT)
# Copyright (C) 2018-19  Aaron J. Heller <heller@ai.sri.com>
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


__fuma_channel_names = [u'W',
                        u'Y', u'Z', u'X',
                        u'V', u'T', u'R', u'S', u'U',
                        u'Q', u'O', u'M', u'K', u'L', u'N', u'P']


def acn2fuma_name(acn):
    try:
        name = __fuma_channel_names[acn]
    except IndexError:
        l, m = acn2lm(acn)
        name = u'%d.%d' % (l, abs(m))
        if m < 0:
            name = name + u'S'
        else:
            name = name + u'C'
    return name

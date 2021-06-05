#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:45:53 2021

@author: heller
"""

import loudspeaker_layout as lsl

# speaker positons from Rec. ITU-R BS.2051-2 Table 1
#  https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.2051-2-201807-I!!PDF-E.pdf
#  0, 0 is the middle of the display
#  azimuths increase to the left, elevations increase up

_m_az = (0, +22.5, -22.5, +30, -30, +45, -45, +60, -60, +90, -90,
         +110, -110, +135, -135, +180)
_m_el = 0
# there are two additional "M" speakers at the left and right edge of the
# display that are omitted here

_u_az = _m_az
_u_el = +30

_uh_az = (+180,)
_uh_el = +45

_t_az = (0, )
_t_el = +90

_b_az = _m_az
_b_el = -30

_lfe_az = (+45, -45)
_lfe_el = -30


def itu_all(r=1):
    m = lsl.from_vectors(_m_az, _m_el, r, name="ITU-M",
                         description="Middle layer of ITU Array",
                         coord_code='AER', unit_code='DDM',
                         ids=[f"M{int(az):+04d}" for az in _m_az])

    u = lsl.from_vectors(_u_az, _u_el, r, name="ITU-U",
                         description="Upper layer of ITU Array",
                         coord_code='AER', unit_code='DDM',
                         ids=[f"U{int(az):+04d}" for az in _u_az])

    uh = lsl.from_vectors(_uh_az, _uh_el, r, name="ITU-UH",
                          description="Upper-High layer of ITU Array",
                          coord_code='AER', unit_code='DDM',
                          ids=[f"UH{int(az):+04d}" for az in _uh_az])

    t = lsl.from_vectors(_t_az, _t_el, r, name="ITU-T",
                         description="Top layer of ITU Array",
                         coord_code='AER', unit_code='DDM',
                         ids=[f"T{int(az):+04d}" for az in _t_az])

    b = lsl.from_vectors(_b_az, _b_el, r, name="ITU-B",
                         description="Bottom layer of ITU Array",
                         coord_code='AER', unit_code='DDM',
                         ids=[f"B{int(az):+04d}" for az in _b_az])

    a = m + u + uh + t + b
    a.name = "ITU-All",
    a.description = "All Speakers in ITU Document"

    return a


def itu_4_5_0(r=1):
    a = itu_all(r)
    a.name = 'ITU-4+5+0'
    a.description = 'ITU-D Array'
    return a[[0, 3, 4, 11, 12, 19, 20, 27, 28]]

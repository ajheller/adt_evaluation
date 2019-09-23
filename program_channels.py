#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 13:07:28 2018

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


import numpy as np
from numpy import pi, inf

from attr import attrs, attrib, validators

"""
 NOTE:
  Ambisonics traditionally calls degree "order", and order "channel". A&S
  [1], Mathematica, and MATLAB use n for degree and m for order. Daniel [2]
  and Malham [3] use m for degree, and n for order. Malham also uses
  \varsigma to indicate cosine (=1) or sine(=-1) for the sectoral
  component. The 2011 AmbiX proposal [4] uses n,m, whereas the earlier one
  [5], Chapman's website [6], and Wikipedia [7] use l,m.

  This toolbox uses l for degree and m for order.

  References:

  [1] I. A. Stegun, "Legendre Functions," in Handbook of Mathematical
  Functions, M. Abramowitz and I. A. Stegun, Eds. Washington, DC: National
  Bureau of Standards, 1964, pp. 331?341.

  [2] J. Daniel, "Spatial Sound Encoding Including Near Field Effect:
  Introducing Distance Coding Filters and a Viable, New Ambisonic Format,"
  Preprints 23rd AES International Conference, Copenhagen, 2003.

  [3] D. G. Malham, "Higher order Ambisonic systems," Space in Music -
  Music in Space, 2003.

  [4] C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, "AMBIX - A
  SUGGESTED AMBISONICS FORMAT," presented at the 3rd International
  Symposium on Ambisonics and Spherical Acoustics, 2011.

  [5] M. Chapman, W. Ritsch, T. Musil, I. Zmölnig, H. Pomberger, F. Zotter,
  and A. Sontacchi, "A standard for interchange of Ambisonic signal sets
  including a file standard with metadata," presented at the Proc. of the
  Ambisonics Symposium, Graz, 2009.

  [6] M. Chapman, "The Ambisonics Association," http://ambisonics.ch

  [7] Wikipedia contributors. "Spherical harmonics." Wikipedia, The Free
  Encyclopedia. Wikipedia, The Free Encyclopedia, 11 Oct. 2017. Web.
  1 Nov. 2017.

"""

# normalization conventions
# the assumption is that the underlying real spherical harmonic code produces
# return full orthonormal values, hence these functions give the gains needed
# to produce the target normalization from those

def normalization_semi(sh_l, sh_m):
    "gains to produce schmidt semi-normalized values from full orthronormal"
    return np.sqrt(2 * sh_l + 1)


def normalization_full(sh_l, sh_m):
    return 1


# mixed-order sets
#  there are two convetions for mixed order sets, HP and HV
#  (get citations)

#    switch upper(scheme)
#        case {'HP', 'AMB'}
#            % used in AMB files, h and v orders independant
#            %  this is what .AMB files use
#            sh_zonal_p    = sh.m == 0;
#            sh_tesseral_p = sh.l == abs(sh.m);
#            sh_sectoral_p = ~sh_zonal_p & ~sh_tesseral_p;
#
#            ch_mask = ...
#               (  sh_tesseral_p & (sh.l <= h_order) ) | ...
#               ( ~sh_tesseral_p & (sh.l <= v_order) );
#            scheme = 'HP';
#
#        case {'HV', 'TRAVIS'}
#            % Travis HV scheme, see [1]
#            ch_mask = sh.l-abs(sh.m) <= v_order & (sh.l<=max(h_order,v_order));
#            scheme = 'HV';
#
#        otherwise
#            error('unknown mixed-order scheme: "%s" ', scheme);


#              W   X   Y   Z \  R   S   T   U   V |  K   L   M   N   O   P   Q
_FuMa_sh_l = np.array(( 0,  1,  1,  1,   2,  2,  2,  2,  2,   3,  3,  3,  3,  3,  3,  3))
_FuMa_sh_m = np.array(( 0,  1, -1,  0,   0,  1, -1,  2, -2,   0,  1, -1,  2, -2,  3, -3))

_FuMa_channel_names = np.array(tuple("W" + "XYZ" + "RSTUV" + "KLMNOPQ"))

_FuMa_channel_normalization = 1/np.sqrt(np.array(
        ((2,) +             # W

         (3,) * 3 +         # X Y Z

         (5,) +             # R
         (5*3/4,) * 4 +     # S T U V

         (7,) +             # K
         (7*32/45,) * 2 +   # L M
         (7*5/9,) * 2 +     # N O
         (7*5/8,) * 2       # P Q
         )))


#
def is_zonal_sh(sh_l, sh_m):
    """Return True for zonal spherical harmonics.
    http://mathworld.wolfram.com/ZonalHarmonic.html"""

    return sh_m == 0


def is_sectoral_sh(sh_l, sh_m):
    """Return True for sectoral spherical harmonics.
    http://mathworld.wolfram.com/SectorialHarmonic.html"""

    return sh_l == np.abs(sh_m)


def is_tesseral_sh(sh_l, sh_m):
    """Return True for tesseral spherical harmonics.
    http://mathworld.wolfram.com/TesseralHarmonic.html"""

    return ~is_sectoral_sh(sh_l, sh_m) & ~is_zonal_sh(sh_l, sh_m)


def channel_mask_HP(sh_l, sh_m, h_order, v_order):
    if h_order != v_order:
        # sectoral harmonics are the horizontal
        sectoral_sh = is_sectoral_sh(sh_l, sh_m)
        ch_mask = (( sectoral_sh & (sh_l <= h_order)) |
                   (~sectoral_sh & (sh_l <= v_order)))
    else:
        ch_mask = sh_l <= h_order
    return ch_mask


def channel_mask_HV(sh_l, sh_m, h_order, v_order):
    if h_order != v_order:
        ch_mask = ((sh_l - np.abs(sh_m)) <= v_order &
                   (sh_l <= max(h_order, v_order)))
    else:
        # this handles the case where sh_l has entries greater than h_order
        ch_mask = sh_l <= h_order

    return ch_mask


def ambisonic_channels_acn(ambisonic_order):
    for l in range(ambisonic_order+1):
        for m in range(-l, l+1):
            yield (l, m)


def ambisonic_channels_sid(ambisonic_order):
    for l in range(ambisonic_order+1):
        for m in range(l, -1, -1):
            yield (l, m)
            if m > 0:
                yield (l, -m)


def h_order_validator(self, attribute, value):
    if not (int(value) == value and 0 <= value):
        raise ValueError("h_order must a non-negative integer")


def v_order_validator(self, attribute, value):
    if not (int(value) == value and 0 <= value):
        raise ValueError("v_order must a non-negative integer no greater than h_order")


@attrs
class Channels(object):

    h_order = attrib()
    v_order = attrib()

    sh_l = attrib()
    sh_m = attrib()

    normalization = attrib()
    cs_phase = attrib()

    channel_mask = attrib()
    channel_names = attrib()
    name = attrib()

    def __str__(self, verbose=False):
        return "<Channels %s %dH%dV>" % (self.name, self.h_order, self.v_order)


class ChannelsAmbisonic(Channels):
    "This class fills in defaults and does sanity checks"

    def __init__(self, h_order, v_order,
                 sh_l, sh_m,
                 normalization,
                 cs_phase=None,
                 mixed_order_scheme='HV',
                 name=None):

        h_order = int(h_order)
        v_order = int(v_order)
        if v_order > h_order:
                pass # FIXME raise a domain exception

        # make sure they're NumPy arrays
        sh_l = np.array(sh_l)
        sh_m = np.array(sh_m)
        normalization = np.array(normalization)

        if cs_phase:
            pass # FIXME check that it is the same length as sh_l
        else:
            cs_phase = np.ones_like(sh_l)

        if mixed_order_scheme.upper().startswith('HV'):
            channel_mask = channel_mask_HV(sh_l, sh_m,
                                           h_order, v_order)
        elif mixed_order_scheme.upper().startswith('HP'):
            channel_mask = channel_mask_HP(sh_l, sh_m,
                                           h_order, v_order)

        super().__init__(
                h_order, v_order,
                sh_l[channel_mask], sh_m[channel_mask],
                normalization[channel_mask],
                cs_phase[channel_mask],
                channel_mask,
                name)


class ChannelsAmbiX(ChannelsAmbisonic):
    def __init__(self, h_order, v_order=None, mixed_order_scheme='HV'):
        if v_order is None:
            v_order = h_order

        h_order = int(h_order)
        v_order = int(v_order)

        sh_l = np.array([l for l in range(h_order+1) for m in range(-l, l+1)])
        sh_m = np.array([m for l in range(h_order+1) for m in range(-l, l+1)])
        norm = normalization_semi(sh_l, sh_m)
        super().__init__(
                h_order, v_order,
                sh_l, sh_m, norm,
                mixed_order_scheme=mixed_order_scheme,
                name="AmbiX")


"""
http://members.tripod.com/martin_leese/Ambisonic/Harmonic.html

Higher-order components
-----------------------

The zero- and first-order components can be augmented by second- and
third-order spherical harmonic components. To date, because of the need for an
impractical number of transmission channels, little work has been carried out
on higher-order Ambisonic systems. However, work has started on the development
of the necessary microphones and decoders.

Second-order Ambisonics requires five transmission channels for horizontal
surround sound and nine for full-sphere. Third-order Ambisonics requires seven
channels for horizontal and sixteen for full-sphere. It is also possible to mix
a high-order horizontal with a lower-order full-sphere; this would require an
intermediate number of transmission channels, as listed in the following table:

Number of channels
Malham notation
Soundfield type
Horizontal order
Height order
Channels

3	h	horizontal	1	0	WXY
4	f	full-sphere	1	1	WXYZ
5	hh	horizontal	2	0	WXYUV
6	fh	mixed-order	2	1	WXYZUV
9	ff	full-sphere	2	2	WXYZRSTUV
7	hhh	horizontal	3	0	WXYUVPQ
8	fhh	mixed-order	3	1	WXYZUVPQ
11	ffh	mixed-order	3	2	WXYZRSTUVPQ
16	fff	full-sphere	3	3	WXYZRSTUVKLMNOPQ
"""

class ChannelsFuMa(ChannelsAmbisonic):
    def __init__(self, h_order, v_order=None, mixed_order_scheme='HP'):
        if h_order > 3:
            pass # FIXME: raise domain error
        if v_order is None:
            v_order = h_order

        norm = _FuMa_channel_normalization

        super().__init__(
                h_order, v_order,
                _FuMa_sh_l, _FuMa_sh_m, norm,
                mixed_order_scheme=mixed_order_scheme,
                name="FuMa")
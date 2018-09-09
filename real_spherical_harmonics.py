# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 09:21:47 2014

@author: heller
"""

from __future__ import print_function
from __future__ import division

from numba import jit
from scipy import special, pi, sin, cos, sqrt
from scipy.integrate import dblquad

from numpy import conj, real, imag, floor

from acn_order import *

# print scipy.special.sph_harm(0,1,0,scipy.pi/2)

# (q,e) = dblquad(lambda y,x: sin(y), 0,2*pi, lambda x: 0, lambda y:pi)

# print q/(4*scipy.pi)
# print e


def dblquad_test():
    # should be 1/3 * 4*pi = 4.1887902047863905
    #  note this is the random energy efficiency of a figure-8 microphone
    qq, ee = dblquad(
        lambda y, x: cos(y)**2 * sin(y),
        0, 2*pi,
        lambda x: 0, lambda y: pi)

    return qq, ee


# quadpack doesn't take complex integrands, so we gotta do it twice. Sheesh!!!
def complex_dblquad(func, a, b, gfun, hfun):

    def real_func(x, y):
        return real(func(x, y))

    def imag_func(x, y):
        return imag(func(x, y))

    real_integral, real_error = dblquad(real_func, a, b, gfun, hfun)
    imag_integral, imag_error = dblquad(imag_func, a, b, gfun, hfun)
    return (complex(real_integral, imag_integral),
            complex(real_error, imag_error))


def check_sph_ortho_c(l1, m1, l2, m2):
    return complex_dblquad(
        # arguments to lambda need to be reversed from args to dblquad (wtf?)
        lambda phi, theta:
            # scipy sph_harm takes order/degree  (wtf?)
            special.sph_harm(m1, l1, theta, phi) *
            conj(special.sph_harm(m2, l2, theta, phi)) * sin(phi),
        0, 2*pi,  # range of theta
        gfun=lambda x: 0, hfun=lambda y: pi  # range of phi
        )


def test_ortho(l1, m1, l2, m2, z, e=1e-10):
    if l1 == l2 and m1 == m2:
        # should be 1
        test = abs(z-1) < abs(e)
    else:
        # should be 0
        test = abs(z) < abs(e)
    return test

def test_ortho_acn(acn1, acn2, z, e=1e-10):
    if acn1 == acn2:
        # should be 1
        test = abs(z-1) < abs(e)
    else:
        # should be 0
        test = abs(z) < abs(e)
    return test


def ortho_test_complex():
    for l1 in range(0, 4):
        for m1 in range(0, l1+1):
            for l2 in range(0, 4):
                for m2 in range(0, l2+1):
                    z, e = check_sph_ortho_c(l1, m1, l2, m2)
                    print(l1, m1, l2, m2, abs(z), abs(e),
                          "Pass = ", test_ortho(l1, m1, l2, m2, z))


@jit
def real_sph_harm(m, l, theta, phi):
    """
    real spherical harmonics w/o Condon-Shortley phase
       m - order -l <= m <= l
       l - degree 0 <= m
       theta - azimuth
       phi - zenith
    """

    # args to sph_harm are order, degree, azimuth, zenith_angle
    Y = special.sph_harm(abs(m), l, theta, phi)
    if m < 0:
        return (-1)**abs(m) * sqrt(2) * imag(Y)
    elif m > 0:
        return (-1)**abs(m) * sqrt(2) * real(Y)
    else:
        return real(Y)


def real_sph_harm_acn(acn, theta, phi):
    (l, m) = acn2lm(acn)
    return real_sph_harm(m, l, theta, phi)


def check_real_sph_ortho(l1, m1, l2, m2):
    return dblquad(
        # arguments to lambda need to be reversed from args to dblquad (wtf?)
        lambda phi, theta:
            # scipy sph_harm takes order/degree  (wtf?)
            real_sph_harm(m1, l1, theta, phi) *
            real_sph_harm(m2, l2, theta, phi) *
            sin(phi),
        0, 2*pi,  # range of theta
        lambda x: 0, lambda y: pi  # range of phi
        )


def check_real_sph_ortho_acn(acn1, acn2):
    return dblquad(
        # arguments to lambda need to be reversed from args to dblquad (wtf?)
        lambda phi, theta:
            # scipy sph_harm takes order/degree  (wtf?)
            real_sph_harm_acn(acn1, theta, phi) *
            real_sph_harm_acn(acn2, theta, phi) *
            sin(phi),
        0, 2*pi,  # range of theta
        lambda x: 0, lambda y: pi  # range of phi
        )


def ortho_test_real(max_degree=3):
    all_ok = True
    for l1 in range(0, max_degree+1):
        for m1 in range(-l1, l1+1):
            for l2 in range(0, max_degree+1):
                for m2 in range(-l2, l2+1):
                    z, e = check_real_sph_ortho(l1, m1, l2, m2)
                    ok = test_ortho(l1, m1, l2, m2, z)
                    all_ok &= ok
                    print(l1, m1, l2, m2, abs(z), abs(e), "Pass=", ok)
    return all_ok


def ortho_test_real_acn(max_degree=3):
    all_ok = True
    for a1 in range(acn(max_degree, max_degree)+1):
        for a2 in range(acn(max_degree, max_degree)+1):
            z, e = check_real_sph_ortho_acn(a1, a2)
            ok = test_ortho_acn(a1, a2, z)
            all_ok &= ok
            print(a1, a2, abs(z), abs(e), "Pass=", ok)
    return all_ok


# at the equator (pi/2 in scipy implementaion)
#   zero crossing for sin (negative m) components at zero should be +
#   zero corssing for cos (positive m) components at -pi/2 should be +
def check_condon_shortley_phase_real(l, m, delta=1e-4):
    if not m & 1:
        return 0
    if m < 0:
        theta = 0
        phi = pi/2
    elif m > 0:
        theta = -pi/2
        phi = pi/2

    ok = (real_sph_harm(m, l, theta+delta, phi) -
          real_sph_harm(m, l, theta-delta, phi)) > 0
    return ok



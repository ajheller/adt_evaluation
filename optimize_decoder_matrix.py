#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 02:48:26 2019

@author: heller
"""

# This file is part of the Ambisonic Decoder Toolbox (ADT)
# Copyright (C) 2018-20  Aaron J. Heller <heller@ai.sri.com>
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


# sanbox for optimizer using Jax for AutoGrad
#   https://github.com/google/jax

# make a conda env for jax, so we can blow it away if we screw up
#  conda create -n jax --clone base

# jax installation:
#   https://github.com/google/jax#installation

# On MacOS 10.13, install jax with
#   pip install jaxlib==0.1.51  # 0.1.52 won't run on MacOS 10.13
#   pip install jax

# on cuda0000x.ai.sri.com install jax like this:
#  PYTHON_VERSION=cp38; CUDA_VERSION=cuda102; PLATFORM=manylinux2010_x86_64
#  BASE_URL='https://storage.googleapis.com/jax-releases'
#  pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.52-$PYTHON_VERSION-none-$PLATFORM.whl
#  pip install --upgrade jax

# Jax Autograd
#  https://github.com/google/jax#automatic-differentiation-with-grad
#  https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
#  https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation


import jax
import jax.numpy as np  # jax overloads numpy
import jax.random as random

import numpy as onp  # 'original' numpy -- this is a convention
from numpy import pi as π  # I get tired of typing np.pi

from Timer import Timer

import scipy.optimize as opt

import spherical_grids as sg
import real_spherical_harmonics as rsh
import basic_decoders as bd
import localization_models as lm
import shelf

import pandas as pd

#  need a local definition so np is jax.np
def rE(M, Su, Y_test):
    G = M @ Y_test
    G2 = G * G  # FIXME: this should be G * G.conj
    E = np.sum(G2, axis=0)
    rExyz = (Su @ G2) / E
    return rExyz, E


def rV(M, Su, Y_test):
    G = M @ Y_test
    P = np.sum(G, axis=0)
    rVxyz = (Su @ G) / P
    return rVxyz, P


def xyz2ur(xyz):
    r = np.linalg.norm(xyz, ord=2, axis=0)
    u = xyz / r
    return u, r


# scipy.optimize needs 64-bit
jax.config.update("jax_enable_x64", True)

# Generate key which is used by JAX to generate random numbers
key = random.PRNGKey(1)

# select ambisonic order of decoder
ambisonic_order = 3
l, m = zip(*rsh.lm_generator(ambisonic_order))


# the test directions
T = sg.t_design5200()

# directtions for plotting
T_azel = sg.az_el()
Y_azel = rsh.real_sph_harm_transform(l, m, T_azel.az, T_azel.el)


# %%

# define a callback for use with opt.minimize
#  Calling this seems to screw up the convergence !?!
ii = 0
def callback(x):
    global ii
    if ii == 0:
        print("Running optimizer")
    ii += 1
    if ii % 50 == 0:
        print(".", end="")
    if ii % 500 == 0:
        print(ii)

# FIXME: this is really the objective funtion, the individual terms are
def loss(M, M_shape0, M_shape1, Su, Y_test, W, tik_lambda=1e-3):
    rExyz, E = rE(M.reshape((M_shape0, M_shape1)), Su, Y_test)
    return (np.sum((rExyz - T.u * 1.0)**2)
            + np.sum((E - W)**2)/10
            + np.sum(M**2) * tik_lambda  # Tikhanov regularization term
            + np.sum(np.abs(M-0.1))  # don't turn off speakers
            #+ np.sum(0.5-M**2)
            )


val_and_grad_fn = jax.jit(jax.value_and_grad(loss),
                          static_argnums=range(1, 7))


def loss_grad(M, M_shape0, M_shape1, Su, Y_test, W, tik_lambda):
    v, g = val_and_grad_fn(M, M_shape0, M_shape1, Su, Y_test, W, tik_lambda)
    # I'm not to happy about having to copy g but L-BGFS needs it in fortran
    # order.  Check with g.flags
    return v, onp.array(g, order='F')  # onp.asfortranarray(g)


# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
def o(M, Su, W=None, ambisonic_order=3, iprint=50, plot=False, tik_lambda=1e-3):

    if W is None:
        W = 1

    l, m = zip(*rsh.lm_generator(ambisonic_order))
    # the test directions
    T = sg.t_design5200()
    Y_test = rsh.real_sph_harm_transform(l, m, T.az, T.el)

    if M is None:
        # infer M_shape from Su and Y
        M_shape = (Su.shape[1],      # number of loudspeakers
                   Y_test.shape[0],  # number of program channels
                   )
        M = random.uniform(key, shape=M_shape, minval=-0.25, maxval=0.25)
    else:
        M_shape = M.shape

    x0 = M.ravel()  # inital guess

    with Timer() as t:
        res = opt.minimize(loss_grad, x0,
                           args=(*M_shape, Su, Y_test, W, tik_lambda),
                           method='L-BFGS-B',
                           jac=True,
                           options=dict(disp=iprint, gtol=1e-8, ftol=1e-12),
                           # callback=callback,
                           )
    if True:
        print()
        print("Execution time: %0.3f" % t.interval)
        print(res)
        print()

    if res.status == 0:
        M_opt = res.x.reshape(M_shape)

        if plot:
            lm.plot_performance(M_opt, Su, ambisonic_order)

    else:
        print('bummer:', res.message)
        raise RuntimeError(res.message)

    return M_opt, res


def unit_test(ambisonic_order=13):
    l, m = zip(*rsh.lm_generator(ambisonic_order))
    # make a  decoder matrix for the 240 speaker t-design via pseudoinverse
    S240 = sg.t_design240()
    Su = S240.u

    # shelf filter gains for max_rE
    # FIXME shelf should return NDARRAY not a list
    gamma = np.diag(np.array(shelf.max_rE_gains_3d(ambisonic_order),
                             dtype=np.float64)[np.array(l)])

    # since this is a spherical design, all three methods should yeild the
    # result

    # inversion
    M240 = bd.inversion(l, m, S240.az, S240.el)

    M240_hf = M240 @ gamma

    lm.plot_performance(M240_hf, Su, ambisonic_order, 'Pinv unit test')
    lm.plot_matrix(M240_hf, title='Pinv unit test')

    # AllRAD
    M240_allrad = bd.allrad(l, m, S240.az, S240.el)
    M240_allrad_hf = M240_allrad @ gamma
    lm.plot_performance(M240_allrad_hf, Su, ambisonic_order, 'AllRAD unit test')
    lm.plot_matrix(M240_allrad_hf, title='AllRAD unit test')

    # NLOpt
    M_opt, res = o(None, Su, 1, ambisonic_order)
    lm.plot_performance(M_opt, Su, ambisonic_order, 'Optimized unit test')
    lm.plot_matrix(M240_allrad, title='Optimized unit test')
    return res


def stage(path='stage.csv'):
    S = pd.read_csv(path)
    S['name'] = S["Name:Stage"]

    # add columns for canonical coordinates
    S['x'], S['y'], S['z'] = \
        sg.sph2cart(S["Azimuth:Degrees"] / 180 * π,
                    S["Elevation:Degrees"] / 180 * π,
                    S["Radius:Inches"] * 2.54 / 100)

    # round trip thru Cartesian to make sure angles are in principal range
    S['az'], S['el'], S['r'] = sg.cart2sph(S.x, S.y, S.z)

    # Nando says this causes an error
    # TODO: where can we put metadata in a Pandas dataframe?
    # S.attrs["Name"] = "Stage"

    # convert "Real" to boolean
    S.Real = (S.Real == "T") | (S.Real == 1)

    return S

import reports
def stage_test(ambisonic_order=3, el_lim=-π/8, tik_lambda=1e-3,
               do_report=False):
    global ii; ii = 0

    l, m = zip(*rsh.lm_generator(ambisonic_order))
    S = stage()
    S_u = (S[['x', 'y', 'z']].T / S.r).values

    gamma = np.diag(np.array(shelf.max_rE_gains_3d(ambisonic_order),
                             dtype=np.float64)[np.array(l)])

    figs = []
    if True:
        # make an AllRAD decoder and plot its performance
        M_allrad = bd.allrad(l, m, S.az, S.el)

        # remove imaginary speaker  FIXME!
        M_allrad = M_allrad[S.Real, :]
        S_u = S_u[:, S.Real]
        Sr = S[S.Real]

        M_allrad_hf = M_allrad @ gamma

        figs.append(
            lm.plot_performance(M_allrad_hf, S_u, ambisonic_order, 'AllRAD'))

        lm.plot_matrix(M_allrad_hf, title='AllRAD')

        print("\n\nDiffuse field gain of each loudspeaker (dB)")
        for n, g in zip(Sr.name.values, 10*np.log10(np.sum(M_allrad**2, axis=1))):
            print(f"{n}: {g:4.2f}")
    else:
        # let optmizer dream up a decoder on it's own
        M_allrad = None

    #M_allrad = None

    # Objective for E
    cap, *_ = sg.spherical_cap(T.u, (0, 0, 1), π/2-el_lim)
    W = np.array([1 if c else 0 for c in cap])

    M_opt, res = o(M_allrad, S_u, W, ambisonic_order,
                   iprint=50, tik_lambda=tik_lambda)
    figs.append(
        lm.plot_performance(M_opt, S_u, ambisonic_order, 'Optimized AllRAD'))

    lm.plot_matrix(M_opt, title='Optimized')

    print("ambisonic_order =", ambisonic_order)
    print("el_lim =", el_lim * 180/π)
    print("tik_lambda =", tik_lambda)

    off = np.isclose(np.sum(M_opt**2, axis=1), 0, rtol=1e-6)  # 60dB down
    print("Using:\n", Sr.name[~off.copy()].values)
    print("Turned off:\n", Sr.name[off.copy()].values)

    print("\n\nDiffuse field gain of each loudspeaker (dB)")
    for n, g in zip(Sr.name.values, 10*np.log10(np.sum(M_opt**2, axis=1))):
        print(f"{n}: {g:4.2f}")

    #print(figs)
    if do_report:
        reports.html_report(zip(*figs),
                            directory="Stage",
                            name=f"Stage-order-{ambisonic_order}")

    return M_opt, M_allrad, off, res


"""
# %% Try to use jax.scipy.optimize.minimize to keep everything in the GPU
#    sadly, this part of Jax apprears to be totally broken
# source code at site-packages/jax/scipy/optimize/_minimize.py,
#  _bfgs.py, _line_search.py

# this one is for jax.scipy.optimize, which has the args in a different order
#  x comes last, this is because it does
#   fun_with_args = partial(fun, *args)  which puts the args first!
# see https://docs.python.org/3.8/library/functools.html#functools.partial


import jax.scipy.optimize as jopt

def loss2(M_shape0, M_shape1, Su, Y, M):
    return loss(M, M_shape0, M_shape1, Su, Y )


def o2(M=None, Su=Su, Y_test=Y_test, iprint=50):
    if M is None:
        # infer M_shape from Su and Y
        M_shape = (Su.shape[1],     # number of loudspeakers
                   Y_test.shape[0], # number of program channels
                   )
        M = random.uniform(key, shape=M_shape, minval=-0.5, maxval=0.5)
    else:
        M_shape = M.shape

    x0 = M.ravel()  # optimize needs a vector

    result = jopt.minimize(fun=loss2, x0=x0,
                           args=(*M.shape, Su, Y_test),
                           method='BFGS',  #'L-BFGS-B',
                           #options=dict(disp=True)
                           )
    return result
"""

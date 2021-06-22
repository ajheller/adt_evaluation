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


# sandbox for a non-linear optimizer using Jax for AutoGrad
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
#  pip install --upgrade \
#     $BASE_URL/$CUDA_VERSION/jaxlib-0.1.52-$PYTHON_VERSION-none-$PLATFORM.whl
#  pip install --upgrade jax

# Jax Autograd
#  https://github.com/google/jax#automatic-differentiation-with-grad
#  https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
#  https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


import warnings

import autograd.numpy as np
from autograd import value_and_grad

import random
import numpy as onp  # 'original' numpy -- this is a convention
import pandas as pd
import scipy.optimize as opt
from matplotlib import pyplot as plt
from numpy import pi as π  # I get tired of typing np.pi

import basic_decoders as bd
import localization_models as lm
import program_channels as pc
import real_spherical_harmonics as rsh
import shelf
import spherical_grids as sg
from Timer import Timer

warnings.filterwarnings(action='once')


#  need a local definition so np is jax.np
def rE(M, Su, Y_test):
    """Return energy-model localization vector and energy."""
    G = M @ Y_test
    G2 = G * G  # FIXME: this should be G * G.conj
    E = np.sum(G2, axis=0)
    rExyz = (Su @ G2) / E
    return rExyz, E


def rV(M, Su, Y_test):
    """Return velocity-model localization vector and pressure."""
    G = M @ Y_test
    P = np.sum(G, axis=0)
    rVxyz = (Su @ G) / P
    return rVxyz, P


def xyz2ur(xyz):
    """Transform cartesian vector to unit vector and magnitude."""
    r = np.linalg.norm(xyz, ord=2, axis=0)
    u = xyz / r
    return u, r


# the test directions
T = sg.t_design5200()


# %%

# define a callback for use with opt.minimize
#  Calling this seems to screw up the convergence !?!
# ii = 0
# def callback(x):
#     """Print progress of optimizer."""
#     global ii
#     if ii == 0:
#         print("Running optimizer")
#     ii += 1
#     if ii % 50 == 0:
#         print(".", end="")
#     if ii % 500 == 0:
#         print(ii)


def optimize(M, Su, sh_l, sh_m,
             E_goal=None,
             iprint=50,
             tikhonov_lambda=1.0e-3,
             sparseness_penalty=1,
             uniform_loudness_penalty=0.1,  # 0.01
             rE_goal=1.0,
             rE_W=1.0,
             maxcor=100,  # how accurate is the Hessian, more is better but slower
             raise_error_on_failure=True):
    """Optimize psychoacoustic criteria."""
    #
    # handle defaults
    if E_goal is None:
        E_goal = 1

    if rE_goal == 'auto' or rE_goal is None:
        # FIXME: This assumes 3D arrays
        rE_goal = shelf.max_rE_3d(np.max(sh_l) + 2)

    print(f"rE_goal min={np.min(rE_goal)} max={np.max(rE_goal)}")

    # the test directions
    T = sg.t_design5200()
    Y_test = rsh.real_sph_harm_transform(sh_l, sh_m, T.az, T.el)

    rExyz_goal = T.u.T * rE_goal

    if M is None:
        # infer M_shape from Su and Y
        M_shape = (Su.shape[1],  # number of loudspeakers
                   Y_test.shape[0],  # number of program channels
                   )
        M = random.uniform(key, shape=M_shape, minval=-1.0, maxval=1.0)
    else:
        M_shape = M.shape

    # the loss function
    def o(x) -> float:
        M = x.reshape(M_shape)
        rExyz, E = rE(M, Su, Y_test)

        # truncation loss due to finite order
        truncation_loss = np.sum(rE_W * ((rExyz - rExyz_goal) ** 2))

        # uniform loudness loss
        uniform_loudness_loss = (np.sum((E - E_goal) ** 2) *
                                 uniform_loudness_penalty)  # was 10

        # Tikhonov regularization term - typical value = 1e-3
        tikhonov_regularization_term = np.sum(M ** 2) * tikhonov_lambda

        # don't turn off speakers
        # pull diffuse-field gain for each speaker away from zero
        sparsness_term = (np.sum(np.abs(1 - np.sum(M**2, axis=1))) *
                          100 * sparseness_penalty)

        # the entire loss function
        f = (truncation_loss + uniform_loudness_loss +
             tikhonov_regularization_term + sparsness_term)
        return f

    # consult the automatic differentiation oracle
    val_and_grad_fn = value_and_grad(o)
    
    def objective_and_gradient(x, *args):
        v, g = val_and_grad_fn(x, *args)
        return v, g

    x0 = M.ravel()  # initial guess
    with Timer() as t:
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
        res = opt.minimize(objective_and_gradient, x0,
                           #bounds=opt.Bounds(-1, 1),
                           method='L-BFGS-B',
                           jac=True,
                           options=dict(
                               maxcor=maxcor,
                               disp=iprint,
                               maxls=500,
                               # maxcor=30,
                               gtol=1e-8,
                               # ftol=1e-12
                               ),
                           # callback=callback,
                           )
    if True:
        print()
        print(f"Execution time: {t.interval:0.3f} sec.")
        print(res.message)
        print(res)
        print()

    if res.status != 0 and raise_error_on_failure:
        print('bummer:', res.message)
        raise RuntimeError(res.message)

    M_opt = res.x.reshape(M_shape)
    return M_opt, res


def optimize_LF(M, Su, sh_l, sh_m, W=1,
                raise_error_on_failure=False):

    M_shape = M.shape
    g_spkr, g_total = lm.diffuse_field_gain(M)

    # the test directions
    T = sg.t_design5200()
    cap = sg.spherical_cap(T.u, (0, 0, 1), 5*np.pi/6)[0]
    W = np.where(cap, 1, 1)

    Y_test = rsh.real_sph_harm_transform(sh_l, sh_m, T.az, T.el)
    rExyz, E = rE(M, Su, Y_test)
    rEu, rEr = xyz2ur(rExyz)

    # define the loss function
    def o(x):
        M = x.reshape(M_shape)
        rVxyz, P = rV(M, Su, Y_test)

        df_gain = np.sum(M*M)

        df_gain_loss = (g_total - df_gain)**2

        # Tikhonov regularization term - typical value = 1e-3
        tikhonov_regularization_term = np.sum(M ** 2) * 1e-2  # tikhonov_lambda

        # dir loss mag(rVxyz) should be 1
        direction_loss = np.sum(W * ((rVxyz - rEu) ** 2))
        P_loss = np.sum(W * ((P - 1)**2))
        return (direction_loss +
                df_gain_loss +
                P_loss/100000 +
                tikhonov_regularization_term
                )

    val_and_grad_fn = value_and_grad(o)


    def objective_and_gradient(x):
        v, g = val_and_grad_fn(x)
        return v, g

    x0 = M.ravel()
    with Timer() as t:
        res = opt.minimize(
            objective_and_gradient, x0,
            #bounds=opt.Bounds(-1, 1),
            method='L-BFGS-B',
            jac=True,
            options=dict(
                 disp=50,
                 # maxls=50,
                 # maxcor=30,
                 # gtol=1e-8,
                 # ftol=1e-12
                 ),
            # callback=callback,
            )
    if True:
        print()
        print(f"Execution time: {t.interval:0.3f} sec.")
        print(res.message)
        print(res)
        print()

    if res.status != 0 and raise_error_on_failure:
        print('bummer:', res.message)
        raise RuntimeError(res.message)

    M_opt = res.x.reshape(M_shape)
    return M_opt, res


def test_optimize_LF(M, C=3):
    import example_speaker_arrays as esa
    h_order, v_order, sh_l, sh_m = pc.ambisonic_channels(C)
    S = esa.nando_dome(False)

    M_opt, ret = optimize_LF(M, S.u.T, sh_l, sh_m)

    return M_opt, ret


def unit_test(C):
    """Run unit test for the optimizer with uniform array."""
    #
    #sh_l, sh_m = zip(*rsh.lm_generator(ambisonic_order))
    h_order, v_order, sh_l, sh_m = pc.ambisonic_channels(C)
    # make a  decoder matrix for the 240 speaker t-design via pseudoinverse
    S240 = sg.t_design240()
    Su = S240.u

    # shelf filter gains for max_rE
    gamma = shelf.gamma(sh_l, decoder_type='max_rE', decoder_3d=True,
                        return_matrix=True)

    # since this is a spherical design, all three methods should yield the
    # same result

    # 1 - inversion
    M240 = bd.inversion(sh_l, sh_m, S240.az, S240.el)

    M240_hf = M240 @ gamma

    lm.plot_performance(M240_hf, Su, sh_l, sh_m,
                        title='Pinv unit test')
    lm.plot_matrix(M240_hf, title='Pinv unit test')

    # 2 - AllRAD
    M240_allrad = bd.allrad(sh_l, sh_m, S240.az, S240.el)
    M240_allrad_hf = M240_allrad @ gamma
    lm.plot_performance(M240_allrad_hf, Su, sh_l, sh_m,
                        title='AllRAD unit test')
    lm.plot_matrix(M240_allrad_hf, title='AllRAD unit test')

    # 3 - NLOpt
    M_opt, res = optimize(None, Su, sh_l, sh_m, E_goal=1, sparseness_penalty=0)
    lm.plot_performance(M_opt, Su, sh_l, sh_m,
                        title='Optimized unit test')
    lm.plot_matrix(M240_allrad, title='Optimized unit test')
    return res


def plot_rE_vs_ambisonic_order():
    """Plot magnitude of rE for uniform loudspeaker arrays."""
    #
    rE_range = np.linspace(0.5, 1, 100)
    plt.plot(rE_range, shelf.rE_to_ambisonic_order_3d(rE_range), label='3D')
    plt.plot(rE_range, shelf.rE_to_ambisonic_order_2d(rE_range), label='2D')
    plt.scatter([shelf.max_rE_3d(o) for o in range(1, 10)], range(1, 10))
    plt.scatter([shelf.max_rE_2d(o) for o in range(1, 10)], range(1, 10))
    plt.grid(True)
    plt.xlabel("Magnitude of rE")
    plt.ylabel("Ambisonic Order")
    plt.legend()
    plt.ylim(0, 10)


def table_ambisonics_order_vs_rE(max_order=20):
    """Return a dataframe with rE as a function of order."""
    order = np.arange(1, max_order+1, dtype=np.int32)
    rE3 = np.array(list(map(shelf.max_rE_3d, order)))
    drE3 = np.append(np.nan, rE3[1:] - rE3[:-1])

    rE2 = np.array(list(map(shelf.max_rE_2d, order)))
    drE2 = np.append(np.nan, rE2[1:] - rE2[:-1])

    df = pd.DataFrame(
        np.column_stack((order,
                         rE2, 100*drE2/rE2, 2*np.arccos(rE2)*180/π,
                         rE3, 100*drE3/rE3, 2*np.arccos(rE3)*180/π,)),
        columns=('order',
                 '2D', '% change', 'asw',
                 '3D', '% change', 'asw'))
    return df


#
#
"""
# %% Try to use jax.scipy.optimize.minimize to keep everything in the GPU
#    sadly, this part of Jax appears to be totally broken
# source code at site-packages/jax/scipy/optimize/_minimize.py,
#  _bfgs.py, _line_search.py

# this one is for jax.scipy.optimize, which has the args in a different order
#  x comes last, this is because it does
#   fun_with_args = partial(fun, *args)  which puts the args first!
# see https://docs.python.org/3.8/library/functools.html#functools.partial


import jax.scipy.optimize as jopt

def loss2(M_shape0, M_shape1, Su, Y, M):
    return loss(M, M_shape0, M_shape1, Su, Y)


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

if __name__ == '__main__':
    # unit_test()
    try:
        for d in range(7, 8):
            unit_test(d)
    except KeyboardInterrupt:
        print('Bye')

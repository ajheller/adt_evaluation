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

import io

import jax
import jax.numpy as np  # jax overloads numpy
import jax.random as random
import numpy as onp  # 'original' numpy -- this is a convention
import pandas as pd
import scipy.optimize as opt
from matplotlib import pyplot as plt
from numpy import pi as π  # I get tired of typing np.pi

import basic_decoders as bd
import localization_models as lm
import real_spherical_harmonics as rsh
import reports
import shelf
import spherical_grids as sg
import program_channels as pc
from Timer import Timer

import warnings
warnings.filterwarnings(action='once')

#from ArrayHash import hashable


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


# scipy.optimize needs 64-bit
jax.config.update("jax_enable_x64", True)

# Generate key which is used by JAX to generate random numbers
key = random.PRNGKey(1)

# the test directions
T = sg.t_design5200()


# directions for plotting
# T_azel = sg.az_el()
# Y_azel = rsh.real_sph_harm_transform(l, m, T_azel.az, T_azel.el)


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


def objective(x,
              # remainder of arguments are static
              M_shape: tuple,
              Su: np.array,
              Y_test: np.array,
              W: float,
              tikhanov_lambda: float,
              sparseness_penalty: float,
              rE_goal: float,
              ) -> float:
    """Return value of objective function for a given decoder matrix."""
    M = x.reshape(M_shape)
    rExyz, E = rE(M, Su, Y_test)

    # truncation loss due to finite order
    tl = np.sum((rExyz - T.u * rE_goal) ** 2)

    # uniform loudness loss
    ulu = np.sum((E - W) ** 2) / 10

    # Tikhanov regularization term
    trt = np.sum(M ** 2) * tikhanov_lambda

    # don't turn off speakers
    sp = np.sum(np.abs(1 - np.sum(M**2, axis=1))) * 100 * sparseness_penalty
    # + np.sum(np.abs(M - 0.1)) * sparseness_penalty
    f = tl + ulu + trt + sp
    return f


def optimize(M, Su, sh_l, sh_m,
             W=None,
             iprint=50,
             tikhanov_lambda=1.0e-3,
             sparseness_penalty=1,
             rE_goal=1.0,
             raise_error_on_failure=True):
    """Optimize psychoacoustic criteria."""
    #
    # handle defaults
    if W is None:
        W = 1

    if rE_goal == 'auto' or rE_goal is None:
        rE_goal = shelf.max_rE_3d(np.max(sh_l) + 2)

    # the test directions
    T = sg.t_design5200()
    Y_test = rsh.real_sph_harm_transform(sh_l, sh_m, T.az, T.el)

    if M is None:
        # infer M_shape from Su and Y
        M_shape = (Su.shape[1],  # number of loudspeakers
                   Y_test.shape[0],  # number of program channels
                   )
        M = random.uniform(key, shape=M_shape, minval=-0.25, maxval=0.25)
    else:
        M_shape = M.shape

    # Need to define these here so JAX's jit recompiles it for each run
    val_and_grad_fn = jax.jit(jax.value_and_grad(objective),
                              static_argnums=range(1, 7))

    def objective_grad(x, *args):
        v, g = val_and_grad_fn(x, *args)
        # I'm not to happy about having to copy g but L-BGFS needs it in
        # fortran order.  Check with g.flags
        return v, onp.array(g, order='F')  # onp.asfortranarray(g)


    # FIXME: make Su and Y_test hashable -- doesn't work...
    # Su.flags.writeable = False
    # Y_test.flags.writeable = False

    Su_h = Su #hashable(Su)
    Y_test_h = Y_test #hashable(Y_test)

    x0 = M.ravel()  # initial guess

    with Timer() as t:
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
        res = opt.minimize(
            objective_grad, x0,
            bounds=opt.Bounds(-1, 1),
            args=(M_shape, Su_h, Y_test_h, W,
                  tikhanov_lambda, sparseness_penalty,
                  rE_goal),
            method='L-BFGS-B',
            jac=True,
            options=dict(
                disp=iprint,
                # maxls=50,
                # maxcor=30,
                gtol=1e-8,
                # ftol=1e-12
                ),
            # callback=callback,
           )
    if True:
        print()
        print("Execution time: %0.3f" % t.interval)
        print(res.message)
        print()

    if res.status != 0 and raise_error_on_failure:
        print('bummer:', res.message)
        raise RuntimeError(res.message)

    M_opt = res.x.reshape(M_shape)
    return M_opt, res


def unit_test(C):
    """Run unit test for the optimizer with uniform array."""
    #
    #sh_l, sh_m = zip(*rsh.lm_generator(ambisonic_order))
    h_order, v_order, sh_l, sh_m = pc.olm(C)
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

    lm.plot_performance(M240_hf, Su, sh_l, sh_m, 'Pinv unit test')
    lm.plot_matrix(M240_hf, title='Pinv unit test')

    # 2 - AllRAD
    M240_allrad = bd.allrad(sh_l, sh_m, S240.az, S240.el)
    M240_allrad_hf = M240_allrad @ gamma
    lm.plot_performance(M240_allrad_hf, Su, sh_l, sh_m, 'AllRAD unit test')
    lm.plot_matrix(M240_allrad_hf, title='AllRAD unit test')

    # 3 - NLOpt
    M_opt, res = optimize(None, Su, sh_l, sh_m, W=1, sparseness_penalty=0)
    lm.plot_performance(M_opt, Su, sh_l, sh_m, 'Optimized unit test')
    lm.plot_matrix(M240_allrad, title='Optimized unit test')
    return res


# TODO: define a class for the speaker array,
#       still use pandas to read the csv files
def stage(path='stage.csv'):
    """Load CCRMA Stage loudspeaker array."""
    #
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


def emb(z_low=-0.2, z_high=1):
    """Return geometry of EMB's loudspeaker array.

    ITU with center with elevated, additional height at left, right, back
    4 meters wide
    As deep as we want
    Ear-level stands are 1.1 meters, acoustic center speaker is .1 meters
    higher

    Ceiling 2.44 meters, acoustic center .2 lower

    """
    #
    S = pd.DataFrame(columns=['name', 'az', 'el', 'r', 'x', 'y', 'z', 'Real'])
    az_deg = np.array([30, 120, -120, -30, 0, 90, 180, -90, 0, 0])
    z = np.array([z_low] * 4 + [z_high] * 4 + [2] + [-2])
    r = np.array([2] * 8 + [0] * 2)

    S.name = ['L', 'LS', 'RS', 'R', 'CU', 'LU', 'BU', 'RU', '*IZ', '*IN']
    S.x, S.y, *_ = \
        sg.sph2cart(az_deg / 180 * π, 0, r)
    S.z = z
    S.az, S.el, S.r = sg.cart2sph(S.x, S.y, S.z)

    S['Real'] = ['*' not in n for n in S.name]

    return S


def csv2spk(path='stage2.csv'):
    """TODO; Load a loudspeaker array form a CSV file. Work in progress."""
    hf = pd.read_table(path, delimiter=',', nrows=1, comment='#')
    units = hf.iloc[0].value
    df = pd.read_table(path,
                       header=1, names=hf.columns,
                       delimiter=',', index_col=False,
                       comment='#', skip_blank_lines=True)
    # remove empty rows
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return hf, df



def stage_test(ambisonic_order=3,
               el_lim=-π / 8,
               tikhanov_lambda=0,  # 1e-3,
               sparseness_penalty=1,
               do_report=False,
               rE_goal=1.1  # 'auto'
               ):
    """Test optimizer with CCRMA Stage array."""
    #
    #
    order_h, order_v, sh_l, sh_m = pc.olm(ambisonic_order)
    #order = max(order_h, order_v)  # FIXME
    is_3D = order_v > 0

    if True:
        S = stage()
        spkr_array_name = 'Stage'
    else:
        # hack to enter Eric's array
        S = emb()
        spkr_array_name = 'EMB'

    S_u = np.array(sg.sph2cart(S.az, S.el, 1))

    gamma = shelf.gamma(sh_l, decoder_type='max_rE', decoder_3d=is_3D,
                        return_matrix=True)

    figs = []
    if True:
        M_start = 'AllRAD'

        M_allrad = bd.allrad(sh_l, sh_m, S.az, S.el)

        # remove imaginary speaker
        # FIXME: this is too messy, need a better way to handle imaginary LSs
        M_allrad = M_allrad[S.Real, :]
        S_u = S_u[:, S.Real.values]
        Sr = S[S.Real.values]
        M_allrad_hf = M_allrad @ gamma

        # performance plots
        plot_title = f"AllRAD, Ambisonic order={order_h}H{order_v}V"
        figs.append(
            lm.plot_performance(M_allrad_hf, S_u, sh_l, sh_m,
                                title=plot_title))

        lm.plot_matrix(M_allrad_hf, title=plot_title)

        print(f"\n\n{plot_title}\nDiffuse field gain of each loudspeaker (dB)")
        for n, g in zip(Sr.name.values,
                        10 * np.log10(np.sum(M_allrad ** 2, axis=1))):
            print(f"{n}: {g:6.2f}")

    else:
        M_start = 'Random'
        # let optimizer dream up a decoder on its own
        M_allrad = None
        S_u = S_u[:, S.Real.values]
        Sr = S[S.Real]

    # M_allrad = None

    # Objective for E
    cap, *_ = sg.spherical_cap(T.u, (0, 0, 1), π / 2 - el_lim)
    E0 = np.array([0.1, 1.0])[cap.astype(np.int8)]
    # objective for rE order+2 inside the cap, order-2 outside
    rE_goal = np.array([shelf.max_rE_3d(max(order-2, 1)),
                        shelf.max_rE_3d(order+2)])[cap.astype(np.int8)]

    M_opt, res = optimize(M_allrad, S_u, sh_l, sh_m, W=E0,
                          iprint=50, tikhanov_lambda=tikhanov_lambda,
                          sparseness_penalty=sparseness_penalty,
                          rE_goal=rE_goal)

    plot_title = f'Optimized {M_start}, Ambisonic order={order_h}H{order_v}V'
    figs.append(
        lm.plot_performance(M_opt, S_u, sh_l, sh_m,
                            title=plot_title
                            ))

    lm.plot_matrix(M_opt, title=plot_title)

    with io.StringIO() as f:
        print(f"ambisonic_order = {order}\n" +
              f"el_lim = {el_lim * 180 / π}\n" +
              f"tikhanov_lambda = {tikhanov_lambda}\n" +
              f"sparseness_penalty = {sparseness_penalty}\n",
              file=f)

        off = np.isclose(np.sum(M_opt ** 2, axis=1), 0, rtol=1e-6)  # 60dB down
        print("Using:\n", Sr.name[~off.copy()].values, file=f)
        print("Turned off:\n", Sr.name[off.copy()].values, file=f)

        print("\n\nDiffuse field gain of each loudspeaker (dB)", file=f)
        for n, g in zip(Sr.name.values,
                        10 * np.log10(np.sum(M_opt ** 2, axis=1))):
            print(f"{n:3}:{g:8.2f} |{'=' * int(60 + g)}", file=f)
        report = f.getvalue()
        print(report)

    if do_report:
        reports.html_report(zip(*figs),
                            text=report,
                            directory=spkr_array_name,
                            name=f"{spkr_array_name}-order-{order}")

    return M_opt, dict(M_allrad=M_allrad, off=off, res=res)


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
    for d in range(1, 8):
        stage_test(d, do_report=True)

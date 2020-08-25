#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 02:48:26 2019

@author: heller
"""

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

import scipy.optimize as opt

import spherical_grids as sg
import real_spherical_harmonics as rsh
import basic_decoders as bd
import localization_models as lm

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
#Y_test = rsh.real_sph_harm_transform(l, m, T.az, T.el)

# directtions for plotting
T_azel = sg.az_el()
Y_azel = rsh.real_sph_harm_transform(l, m, T_azel.az, T_azel.el)




# %%
cap = sg.spherical_cap(T.u, (0, 0, 1), np.pi/2+np.pi/8)[0]
W = np.array([1 if c else 0 for c in cap])

# %%
def loss(M, M_shape0, M_shape1, Su, Y_test, W):
    rExyz, E = rE(M.reshape((M_shape0, M_shape1)), Su, Y_test)
    return (np.sum((rExyz - T.u * 1.1)**2)
            + np.sum((E - W)**2)/10
            + np.sum(M**2)/1000  # regularization term
            )


val_and_grad_fn = jax.jit(jax.value_and_grad(loss), static_argnums=range(1, 6))


def loss_grad(M, M_shape0, M_shape1, Su, Y_test, W):
    v, g = val_and_grad_fn(M, M_shape0, M_shape1, Su, Y_test, W)
    # I'm not to happy about having to copy g but L-BGFS needs it in fortran
    # order.  Check with g.flags
    return v.copy(), onp.array(g, order='F')  # onp.asfortranarray(g)


# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
def o(M, Su, W=None, ambisonic_order=3, iprint=50, plot=False):

    if W == None:
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
        M = random.uniform(key, shape=M_shape, minval=-0.5, maxval=0.5)
    else:
        M_shape = M.shape

    x0 = M.ravel()  # inital guess

    res = opt.minimize(loss_grad, x0,
                       args=(*M_shape, Su, Y_test, W),
                       method='L-BFGS-B',
                       jac=True,
                       options=dict(disp=iprint, gtol=1e-8, ftol=1e-12)
                       )
    if res.status == 0:
        M_opt = res.x.reshape(M_shape)

        if plot:
            lm.plot_performance(M_opt, Su, ambisonic_order)

    else:
        print('bummer:', res.message)

    return M_opt, res


def unit_test(ambisonic_order=3):
    # make a  decoder matrix for the 240 speaker t-design via pseudoinverse
    S240 = sg.t_design240()
    Su = S240.u
    M240 = bd.inversion(l, m, S240.az, S240.el)
    M240_shape = M240.shape

    M240_allrad = bd.allrad(l, m, S240.az, S240.el)

    M_opt, res = o(None, Su, 1, ambisonic_order)
    lm.plot_performance(M_opt, Su, ambisonic_order, 'Optimized unit test')
    return


def stage_test(ambisonic_order=3):
    l, m = zip(*rsh.lm_generator(ambisonic_order))

    df = pd.read_csv('stage.csv')
    S_az = df["Azimuth:Degrees"] / 180 * np.pi
    S_el = df["Elevation:Degrees"] / 180 * np.pi
    S_r = df["Radius:Inches"] * 2.54 / 100
    S_u = np.vstack(sg.sph2cart(S_az, S_el))

    if True:
        # make an AllRAD decoder and plot its performances
        M_allrad = bd.allrad(l, m, S_az, S_el)
        lm.plot_performance(M_allrad, S_u, ambisonic_order, 'AllRAD')
        lm.plot_matrix(M_allrad, title='AllRAD')
    else:
        M_allrad = None

    M_opt, res = o(M_allrad, S_u, W, ambisonic_order)
    lm.plot_performance(M_opt, S_u, ambisonic_order, 'Optimized AllRAD')

    lm.plot_matrix(M_opt, title='Optimized')

    off = np.isclose(np.sum(M_opt**2, axis=1), 0, rtol=1e-6) # 60dB down
    print("Using:\n", df["Name:Stage"][~off.copy()].values)
    print("Turned off:\n", df["Name:Stage"][off.copy()].values)

    return M_opt, M_allrad, off


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

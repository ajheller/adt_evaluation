#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 02:48:26 2019

@author: heller
"""

# sanbox for optimizer using Jax for AutoGrad
#   https://github.com/google/jax

#  conda create -n jax --clone base

# jax installation:
#   https://github.com/google/jax#installation

# install jax with
#   pip install jaxlib==0.1.51  # 0.1.52 won't run on MacOS 10.13
#   pip install jax

# Jax Autograd
#  https://github.com/google/jax#automatic-differentiation-with-grad
#  https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
#  https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation
# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)


import jax
import jax.numpy as np  # jax overloads numpy
import numpy as onp

import scipy.optimize as opt

import spherical_grids as sg
import real_spherical_harmonics as rsh
import basic_decoders as bd

import localization_models as locm



# 3rd-order decoding
l, m = zip(*rsh.lm_generator(3))


# the test directions
T = sg.t_design5200()
Y = rsh.real_sph_harm_transform(l, m, T.az, T.el)

# make a decoder matrix for the 240 speaker t-design
S240 = sg.t_design240()
Su = S240.u
M = bd.inversion(l, m, S240.az, S240.el)
M_shape = M.shape


#@jax.jit(static_argnums=(1,2,3))
def rE(M, Su, Y, full=False):
    #G = M.reshape(M_shape) @ Y
    G = M @ Y
    G2 = G * G
    E = np.sum(G2, axis=0)
    rExyz = (Su @ G2) / E
    if full:
        rEr = np.linalg.norm(rExyz, ord=2, axis=0)
        rEu = rExyz / rEr
        return rExyz, rEu, rEr
    else:
        return rExyz

def loss(M, args):
    return np.sum( (rE(M.reshape(args), Su, Y) - T.u * 0.75)**2 ) \
            + np.sum(M**2)/1000  # regularization term

val_and_grad_fn = jax.value_and_grad(loss)


def loss_grad(M, args):
    v, g = val_and_grad_fn(M, args)
    # I'm not to happy about having to copy g but L-BGFS needs it in fortran
    # order.  Check with g.flags
    return v.copy(), onp.array(g, order='F') #onp.asfortranarray(g)



def rms_dir_error(M):
    #print(M.shape)
    G = M.reshape(M_shape) @ Y
    G2 = G * G
    E = np.sum(G2, axis=0)
    rExyz = (Su @ G2) / E

    # magnitude and direction of rE
    rEr = np.sqrt(np.sum(rExyz * rExyz, axis=0))
    rEu = rExyz/rEr

    # the direction error vector
    rE_err_xyz = T.u - rEu
    rE_err_mag = np.sum((rE_err_xyz * rE_err_xyz).ravel())/T.shape[0]


    return rE_err_mag

def gv(M):
    v, g = jax.value_and_grad(rms_dir_error)(M)
    return onp.asarray(v), onp.array(g, order='F')

# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

def o(M):
    x, f, d = opt.fmin_l_bfgs_b(loss_grad, M.ravel(),
                                args=(M.shape,),
                                fprime=None,
                                iprint=50)
    M_opt = x.reshape(M.shape)

    r = rE(M_opt, Su, Y, full=true)

    return M_opt, f, d, r

def o2(M):
    return opt.fmin_l_bfgs_b(rms_dir_error, M, approx_grad=True, iprint=1)





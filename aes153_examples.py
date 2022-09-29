#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:38:05 2022

@author: heller
"""

from pathlib import Path

from numpy import pi as π

import example_speaker_arrays as esa
import optimize_dome as od
import program_channels as pc

import numpy as np

# %% globals

_here = Path(__file__).parent

el_lim = np.array((-60, 60)) * np.pi / 180
eval_el_lim = np.array((-89, 89)) * np.pi / 180
quiet = False

# if __name__ == "__main__":
#     import matplotlib

#     matplotlib.use("Agg")

# %%
S = esa.emb_cmap484()

for vo in (1, 2, 3):
    res = od.optimize_dome(
        S,
        ambisonic_order=pc.ChannelsAmbiX(3, vo),
        # eval_order=C,
        sparseness_penalty=0,
        el_lim=el_lim,
        do_report=True,
        random_start=False,
        eval_el_lim=eval_el_lim,
        quiet=quiet,
    )

# %%
S = esa.emb_cmap888()

res = od.optimize_dome(
    S,
    ambisonic_order=pc.ChannelsAmbiX(3, 3),
    # eval_order=C,
    sparseness_penalty=0,
    el_lim=el_lim,
    do_report=True,
    random_start=False,
    eval_el_lim=eval_el_lim,
    quiet=quiet,
)

# %%

S = esa.emb_cmap686(bottom_center=False)

res = od.optimize_dome(
    S,
    ambisonic_order=pc.ChannelsAmbiX(3, 3),
    # eval_order=C,
    sparseness_penalty=0,
    el_lim=el_lim,
    do_report=True,
    random_start=False,
    eval_el_lim=eval_el_lim,
    quiet=quiet,
)

# %%

S = esa.emb_cmap686(bottom_center=True)

res = od.optimize_dome(
    S,
    ambisonic_order=pc.ChannelsAmbiX(3, 3),
    # eval_order=C,
    sparseness_penalty=0,
    el_lim=el_lim,
    do_report=True,
    random_start=False,
    eval_el_lim=eval_el_lim,
    quiet=quiet,
)

# %%

S = esa.amb_10_8_4()

res = od.optimize_dome(
    S,
    ambisonic_order=pc.ChannelsAmbiX(4, 4),
    # eval_order=C,
    sparseness_penalty=0,
    el_lim=el_lim,
    do_report=True,
    random_start=False,
    eval_el_lim=eval_el_lim,
    quiet=quiet,
)


# %%

res = od.optimize_dome(
    S,
    ambisonic_order=pc.ChannelsAmbiX(3, 3),
    # eval_order=C,
    sparseness_penalty=0,
    el_lim=el_lim,
    do_report=True,
    random_start=False,
    eval_el_lim=eval_el_lim,
    quiet=quiet,
)

# %% rsync to dreamhosters
import subprocess as sp

sp.run(f"cd {_here/'reports'}; zip -9r reports-cmap.zip cmap*", shell=True)
print(
    sp.run(
        "rsync -avurPz --delete-after ./reports ajh-dh:ambisonics/AES153", shell=True
    )
)

# %%
from matplotlib import pyplot as plt
import shelf
from scipy import special as spec


def erics_plot():
    x = np.linspace(0.5, 1, 100)
    y = shelf.rE_to_ambisonic_order_3d(x)
    plt.plot(x, np.round(y))
    plt.plot(x, y)
    plt.ylim((0, 10))
    plt.xlabel("magnitude of $r_E$")
    plt.ylabel("3D Ambisonic order")
    plt.grid()
    plt.show()


def plot_legendre(l_max):
    for l in range(l_max + 1):
        x = np.linspace(-1, 1, 100)
        y = spec.eval_legendre(l, x)
        plt.plot(x, y)
    plt.show()


def shelf_calc(l_max):
    for l in range(l_max + 1):
        roots, weights = spec.roots_legendre(l + 1)
        # print(roots, weights)
        max_root = np.max(roots)
        g = spec.eval_legendre(range(l + 1), max_root)
        print(l, max_root, g)

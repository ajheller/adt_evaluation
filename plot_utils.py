#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 13:39:10 2020

@author: heller
"""
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as π


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def axis_demo():
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # ax.set_aspect('equal')

    X = np.random.rand(100) * 10 + 5
    Y = np.random.rand(100) * 5 + 2.5
    Z = np.random.rand(100) * 50 + 25

    scat = ax.scatter(X, Y, Z)

    set_axes_equal(ax)
    plt.show()


def plot_lsl(S, speaker_stands=True, title=None, axis=None, show=True):
    if axis is None:
        fig, axis = plt.subplots(1, 1, figsize=(10, 10))
    # ax = fig.add_subplot(1, 1, 1)
    # ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')

    ax = plt.axes(projection="3d")

    ax.scatter(*S.xyz.T, c=S.z, s=200)
    ax.scatter(0, 0, 0, marker="d")

    z_floor = np.min(S.z) - np.abs(np.min(S.z) / 10)
    for x, y, z, id in zip(*S.xyz.T, S.ids):
        ax.text(x, y, z, id, zorder=1000, weight="bold", size=14)
        plt.plot([x, x], [y, y], [z, z_floor], "-.k")
        plt.plot([0, x], [0, y], [z_floor, z_floor], "-.k")

    set_axes_equal(ax)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")

    ax.text(
        np.max(S.x)*1.1,
        0,
        0,
        "\n".join("FRONT"),
        fontstyle="italic",
        bbox=dict(facecolor="lightblue", alpha=0.4),
        horizontalalignment="center",
        verticalalignment="center",
    )

    if title is not None:
        ax.set(title=title)
    if show:
        plt.show()
    return fig


def plot_lsl_plan(S, title=None, show=False, axis=None, **kwargs):
    if axis is None:
        fig, axis = plt.subplots(1, 1, **kwargs)
    else:
        fig = axis.figure

    axis.scatter(S.x, S.y, c=S.z, marker="o")
    axis.axis("equal")
    axis.grid()
    # plt.colorbar()
    # plt.title(title)
    if show:
        fig.show()
    return fig


def plot_lsl_azel(S, title=None, show=False, axis=None, **kwargs):
    if axis is None:
        fig, axis = plt.subplots(1, 1, **kwargs)
    else:
        fig = axis.figure
    # fig = plt.figure(figsize=(12, 6))

    if True:
        for x, y, r, t in zip(S.az * 180 / π, S.el * 180 / π, S.r, S.ids):
            t += "\n" f"r={r:0.2f}"
            axis.text(
                x,
                y,
                t,
                bbox=dict(facecolor="lightblue", alpha=0.4),
                horizontalalignment="center",
                verticalalignment="center",
            )
    if True:
        axis.scatter(S.az * 180 / π, S.el * 180 / π, c="white", marker="+")

    axis.set_xlim(-180, 180)
    axis.set_ylim(-90, 90)
    axis.set_xlabel("Azimuth [degrees]")
    axis.set_ylabel("Elevation [degrees]")
    axis.set_title(title)
    axis.grid(True)
    # invert x-axis to show from listeners point of view
    axis.invert_xaxis()
    # plt.colorbar()
    if show:
        fig.show()
    return fig


"""
======================
Text annotations in 3D
======================

Demonstrates the placement of text annotations on a 3D plot.

Functionality shown:
- Using the text function with three types of 'zdir' values: None,
  an axis name (ex. 'x'), or a direction tuple (ex. (1, 1, 0)).
- Using the text function with the color keyword.
- Using the text2D function to place text on a fixed position on the ax object.
"""


# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
def text_annotations_3d():

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Demo 1: zdir
    zdirs = (None, "x", "y", "z", (1, 1, 0), (1, 1, 1))
    xs = (1, 4, 4, 9, 4, 1)
    ys = (2, 5, 8, 10, 1, 2)
    zs = (10, 3, 8, 9, 1, 8)

    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        label = "(%d, %d, %d), dir=%s" % (x, y, z, zdir)
        ax.text(x, y, z, label, zdir)

    # Demo 2: color
    ax.text(9, 0, 0, "red", color="red")

    # Demo 3: text2D
    # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.
    ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)

    # Tweaking display region and labels
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    plt.show()

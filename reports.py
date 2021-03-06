#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:28:07 2020

@author: heller
"""

# dominate is not avaiable via Conda, install it with:
#    pip install dominate

import os
try:
    import dominate
except ModuleNotFoundError as ie:
    print("run 'pip install dominate' for reports")
    dominate = False
else:
    from dominate.tags import div, table, tbody, tr, td, img
    from dominate.tags import html, body, h1, pre


# TODO: rework using pathlib
def html_report(figs, text=None, name='report', directory=None,
                dpi=75, fig_dir='figs'):
    """Produce an HTML report containing figs and text."""
    #
    # if dominate not installed, dive out here
    if not dominate:
        return

    if directory is None:
        directory = name
    try:
        os.mkdir(directory)
    except FileExistsError as e:
        print(e)

    try:
        os.mkdir(os.path.join(directory, fig_dir))
    except FileExistsError as e:
        print(e)

    h = html()
    with h.add(body()).add(div(id='content')):
        h1(f'Performance Plots: {name}')
        if text:
            pre(text)
        with table().add(tbody()):
            for j, row in enumerate(figs):
                r = tr()
                for i, item in enumerate(row):
                    url = os.path.join(fig_dir, f"{name}-fig-{j}_{i}.png")
                    item.savefig(os.path.join(directory, url),
                                 dpi=dpi)
                    r += td(img(src=url))

    with open(os.path.join(directory, name + '.html'), 'w') as f:
        print(h, file=f)
    return h

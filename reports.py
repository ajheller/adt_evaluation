#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:28:07 2020

@author: heller
"""

# dominate is not avaiable via Conda, install it with:
#    pip install dominate

import os
from pathlib import Path
try:
    import dominate
except ModuleNotFoundError as ie:
    print("run 'pip install dominate' for reports")
    dominate = False
else:
    from dominate.tags import div, table, tbody, tr, td, img
    from dominate.tags import html, body, h1, pre

try:
    import slugify
except ModuleNotFoundError as ie:
    print("run 'pip install python-slugify' for reports")
    slugify = False
else:
    from slugify import slugify

_here = Path(__file__).parent
_report_dir = _here/'reports'


# TODO: rework using pathlib
def html_report(figs, text=None, name='report', directory=None,
                dpi=300, fig_dir='figs'):
    """Produce an HTML report containing figs and text."""
    #
    # if dominate not installed, dive out here
    if not (dominate and slugify):
        return

    safe_name = slugify(name)
    safe_fig_dir = slugify(fig_dir)

    if directory is None:
        directory = safe_name

    directory = Path(directory)
    print(directory, directory.is_absolute())
    if not directory.is_absolute():
        directory = _report_dir/slugify(str(directory))
    try:
        os.makedirs(directory)
    except FileExistsError as e:
        print(e)

    try:
        os.mkdir(directory/safe_fig_dir)
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
                    if item:
                        url = os.path.join(safe_fig_dir,
                                           f"{safe_name}-fig-{j}_{i}.png")
                        item.savefig(os.path.join(directory, url),
                                     dpi=dpi, bbox_inches="tight")
                        # width=100% makes browser scale image
                        r += td(img(src=url, width="100%"))
                    else:
                        r += td()

    with open(os.path.join(directory, safe_name + '.html'), 'w') as f:
        print(h, file=f)
    return h

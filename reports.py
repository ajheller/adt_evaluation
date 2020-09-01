#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:28:07 2020

@author: heller
"""
import os
from dominate.tags import *

def html_report(figs, name='report', directory=None, dpi=75):

    if directory is None:
        directory = name
    try:
        os.mkdir(directory)
    except FileExistsError as e:
        print(e)

    h = html()
    with h.add(body()).add(div(id='content')):
        h1(f'Performance Plots: {name}')
        #p('Lorem ipsum ...')
        with table().add(tbody()):
            for j, row in enumerate(figs):
                l = tr()
                for i, item in enumerate(row):
                    url = f"{name}-fig-{j}_{i}.png"
                    item.savefig(os.path.join(directory, url), dpi=dpi)
                    l += td(img(src=url))

    with open(os.path.join(directory, name + '.html'), 'w') as f:
        print(h, file=f)
    return h

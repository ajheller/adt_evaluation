#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:35:22 2020

@author: heller
"""

from dataclasses import dataclass, field
from loudspeaker_layout import LoudspeakerLayout
from program_channels import Channels


@dataclass
class Decoder:
    C: Channels
    S: LoudspeakerLayout

    name: str = ""
    description: str = ""


import requests
import json


def get_mixer_value(i, j):
    r = requests.get(f"http://localhost:5510/matrix_mixer/g-o{i}-i{j}")
    c = r.content
    value = c.split(b" ")
    return float(value[1])


def set_mixer_value(i, j, v):
    r = requests.get(f"http://localhost:5510/matrix_mixer/g-o{i}-i{j}?value={v}")
    c = r.content
    value = c.split(b" ")
    try:
        v = float(value[1])
    except IndexError:
        v = None
    return v


def get_mixer_schema():
    r = requests.get(f"http://localhost:5510/JSON")
    c = r.content
    j = json.loads(c)
    return j


def set_mixer_matrix(M):
    for i, u in enumerate(M):
        for j, v in enumerate(u):
            w = set_mixer_value(i, j, v)
            if w is None:
                print("v != w", w, v)


import http.client

# %timeit set_mixer_matrix2(np.random.rand(49,64))
# 33.7 s ± 2.73 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

# %timeit -n 1 -r 1 set_mixer_matrix2(np.random.rand(49,64))
# 30 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)


def set_mixer_matrix2(M, host="localhost", port=5510):
    try:
        c = http.client.HTTPConnection(host, port, timeout=10)
        for i, u in enumerate(M):
            for j, v in enumerate(u):
                c.request("GET", f"/matrix_mixer/out{j}/g-o{i}-i{j}?value={v}")
                res = c.getresponse()
                if not res.status == 200:
                    print(i, j, res.status)
    finally:
        c.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:35:22 2020

@author: heller
"""

import requests
import json

# prodocol, userinfo, subdomain, domain_name, port, path, query, parameters, fragment

host = "localhost"
port = 5510

def mixer_url(i, j):
    return f"http://{host}:{port}/matrix_mixer/out-{i}/in-{j}"


def get_mixer_value(i, j):
    r = requests.get(mixer_url(i, j))
    c = r.content
    print(c)
    value = c.split(b" ")
    return float(value[1])


def set_mixer_value(i, j, v):
    r = requests.get(mixer_url(i, j) + f"?value={v}")
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
# 30/(49*64) = 100 ms/element, is this the timing of the "slow" loop?


def set_mixer_matrix2(M, host="localhost", port=5510):
    try:
        c = http.client.HTTPConnection(host, port, timeout=10)
        for i, u in enumerate(M):
            for j, v in enumerate(u):
                c.request("GET", f"/matrix_mixer/out{i}/in{j}?value={v}")
                res = c.getresponse()
                if not res.status == 200:
                    print(i, j, res.status)
    finally:
        c.close()

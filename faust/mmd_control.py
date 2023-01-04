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
import numpy as np
import timeit, time

# %timeit set_mixer_matrix2(np.random.rand(49,64))
# 33.7 s ± 2.73 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

# %timeit -n 1 -r 1 set_mixer_matrix2(np.random.rand(49,64))
# 30 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
# 30/(49*64) = 10 ms/element, is this the timing of the "slow" loop?


request_count = 0

def set_mixer_matrix2(M, host="localhost", port=5510):
    global request_count
    try:
        c = http.client.HTTPConnection(host, port, timeout=1)
        for i, u in enumerate(M):
            for j, v in enumerate(u):
                try:
                    request_count += 1
                    c.request("GET", f"/matrix_mixer/out-{i}/in-{j}?value={v}")
                    res = c.getresponse()
                    if res.status in [200, 201]:
                        results = res.read().split(b" ")
                        if np.isclose(float(results[1]), v, atol=1e-6):
                            pass
                        else:
                            print(float(results[1]), v)
                    else:
                        print(i, j, res.status)
                except http.client.CannotSendRequest as e:
                    print("CSR", e, M.shape, i, j, request_count)
                except ConnectionRefusedError as e:
                    print("CRE", e, M.shape, i, j, request_count)
    finally:
        c.close()
        #print(c)


def benchmark(repeats=10, sizes=(1, 2, 4, 8, 16, 32, 64)):
    global request_count
    request_count = 0
    s = get_mixer_schema()
    n_in = int(s['inputs'])
    n_out = int(s['outputs'])
    stats = []
    for i in sizes:
        times = [timeit.timeit(lambda: set_mixer_matrix2(np.random.random((i, n_in))),
                               number=1,
                               setup=lambda:time.sleep(0.5))
                 for _ in range(repeats)]
        stats.append((i, np.mean(times), np.std(times)))
    return stats


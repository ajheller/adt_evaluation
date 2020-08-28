#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:33:00 2020

@author: heller
"""

# from https://preshing.com/20110924/timing-your-code-using-pythons-with-statement/

# TODO: reimplement with:
#  https://docs.python.org/dev/library/contextlib.html#contextlib.ContextDecorator
# so it can be used as a context manager and decorator (and a dessert topping!)

import time


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


"""

Example use:

with Timer() as t:
    conn = httplib.HTTPConnection('google.com')
    conn.request('GET', '/')

print('Request took %.03f sec.' % t.interval)


try:
    with Timer() as t:
        conn = httplib.HTTPConnection('google.com')
        conn.request('GET', '/')
finally:
    print('Request took %.03f sec.' % t.interval)

"""
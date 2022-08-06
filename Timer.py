#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:33:00 2020

@author: heller
"""

# This file is part of the Ambisonic Decoder Toolbox (ADT)
# Copyright (C) 2018-19  Aaron J. Heller <heller@ai.sri.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


#
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

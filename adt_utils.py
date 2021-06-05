#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:49:39 2021

@author: heller
"""

import getpass
import platform
from datetime import datetime


def timestamp(join='\n'):
    ret = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           f"{platform.node()} ({platform.platform()})",
           getpass.getuser())
    if join:
        ret = join.join(ret)
    return ret

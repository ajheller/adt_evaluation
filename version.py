#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 12:55:39 2022

@author: heller
"""

import subprocess
from pathlib import Path


def version():
    return (
        subprocess.check_output(
            ["git", "describe", "--always"], cwd=Path(__file__).resolve().parent
        )
        .strip()
        .decode()
    )

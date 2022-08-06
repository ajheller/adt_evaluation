#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:01:24 2020

@author: heller
"""
# This file is part of the Ambisonic Decoder Toolbox (ADT)
# Copyright (C) 2018-20  Aaron J. Heller <heller@ai.sri.com>
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

import csv
import json
from collections.abc import Sequence  # for type declarations
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import pi as π

import spherical_data as SphD
from plot_utils import plot_lsl, plot_lsl_plan, plot_lsl_azel


@dataclass
class LoudspeakerLayout(SphD.SphericalData):
    """A class to represent loudspeaker arrays."""

    description: str = ""
    is_real: np.array = field(default_factory=lambda: np.array(0, dtype=bool))
    ids: list = field(default_factory=lambda: [])

    _primary_attrs = ["x", "y", "z", "is_real", "name", "ids", "description"]

    def __add__(self, other):
        """Append two layouts."""
        return append_layouts(self, other)

    def __getitem__(self, index, new_name=None, new_description=None):
        """Return a new LL with only the indexed items."""

        # look for magic indices
        # TODO: replace this with a dictionary of speaker ids and
        # TODO: and magic indices
        if isinstance(index, str):
            if "real".startswith(index.lower()):
                index = self.is_real
            elif "imaginary".startswith(index.lower()):
                index = ~self.is_real
            else:
                # is it a spkr id?
                index_bool = self.ids == index
                if np.sum(index_bool) == 1:
                    index = index_bool
                else:
                    raise ValueError(f"Unknown speaker id: {index}.")

        xyz = self.xyz[index]
        ids = np.asarray(self.ids)[index]
        is_real = np.asarray(self.is_real)[index]
        #
        name = new_name or self.name
        description = new_description or self.description

        l3 = LoudspeakerLayout(
            *xyz.T, name=name, description=description, ids=ids, is_real=is_real
        )
        return l3

    def real_only(self):
        return self[self.is_real]

    def set_is_real(self, is_real=True):
        """Set the is_real attribute of self, broadcasting if necessary."""
        try:
            # assume it is a sequence
            if len(is_real) == len(self.is_real):
                self.is_real = is_real
            else:
                raise ValueError("len(is_real) != len(self.is_real)")
        except TypeError:
            # assume it is a scalar
            self.is_real = np.full(self.shape, is_real, dtype=np.bool_)
        return self

    def set_imaginary(self):
        """Set the is_real attribute of the array to False."""
        self.set_is_real(False)
        return self

    def to_json(self, channels=None, gains=None):
        """Return a JSON dictionary in IEM plugin format."""
        #
        if channels is None:
            channels = range(1, len(self.ids) + 1)  # 1-based
        if gains is None:
            gains = np.ones_like(self.az)

        # IEM format documented here:
        #  https://plugins.iem.at/docs/configurationfiles/#the-loudspeakerlayout-object

        records = zip(
            self.az * 180 / π,
            self.el * 180 / π,
            self.r,
            map(bool, ~self.is_real),  # json needs bool not bool_
            channels,
            gains,
            self.ids,
        )
        columns = (
            "Azimuth",
            "Elevation",
            "Radius",
            "IsImaginary",
            "Channel",
            "Gain",
            "id",
        )
        ls_dict = [dict(zip(columns, rec)) for rec in records]
        return dict(
            {
                "LoudspeakerLayout": {
                    "Name": self.name,
                    "Description": self.description,
                    "Loudspeakers": ls_dict,
                }
            }
        )

    def to_iem_file(self, file, **kwargs):
        """Write LoudspeakerLayout to IEM JSON file."""
        with open(file, "w") as f:
            json.dump(obj=self.to_json(**kwargs), indent=4, fp=f)

    def plot3D(self, **kwargs):
        backend = kwargs.get("backend", "matplotlib")
        if backend == "matplotlib":
            ret = plot_lsl(self, title=f"Speaker Array: {self.name}", **kwargs)
        elif backend == "plotly":
            pass
        else:
            raise ValueError(f"Unknown plot backend {backend}")
        return ret

    def plot(self, **kwargs):
        return self.plot3D(**kwargs)

    def plot_plan(self, **kwargs):
        return plot_lsl_plan(self, **kwargs)

    def plot_azel(self, **kwargs):
        return plot_lsl_azel(self, **kwargs)


def append_layouts(l1, l2, name=None, description=None):
    """Append two layouts."""
    #
    if name is None:
        name = l1.name
    if description is None:
        description = l1.description

    print(l1, l2)

    xyz = np.append(l1.xyz, l2.xyz, axis=0)
    ids = np.append(l1.ids, l2.ids, axis=0)
    is_real = np.append(l1.is_real, l2.is_real, axis=0)

    l3 = LoudspeakerLayout(
        *xyz.T, name=name, description=description, ids=ids, is_real=is_real
    )
    return l3


#
# TODO: should there be ignore and no-op codes?
# unit codes and conversion factors to meters and radians
to_base = {
    "R": 1,  # Radians
    "D": π / 180,  # Degrees
    "G": π / 200,  # Gradians (grad, grade, gons, metric degrees)
    "M": 1,  # Meters
    "C": 1 / 100,  # Centimeters
    "I": 2.54 / 100,  # Inches
    "F": 12 * 2.54 / 100,  # Feet
    "L": 660 * 12 * 2.54 / 100,  # furLongs
    "S": 67 * 2.54 / 100,
}  # Smoots


def convert_units(quantity, from_unit, to_unit=None):
    """Convert a quantity between two units. No dimensional analysis."""
    q = quantity * to_base[from_unit.upper()]
    if to_unit:
        q /= to_base[to_unit]
    return q


# canonical ordering of coordinates
# coords that made sense: 'XYZ', 'AER', 'ANR', 'ARZ', but...
# we need a unique location for each; horizontal, vertical, radial
# gives: 'XZY', 'AER', 'ANR', 'AZR'
to_canonical = {
    "X": 0,
    "Y": 2,
    "Z": 1,
    "A": 0,
    "E": 1,
    "R": 2,  # Azimuth, Elevation, Radius
    "N": 1,
}  # zeNith angle (can't use Z)


"""
% TODO: copied from the MATLAB ADT.

%AMBI_MAT2SPKR_ARRAY convert matrix of speaker coordinates to SPKR_ARRAY struct
%   A is nx3 matrix of speaker coordinates
%     if A is a string, read coordinates from a file
%
%   coord_code identifies each coordinate (default: AER)
%     A, E, R = azimuth, elevation, radius
%     N = zeNith angle, angle fron North pole (can't use Z)
%     X, Y, Z = cartesian coordinates in Ambisonic convention:
%                  +/- X = front/back
%                  +/- Y = left/right
%                  +/- Z = up/down
%     U, V, W = cartesian coordinates, but with sign negated
%               (see SPKR_ARRAY_Boardroom for example)
%     can be in any order and mixed, e.g. ARZ for cylindrical.
%
%   unit_code identifies units for each coordinate (default: DDM)
%     D = Degrees
%     R = Radians
%     G = Gradians (aka - gon, grad, grade, metric degree)
%     C = Centimeters
%     M = Meters
%     F = Feet
%     I = Inches
%     S = Smoots
%     L = furLongs
%
%   name optional name of speaker array, defaults to filename
%
"""


def from_array(
    a: Sequence,
    coord_code: Sequence = "AER",
    unit_code: Sequence = "DDM",
    name: str = None,
    description=None,
    ids="S",
    is_real=True,
) -> LoudspeakerLayout:
    """

    :type unit_code: object
    """
    # make sure it's an Nx3 numpy array
    a = np.asarray(a).reshape(-1, 3)
    num_spkrs = len(a)

    if np.isscalar(ids):
        ids = [ids + "%02d" % i for i in range(num_spkrs)]
    elif len(ids) != num_spkrs:
        raise ValueError("len(ids) != num_spkrs")

    if np.isscalar(is_real):
        is_real = np.full(num_spkrs, is_real, dtype=bool)
    elif len(is_real) != num_spkrs:
        raise ValueError("len(is_real) != num_spkrs")

    if name is None:
        name = "Amb" + str(num_spkrs)

    # convert the columns to base units -- meter, radians
    for (col, code) in enumerate(unit_code):
        a[:, col] *= to_base[code[0].upper()]

    # coordinate untangling
    aa = np.zeros_like(a)
    ac = [" "] * 3
    for col, code in enumerate(coord_code):
        c = code[0].upper()
        to_col = to_canonical[c]
        aa[:, to_col] = a[:, col]
        ac[to_col] = c
    ac = "".join(ac)
    print("ac:", ac)

    # make the SA object
    s = LoudspeakerLayout(ids=ids, is_real=is_real, name=name, description=description)
    # TODO: is there a slicker way to do this?
    if ac == "XZY":  # see comment on to_cannonical for why this is XZY
        s.set_from_cart(*[aa[:, to_canonical[c]] for c in "XYZ"])
    elif ac == "AER":
        s.set_from_aer(*[aa[:, to_canonical[c]] for c in "AER"])
    elif ac == "ANR":
        s.set_from_sph(*[aa[:, to_canonical[c]] for c in "ANR"])
    elif ac == "AZR":
        s.set_from_cyl(*[aa[:, to_canonical[c]] for c in "ARZ"])
    else:
        raise NotImplementedError(f"Sorry, {ac} not implemented.")

    return s


# convenience function that takes three vectors (or scalars) of coordinates
def from_vectors(c0, c1, c2, **kwargs) -> LoudspeakerLayout:
    if np.isscalar(c1):
        c1 = np.full_like(c0, c1, dtype=float)
    if np.isscalar(c2):
        c2 = np.full_like(c0, c2, dtype=float)

    if len(c0) == len(c1) == len(c2):
        return from_array(np.column_stack((c0, c1, c2)).astype(float), **kwargs)
    else:
        raise ValueError(
            "c0, c1, c2 must be the same length, "
            f"but were {list(map(len, (c0, c1, c2)))}."
        )


# def from_iem_file(file):
#     """Load a layout from an IEM-format file."""
#     obj = json.load(open(file, 'r'))
#     lsl_dict = obj["LoudspeakerLayout"]

#     name = lsl_dict.get('Name', Path(file).stem)
#     description = lsl_dict.get('Description', name)
#     spkr_keys = ('Azimuth', 'Elevation', 'Radius',
#                  'IsImaginary', 'Channel', 'Gain', 'id')
#     az, el, r, is_imaginary, channel, gain, id = \
#         zip(*[itemgetter(*spkr_keys)(ls)
#               for ls in lsl_dict["Loudspeakers"]])

#     lsl = from_vectors(az, el, r,
#                        coord_code='AER', unit_code='DDM',
#                        is_real=~np.array(is_imaginary),
#                        name=name, description=description)

#     return lsl, obj

_iem_loudspeaker_layout_keys = (
    "Azimuth",
    "Elevation",
    "Radius",
    "IsImaginary",
    "Channel",
    "Gain",
)
_iem_loudspeaker_layout_getter = itemgetter(*_iem_loudspeaker_layout_keys)


def from_iem_file(file_name, return_json=False) -> LoudspeakerLayout:
    """Load a layout from an IEM-format file."""
    #
    with open(file_name, "r", encoding="utf-8") as f:
        obj = json.load(f)
    lsl_dict = obj.get("LoudspeakerLayout")
    if not lsl_dict:
        print(f'The "{file_name}" json file is invalid')
        return

    dec_dict = obj.get("Decoder", dict())

    name = lsl_dict["Name"]
    description = lsl_dict.get("Description", dec_dict.get("Description"))

    az, el, r, is_imaginary, channel, gain = zip(
        *[_iem_loudspeaker_layout_getter(ls) for ls in lsl_dict["Loudspeakers"]]
    )

    # make ids from channels numbers
    ids = [
        f"S{ch:02d}" if not is_imag else f"I:{ch:02d}"
        for ch, is_imag in zip(channel, is_imaginary)
    ]

    lsl = from_vectors(
        az,
        el,
        r,
        coord_code="AER",
        unit_code="DDM",
        ids=ids,
        is_real=~np.array(is_imaginary),
        name=name,
        description=description,
    )
    if return_json:
        return lsl, obj
    else:
        return lsl


def from_csv_file(file_name) -> LoudspeakerLayout:
    with open(file_name, "r", encoding="utf-8") as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = csv.reader(read_obj, skipinitialspace=True)
        # Iterate over each row in the csv using reader object
        spkrs = []
        for row in csv_reader:
            try:
                op, *args = row
                # print(op, args)
            except ValueError:
                pass
            else:
                op = op.lower()
                # print(op)
                if "name".startswith(op):
                    name = args[0]
                elif "description".startswith(op):
                    description = args[0]
                elif "fields".startswith(op):
                    fields = args
                elif "units".startswith(op):
                    units = args
                elif "speaker".startswith(op) or "spkr".startswith(op):
                    spkrs.append(args)
                elif op.startswith("#"):
                    pass
                elif op.startswith("!"):
                    print(args)
                else:
                    print(f"Ignoring op: {op}")
    # use pandas superpowers
    spkr_df = pd.DataFrame(spkrs, columns=fields)
    # convert columns to numeric if possible
    for key in spkr_df.keys():
        if key in (
            "Azimuth",
            "Elevation",
            "Zenith",
            "Radius",
            "X",
            "Y",
            "Z",
            "U",
            "V",
            "W",
        ):
            try:
                spkr_df[key] = pd.to_numeric(spkr_df[key])
            except ValueError as e:
                raise ValueError(f"{e} in {key}")
    # normalize booleans
    spkr_df.replace(
        ("T", "True", "Yes", "Dah", "Oui", "Ja", "Yo", "Gnarly"), True, inplace=True
    )
    spkr_df.replace(
        ("F", "False", "No", "Nyet", "Non", "Nein", "Bupkis", "Whatev"),
        False,
        inplace=True,
    )

    for keys, codes in (
        (["Azimuth", "Elevation", "Radius"], "AER"),
        (["Azimuth", "Zenith", "Radius"], "ANR"),
        (["X", "Z", "Y"], "XZY"),
        (["Azimuth", "Z,", "R"], "AZR"),
    ):

        try:
            x = spkr_df[keys]
        except KeyError as e:
            print(keys, e)

        else:
            code = codes
            # print(code, x)
            # get the units

            break
    else:
        raise ValueError(f"Can't make sense of {spkr_df.keys()}")

    # units
    unit_coord_dict = {c: u.upper() for c, u in zip(fields, units)}
    print(unit_coord_dict)
    unit_code = []
    for k in keys:
        unit = unit_coord_dict[k]
        # special case for furlongs
        if unit.startswith("FUR"):
            unit_code.append("L")
            break
        # otherwise the first letter is the code
        for u in ("M", "F", "I", "S", "C", "R", "D", "G"):
            if unit.startswith(u):
                unit_code.append(u)
                print(k, unit, unit_code)
                break

    lsl = from_array(
        x,
        name=name,
        description=description,
        coord_code=code,
        unit_code=unit_code,
        ids=spkr_df["Name"].values,
        is_real=spkr_df["Real"].values,
    )

    return lsl


#
#
def unit_test_iem():
    import example_speaker_arrays as esa

    s = esa.stage2017()
    # json dump, read
    s.to_iem_file("test_iem_stage.json")
    lsl = from_iem_file("test_iem_stage.json")

    return lsl


#
def unit_test():
    """Run tests for this module."""
    from matplotlib import pyplot as plt

    import example_speaker_arrays as esa

    s = esa.stage2017()
    s.plot_plan()

    plt.figure(figsize=(12, 6))
    plt.scatter(s.az * 180 / π, s.el * 180 / π, c="white", marker="o")
    for x, y, r, t in zip(s.az * 180 / π, s.el * 180 / π, s.r, s.ids):
        plt.text(
            x,
            y,
            t,
            bbox=dict(facecolor="lightblue", alpha=0.4),
            horizontalalignment="center",
            verticalalignment="center",
        )
    plt.grid()
    # plt.colorbar()
    plt.show()

    lsl = unit_test_iem()

    return s, lsl

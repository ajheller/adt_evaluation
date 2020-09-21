from dataclasses import dataclass, field
import numpy as np
from functools import cached_property

from numpy import pi
from numpy import pi as π


# these follow the MATLAB convention for spherical coordinates
def cart2sph(x, y, z):
    """Convert from Cartesian to spherical coordinates, using MATLAB convention
    for spherical coordinates.

    Parameters
    ----------
        x, y, z: array-like
           Cartesian coordinates

    Returns
    -------
        az, el, r: nd-array
            azimuth, elevation (radians)
            radius (input units)

    """
    r_xy = np.hypot(x, y)
    r = np.hypot(r_xy, z)
    el = np.arctan2(z, r_xy)
    az = np.arctan2(y, x)
    return az, el, r


def sph2cart(az, el, r=1):
    """Convert from spherical to Cartesian coordinates, using MATLAB convention
    for spherical coordinates.

    Parameters
    ----------
        az, e, r: ndarray
            azimuth, elevation (radians)
            radius (input units), optional (default 1)

    Returns
    -------
        x, y, z: ndarray
           Cartesian coordinates (in input units)

    Notes
    -----
        The function is vectorized and return values will be the same shape as
        the inputs.

    """
    z = r * np.sin(el) * np.ones_like(az)
    r_cos_el = r * np.cos(el)
    x = r_cos_el * np.cos(az)
    y = r_cos_el * np.sin(az)
    return x, y, z


# these follow the physics convention of zenith angle, azimuth
def sphz2cart(zen, az, r=1):
    "Spherical to cartesian using Physics conventxion, e.g. Arfkin"
    return sph2cart(az, pi / 2 - zen, r)


def cart2sphz(x, y, z):
    """Cartesian to spherical using Physics convention

    Parameters
    ----------
        x, y, z: ndarray
           Cartesian coordinates (in input units)


    Returns
    -------
        zentih angle: ndarray
            angle from +z-axis in radians

        azimuth: ndarray
            angle from the x-axis in the x-y plane in radians

        radius: ndarray
           distance from the origin in input units
    """
    az, el, r = cart2sph(x, y, z)
    return (pi / 2 - el), az, r

# TODO: rework this so the shape of the elements are preserved.
# add a shape property.  x, y, z beocome primary representation
# xyz and u become cached properties
@dataclass
class SphericalData():
    xyz: np.ndarray = field(default_factory=lambda: np.array(None))
    name: str = 'data'

    _primary_attrs = ['xyz', 'name']

    # def __init__(self, name="data"):
    #     self.xyz = None
    #     self.name = name

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'xyz':
            print('clearing caches')
            self._clear_cached_properties()

    def _clear_cached_properties(self):
        keys = list(self.__dict__.keys())
        for key in keys:
            if key not in ('xyz',):
                print(f"   clearing: {key}")
                delattr(self, key)

    def set_from_cart(self, x_or_xyz, y=None, z=None):
        xyz = np.asarray(x_or_xyz)
        if y is not None:
            xyz = np.c_[xyz, y, z]
        self.xyz = xyz
        return self

    def set_from_sph(self, theta, phi, rho=1, phi_is_zenith=False):
        if phi_is_zenith:
            phi = pi/2 - phi
        return self.set_from_cart(sph2cart(theta, phi, rho))

    def set_from_aer(self, az, el, r=1):
        self.xyz = np.column_stack(*sph2cart(az, el, r))
        return self

    def set_from_cyl(self, theta, rho, z):
        raise NotImplementedError
        return self

    @cached_property
    def u(self):
        return self.xyz / np.linalg.norm(self.xyz, axis=1)[None].T

    @cached_property
    def aer(self):
        return np.column_stack(cart2sph(*self.xyz.T))

    @cached_property
    def azr(self):
        return np.column_stack(cart2sphz(*self.xyz.T))

    @property
    def az(self):
        return self.aer[:, 0]

    @property
    def el(self):
        return self.aer[:, 1]

    @property
    def r(self):
        return self.aer[:, 2]


def from_cart(*args, sd_class=SphericalData):
    return sd_class().set_from_cart(*args)

def from_sph(*args, sd_class=SphericalData):
    return sd_class().set_from_sph(*args)


def unit_test():
    import spherical_grids
    sg240 = spherical_grids.t_design240()
    q = from_sph(sg240.az, sg240.el)
    u = q.u
    v = q.az
    w = q.aer
    q.set_from_cart(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    # cache should be cleared
    if q.u == u:
        print('FAIL: caches were not cleared!!')
    return q, (u, v, w)



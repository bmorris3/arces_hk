"""
Tools for measuring equivalent widths, S-indices.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time

from .catalog import query_catalog_for_object

__all__ = ['integrate_spectrum_trapz', 'true_h_centroid', 'true_k_centroid',
           'uncalibrated_s_index', 'StarProps', 'Measurement']

true_h_centroid = 3968.47 * u.Angstrom
true_k_centroid = 3933.66 * u.Angstrom


def integrate_spectrum_trapz(spectrum, center_wavelength, width,
                             weighting=False, plot=False):
    """
    Integrate the area under a spectrum.

    Parameters
    ----------
    spectrum : `EchelleSpectrum`
        Spectrum to integrate under
    center_wavelength : `~astropy.units.Quantity`
        Center of region to integrate
    width : `~astropy.units.Quantity`
        Width about the center to integrate
    wavelength_offset : float
        Offset wavelengths by this amount, which is useful if a wavelength
        solution refinement as been made.
    weighting : bool
        Apply a triangular weighting function to the fluxes

    Returns
    -------
    integral : float
        Integral under the spectrum
    error : float
        Square-root of the sum of the fluxes within the bounds of the integral
    """
    wavelength = spectrum.wavelength
    wavelengths_increasing = wavelength[1] > wavelength[0]

    if (not center_wavelength < wavelength.max() and
            not center_wavelength > wavelength.min()):
        raise ValueError("This spectral order does not contain"
                         "the center_wavelength given.")

    flux = spectrum.flux

    within_bounds = ((wavelength > center_wavelength - width/2) &
                     (wavelength < center_wavelength + width/2))

    if not weighting:
        integral = np.trapz(flux[within_bounds].value,
                            wavelength[within_bounds].value)
        error = np.sqrt(np.sum(flux[within_bounds].value))
    else:
        triangle_weights = triangle_weighting(wavelength[within_bounds],
                                              center_wavelength)
        integral = np.trapz(flux[within_bounds].value * triangle_weights,
                            wavelength[within_bounds].value)
        error = np.sqrt(np.sum(flux[within_bounds].value * triangle_weights))

    if not wavelengths_increasing:
        integral *= -1

    if plot:
        plt.figure()
        plt.plot(wavelength[spectrum.mask], flux[spectrum.mask])
        if weighting:
            triangle = triangle_weighting(wavelength[within_bounds],
                                          center_wavelength)
            plt.plot(wavelength[within_bounds], triangle, 'r', lw=2)
        plt.show()

    return integral, error


def triangle_weighting(x, x0, fwhm=1.09*u.Angstrom):
    """
    Compute the triangular weighting function used in CaII cores

    Parameters
    ----------
    x : `~astropy.units.Quantity`, array-like
        Wavelengths
    x0 : `~astropy.units.Quantity`
        Central wavelength of emission feature
    fwhm : `~astropy.units.Quantity`
        Full-width at half maximum of the weighting function
    """

    left_half = (x <= x0) & (x > x0 - fwhm)
    right_half = (x > x0) & (x < x0 + fwhm)

    weights = np.zeros_like(x.value)
    # float typecasting below resolves the unitless quantities
    weights[left_half] = ((x[left_half] - x0) / fwhm).value + 1
    weights[right_half] = ((x0 - x[right_half]) / fwhm).value + 1

    return weights


def uncalibrated_s_index(spectrum):
    """
    Calculate the uncalibrated S-index from an Echelle spectrum.

    Parameters
    ----------
    spectrum : `EchelleSpectrum`
        Normalized target spectrum

    Returns
    -------
    s_ind : `SIndex`
        S-index. This value is intrinsic to the instrument you're using.
    """

    order_h = spectrum.get_order(89)
    order_k = spectrum.get_order(90)
    order_r = spectrum.get_order(91)
    order_v = spectrum.get_order(88)

    r_centroid = 3900 * u.Angstrom
    v_centroid = 4000 * u.Angstrom
    hk_fwhm = 1.09 * u.Angstrom
    hk_width = 2 * hk_fwhm
    rv_width = 20 * u.Angstrom

    h, h_err = integrate_spectrum_trapz(order_h, true_h_centroid, hk_width,
                                        weighting=True)
    k, k_err = integrate_spectrum_trapz(order_k, true_k_centroid, hk_width,
                                        weighting=True)
    r, r_err = integrate_spectrum_trapz(order_r, r_centroid, rv_width)
    v, v_err = integrate_spectrum_trapz(order_v, v_centroid, rv_width)

    s_ind = SIndex(h=h, k=k, r=r, v=v, time=spectrum.time,
                   h_err=h_err, k_err=k_err, r_err=r_err, v_err=v_err)
    return s_ind


class SIndex(object):
    def __init__(self, h, k, r, v, k_factor=0.84, v_factor=1.0, time=None,
                 h_err=None, k_err=None, r_err=None, v_err=None):
        """
        The pre-factors have been chosen to make the ``h`` and ``k`` values
        of the same order of magnitude; same for ``r`` and ``v``.

        Parameters
        -----------
        h : float
            CaII H feature emission flux
        k : float
            CaII K feature emission flux
        r : float
            Pseudo-continuum flux redward of CaII H&K
        v : float
            Pseudo-continuum flux blueward of CaII H&K
        k_factor : float
            Multiplicative factor for the K emission feature flux
            to make the H & K fluxes similar
        v_factor : float
            Multiplicative factor for the blue continuum region flux
            to make the r & v fluxes similar
        time : `~astropy.time.Time`
            Time this S-index measurement was taken.
        """
        self.r = r
        self.v = v
        self.h = h
        self.k = k

        self.r_err = r_err
        self.v_err = v_err
        self.h_err = h_err
        self.k_err = k_err

        self.k_factor = k_factor
        self.v_factor = v_factor

        self.time = time

    @property
    def uncalibrated(self):
        """
        Compute Eqn 2 of Isaacson+ 2010, for C1=1 and C2=0. This can be used
        to solve for C1 and C2.
        """

        scale_down_err = 100

        uncalibrated_s_ind = ((self.h + self.k_factor * self.k) /
                              (self.r + self.v_factor * self.v))

        s_ind_err = (1 / (self.r + self.v_factor*self.v)**2 *
                     (self.h_err**2 + self.k_factor**2 * self.k**2) +
                     (self.h + self.k_factor*self.k)**2 /
                     (self.r + self.v_factor * self.v)**4 *
                     (self.r_err**2 + self.v_factor**2 * self.v_err**2)
                     )**0.5 / scale_down_err

        return Measurement(uncalibrated_s_ind, err_upper=s_ind_err,
                           err_lower=s_ind_err)

    def calibrated(self, c1, c2):
        """
        Calibrated S-index measurement (comparable with MWO S-indices).

        Uses the scaling constants as defined in Isaacson 2010+ (c1 and c2).

        Parameters
        ----------
        c1 : float
        c2 : float

        Returns
        -------
        Calibrated S-index.
        """
        return c1 * self.uncalibrated + c2

    @classmethod
    def from_dict(cls, dictionary):
        d = dictionary.copy()
        for key in dictionary:
            if key == 'time':
                d[key] = Time(float(d[key]), format='jd')
            else:
                d[key] = float(d[key])
        return cls(**d)


class StarProps(object):
    def __init__(self, name=None, s_apo=None, s_mwo=None, time=None):
        self.name = name
        self.s_apo = s_apo
        self._s_mwo = s_mwo
        self.time = time

    @property
    def s_mwo(self):
        if self._s_mwo is None:
            obj = query_catalog_for_object(self.name)
            self._s_mwo = Measurement.from_min_mean_max(obj['Smean'],
                                                        obj['Smin'],
                                                        obj['Smax'])
        return self._s_mwo

    @classmethod
    def from_dict(cls, dictionary):
        s_apo = SIndex.from_dict(dictionary['s_apo'])
        if 's_mwo' in dictionary:
            s_mwo = Measurement.from_dict(dictionary['s_mwo'])
        else:
            s_mwo = None

        if dictionary['time'] != 'None':
            dictionary['time'] = Time(float(dictionary['time']), format='jd')
        else:
            dictionary['time'] = None



        return cls(s_apo=s_apo, s_mwo=s_mwo, name=dictionary['name'],
                   time=dictionary['time'])


class Measurement(object):
    def __init__(self, value, err_upper=None, err_lower=None, default_err=0.1):

        if hasattr(value, '__len__'):
            value = np.asarray(value)
            err_upper = np.asarray(err_upper)
            err_lower = np.asarray(err_lower)
            self.value = value
            self.err_upper = err_upper
            self.err_lower = err_lower

            self.err_upper[self.err_upper == 0] = default_err
            self.err_lower[self.err_lower == 0] = default_err

        else:

            self.value = value
            if err_lower == 0 or err_upper == 0:
                self.err_upper = self.err_lower = default_err
            else:
                self.err_upper = err_upper
                self.err_lower = err_lower

    @classmethod
    def from_min_mean_max(cls, mean, min, max):
        return cls(value=mean, err_upper=max-mean, err_lower=mean-min)

    @classmethod
    def from_dict(cls, dictionary):
        kwargs = {key: float(dictionary[key]) for key in dictionary}
        return cls(**kwargs)


"""
Tools for measuring equivalent widths, S-indices.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.modeling.models import Voigt1D, Lorentz1D
from astropy.modeling import fitting

from .catalog import query_catalog_for_object

__all__ = ['fit_emission_feature', 'integrate_spectrum_trapz',
           'plot_spectrum_for_s_index', 'true_h_centroid', 'true_k_centroid',
           'uncalibrated_s_index', 'Star']

fit_model = fitting.SLSQPLSQFitter()  # fitting.LevMarLSQFitter()

true_h_centroid = 3968.47 * u.Angstrom
true_k_centroid = 3933.66 * u.Angstrom


def fit_emission_feature(normalized_spectrum, approx_wavelength, spectral_order,
                         background_width=8*u.Angstrom, name=None, plot=False,
                         use_right_wings=False):
    """
    Fit an emission feature in a normalized spectrum.

    Parameters
    ----------
    normalized_spectrum : `EchelleSpectrum`
        Normalized spectrum of the target
    approx_wavelength : `~astropy.units.Quantity`
        Approximate wavelength of the emission feature. The solution will be
        bounded to within two angstroms of the ``approx_wavelength`` guess.
    spectral_order : int
        Spectral order to fit
    background_width : `~astropy.units.Quantity`
        Set the width of the region around the the emission feature to fit
    name : str
        Name of the emission feature being fit (used in the optional plots)
    plot : bool
        Display a plot of the fit
    use_right_wings : bool
        Fit only the wavelengths redward of the emission feature. This is useful
        if there are significant absorption features on the blue absorption wing.
    """
    spectrum = normalized_spectrum.get_order(spectral_order)
    near_core = ((np.abs(spectrum.wavelength - approx_wavelength) <
                  background_width))
    time = normalized_spectrum.header['DATE-OBS']

    if use_right_wings:
        near_core &= (spectrum.wavelength > approx_wavelength - 2*u.angstrom)

    wavelength = spectrum.wavelength[near_core]
    flux = spectrum.flux[near_core]

    # Normalize to the off-core flux:
    core_width = 1.5 * u.angstrom
    in_core = np.abs(wavelength - approx_wavelength) < core_width
    flux /= np.max(flux[~in_core])

    # Construct a model which has a Lorentzian component for absorption,
    # a Voigt component for the emission
    init_params = (Lorentz1D(amplitude=-0.8, x_0=approx_wavelength.value, fwhm=7.5) +
                   Voigt1D(x_0=approx_wavelength.value,
                           amplitude_L=1.0, fwhm_L=0.3, fwhm_G=0.2))

    # Force the center of absorption and emission to be the same wavelength
    init_params.x_0_1.tied = lambda m: m.x_0_0
    # Force the wavelength center to be near the initial guess
    init_params.x_0_0.bounds = [approx_wavelength.value - 2.0,
                                approx_wavelength.value + 2.0]
    # Force the amplitude of the voigt profile to be positive
    init_params.amplitude_L_1.bounds = [0, np.inf]
    # Force the amplitude of the lorentz profile to be negative
    init_params.amplitude_0.bounds = [-np.inf, 0]

    init_model = init_params.evaluate(wavelength.value, *init_params.parameters)
    composite_model = fit_model(init_params, wavelength.value, flux - 1,
                                maxiter=1000, acc=1e-8, disp=False)
    best_fit_model = composite_model(wavelength.value)

    # get just the best-fit Voigt profile, use it to measure the equivalent width:
    composite_model_params = {i: j for i, j in zip(composite_model.param_names,
                                                   composite_model.parameters)}

    best_fit_core_model = Voigt1D(x_0=composite_model_params['x_0_1'],
                                  amplitude_L=composite_model_params['amplitude_L_1'],
                                  fwhm_L=composite_model_params['fwhm_L_1'],
                                  fwhm_G=composite_model_params['fwhm_G_1'])

    # Equivalent width is multiplied by -1 because wavelength array is in reverse order
    equiv_width_trapz = -1 * np.trapz(best_fit_core_model(wavelength.value),
                                      x=wavelength.value)

    if plot:
        fig, ax = plt.subplots(1)
        ax.plot(wavelength, flux)
        ax.plot(wavelength, best_fit_model + 1, 'r')
        ax.plot(wavelength, init_model+1, 'm')
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.set_title(name + " EW={0:.4f}".format(equiv_width_trapz))

    return time, equiv_width_trapz, composite_model_params


def integrate_spectrum_trapz(spectrum, center_wavelength,
                             width, weighting=False):
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
    else:
        integral = np.trapz(flux[within_bounds].value *
                            triangle_weighting(wavelength[within_bounds],
                                               center_wavelength),
                            wavelength[within_bounds].value)

    if not wavelengths_increasing:
        integral *= -1

    return integral


def plot_spectrum_for_s_index(all_normalized_spectra):

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))


    for spectrum in all_normalized_spectra:

        order_h = spectrum.get_order(89)
        order_k = spectrum.get_order(90)
        order_r = spectrum.get_order(91)
        order_v = spectrum.get_order(88)

        alpha = 0.5

        ax[0].plot(order_k.wavelength, order_k.flux, alpha=alpha)
        ax[0].set_title('K')

        ax[1].plot(order_h.wavelength, order_h.flux, alpha=alpha)
        ax[1].set_title('H')

        ax[2].plot(order_r.wavelength, order_r.flux, alpha=alpha)
        ax[2].set_title('R')

        ax[3].plot(order_v.wavelength, order_v.flux, alpha=alpha)
        ax[3].set_title('V')


    r_centroid = 3900 * u.Angstrom
    v_centroid = 4000 * u.Angstrom
    width = 12 * u.Angstrom

    ax[0].set_xlim([(true_k_centroid - width).value,
                    (true_k_centroid + width).value])
    ax[1].set_xlim([(true_h_centroid - width).value,
                    (true_h_centroid + width).value])
    ax[2].set_xlim([(r_centroid - width).value, (r_centroid + width).value])
    ax[3].set_xlim([(v_centroid - width).value, (v_centroid + width).value])

    # K badpass:
    hk_width = 1.05*u.Angstrom
    rv_width = 20*u.Angstrom
    ax[0].fill_between([(true_k_centroid - hk_width/2).value,
                        (true_k_centroid + hk_width/2).value],
                       0, 2, alpha=0.2, color='k', zorder=-100)

    ax[1].fill_between([(true_h_centroid - hk_width/2).value,
                        (true_h_centroid + hk_width/2).value],
                       0, 2, alpha=0.2, color='k', zorder=-100)

    ax[2].fill_between([(r_centroid - rv_width/2).value,
                        (r_centroid + rv_width/2).value],
                       0, 2, alpha=0.2, color='k', zorder=-100)

    ax[3].fill_between([(v_centroid - rv_width/2).value,
                        (v_centroid + rv_width/2).value],
                       0, 2, alpha=0.2, color='k', zorder=-100)

    for axes in ax:
        axes.set_ylim([0, 2])

    plt.show()


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
    hk_width = 1.09 * u.Angstrom
    rv_width = 20 * u.Angstrom

    h = integrate_spectrum_trapz(order_h, true_h_centroid, hk_width,
                                 weighting=False)
    k = integrate_spectrum_trapz(order_k, true_k_centroid, hk_width,
                                 weighting=False)
    r = integrate_spectrum_trapz(order_r, r_centroid, rv_width)
    v = integrate_spectrum_trapz(order_v, v_centroid, rv_width)

    #s_uncalibrated = (h + k) / (r + v)
    s_ind = SIndex(h=h, k=k, r=r, v=v)
    return s_ind


class SIndex(object):
    def __init__(self, h, k, r, v, k_factor=2.0, v_factor=1.0, time=None):
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

        self.k_factor = k_factor
        self.v_factor = v_factor

        self.time = time

    @property
    def uncalibrated(self):
        """
        Compute Eqn 2 of Isaacson+ 2010, for C1=1 and C2=0. This can be used
        to solve for C1 and C2.
        """
        return ((self.h + self.k_factor * self.k) /
                (self.r + self.v_factor * self.v))

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
        """
        return c1 * self.uncalibrated + c2

class Star(object):
    def __init__(self, name=None, s_apo=None, s_mwo=None, time=None):
        self.name = name
        self.s_apo = s_apo
        self._s_mwo = s_mwo
        self.time = time

    @property
    def s_mwo(self):
        if self._s_mwo is None:
            obj = query_catalog_for_object(self.name)
            self._s_mwo = obj['Smean']
        return self._s_mwo



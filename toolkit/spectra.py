"""
Tools for organizing, normalizing echelle spectra.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
import astropy.units as u
from specutils.io import read_fits
from specutils import Spectrum1D as Spec1D

from .spectral_type import query_for_T_eff
from .phoenix import get_phoenix_model_spectrum
from .masking import get_spectrum_mask

__all__ = ["EchelleSpectrum", "plot_spectrum", "continuum_normalize",
           "slice_spectrum", "interpolate_spectrum", "cross_corr"]


class Spectrum1D(Spec1D):
    """
    Inherits from the `~specutils.Spectrum1D` object.

    Adds a plot method.
    """
    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        if self.mask is None:
            mask = np.ones_like(self.flux.value).astype(bool)
        else:
            mask = self.mask
        ax.plot(self.wavelength[mask], self.flux[mask], **kwargs)


class EchelleSpectrum(object):
    """
    Echelle spectrum of one or more spectral orders
    """
    def __init__(self, spectrum_list, header=None, name=None):
        self.spectrum_list = spectrum_list
        self.header = header
        self.name = name
        self.model_spectrum = None
        
    @classmethod
    def from_fits(cls, path):
        """
        Load an echelle spectrum from a FITS file.

        Parameters
        ----------
        path : str
            Path to the FITS file
        """
        spectrum_list = read_fits.read_fits_spectrum1d(path)
        header = fits.getheader(path)

        name = header.get('OBJNAME', None)
        return cls(spectrum_list, header=header, name=name)
    
    def get_order(self, order):
        """
        Get the spectrum from a specific spectral order

        Parameter
        ---------
        order : int
            Echelle order to return

        Returns
        -------
        spectrum : `~specutils.Spectrum1D`
            One order from the echelle spectrum
        """
        return self.spectrum_list[order]

    def fit_order(self, spectral_order, polynomial_order):
        """
        Fit a spectral order with a polynomial.

        Parameters
        ----------
        spectral_order : int
            Spectral order index
        polynomial_order : int
            Polynomial order

        Returns
        -------
        fit_params : `~numpy.ndarray`
            Best-fit polynomial coefficients
        """
        spectrum = self.get_order(spectral_order)
        mean_wavelength = spectrum.wavelength.mean()
        fit_params = np.polyfit(spectrum.wavelength - mean_wavelength, 
                                spectrum.flux, polynomial_order)
        return fit_params
    
    def predict_continuum(self, spectral_order, fit_params):
        """
        Predict continuum spectrum given results from a polynomial fit from
        `EchelleSpectrum.fit_order`.

        Parameters
        ----------
        spectral_order : int
            Spectral order index
        fit_params : `~numpy.ndarray`
            Best-fit polynomial coefficients

        Returns
        -------
        flux_fit : `~numpy.ndarray`
            Predicted flux in the continuum for this order
        """
        spectrum = self.get_order(spectral_order)
        mean_wavelength = spectrum.wavelength.mean()
        flux_fit = np.polyval(fit_params, 
                              spectrum.wavelength - mean_wavelength)
        return flux_fit

    def offset_wavelength_solution(self, wavelength_offset):
        """
        Offset the wavelengths by a constant amount in a specific order.

        Parameters
        ----------
        spectral_order : int
            Echelle spectrum order to correct
        wavelength_offset : `~astropy.units.Quantity`
            Offset the wavelengths by this amount
        """
        for spectrum in self.spectrum_list:
            spectrum.wavelength += wavelength_offset

    def rv_wavelength_shift(self, spectral_order):
        """
        Solve for the radial velocity wavelength shift.

        Parameters
        ----------
        spectral_order : int
            Echelle spectrum order to shift
        """
        order = self.spectrum_list[spectral_order]

        if self.model_spectrum is None:
            T_eff = query_for_T_eff(self.name)
            self.model_spectrum = get_phoenix_model_spectrum(T_eff)

        order_width = order.wavelength.ptp()
        target_slice = slice_spectrum(order, order.wavelength.min() + order_width/4,
                                      order.wavelength.max() - order_width/4)
        model_slice = slice_spectrum(self.model_spectrum, target_slice.wavelength.min(),
                                     target_slice.wavelength.max(),
                                     norm=target_slice.flux.max())

        interp_target_slice = interpolate_spectrum(target_slice, model_slice.wavelength)

        rv_shift = cross_corr(interp_target_slice, model_slice)
        #self.offset_wavelength_solution(spectral_order, rv_shift)
        return rv_shift


def plot_spectrum(spectrum, norm=None, ax=None, offset=0, margin=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if norm is None:
        norm = np.ones_like(spectrum.flux)
    elif hasattr(norm, 'flux'): 
        norm = norm.flux
    if margin is None:
        ax.plot(spectrum.wavelength, spectrum.flux/norm + offset, **kwargs)
    else: 
        ax.plot(spectrum.wavelength[margin:-margin], 
                spectrum.flux[margin:-margin]/norm[margin:-margin] + offset, 
                **kwargs)

        
def continuum_normalize(target_spectrum, standard_spectrum, polynomial_order):
    """
    Normalize the target's spectrum by a polynomial fit to the the standard's
    spectrum.

    Parameters
    ----------
    target_spectrum : `EchelleSpectrum`
        Spectrum of the target object
    standard_spectrum : `EchelleSpectrum`
        Spectrum of the standard object
    polynomial_order : int
        Fit the standard's spectrum with a polynomial of this order

    Returns
    -------
    spectrum : `EchelleSpectrum`
        Normalized spectrum of the target star
    """
    normalized_spectrum_list = []
    
    for spectral_order in range(len(target_spectrum.spectrum_list)):
        # Extract one spectral order at a time to normalize
        standard_order = standard_spectrum.get_order(spectral_order)
        target_order = target_spectrum.get_order(spectral_order)

        target_mask = get_spectrum_mask(target_order)

        # Fit the standard's flux in this order with a polynomial
        fit_params = standard_spectrum.fit_order(spectral_order, polynomial_order)
        standard_continuum_fit = standard_spectrum.predict_continuum(spectral_order, 
                                                                     fit_params)
        
        # Normalize the target's flux with the continuum fit from the standard
        target_continuum_fit = target_spectrum.predict_continuum(spectral_order, 
                                                                 fit_params)
        target_continuum_normalized_flux = target_order.flux/target_continuum_fit
        target_continuum_normalized_flux /= np.median(target_continuum_normalized_flux)

        normalized_target_spectrum = Spectrum1D(target_continuum_normalized_flux, 
                                                target_order.wcs, mask=target_mask)

        normalized_spectrum_list.append(normalized_target_spectrum)
        
    return EchelleSpectrum(normalized_spectrum_list, header=target_spectrum.header,
                           name=target_spectrum.name)


def slice_spectrum(spectrum, start_wavelength, end_wavelength, norm=None):
    in_range = ((spectrum.wavelength < end_wavelength) &
                (spectrum.wavelength > start_wavelength))

    wavelength = spectrum.wavelength[in_range]

    if norm is None:
        flux = spectrum.flux[in_range]
    else:
        flux = spectrum.flux[in_range] * norm / spectrum.flux[in_range].max()


    return Spectrum1D.from_array(wavelength, flux,
                                 dispersion_unit=spectrum.wavelength_unit)


def interpolate_spectrum(spectrum, new_wavelengths):
    # start_ind = np.argmin(np.abs(start_wavelength - spectrum.wavelength))
    # end_ind = np.argmin(np.abs(end_wavelength - spectrum.wavelength))
    # print(start_ind, end_ind)
    #
    # if end_ind < start_ind:
    #     start_ind, end_ind = end_ind, start_ind
    #
    # return spectrum.slice_index(start_ind, end_ind)
    sort_order = np.argsort(spectrum.wavelength.to(u.Angstrom).value)
    sorted_spectrum_wavelengths = spectrum.wavelength.to(u.Angstrom).value[sort_order]
    sorted_spectrum_fluxes = spectrum.flux.value[sort_order]

    new_flux = np.interp(new_wavelengths.to(u.Angstrom).value,
                         sorted_spectrum_wavelengths,
                         sorted_spectrum_fluxes)

    return Spectrum1D.from_array(new_wavelengths, new_flux,
                                 dispersion_unit=spectrum.wavelength_unit)


def cross_corr(target_spectrum, model_spectrum):

    corr = np.correlate(target_spectrum.flux.value,
                        model_spectrum.flux.value, mode='same')

    max_corr_ind = np.argmax(corr)
    index_shift = corr.shape[0]/2 - max_corr_ind
    delta_wavelength = np.abs(target_spectrum.wavelength[1] -
                              target_spectrum.wavelength[0])
    wavelength_shift = index_shift * delta_wavelength
    return wavelength_shift



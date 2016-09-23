"""
Tools for organizing, normalizing echelle spectra.
"""

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

from specutils.io import read_fits
from specutils import Spectrum1D

__all__ = ["EchelleSpectrum", "plot_spectrum", "continuum_normalize"]


class EchelleSpectrum(object):
    """
    Echelle spectrum of one or more spectral orders
    """
    def __init__(self, spectrum_list, header=None):
        self.spectrum_list = spectrum_list
        self.header = header
        
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
        return cls(spectrum_list, header=header)
    
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
                                                target_order.wcs)

        normalized_spectrum_list.append(normalized_target_spectrum)
        
    return EchelleSpectrum(normalized_spectrum_list, header=target_spectrum.header)

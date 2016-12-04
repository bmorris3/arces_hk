"""
Tools for organizing, normalizing echelle spectra.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
import astropy.units as u
from astropy.time import Time
from specutils.io import read_fits
from specutils import Spectrum1D as Spec1D
from scipy.ndimage import gaussian_filter1d

from .spectral_type import query_for_T_eff
from .phoenix import get_phoenix_model_spectrum
from .masking import get_spectrum_mask
from .activity import true_h_centroid, true_k_centroid

__all__ = ["EchelleSpectrum", "plot_spectrum", "continuum_normalize",
           "slice_spectrum", "interpolate_spectrum", "cross_corr"]


class Spectrum1D(Spec1D):
    """
    Inherits from the `~specutils.Spectrum1D` object.

    Adds a plot method.
    """
    def plot(self, ax=None, **kwargs):
        """
        Plot the spectrum.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` (optional)
            The `~matplotlib.axes.Axes` to draw on, if provided.
        kwargs
            All other keyword arguments are passed to `~matplotlib.pyplot.plot`

        """
        if ax is None:
            ax = plt.gca()

        ax.plot(self.masked_wavelength, self.masked_flux, **kwargs)
        ax.set_xlim([self.masked_wavelength.value.min(),
                     self.masked_wavelength.value.max()])

    @property
    def masked_wavelength(self):
        if self.mask is not None:
            return self.wavelength[self.mask]
        else:
            return self.wavelength

    @property
    def masked_flux(self):
        if self.mask is not None:
            return self.flux[self.mask]
        else:
            return self.flux


class EchelleSpectrum(object):
    """
    Echelle spectrum of one or more spectral orders
    """
    def __init__(self, spectrum_list, header=None, name=None, fits_path=None,
                 time=None):
        self.spectrum_list = spectrum_list
        self.header = header
        self.name = name
        self.fits_path = fits_path
        self.standard_star_props = {}
        self.model_spectrum = None

        if header is not None and time is None:
            time = Time(header['JD'], format='jd')

        self.time = time

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
        return cls(spectrum_list, header=header, name=name, fits_path=path)
    
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

    def fit_order(self, spectral_order, polynomial_order, plots=False):
        """
        Fit a spectral order with a polynomial.

        Ignore fluxes near the CaII H & K wavelengths.

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

        mask_wavelengths = ((abs(spectrum.wavelength - true_h_centroid) > 6.5*u.Angstrom) &
                            (abs(spectrum.wavelength - true_k_centroid) > 6.5*u.Angstrom))

        fit_params = np.polyfit(spectrum.wavelength[mask_wavelengths] - mean_wavelength,
                                spectrum.flux[mask_wavelengths], polynomial_order)

        if plots:
            if spectral_order in [88, 89, 90, 91]:
                plt.figure()
                plt.plot(spectrum.wavelength, spectrum.flux)
                plt.plot(spectrum.wavelength[mask_wavelengths],
                         spectrum.flux[mask_wavelengths])
                plt.plot(spectrum.wavelength,
                         np.polyval(fit_params,
                                    spectrum.wavelength - mean_wavelength))
                plt.show()
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

    def continuum_normalize(self, standard_spectrum, polynomial_order,
                            only_orders=None, plot_masking=False):
        """
        Normalize the spectrum by a polynomial fit to the standard's
        spectrum.

        Parameters
        ----------
        standard_spectrum : `EchelleSpectrum`
            Spectrum of the standard object
        polynomial_order : int
            Fit the standard's spectrum with a polynomial of this order
        """

        # Copy some attributes of the standard star's EchelleSpectrum object into
        # a dictionary on the target star's EchelleSpectrum object.
        attrs = ['name', 'fits_path', 'header']
        for attr in attrs:
            self.standard_star_props[attr] = getattr(standard_spectrum, attr)

        if only_orders is None:
            only_orders = range(len(self.spectrum_list))

        for spectral_order in only_orders:
            # Extract one spectral order at a time to normalize
            standard_order = standard_spectrum.get_order(spectral_order)
            target_order = self.get_order(spectral_order)

            target_mask = get_spectrum_mask(standard_order, plot=plot_masking)

            # Fit the standard's flux in this order with a polynomial
            # fit_params = standard_spectrum.fit_order(spectral_order,
            #                                          polynomial_order)

            fit_params = standard_spectrum.fit_order(spectral_order,
                                                     polynomial_order)

            # Normalize the target's flux with the continuum fit from the standard
            target_continuum_fit = self.predict_continuum(spectral_order,
                                                          fit_params)
            #
            # if spectral_order in [88, 89, 90, 91]:
            #     plt.figure()
            #     plt.plot(target_continuum_fit)
            #     plt.plot(target_order.flux)
            #     plt.show()

            target_continuum_normalized_flux = target_order.flux / target_continuum_fit
            # target_continuum_normalized_flux /= np.median(target_continuum_normalized_flux)

            flux_80th_percentile = np.percentile(target_continuum_normalized_flux[target_mask], 80)
            target_continuum_normalized_flux /= flux_80th_percentile

            # if spectral_order in [88, 89, 90, 91]:
            #     fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
            #     # plt.plot(target_order.flux / target_continuum_fit)
            #     ax[0].plot(target_order.wavelength, target_continuum_normalized_flux)
            #     ax[0].plot(target_order.wavelength[target_mask],
            #              target_continuum_normalized_flux[target_mask])
            #
            #     yrange = [0, 3]
            #     ax[0].set_ylim(yrange)
            #
            #     ax[1].hist(target_continuum_normalized_flux, 70, range=yrange,
            #                orientation='horizontal', histtype='stepfilled',
            #                color='k')
            #
            #     for axis in ax:
            #         props = dict(ls='--', lw=2, color='r')
            #         axis.axhline(np.percentile(target_continuum_normalized_flux, 80), **props)
            #         # axis.axhline(np.percentile(target_continuum_normalized_flux, 90), **props)
            #
            #     plt.show()

            normalized_target_spectrum = Spectrum1D(target_continuum_normalized_flux,
                                                    target_order.wcs, mask=target_mask)

            # Replace this order's spectrum with the continuum-normalized one
            self.spectrum_list[spectral_order] = normalized_target_spectrum

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

    def rv_wavelength_shift(self, spectral_order, plot=False):
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

        model_slice = slice_spectrum(self.model_spectrum,
                                     order.masked_wavelength.min(),
                                     order.masked_wavelength.max(),
                                     norm=order.masked_flux.max())

        delta_lambda_obs = np.abs(np.diff(order.wavelength.value[0:2]))[0]
        delta_lambda_model = np.abs(np.diff(model_slice.wavelength.value[0:2]))[0]
        smoothing_kernel_width = delta_lambda_obs/delta_lambda_model

        interp_target_slice = interpolate_spectrum(order,
                                                   model_slice.wavelength)

        rv_shift = cross_corr(interp_target_slice, model_slice,
                              kernel_width=smoothing_kernel_width)

        if plot:
            plt.figure()
            plt.plot(order.masked_wavelength + rv_shift, order.masked_flux,
                     label='shifted spectrum')
            # plt.plot(interp_target_slice.wavelength, interp_target_slice.flux,
            #          label='spec interp')
            # plt.plot(model_slice.wavelength,
            #          gaussian_filter1d(model_slice.flux, smoothing_kernel_width),
            #          label='smoothed model')
            plt.plot(model_slice.wavelength,
                     gaussian_filter1d(model_slice.flux, smoothing_kernel_width),
                     label='smooth model')

            plt.legend()
            plt.show()

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
                                                target_order.wcs,
                                                mask=target_mask)

        normalized_spectrum_list.append(normalized_target_spectrum)
        
    return EchelleSpectrum(normalized_spectrum_list,
                           header=target_spectrum.header,
                           name=target_spectrum.name)


def slice_spectrum(spectrum, min_wavelength, max_wavelength, norm=None):
    """
    Return a slice of a spectrum on a smaller wavelength range.

    Parameters
    ----------
    spectrum : `Spectrum1D`
        Spectrum to slice.
    min_wavelength : `~astropy.units.Quantity`
        Minimum wavelength to include in new slice
    max_wavelength : `~astropy.units.Quantity`
        Maximum wavelength to include in new slice
    norm : float
        Normalize the new slice fluxes by ``norm`` divided by the maximum flux
        of the new slice.

    Returns
    -------
    sliced_spectrum : `Spectrum1D`
    """
    in_range = ((spectrum.wavelength < max_wavelength) &
                (spectrum.wavelength > min_wavelength))

    wavelength = spectrum.wavelength[in_range]

    if norm is None:
        flux = spectrum.flux[in_range]
    else:
        flux = spectrum.flux[in_range] * norm / spectrum.flux[in_range].max()

    return Spectrum1D.from_array(wavelength, flux,
                                 dispersion_unit=spectrum.wavelength_unit)


def interpolate_spectrum(spectrum, new_wavelengths):
    """
    Linearly interpolate a spectrum onto a new wavelength grid.

    Parameters
    ----------
    spectrum : `Spectrum1D`
        Spectrum to interpolate onto new wavelengths
    new_wavelengths : `~astropy.units.Quantity`
        New wavelengths to interpolate the spectrum onto

    Returns
    -------
    interp_spec : `Spectrum1D`
        Interpolated spectrum.
    """
    # start_ind = np.argmin(np.abs(start_wavelength - spectrum.wavelength))
    # end_ind = np.argmin(np.abs(end_wavelength - spectrum.wavelength))
    # print(start_ind, end_ind)
    #
    # if end_ind < start_ind:
    #     start_ind, end_ind = end_ind, start_ind
    #
    # return spectrum.slice_index(start_ind, end_ind)
    sort_order = np.argsort(spectrum.masked_wavelength.to(u.Angstrom).value)
    sorted_spectrum_wavelengths = spectrum.masked_wavelength.to(u.Angstrom).value[sort_order]
    sorted_spectrum_fluxes = spectrum.masked_flux.value[sort_order]

    new_flux = np.interp(new_wavelengths.to(u.Angstrom).value,
                         sorted_spectrum_wavelengths,
                         sorted_spectrum_fluxes)

    return Spectrum1D.from_array(new_wavelengths, new_flux,
                                 dispersion_unit=spectrum.wavelength_unit)


def cross_corr(target_spectrum, model_spectrum, kernel_width):
    """
    Cross correlate an observed spectrum with a model.

    Convolve the model with a Gaussian kernel.

    Parameters
    ----------
    target_spectrum : `Spectrum1D`
        Observed spectrum of star
    model_spectrum : `Spectrum1D`
        Model spectrum of star
    kernel_width : float
        Smooth the model spectrum with a kernel of this width, in units of the
        wavelength step size in the model
    Returns
    -------
    wavelength_shift : `~astropy.units.Quantity`
        Wavelength shift required to shift the target spectrum to the rest-frame
    """

    smoothed_model_flux = gaussian_filter1d(model_spectrum.masked_flux.value,
                                            kernel_width)

    corr = np.correlate(target_spectrum.masked_flux.value,
                        smoothed_model_flux, mode='same')

    max_corr_ind = np.argmax(corr)
    index_shift = corr.shape[0]/2 - max_corr_ind

    # delta_wavelength = np.abs(target_spectrum.masked_wavelength[1] -
    #                           target_spectrum.masked_wavelength[0])

    delta_wavelength = np.median(np.abs(np.diff(target_spectrum.masked_wavelength)))

    wavelength_shift = index_shift * delta_wavelength
    return wavelength_shift



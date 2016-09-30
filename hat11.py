from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from toolkit import (EchelleSpectrum, continuum_normalize, fit_emission_feature,
                     plot_spectrum_for_s_index, true_h_centroid,
                     true_k_centroid, uncalibrated_s_index)

root_dir = '/media/PASSPORT/APO/Q3UW04/'
dates = ['UT160703', 'UT160706', 'UT160707', 'UT160709', 'UT160918']
standard = ['BD28_4211', 'BD28_4211', 'BD28_4211', 'BD28_4211', 'hr6943']

all_normalized_spectra = []

approx_k = 3932.5 * u.Angstrom
approx_h = 3967.5 * u.Angstrom

date_index = 0
for date_index in range(len(dates)):
    data_dir = os.path.join(root_dir, dates[date_index])
    hat11_spectra_paths = (glob(os.path.join(data_dir, 'HAT*.wfrmcpc.fits')) +
                           glob(os.path.join(data_dir, 'hat*.wfrmcpc.fits')))
    standard_spectra_paths = glob(os.path.join(data_dir,
                                               "{0}*.wfrmcpc.fits"
                                               .format(standard[date_index])))
    for spectrum_index in range(len(hat11_spectra_paths)):

        # Skip one bad observation:
        if not (spectrum_index == 3 and dates[date_index] == 'UT160707'):

            hat11_spectrum = EchelleSpectrum.from_fits(hat11_spectra_paths[spectrum_index])
            standard_spectrum = EchelleSpectrum.from_fits(standard_spectra_paths[0])

            normed_spectrum = continuum_normalize(hat11_spectrum, standard_spectrum,
                                                  polynomial_order=8)
            all_normalized_spectra.append(normed_spectrum)

# Fit the H & K features to refine wavelength solution:

times = []

for spectrum in all_normalized_spectra:
    time, ew, params = fit_emission_feature(spectrum, approx_k, 90,
                                            name='CaII K', plot=False,
                                            use_right_wings=True,
                                            background_width=8*u.Angstrom)
    spectrum.offset_wavelength_solution(90, true_k_centroid -
                                        params['x_0_0']*u.angstrom)

    time, ew, params = fit_emission_feature(spectrum, approx_h, 89,
                                            name='CaII H', plot=False,
                                            use_right_wings=True,
                                            background_width=8*u.Angstrom)
    spectrum.offset_wavelength_solution(89, true_h_centroid -
                                        params['x_0_0']*u.angstrom)
    times.append(time)

plot_spectrum_for_s_index(all_normalized_spectra)

s = []
for spectrum in all_normalized_spectra:
    s.append(uncalibrated_s_index(spectrum))

from astropy.time import Time
times = Time(times)

plt.plot_date(times.plot_date, s)
plt.show()
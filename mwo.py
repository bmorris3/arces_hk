"""
HR 6943       18:29:35  +23:51:58   5.0  B6 telluric standard - done
KIC 12418816  19:45:46	+51:13:27  12.9  K EB for RV - done
KIC 9652680   19:33:05  +46:19:09  11.5  G super flare spots - done
HAT-P-11      19:50:50  +48:04:51   9.5  K4 spots/planet - done

"HD 187642"   19:50:46  +08:52:02 this is a delta Scuti, A7 star??
"HD 192577"   20:13:37  +46:44:28  can't separate this binary
"HD 192578"   20:13:37  +46:44:28     "
"HD 194093"   20:22:13  +40:15:24  F8 Ib - skipping

"HD 201251"   21:06:36  +47:38:54   S(mean) 0.34 - done
"HD 217906"   23:03:46  +28:04:57   0.23 - done
"HD 218356"   23:07:06  +25:28:05   0.68 - done
"HD 222107"   23:37:33  +46:27:33   1.28 - done

HD 210905 22:11:56.89057 +59:05:04.4907 K0III V=6.296 S(mean)=0.1154 - done
HD 220182 23:21:36.51306 +44:05:52.3818 G9V V=7.36 S(mean)=0.4545 - done
GJ 9781A 22:24:45.526 +22:33:03.85 K7 V=9.02 S(mean)=0.8954 - done
HR 8781 23:04:46  15:12:19  B9V V=2.5 telluric standard - done
"""

from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from toolkit import (EchelleSpectrum, continuum_normalize, fit_emission_feature,
                     plot_spectrum_for_s_index, true_h_centroid,
                     true_k_centroid, uncalibrated_s_index)

root_dir = '/media/PASSPORT/APO/Q3UW04/'
dates = ['UT160918']
standard = ['hr6943']

target_names = ['hd201251', 'hd217906', 'hd218356', 'hd222107',
                'hd210905', 'hd220182', 'gj9781a', 'hr8781']

all_normalized_spectra = []

approx_k = 3933.6 * u.Angstrom
approx_h = 3968.0 * u.Angstrom

date_index = 0
for date_index in range(len(dates)):
    data_dir = os.path.join(root_dir, dates[date_index])

    spectra_paths = reduce(list.__add__,
                           [glob(os.path.join(data_dir,
                                       '{0}*.wfrmcpc.fits'.format(name)))
                            for name in target_names])

    standard_spectra_paths = glob(os.path.join(data_dir,
                                               "{0}*.wfrmcpc.fits"
                                               .format(standard[date_index])))
    for spectrum_index in range(len(spectra_paths)):

        hat11_spectrum = EchelleSpectrum.from_fits(spectra_paths[spectrum_index])
        standard_spectrum = EchelleSpectrum.from_fits(standard_spectra_paths[0])

        normed_spectrum = continuum_normalize(hat11_spectrum, standard_spectrum,
                                              polynomial_order=8)
        all_normalized_spectra.append(normed_spectrum)

# Fit the H & K features to refine wavelength solution:

times = []

for spectrum in all_normalized_spectra:
    time, ew, params = fit_emission_feature(spectrum, approx_k, 90,
                                            name='CaII K', plot=True,
                                            use_right_wings=True,
                                            background_width=8*u.Angstrom)
    spectrum.offset_wavelength_solution(90, true_k_centroid -
                                        params['x_0_0']*u.angstrom)

    time, ew, params = fit_emission_feature(spectrum, approx_h, 89,
                                            name='CaII H', plot=True,
                                            use_right_wings=True,
                                            background_width=8*u.Angstrom)
    spectrum.offset_wavelength_solution(89, true_h_centroid -
                                        params['x_0_0']*u.angstrom)
    times.append(time)
    plt.show()

plot_spectrum_for_s_index(all_normalized_spectra)

s = []
for spectrum in all_normalized_spectra:
    s.append(uncalibrated_s_index(spectrum))

from astropy.time import Time
times = Time(times)

plt.plot_date(times.plot_date, s)
plt.show()
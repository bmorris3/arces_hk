from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from toolkit import (EchelleSpectrum, glob_spectra_paths, uncalibrated_s_index,
                     Star)

root_dir = '/run/media/bmmorris/PASSPORT/APO/Q3UW04/'
dates = ['UT160918']
standards = ['hr6943']
# hd222107 seems to have an anamolously low S_apo
target_names = ['hd201251', 'hd217906', 'hd218356', #'hd222107',
                'hd210905', 'hd220182', 'gj9781a']

all_spectra = []
stars = []

approx_k = 3933.6 * u.Angstrom
approx_h = 3968.25 * u.Angstrom

fig, ax = plt.subplots(1, 2, figsize=(14, 7))

for date_name, standard_name in zip(dates, standards):
    data_dir = os.path.join(root_dir, date_name)

    spectra_paths = glob_spectra_paths(data_dir, target_names)

    standard_spectra_paths = glob(os.path.join(data_dir,
                                               "{0}*.wfrmcpc.fits"
                                               .format(standard_name)))

    for spectrum_path in spectra_paths:
        target_spectrum = EchelleSpectrum.from_fits(spectrum_path)
        standard_spectrum = EchelleSpectrum.from_fits(standard_spectra_paths[0])

        target_spectrum.continuum_normalize(standard_spectrum,
                                            polynomial_order=8,
                                            plot_masking=False)

        rv_shifts = u.Quantity([target_spectrum.rv_wavelength_shift(order)
                                for order in range(81, 91)])
        median_rv_shift = np.median(rv_shifts)

        target_spectrum.offset_wavelength_solution(median_rv_shift)

        s89 = target_spectrum.get_order(89)
        # plt.plot(target_spectrum.model_spectrum.wavelength,
        #          target_spectrum.model_spectrum.flux * s.flux[s.mask].max() /
        #          target_spectrum.model_spectrum.flux.max())

        s90 = target_spectrum.get_order(90)
        s89.plot(ax=ax[0], label=target_spectrum.name)
        s90.plot(ax=ax[1], label=target_spectrum.name)

        # plt.legend()
        # plt.show()

        all_spectra.append(target_spectrum)

        s_apo = uncalibrated_s_index(target_spectrum)

        star = Star(name=target_spectrum.name, s_apo=s_apo)
        stars.append(star)

ax[0].set_ylabel('Flux')
ax[1].set_ylabel('Flux')
ax[1].set_xlabel('Wavelength [Angstroms]')
ax[0].set_xlabel('Wavelength [Angstroms]')
ax[0].set_title('CaII H')
ax[1].set_title('CaII K')
ax[0].set_xlim([3960, 3980])
ax[1].set_xlim([3925, 3945])

for axis in ax:
    axis.set_ylim([0, 5])
    axis.legend()
plt.show()

# from toolkit.utils import construct_standard_star_table
# construct_standard_star_table(target_names)


s_mwo = np.array([s.s_mwo for s in stars])
s_apo = np.array([s.s_apo.uncalibrated for s in stars])

Xdata = np.vander(s_apo, 2)
ydata = s_mwo
theta_best, resid, rank, singvals = np.linalg.lstsq(Xdata, ydata)

best_model = theta_best[0] * s_apo + theta_best[1]

plt.text(0.015, 0.7, "c1 = {0:.2f}, \nc2 = {1:.2f}".format(*theta_best))

plt.plot(s_apo, s_mwo, '.')
plt.plot(s_apo, best_model, 'r')
plt.xlabel('APO')
plt.ylabel('MWO')
plt.savefig('plots/s-index_calibration.png', bbox_inches='tight', dpi=200)
plt.show()

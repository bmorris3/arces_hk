from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from toolkit import (EchelleSpectrum, glob_spectra_paths, uncalibrated_s_index,
                     StarProps, stars_to_json)

root_dir = '/Users/bmmorris/data/Q3UW04'
dates = ['UT160703', 'UT160706', 'UT160707', 'UT160709', 'UT160918']
standards = ['BD28_4211', 'BD28_4211', 'BD28_4211', 'BD28_4211', 'hr6943']


# hd222107 seems to have an anamolously low S_apo
target_names = ['hat', 'HAT']

all_spectra = []
stars = []

fig, ax = plt.subplots(2, 2, figsize=(12, 10))

for date_name, standard_name in zip(dates, standards):
    data_dir = os.path.join(root_dir, date_name)

    spectra_paths = glob_spectra_paths(data_dir, target_names)

    standard_spectra_paths = glob(os.path.join(data_dir,
                                               "{0}*.wfrmcpc.fits"
                                               .format(standard_name)))

    for spectrum_path in spectra_paths:
        target_spectrum = EchelleSpectrum.from_fits(spectrum_path)
        standard_spectrum = EchelleSpectrum.from_fits(standard_spectra_paths[0])

        only_orders = list(range(81, 91+1))
        target_spectrum.continuum_normalize(standard_spectrum,
                                            polynomial_order=10,
                                            only_orders=only_orders,
                                            plot_masking=False)

        rv_shifts = u.Quantity([target_spectrum.rv_wavelength_shift(order)
                                for order in only_orders])
        median_rv_shift = np.median(rv_shifts)

        target_spectrum.offset_wavelength_solution(median_rv_shift)

        order_h = target_spectrum.get_order(89)
        order_k = target_spectrum.get_order(90)
        order_h.plot(ax=ax[0, 0], label=target_spectrum.time.datetime.date())
        order_k.plot(ax=ax[0, 1], label=target_spectrum.time.datetime.date())

        order_r = target_spectrum.get_order(91)
        order_v = target_spectrum.get_order(88)

        order_r.plot(ax=ax[1, 0], label=target_spectrum.time.datetime.date())
        order_v.plot(ax=ax[1, 1], label=target_spectrum.time.datetime.date())

        all_spectra.append(target_spectrum)

        s_apo = uncalibrated_s_index(target_spectrum)

        star = StarProps(name=target_spectrum.name, s_apo=s_apo,
                         time=target_spectrum.time)
        stars.append(star)

stars_to_json(stars, output_path='hat11.json')

ax[0, 0].set_ylabel('Flux')
ax[1, 0].set_ylabel('Flux')
ax[-1, 1].set_xlabel('Wavelength [Angstroms]')
ax[-1, 0].set_xlabel('Wavelength [Angstroms]')
ax[0, 0].set_title('CaII H')
ax[0, 1].set_title('CaII K')
ax[0, 0].set_xlim([3960, 3980])
ax[0, 1].set_xlim([3925, 3945])

ax[1, 0].set_title('R (pseudocontinuum)')
ax[1, 1].set_title('V (pseudocontinuum)')
ax[1, 0].set_xlim([3890, 3910])
ax[1, 1].set_xlim([3990, 4010])

for axis in ax[0, :]:
    axis.set_ylim([0, 2.0])
    axis.legend(fontsize=8)

fig.savefig('plots/spectra_hat11.png', bbox_inches='tight', dpi=200)

plt.show()


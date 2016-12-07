from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from toolkit import (EchelleSpectrum, glob_spectra_paths, uncalibrated_s_index,
                     StarProps, Measurement, stars_to_json)

root_dir = '/Users/bmmorris/data/'
dates = ['UT160918', 'UT161202']
standards = ['hr6943', 'HR3454'] # ['hr6943']

# Night: UT160918
# hd222107 seems to have an anamolously low S_apo
target_names = ['hd201251', 'hd217906', 'hd218356', #'hd222107',
                'hd210905', 'hd220182', 'gj9781a']

# Night: UT161201
target_names += ['HD41593', 'HD45088', 'HD68017', 'HD34411', 'HD39587',
                 'HD86728']

# target_names = ['HD34411', 'HD86728']

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
        order_h.plot(ax=ax[0, 0], label=target_spectrum.name)
        order_k.plot(ax=ax[0, 1], label=target_spectrum.name)

        order_r = target_spectrum.get_order(91)
        order_v = target_spectrum.get_order(88)

        order_r.plot(ax=ax[1, 0], label=target_spectrum.name)
        order_v.plot(ax=ax[1, 1], label=target_spectrum.name)

        all_spectra.append(target_spectrum)

        s_apo = uncalibrated_s_index(target_spectrum)

        star = StarProps(name=target_spectrum.name, s_apo=s_apo,
                         time=target_spectrum.time)
        stars.append(star)

stars_to_json(stars, output_path='mwo_stars.json')

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
    axis.set_ylim([0, 3.5])
    axis.legend(fontsize=8)

fig.savefig('plots/spectra.png', bbox_inches='tight', dpi=200)
plt.show()

# from toolkit.utils import construct_standard_star_table
# construct_standard_star_table(target_names)

s_mwo = Measurement([s.s_mwo.value for s in stars],
                    err=[s.s_mwo.err for s in stars])

s_apo = Measurement([s.s_apo.uncalibrated.value for s in stars],
                    err=[s.s_apo.uncalibrated.err for s in stars])

Xdata = np.vander(s_apo.value, 2)
ydata = s_mwo.value
theta_best, resid, rank, singvals = np.linalg.lstsq(Xdata, ydata)

best_model = theta_best[0] * s_apo.value + theta_best[1]

plt.text(0.015, 0.7, "c1 = {0:.2f}, \nc2 = {1:.2f}".format(*theta_best))

# for s in stars:
    # plt.text(s.s_apo.uncalibrated, s.s_mwo.value, s.name)

plt.errorbar(s_apo.value, s_mwo.value,
             xerr=s_apo.err,
             yerr=s_mwo.err,
             fmt='.', color='k')
plt.plot(s_apo.value, best_model, 'r')
plt.xlabel('APO')
plt.ylabel('MWO')
plt.savefig('plots/s-index_calibration.png', bbox_inches='tight', dpi=200)
plt.show()





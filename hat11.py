from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from toolkit import (EchelleSpectrum, glob_spectra_paths, uncalibrated_s_index,
                     StarProps, stars_to_json)

# Note that not all observations in this dir are actually from Q3
root_dir = '/Users/bmmorris/data/Q3UW04'
dates = ['UT160703', 'UT160706', 'UT160707', 'UT160709', 'UT160918',
         'UT170411', 'UT170612']
standard_path = ('/Users/bmmorris/data/Q3UW04/UT160706/'
                 'BD28_4211.0034.wfrmcpc.fits')
# dates = ['UT170411']
# standards = ['HR5501']

from astropy.time import Time
date_min = Time('2016-07-03')
date_max = Time('2017-04-11')
colormap = lambda x: plt.cm.winter(float(x - date_min.jd) /
                                   (date_max.jd-date_min.jd))

# hd222107 seems to have an anamolously low S_apo
target_names = ['hat', 'HAT']

all_spectra = []
stars = []

fig, ax = plt.subplots(2, 2, figsize=(12, 10))

for date_name in dates:
    data_dir = os.path.join(root_dir, date_name)

    spectra_paths = glob_spectra_paths(data_dir, target_names)

    for spectrum_path in spectra_paths:
        target_spectrum = EchelleSpectrum.from_fits(spectrum_path)
        standard_spectrum = EchelleSpectrum.from_fits(standard_path)

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
        date_label = target_spectrum.time.datetime.date()
        plot_kwargs = dict(label=date_label,
                           color=colormap(target_spectrum.time.jd),
                           alpha=0.7)
        order_h.plot(ax=ax[0, 0], **plot_kwargs)
        order_k.plot(ax=ax[0, 1], **plot_kwargs)

        order_r = target_spectrum.get_order(91)
        order_v = target_spectrum.get_order(88)

        order_r.plot(ax=ax[1, 0], **plot_kwargs)
        order_v.plot(ax=ax[1, 1], **plot_kwargs)

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


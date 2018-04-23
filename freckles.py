import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.table import Table
from toolkit import (EchelleSpectrum, glob_spectra_paths, uncalibrated_s_index,
                     StarProps, stars_to_json, h_alpha_ew)
import json

# Note that not all observations in this dir are actually from Q3
root_dir = '/Users/bmmorris/git/nephelion/data/wavelength_calibrated/'
standard_path = ('/Users/bmmorris/data/Q3UW04/UT160706/'
                 'BD28_4211.0034.wfrmcpc.fits')

date_min = Time('2017-12-04')
date_max = Time('2017-12-31')
colormap = lambda x: plt.cm.winter(float(x - date_min.jd) /
                                   (date_max.jd-date_min.jd))

def table_to_python(table):
    """Convert Astropy Table to Python dict.

    Numpy arrays are converted to lists, so that
    the output is JSON serialisable.

    Can work with multi-dimensional array columns,
    by representing them as list of list.
    """
    total_data = {}
    for name in table.colnames:
        data = table[name].tolist()
        total_data[name] = data
    return total_data

targets = ['stack_EPIC211928486', 'stack_EPIC211966629']
target_names = ['EPIC211928486', 'EPIC211966629']

for target, target_name in zip(targets, target_names):

    all_spectra = []
    stars = []

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    spectra_paths = glob_spectra_paths(root_dir, target)
    print(spectra_paths)
    ha_eqw = []

    for spectrum_path in sorted(spectra_paths):
        target_spectrum = EchelleSpectrum.from_fits(spectrum_path)
        standard_spectrum = EchelleSpectrum.from_fits(standard_path)

        only_orders = list(range(81, 91+1))
        target_spectrum.continuum_normalize(standard_spectrum,
                                            polynomial_order=10,
                                            only_orders=only_orders,
                                            plot_masking=False)

        rv_shifts = u.Quantity([target_spectrum.rv_wavelength_shift(order,
                                                                    T_eff=4700)
                                for order in only_orders])
        median_rv_shift = np.median(rv_shifts)

        target_spectrum.offset_wavelength_solution(median_rv_shift)

        order_h = target_spectrum.get_order(89)
        order_k = target_spectrum.get_order(90)
        date_label = target_spectrum.time.datetime.date()
        plot_kwargs = dict(label=date_label,
                           color=colormap(target_spectrum.time.jd),
                           alpha=0.7, lw=1)
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

        ew = h_alpha_ew(target_spectrum)
        ha_eqw.append(ew.value)


    json.dump(dict(time=[s.time.jd for s in all_spectra], ha=ha_eqw),
             open('data/{0}_ha.json'.format(target_name), 'w'))
    print(dict(time=[s.time.jd for s in all_spectra], ha=ha_eqw))

    stars_to_json(stars, output_path='data/{0}.json'.format(target_name))

    ax[0, 0].set_ylabel('Flux')
    ax[1, 0].set_ylabel('Flux')
    ax[-1, 1].set_xlabel('Wavelength [Angstroms]')
    ax[-1, 0].set_xlabel('Wavelength [Angstroms]')
    ax[0, 0].set_title('CaII H')
    ax[0, 1].set_title('CaII K')
    #ax[0, 0].set_xlim([3960, 3968])
    #ax[0, 1].set_xlim([3926, 3934])
    cah = 3968.468
    cak = 3933.6614
    window = 1.5
    ax[0, 0].set_xlim([cah - window, cah + window])
    ax[0, 1].set_xlim([cak - window, cak + window])

    ax[1, 0].set_title('R (pseudocontinuum)')
    ax[1, 1].set_title('V (pseudocontinuum)')
    #ax[1, 0].set_xlim([3890, 3910])
    #ax[1, 1].set_xlim([3990, 4010])

    for axis in ax[0, :]:
        axis.set_ylim([0, 3.5])
        axis.legend(fontsize=8)

    fig.savefig('plots/spectra_{0}.png'.format(target_name),
                bbox_inches='tight', dpi=200)
    fig.savefig('plots/spectra_{0}.pdf'.format(target_name),
                bbox_inches='tight', dpi=200)

    plt.show()


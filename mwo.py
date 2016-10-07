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

from toolkit import (EchelleSpectrum, glob_spectra_paths, uncalibrated_s_index,
                     Star)

root_dir = '/run/media/bmmorris/PASSPORT/APO/Q3UW04/'
dates = ['UT160918']
standards = ['hr6943']
target_names = ['hd201251', 'hd217906', 'hd218356', 'hd222107',
                'hd210905', 'hd220182', 'gj9781a']

all_spectra = []
stars = []

approx_k = 3933.6 * u.Angstrom
approx_h = 3968.25 * u.Angstrom

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

        s = target_spectrum.get_order(89)
        # plt.plot(target_spectrum.model_spectrum.wavelength,
        #          target_spectrum.model_spectrum.flux * s.flux[s.mask].max() /
        #          target_spectrum.model_spectrum.flux.max())
        # s.plot()
        # plt.legend()
        # plt.show()

        all_spectra.append(target_spectrum)

        s_apo = uncalibrated_s_index(target_spectrum)

        star = Star(name=target_spectrum.name, s_apo=s_apo)
        stars.append(star)

from astropy.time import Time
# times = Time(times)

#plt.plot_date(times.plot_date)

s_mwo = [s.s_mwo for s in stars]
s_apo = [s.s_apo for s in stars]

plt.plot(s_mwo, s_apo, '.')
plt.xlabel('MWO')
plt.ylabel('APO')
plt.show()
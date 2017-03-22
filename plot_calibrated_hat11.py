import matplotlib.pyplot as plt
import numpy as np

from toolkit import (json_to_stars, Measurement, stars_to_json, FitParameter,
                     parse_hires, StarProps)

props = dict(fontsize=15)

hat11 = json_to_stars('hat11.json')

f = FitParameter.from_text('calibrated_f.txt')
c1 = FitParameter.from_text('calibrated_c1.txt')
c2 = FitParameter.from_text('calibrated_c2.txt')

hat11_s_apo = Measurement([s.s_apo.uncalibrated.value for s in hat11],
                          err=[s.s_apo.uncalibrated.err for s in hat11],
                          time=[s.s_apo.time.jd for s in hat11])

##############################################################################
# Solve for HAT-P-11 S-indices:
hat11_s_mwo_err = np.sqrt((f.value * hat11_s_apo.value * c1.err_lower)**2 +
                          (c1.value * f.value * hat11_s_apo.err)**2 +
                          c2.err_lower**2)

hat11_s_mwo = Measurement(c1.value * hat11_s_apo.value + c2.value,
                          err=hat11_s_mwo_err,
                          time=hat11_s_apo.time)

hat11_apo_calibrated = [StarProps(name='HAT-P-11', s_apo=sapo, s_mwo=smwo,
                                  time=sapo.time)
                        for sapo, smwo in zip(hat11_s_apo, hat11_s_mwo)]

stars_to_json(hat11_apo_calibrated, 'hat11_apo_calibrated.json')

#############################################################################
# Plot the S-index time series for HAT-P-11 with HIRES data too
hires = parse_hires('hat-p-11_svals.txt')

hat11_s_mwo_mean = c1.value * hat11_s_apo.value.mean() + c2.value
hat11_s_apo_err_mean = np.sqrt(np.sum(hat11_s_apo.err**2/len(hat11_s_apo.err)**2))
hat11_s_mwo_mean_err = np.sqrt((hat11_s_apo.value.mean() * c1.err_lower)**2 +
                               (c1.value * hat11_s_apo_err_mean)**2 +
                               c2.err_lower**2)

rm = (hires['time'].decimalyear < 2010.7) & (hires['time'].decimalyear > 2010.6)
rough_hires_err = np.std(hires[rm]['S-value'])

hat11_s_keck = Measurement(value=hires['S-value'].data,
                           err=rough_hires_err * np.ones(len(hires['S-value'].data)),
                           time=hires['time'])

hat11_keck_calibrated = [StarProps(name='HAT-P-11', s_mwo=s, time=s.time)
                         for s in hat11_s_keck]
stars_to_json(hat11_keck_calibrated, 'hat11_keck_calibrated.json')

fig, ax = plt.subplots()
ax.axvspan(2009.4, 2013.4, color='k', alpha=0.1, label='Kepler')

ax.errorbar(hires['time'].decimalyear, hires['S-value'], yerr=rough_hires_err,
            color='k', fmt='.', label='Keck/HIRES', capsize=0,
            ecolor='gray')
ax.errorbar(hat11_s_apo.time.decimalyear, hat11_s_mwo.value, hat11_s_mwo.err,
            fmt='.', color='gray', capsize=0, label='ARC 3.5m/ARCES')
ax.errorbar(hat11_s_apo.time.decimalyear.mean(), hat11_s_mwo_mean,
            hat11_s_mwo_mean_err, fmt='s', color='r', capsize=0, markersize=10,
            elinewidth=4, label='ARC 3.5m/ARCES (mean)')
ax.set_xlabel('Date', **props)
ax.set_ylabel('$S$-index', **props)
ax.set_title('CaII H & K', **props)
ax.legend(loc='lower left', numpoints=1)
fig.savefig('plots/s-index_hat11.png', bbox_inches='tight', dpi=200)
plt.show()

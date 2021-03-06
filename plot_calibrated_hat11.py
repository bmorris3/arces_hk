import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from astropy.table import Table, Column

from toolkit import (json_to_stars, Measurement, stars_to_json, FitParameter,
                     parse_hires, StarProps)

props = dict(fontsize=15)

hat11 = json_to_stars('data/hat11.json')

f = FitParameter.from_text('calibration_constants/calibrated_f.txt')
c1 = FitParameter.from_text('calibration_constants/calibrated_c1.txt')
c2 = FitParameter.from_text('calibration_constants/calibrated_c2.txt')

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

stars_to_json(hat11_apo_calibrated, 'data/hat11_apo_calibrated.json')

#############################################################################
# Write out results to table

t = Column([o.s_apo.time.jd for o in hat11_apo_calibrated])
e = Column([o.s_mwo.err for o in hat11_apo_calibrated])
m = Column([o.s_mwo.value for o in hat11_apo_calibrated])
table = Table([t, m, e], names=['JD', '$S_{APO}$', 'Uncertainty'])
table.sort("JD")
formats={"JD": "%.4f", "$S_{APO}$":"%.2f", "Uncertainty":"%.2f"}
ascii.write(table, output='hat11_s_mwo.tex', format='latex', formats=formats,
            overwrite=True)

#############################################################################
# Plot the S-index time series for HAT-P-11 with HIRES data too
hires = parse_hires('hat-p-11_svals.txt')

last_week_inds = np.argsort(hat11_s_apo.time.jd)[-8:]
hat11_s_mwo_mean = c1.value * hat11_s_apo.value[last_week_inds].mean() + c2.value
hat11_s_apo_err_mean = np.sqrt(np.sum(hat11_s_apo.err[last_week_inds]**2 /
                                      len(last_week_inds)**2))
hat11_s_mwo_mean_err = np.sqrt((hat11_s_apo.value[last_week_inds].mean() *
                                c1.err_lower)**2 +
                               (c1.value * hat11_s_apo_err_mean)**2 +
                               c2.err_lower**2)

# Estimate Keck uncertainty from scatter during rapid repeated RM observations
rm = (hires['time'].decimalyear < 2010.7) & (hires['time'].decimalyear > 2010.6)
rough_hires_err = np.std(hires[rm]['S-value'])

hat11_s_keck = Measurement(value=hires['S-value'].data,
                           err=rough_hires_err * np.ones(len(hires['S-value'].data)),
                           time=hires['time'])

hat11_keck_calibrated = [StarProps(name='HAT-P-11', s_mwo=s, time=s.time)
                         for s in hat11_s_keck]
stars_to_json(hat11_keck_calibrated, 'data/hat11_keck_calibrated.json')

fig, ax = plt.subplots()
ax.axvspan(2009.4, 2013.4, color='k', alpha=0.1, label='Kepler')

ax.errorbar(hat11_s_apo.time.decimalyear, hat11_s_mwo.value, hat11_s_mwo.err/2,
            fmt='s', markersize=3, color='r', capsize=0,
            ecolor='gray', label='ARC 3.5m/ARCES')

ax.errorbar(hires['time'].decimalyear, hires['S-value'], yerr=rough_hires_err,
            color='k', fmt='.', label='Keck/HIRES', capsize=0,
            ecolor='gray')

ax.set_xticks(range(2008, 2020))
ax.set_xlabel('Date', **props)
ax.set_ylabel('$S$-index', **props)
ax.grid(ls=':', color='gray')
ax.set_title('CaII H & K', **props)
ax.legend(loc='lower left', numpoints=1)
fig.savefig('plots/s-index_hat11.png', bbox_inches='tight', dpi=200)
fig.savefig('plots/s-index_hat11.pdf', bbox_inches='tight', dpi=200)
plt.show()

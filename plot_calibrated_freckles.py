import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from astropy.table import Table, Column
from astropy.time import Time
import astropy.units as u
import json
from toolkit import (json_to_stars, Measurement, stars_to_json, FitParameter,
                     StarProps)

props = dict(fontsize=12)
fig, ax = plt.subplots(2, 1, figsize=(4, 6), sharex=True)

target_names = ['EPIC211928486', 'EPIC211966629']
colors = ['DodgerBlue', 'r']

ha_EPIC211966629 = json.load(open('data/EPIC211966629_ha.json'))
ha_EPIC211928486 = json.load(open('data/EPIC211928486_ha.json'))
decimal_year_6629 = (Time(sorted(ha_EPIC211966629['time']), format='jd') -
                     Time(min(ha_EPIC211966629['time']), format='jd')).to(u.day).value
decimal_year_8486 = (Time(sorted(ha_EPIC211928486['time']), format='jd') -
                     Time(min(ha_EPIC211928486['time']), format='jd')).to(u.day).value

days_since = min(ha_EPIC211928486['time'] + ha_EPIC211966629['time'])

for target, color, times in zip(target_names, colors,
                                [decimal_year_8486, decimal_year_6629]):
    starprops = json_to_stars('data/{0}.json'.format(target))

    f = FitParameter.from_text('calibration_constants/calibrated_f.txt')
    c1 = FitParameter.from_text('calibration_constants/calibrated_c1.txt')
    c2 = FitParameter.from_text('calibration_constants/calibrated_c2.txt')

    target_s_apo = Measurement([s.s_apo.uncalibrated.value for s in starprops],
                               err=[s.s_apo.uncalibrated.err for s in starprops],
                               time=[s.s_apo.time.jd for s in starprops])

    ##############################################################################
    # Solve for HAT-P-11 S-indices:
    target_s_mwo_err = np.sqrt((f.value * target_s_apo.value * c1.err_lower)**2 +
                               (c1.value * f.value * target_s_apo.err)**2 +
                                c2.err_lower**2)

    target_s_mwo = Measurement(c1.value * target_s_apo.value + c2.value,
                               err=target_s_mwo_err,
                               time=target_s_apo.time)

    target_apo_calibrated = [StarProps(name=target, s_apo=sapo, s_mwo=smwo,
                                       time=sapo.time)
                             for sapo, smwo in zip(target_s_apo, target_s_mwo)]

    stars_to_json(target_apo_calibrated,
                  'data/{0}_apo_calibrated.json'.format(target))

    #############################################################################
    # Write out results to table
    t = Column([o.s_apo.time.jd for o in target_apo_calibrated])
    e = Column([o.s_mwo.err for o in target_apo_calibrated])
    m = Column([o.s_mwo.value for o in target_apo_calibrated])
    table = Table([t, m, e], names=['JD', '$S_{APO}$', 'Uncertainty'])
    table.sort("JD")
    formats = {"JD": "%.4f", "$S_{APO}$":"%.2f", "Uncertainty":"%.2f"}
    ascii.write(table, output='{0}_s_mwo.tex'.format(target),
                format='latex', formats=formats, overwrite=True)

    #print(times.shape, target_s_mwo.value)

    ax[0].errorbar(times, target_s_mwo.value,
                   target_s_mwo.err/2, fmt='s', markersize=6, color=color,
                   capsize=0, ecolor='gray', label=target)

# plot H-alpha
ax[1].scatter(decimal_year_6629, ha_EPIC211966629['ha'], color='r', s=25)
ax[1].scatter(decimal_year_8486, ha_EPIC211928486['ha'], color='DodgerBlue', s=25)

for axis in ax:
    axis.grid(ls=':', color='gray')

ax[-1].set_xlabel('Days since {}'.format(
                  Time(days_since, format='jd').datetime.date()), **props)
ax[0].set_ylabel('$S$-index', **props)
ax[1].set_ylabel(r'EW(H$\alpha$) [$\AA$]', **props)


ax[0].set_title('CaII H & K', **props)
ax[0].legend(loc='lower right', numpoints=1, fontsize=8)
# fig.savefig('plots/s-index_hat11.png', bbox_inches='tight', dpi=200)
fig.tight_layout()
fig.savefig('plots/s-index_EPIC.pdf', bbox_inches='tight', dpi=200)
plt.show()

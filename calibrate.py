import matplotlib.pyplot as plt
import numpy as np
from corner import corner

from toolkit import (json_to_stars, Measurement, mcmc_fit, stars_to_json,
                     FitParameter, parse_hires, StarProps)

props = dict(fontsize=15)

calstars = json_to_stars('mwo_stars.json')
hat11 = json_to_stars('hat11.json')

cals_s_mwo = Measurement([s.s_mwo.value for s in calstars],
                         err=[s.s_mwo.err for s in calstars],
                         time=[s.time.jd for s in calstars])

# Make these uncertainties the quadrature sum of the fractional uncertainty
# measured by MWO and the Poisson flux uncertainty that I calculated
cal_s_apo_values = np.array([s.s_apo.uncalibrated.value for s in calstars])
scaled_apo_err = cals_s_mwo.err / cals_s_mwo.value * cal_s_apo_values

cals_s_apo = Measurement([s.s_apo.uncalibrated.value for s in calstars],
                         err=[np.sqrt(s.s_apo.uncalibrated.err**2 +
                                      scaled_err**2)
                              for s, scaled_err in zip(calstars,
                                                       scaled_apo_err)],
                         time=[s.time.jd for s in calstars])

hat11_s_apo = Measurement([s.s_apo.uncalibrated.value for s in hat11],
                          err=[s.s_apo.uncalibrated.err for s in hat11],
                          time=[s.s_apo.time.jd for s in hat11])

sort = np.argsort(cals_s_apo.value)

##############################################################################
# Simple linear regression with errors in one dimension for initial fit
Xdata = np.vander(cals_s_apo.value, 2)
ydata = cals_s_mwo.value
params, resid, rank, singvals = np.linalg.lstsq(Xdata, ydata)
m_init, b_init = params
init_guess = [np.arctan(m_init), b_init, 1]

##############################################################################
# Run MCMC fit
samples = mcmc_fit(cals_s_apo, cals_s_mwo, init_guess,
                   n_steps_burnin=4000,
                   n_steps_postburnin=2000, nwalkers=12)

# Create a corner plot with the original parameterization:
corner(samples, labels=[r'$\theta$', '$b$', '$lnf$'])

plt.savefig('plots/corner_thetablnf.png', bbox_inches='tight', dpi=200)

# Convert theta parameter to slope:
samples[:, 0] = np.tan(samples[:, 0])
# Convert lnf to f
samples[:, 2] = np.exp(samples[:, 2])

m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                        axis=0)))

c1 = FitParameter(m_mcmc[0], err_upper=m_mcmc[1], err_lower=m_mcmc[2])
c2 = FitParameter(b_mcmc[0], err_upper=b_mcmc[1], err_lower=b_mcmc[2])

model_best = c1.value * cals_s_apo.value + c2.value

##############################################################################
# Solve for HAT-P-11 S-indices:
hat11_s_mwo_err = np.sqrt((f_mcmc[0] * hat11_s_apo.value * c1.err_lower)**2 +
                          (c1.value * f_mcmc[0] * hat11_s_apo.err)**2 +
                          c2.err_lower**2)

hat11_s_mwo = Measurement(c1.value * hat11_s_apo.value + c2.value,
                          err=hat11_s_mwo_err,
                          time=hat11_s_apo.time)

##############################################################################
# Plot fit results
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(cals_s_apo.value[sort], model_best[sort],
        color='r', label='MCMC model')

ax.errorbar(cals_s_apo.value, cals_s_mwo.value,
            xerr=f_mcmc[0] * cals_s_apo.err,
            yerr=cals_s_mwo.err, fmt='o', color='k',
            ecolor='gray', capsize=0)

ax.errorbar(hat11_s_apo.value, hat11_s_mwo.value,
            xerr=f_mcmc[0] * hat11_s_apo.err, yerr=hat11_s_mwo.err,
            fmt='o', color='r', mec='none', ecolor='r', capsize=0)

fontsize = 16
ax.set_xlabel("$S_{APO}$", fontsize=fontsize)
ax.set_ylabel("$S_{MWO}$", fontsize=fontsize)

note = ("$C_1 = {0:.2f}^{{+ {1:.2f} }}_{{- {2:.2f} }}$\n".format(*m_mcmc) +
        "$C_2 = {0:.2f}^{{+ {1:.2f} }}_{{- {2:.2f} }}$".format(*b_mcmc))

ax.text(0.05, 0.1, note, ha='right', fontsize=fontsize)
ax.set_aspect('auto', 'datalim')
ax.set_xlim([0, 0.075])
ax.set_ylim([0, 2])
fig.savefig('plots/s-index_calibration.png', bbox_inches='tight', dpi=200)

corner(samples[:, :], labels=[r'$C_1$', '$C_2$', 'f'])

plt.savefig('plots/corner.png', bbox_inches='tight', dpi=200)

plt.show()

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
ax.set_xlabel('Date', fontsize=fontsize)
ax.set_ylabel('$S$-index', fontsize=fontsize)
ax.set_title('CaII H & K', fontsize=fontsize)
ax.legend(loc='lower left', numpoints=1)
fig.savefig('plots/s-index_hat11.png', bbox_inches='tight', dpi=200)
plt.show()

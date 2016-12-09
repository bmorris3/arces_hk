import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from corner import corner

from toolkit import json_to_stars, Measurement, mcmc_fit, FitParameter, parse_hires

props = dict(fontsize=15)

calstars = json_to_stars('mwo_stars.json')
hat11 = json_to_stars('hat11.json')

cals_s_mwo = Measurement([s.s_mwo.value for s in calstars],
                         err=[s.s_mwo.err for s in calstars])

cals_s_apo = Measurement([s.s_apo.uncalibrated.value for s in calstars],
                         err=[s.s_apo.uncalibrated.err for s in calstars])

hat11_s_apo = Measurement([s.s_apo.uncalibrated.value for s in hat11],
                          err=[s.s_apo.uncalibrated.err for s in hat11])

sort = np.argsort(cals_s_apo.value)

##############################################################################
# Simple linear regression with errors in one dimension for initial fit
Xdata = np.vander(cals_s_apo.value, 2)
ydata = cals_s_mwo.value
params, resid, rank, singvals = np.linalg.lstsq(Xdata, ydata)
m_init, b_init = params
init_guess = [np.arctan(m_init), b_init, np.log(10), 0.01]

##############################################################################
# Run MCMC fit
samples = mcmc_fit(cals_s_apo, cals_s_mwo, init_guess,
                   n_steps_burnin=4000,
                   n_steps_postburnin=1500, nwalkers=12)

# Create a corner plot with the original parameterization:
corner(samples, labels=[r'$\theta$', '$b$', '$lnf$', '$V$'])

plt.savefig('plots/corner_thetablnf.png', bbox_inches='tight', dpi=200)

# Convert theta parameter to slope:
samples[:, 0] = np.tan(samples[:, 0])
# Convert lnf to f
samples[:, 2] = np.exp(samples[:, 2])

# m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                              zip(*np.percentile(samples, [16, 50, 84],
#                                                 axis=0)))
m_mcmc, b_mcmc, f_mcmc, v_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

c1 = FitParameter(m_mcmc[0], err_upper=m_mcmc[1], err_lower=m_mcmc[2])
c2 = FitParameter(b_mcmc[0], err_upper=b_mcmc[1], err_lower=b_mcmc[2])

model_best = c1.value * cals_s_apo.value + c2.value

##############################################################################
# Plot fit results
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(cals_s_apo.value[sort], model_best[sort],
        color='r', label='MCMC model')

ax.fill_between(cals_s_apo.value[sort], model_best[sort]-v_mcmc[0],
                model_best[sort]+v_mcmc[0], color='r', alpha=0.5)

ax.errorbar(cals_s_apo.value, cals_s_mwo.value,
            xerr=f_mcmc[0] * cals_s_apo.err,
            yerr=cals_s_mwo.err, fmt='o', color='k',
            ecolor='gray', capsize=0)

# Solve for HAT-P-11 S-indices:
hat11_s_mwo = c1.value * hat11_s_apo.value + c2.value
# hat11_s_mwo_err = (0.5 * (c1.err_lower + c1.err_upper) *
#                    np.ones_like(hat11_s_apo.value))

hat11_s_mwo_err = np.sqrt((hat11_s_apo.value * c1.err_lower)**2 +
                          (c1.value * hat11_s_apo.err)**2 +
                          c2.err_lower**2)

hat11_times = Time([s.time for s in hat11])
ax.errorbar(hat11_s_apo.value, hat11_s_mwo,
            xerr=f_mcmc[0] * hat11_s_apo.err,
            yerr=hat11_s_mwo_err, fmt='o', color='r', mec='none',
            ecolor='gray', capsize=0)

fontsize = 16
ax.set_xlabel("$S_{APO}$", fontsize=fontsize)
ax.set_ylabel("$S_{MWO}$", fontsize=fontsize)

note = ("$C_1 = {0:.2f}^{{+ {1:.2f} }}_{{- {2:.2f} }}$\n".format(*m_mcmc) +
        "$C_2 = {0:.2f}^{{+ {1:.2f} }}_{{- {2:.2f} }}$".format(*b_mcmc))

ax.text(0.05, 0.1, note, ha='right', fontsize=fontsize)
ax.set_aspect('auto', 'datalim')

fig.savefig('plots/s-index_calibration.png', bbox_inches='tight', dpi=200)

corner(samples[:, :-1], labels=[r'$C_1$', '$C_2$', '$f$'])

plt.savefig('plots/corner.png', bbox_inches='tight', dpi=200)

plt.show()

# ##############################################################################
# # Plot the S-index time series for HAT-P-11 with HIRES data too
# hires = parse_hires('hat-p-11_svals.txt')
#
# hat11_s_mwo_mean = c1.value * hat11_s_apo.value.mean() + c2.value
# # hat11_s_mwo_mean_err = np.sqrt(np.sum(hat11_s_mwo_err**2 / len(hat11_s_mwo)))
# hat11_s_apo_err_mean = np.sum(np.sqrt(hat11_s_apo.err**2 / len(hat11_s_apo.err)))
# hat11_s_mwo_mean_err = np.sqrt((hat11_s_apo.value.mean() * c1.err_lower)**2 +
#                                (c1.value * hat11_s_apo_err_mean)**2 +
#                                c2.err_lower**2)
# fig, ax = plt.subplots()
# ax.axvspan(2009.4, 2013.4, color='k', alpha=0.1, label='Kepler')
#
# ax.plot(hires['time'].decimalyear, hires['S-value'], 'k.', label='Keck/HIRES')
# ax.errorbar(hat11_times.decimalyear, hat11_s_mwo, hat11_s_mwo_err,
#             fmt='s', color='gray', capsize=0, label='ARC 3.5m/ARCES')
# ax.errorbar(hat11_times.decimalyear.mean(), hat11_s_mwo_mean,
#             hat11_s_mwo_mean_err, fmt='s', color='r', capsize=0, markersize=10,
#             elinewidth=2, label='ARC 3.5m/ARCES (mean)')
# ax.set_xlabel('Date', fontsize=fontsize)
# ax.set_ylabel('$S$-index', fontsize=fontsize)
# ax.set_title('CaII H & K', fontsize=fontsize)
# ax.legend(loc='lower left', numpoints=1)
# fig.savefig('plots/s-index_hat11.png', bbox_inches='tight', dpi=200)
# plt.show()

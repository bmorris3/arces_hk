import matplotlib.pyplot as plt
import numpy as np
from corner import corner

from toolkit import json_to_stars, Measurement, mcmc_fit, FitParameter


calstars = json_to_stars('mwo_stars.json')

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
f = FitParameter(f_mcmc[0], err_upper=f_mcmc[1], err_lower=f_mcmc[2])

model_best = c1.value * cals_s_apo.value + c2.value

c1.to_text('calibrated_c1.txt')
c2.to_text('calibrated_c2.txt')
f.to_text('calibrated_f.txt')

##############################################################################
# Plot fit results
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(cals_s_apo.value[sort], model_best[sort],
        color='r', label='MCMC model')

ax.errorbar(cals_s_apo.value, cals_s_mwo.value,
            xerr=f.value * cals_s_apo.err,
            yerr=cals_s_mwo.err, fmt='o', color='k',
            ecolor='gray', capsize=0)

fontsize = 16
ax.set_xlabel("$S_{APO}$", fontsize=fontsize)
ax.set_ylabel("$S_{MWO}$", fontsize=fontsize)

note = ("$C_1 = {0:.2f}^{{+ {1:.2f} }}_{{- {2:.2f} }}$\n".format(*m_mcmc) +
        "$C_2 = {0:.2f}^{{+ {1:.2f} }}_{{- {2:.2f} }}$".format(*b_mcmc))

#ax.text(0.03, 1.5, note, ha='right', fontsize=fontsize)
ax.set_aspect('auto', 'datalim')
ax.set_xlim([0, 0.075])
ax.set_ylim([0, 2])
fig.savefig('plots/s-index_calibration.png', bbox_inches='tight', dpi=200)

corner(samples[:, :], labels=[r'$C_1$', '$C_2$', 'f'])

plt.savefig('plots/corner.png', bbox_inches='tight', dpi=200)

plt.show()


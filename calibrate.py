import matplotlib.pyplot as plt
import numpy as np
import emcee
from corner import corner

from toolkit import json_to_stars, Measurement
from toolkit.analysis import initial_odr_fit, mcmc_fit

props = dict(fontsize=15)

calstars = json_to_stars('mwo_stars.json')

cals_s_mwo = Measurement([s.s_mwo.value for s in calstars],
                         err=[s.s_mwo.err for s in calstars])

cals_s_apo = Measurement([s.s_apo.uncalibrated.value for s in calstars],
                         err=[s.s_apo.uncalibrated.err for s in calstars])

sort = np.argsort(cals_s_apo.value)

##############################################################################
# Simple linear regression with errors in one dimension for initial fit
Xdata = np.vander(cals_s_apo.value, 2)
ydata = cals_s_mwo.value
params, resid, rank, singvals = np.linalg.lstsq(Xdata, ydata)
m_init, b_init = params
init_guess = [np.arctan(m_init), b_init, np.log(15)]

##############################################################################
# Run MCMC fit
samples = mcmc_fit(cals_s_apo, cals_s_mwo, init_guess,
                   n_steps_postburnin=4000, nwalkers=12)

# Create a corner plot with the original parameterization:
corner(samples, labels=[r'$\theta$', '$b$', '$lnf$'])

plt.savefig('plots/corner_thetablnf.png', bbox_inches='tight', dpi=200)

# Convert theta parameter to slope:
samples[:, 0] = np.tan(samples[:, 0])
# Convert lnf to f
samples[:, -1] = np.exp(samples[:, -1])

m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

model_best = m_mcmc[0] * cals_s_apo.value + b_mcmc[0]

##############################################################################
# Plot results
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(cals_s_apo.value[sort], model_best[sort],
        color='r', label='MCMC model')

ax.errorbar(cals_s_apo.value, cals_s_mwo.value,
            xerr=f_mcmc[0] * cals_s_apo.err,
            yerr=cals_s_mwo.err, fmt='.', color='k',
            ecolor='gray', capsize=0)

fontsize = 16
ax.set_xlabel("$S_{APO}$", fontsize=fontsize)
ax.set_ylabel("$S_{MWO}$", fontsize=fontsize)

note = ("$C_1 = {0:.2f}^{{+ {1:.2f} }}_{{- {2:.2f} }}$\n".format(*m_mcmc) +
        "$C_2 = {0:.2f}^{{+ {1:.2f} }}_{{- {2:.2f} }}$".format(*b_mcmc))

ax.text(0.05, 0.1, note, ha='right', fontsize=fontsize)
ax.set_aspect('auto', 'datalim')

fig.savefig('plots/s-index_calibration.png', bbox_inches='tight', dpi=200)

corner(samples, labels=[r'$C_1$', '$C_2$', '$f$'])

plt.savefig('plots/corner.png', bbox_inches='tight', dpi=200)
plt.show()

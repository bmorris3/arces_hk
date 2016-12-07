import matplotlib.pyplot as plt
import numpy as np

from toolkit import json_to_stars, Measurement

props = dict(fontsize=15)

calstars = json_to_stars('mwo_stars.json')
hat11 = json_to_stars('hat11.json')

cals_s_mwo = Measurement([s.s_mwo.value for s in calstars],
                          err_upper=[s.s_mwo.err_upper for s in calstars],
                          err_lower=[s.s_mwo.err_lower for s in calstars])

cals_s_apo = Measurement([s.s_apo.uncalibrated.value for s in calstars],
                          err_upper=[s.s_apo.uncalibrated.err_upper for s in calstars],
                          err_lower=[s.s_apo.uncalibrated.err_lower for s in calstars])

sort = np.argsort(cals_s_apo.value)
# plt.errorbar(cals_s_apo.value, cals_s_mwo.value,
#              xerr=cals_s_apo.err_lower, yerr=cals_s_mwo.err_lower, fmt='.')

##############################################################################
# Simple linear regression with errors in one dimension for initial fit
Xdata = np.vander(cals_s_apo.value, 2)
ydata = cals_s_mwo.value
params, resid, rank, singvals = np.linalg.lstsq(Xdata, ydata)
m_init, b_init = params

init_model = m_init * cals_s_apo.value + b_init

# plt.plot(cals_s_apo.value[sort], init_model[sort],
#          color='gray', label='init model')

##############################################################################
# Wrote my own orthogonal distance regression likelihood for optimization

from toolkit.analysis import ln_likelihood
from scipy.optimize import fmin


def chi2(p, x, y, xerr, yerr):
    return -ln_likelihood(p, x, y, xerr, yerr)

theta_init = np.arctan(m_init)
lnf = np.log(15)
init_params = [theta_init, b_init, lnf]

args = (cals_s_apo.value, cals_s_mwo.value, cals_s_apo.err_lower,
        cals_s_mwo.err_lower)

init_likelihood = ln_likelihood(init_params, *args)
print('init ln_likelihood', init_likelihood)

result = fmin(chi2, init_params, args=args)
m_best = np.tan(result[0])
b_best = result[1]

model_best = m_best * cals_s_apo.value + b_best

final_likelihood = ln_likelihood(result, *args)
print('final ln_likelihood', final_likelihood)

import emcee
from corner import corner

ndim, nwalkers = len(init_params), 10
n_steps = 5000
p0 = [[init_params[0] + 0.05 * np.random.randn(),
       init_params[1] + 0.01 * np.random.randn(),
       init_params[2] + 0.001 * np.random.randn()]
      for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_likelihood, args=args,
                                threads=3)
p1 = sampler.run_mcmc(p0, 100)[0]
sampler.reset()
sampler.run_mcmc(p1, n_steps)

samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))

# Convert theta parameter to slope:
samples[:, 0] = np.tan(samples[:, 0])
# Convert lnf to f
samples[:, -1] = np.exp(samples[:, -1])

m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(cals_s_apo.value[sort], model_best[sort],
        color='r', label='MCMC model')

ax.errorbar(cals_s_apo.value, cals_s_mwo.value,
            xerr=f_mcmc[0] * cals_s_apo.err_lower,
            yerr=cals_s_mwo.err_lower, fmt='.', color='k',
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

# ##############################################################################
# # Double check results with scipy's ODR package
# # ODR docs: https://docs.scipy.org/doc/external/odrpack_guide.pdf
# from scipy.odr import Model, RealData, ODR
#
# # Create a Model, create a RealData instance.
# linear = Model(lambda b, x: b[0] * x + b[1])
# mydata = RealData(cals_s_apo.value, cals_s_mwo.value,
#                   sx=cals_s_apo.err_lower, sy=cals_s_mwo.err_lower)
# myodr = ODR(mydata, linear, beta0=[m_init, b_init])
# myoutput = myodr.run()
# myoutput.pprint()
# m_odr, b_odr = myoutput.beta
#
# # Plot results
# model_odr = m_odr * cals_s_apo.value + b_odr
# plt.plot(cals_s_apo.value[sort], model_odr[sort],
#          color='g', ls='--', label='scipy ODR')
#
# plt.legend(loc='lower right')
# plt.show()
#
#

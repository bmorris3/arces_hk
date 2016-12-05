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

plt.errorbar(cals_s_apo.value, cals_s_mwo.value,
             xerr=cals_s_apo.err_lower, yerr=cals_s_mwo.err_lower, fmt='.')

# Simple linear regression with errors in one dimension for initial fit
Xdata = np.vander(cals_s_apo.value, 2)
ydata = cals_s_mwo.value
params, resid, rank, singvals = np.linalg.lstsq(Xdata, ydata)
m_init, b_init = params

init_model = m_init * cals_s_apo.value + b_init

plt.plot(cals_s_apo.value, init_model, color='gray', label='init model')

##############################################################################


from toolkit.analysis import ln_likelihood
from scipy.optimize import fmin


def chi2(p, x, y, xerr, yerr):
    return -np.exp(ln_likelihood(p, x, y, xerr, yerr))

theta_init = np.arctan(m_init)
lnf = 0.0
init_params = [theta_init, b_init]

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

plt.plot(cals_s_apo.value, model_best, color='r', label='init_model')
plt.legend(loc='lower right')
plt.show()


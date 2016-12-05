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
plt.errorbar(cals_s_apo.value, cals_s_mwo.value,
             xerr=cals_s_apo.err_lower, yerr=cals_s_mwo.err_lower, fmt='.')

##############################################################################
# Simple linear regression with errors in one dimension for initial fit
Xdata = np.vander(cals_s_apo.value, 2)
ydata = cals_s_mwo.value
params, resid, rank, singvals = np.linalg.lstsq(Xdata, ydata)
m_init, b_init = params

init_model = m_init * cals_s_apo.value + b_init

plt.plot(cals_s_apo.value[sort], init_model[sort],
         color='gray', label='init model')

##############################################################################
# Wrote my own orthogonal distance regression likelihood for optimization

from toolkit.analysis import ln_likelihood
from scipy.optimize import fmin


def chi2(p, x, y, xerr, yerr):
    return -ln_likelihood(p, x, y, xerr, yerr)

theta_init = np.arctan(m_init)
lnf = 0.0
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
print(result, m_best, b_best)
plt.plot(cals_s_apo.value[sort], model_best[sort],
         color='r', label='custom ODR')
# plt.show()

##############################################################################
# Double check results with scipy's ODR package
# ODR docs: https://docs.scipy.org/doc/external/odrpack_guide.pdf
from scipy.odr import Model, RealData, ODR

# Create a Model, create a RealData instance.
linear = Model(lambda b, x: b[0] * x + b[1])
mydata = RealData(cals_s_apo.value, cals_s_mwo.value,
                  sx=cals_s_apo.err_lower, sy=cals_s_mwo.err_lower)
myodr = ODR(mydata, linear, beta0=[m_init, b_init])
myoutput = myodr.run()
myoutput.pprint()
m_odr, b_odr = myoutput.beta

# Plot results
model_odr = m_odr * cals_s_apo.value + b_odr
plt.plot(cals_s_apo.value[sort], model_odr[sort],
         color='g', ls='--', label='scipy ODR')

plt.legend(loc='lower right')
plt.show()



import matplotlib.pyplot as plt
import numpy as np

from toolkit import json_to_stars

props = dict(fontsize=15)

calstars = json_to_stars('mwo_stars.json')
hat11 = json_to_stars('hat11.json')

calstars_s_apo = np.array([star.s_apo.uncalibrated for star in calstars])
calstars_s_mwo = np.array([star.s_mwo.value for star in calstars])
calstars_s_mwo_err = [[star.s_mwo.err_lower for star in calstars],
                      [star.s_mwo.err_upper for star in calstars]]

hat11_s_apo = [obs.s_apo.uncalibrated for obs in hat11]

# Xdata = np.vander(calstars_s_apo, 2)
# ydata = calstars_s_mwo
# theta_best, resid, rank, singvals = np.linalg.lstsq(Xdata, ydata)
#
# best_model = theta_best[0] * calstars_s_apo + theta_best[1]

sort = np.argsort(calstars_s_apo)
x = calstars_s_apo
y = calstars_s_mwo
yerr = np.mean(calstars_s_mwo_err, axis=0)

A = np.vander(calstars_s_apo, 2)
C = np.diag(yerr**2)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
theta_best = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
theta_uncertainties = np.sqrt(np.diag(cov))
best_model = theta_best[1] + theta_best[0] * x

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
formatvals = dict(c1=theta_best[0], c1err=theta_uncertainties[0],
                  c2=theta_best[1], c2err=theta_uncertainties[1])
coeffs = ("$c_1$ = {c1:.2f} $\pm$ {c1err:.2f}"
          "\n$c_2$ = {c2:.2f} $\pm$ {c2err:.2f}").format(**formatvals)
ax[0].text(0.067, 0.1, coeffs,
           ha='right', **props)

ax[0].errorbar(calstars_s_apo, calstars_s_mwo, yerr=calstars_s_mwo_err, fmt='.')
ax[0].plot(calstars_s_apo, best_model, 'r')

for obs in hat11_s_apo:
    ax[0].axvline(obs, color='gray')

ax[0].set_xlabel('$S_{APO}$', **props)
ax[0].set_ylabel('$S_{MWO}$', **props)

hat11_s_mwo = [obs.s_apo.calibrated(*theta_best) for obs in hat11]
ax[1].hist(hat11_s_mwo)
ax[1].set_xlabel('$S_{MWO}$', **props)
ax[1].set_title('HAT-P-11 (APO)')
plt.show()
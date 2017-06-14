import numpy as np
from toolkit import json_to_stars, fit_gp, plot_corner, plot_draws
import matplotlib.pyplot as plt
from astropy.time import Time

hat11_keck = json_to_stars('hat11_keck_calibrated.json')
hat11_apo = json_to_stars('hat11_apo_calibrated.json')

x = []
y = []
yerr = []

for starprops in [hat11_keck, hat11_apo]:
    for star in starprops:
        x.append(star.time.jd)
        y.append(star.s_mwo.value)
        yerr.append(star.s_mwo.err)

sort = np.argsort(x)

x = np.array(Time(x, format='jd').decimalyear)[sort]
y = np.array(y)[sort]
yerr = np.ones_like(x) * np.median(yerr) #np.array(yerr)[sort]

#initp = [0.59, 0.45, 11.0, 0.14, 2.85, 7.62, -7.28, 0.0002]
initp = [0.6, -1, -1, 11]

args = (x, y, yerr)
sampler = fit_gp(initp, args, nsteps=500)
samples = sampler.flatchain

fig, ax = plot_corner(samples)
fig.savefig('plots/corner_hat11.png', bbox_inches='tight', dpi=200)
fig, ax = plot_draws(samples, x, y, yerr)
fig.savefig('plots/model_hat11.png', bbox_inches='tight', dpi=200)
plt.show()
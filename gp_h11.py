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
yerr = np.array(yerr)[sort]

initp = [0.59, 0.45, 11.0, 1, 1, 1, 0.01]

args = (x, y, yerr)
sampler = fit_gp(initp, args, nsteps=1000)
samples = sampler.flatchain

fig, ax = plot_corner(samples)
fig.savefig('plots/corner_hat11.png', bbox_inches='tight', dpi=200)

## Plot samples
t = np.concatenate((x, np.linspace(x.min()-x.ptp()/2,
                                   x.max()+x.ptp()/2, 300)))
t = np.sort(t)

n_draws = 100
fig, ax = plt.subplots()
from toolkit.cycle import model
for s in samples[np.random.randint(len(samples), size=n_draws)]:
#    high, low, period, duration_low, duration_slope, phase, var  = s
    m = model(s, t)
    ax.plot(t, m, '-', color="#4682b4", alpha=0.05)
ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0,
            zorder=10)
ax.set_ylabel(r"$S$-index")
ax.set_title("Gaussian process model")
#fig, ax = plot_draws(samples, x, y, yerr)
#fig.savefig('plots/model_hat11.png', bbox_inches='tight', dpi=200)
plt.show()
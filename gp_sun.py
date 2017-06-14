import numpy as np
from toolkit import fit_gp, plot_corner, plot_draws, trap_model
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.io import ascii
from scipy.stats import binned_statistic

# Import raw data
table = ascii.read('solar/sp_iss.dat')
solar_times = Time(table['col2'], format='jd').decimalyear
# Convert to S-index with Eqn 12 of Egeland 2017
solar_sind = 1.5 * table['col3'] + 0.031

# Median bin the data
nbins = 1000
medians = binned_statistic(solar_times, solar_sind, statistic='median', bins=nbins)
std = binned_statistic(solar_times, solar_sind, statistic='std', bins=nbins)
solar_times_binned = 0.5*(medians.bin_edges[:-1] + medians.bin_edges[1:])
solar_sind_binned = medians.statistic
solar_sind_err_binned = np.median(std.statistic) * np.ones_like(solar_times_binned)

x = solar_times_binned
y = solar_sind_binned
yerr = solar_sind_err_binned

not_nan = np.logical_not(np.isnan(y))
nonzero_err = yerr != 0
year_range = (solar_times_binned > 1985) & (solar_times_binned < 2010)
x = x[not_nan & nonzero_err & year_range]
y = y[not_nan & nonzero_err & year_range]
yerr = yerr[not_nan & nonzero_err & year_range]

sort = np.argsort(x)

x = x[sort]
y = np.array(y)[sort]
yerr = np.array(yerr)[sort]

initp = [0.168, 0.156, 11.2, 2, 2, 1, -10, 0.1]

# from scipy.optimize import fmin
#
# def minimize(p, t):
#     return np.sum((trap_model(p, t) - solar_sind_binned)**2/solar_sind_err_binned**2)
#
# bestp = fmin(minimize, initp, args=(solar_times_binned,))
#
# model_times = np.linspace(1975, 2017, 300)
# plt.plot(model_times, trap_model(bestp, model_times))
# print(x.mean(), model_times.mean())
# plt.errorbar(x, y, yerr=yerr, fmt='o')
# plt.show()

args = (x, y, yerr)
sampler = fit_gp(initp, args, nsteps=500)
samples = sampler.flatchain

fig, ax = plot_corner(samples)
fig.savefig('plots/corner_sun.png', bbox_inches='tight', dpi=200)
fig, ax = plot_draws(samples, x, y, yerr)
fig.savefig('plots/model_sun.png', bbox_inches='tight', dpi=200)
plt.show()
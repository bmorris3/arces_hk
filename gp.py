import george
from george.kernels import Matern32Kernel
import numpy as np
from toolkit import json_to_stars, get_duncan_catalog
import matplotlib.pyplot as plt
import emcee
from corner import corner
from astropy.time import Time

hat11_keck = json_to_stars('hat11_keck_calibrated.json')
hat11_apo = json_to_stars('hat11_apo_calibrated.json')

catalog_s_min = get_duncan_catalog()['Smin'].data.min()

x = []
y = []
yerr = []

for starprops in [hat11_keck, hat11_apo]:
    for star in starprops:
        x.append(star.time.jd)
        y.append(star.s_mwo.value)
        yerr.append(star.s_mwo.err)

sort = np.argsort(x)

x = np.array(x)[sort]
y = np.array(y)[sort] #- np.mean(y)
yerr = np.array(yerr)[sort]


def model(p, x):
    s0, alpha, per, t0, lna, tau = p
    return s0 + alpha * np.sin(2*np.pi/per * (x - t0))


def lnlike_gp(p, x, y, yerr):
    s0, alpha, per, t0, lna, tau = p
    gp = george.GP(np.exp(lna) * Matern32Kernel(tau))
    gp.compute(x, yerr)
    return gp.lnlikelihood(y - model(p, x))


def lnprior(p):
    s0, alpha, per, t0, lna, tau = p
    if not ((alpha < s0 < y.max()) and (5*365 < per < 50*365) and
            (-50 < lna < 10) and (1 < tau < 200) and (alpha > 0) and
            s0-alpha >= catalog_s_min):
        return -np.inf
    else:
        return 0


def lnprob_gp(p, x, y, yerr):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(p, x, y, yerr)


def fit_gp(initial, data, nwalkers=16):
    ndim = len(initial)

    p0 = []

    s0, alpha, per, t0, lna, tau = initial

    while len(p0) < nwalkers:

        p0_trial = [s0 + 0.05 * np.random.randn(),
                    alpha + 0.01 * np.random.randn(),
                    per + 300 * np.random.randn(),
                    t0 + 50 * np.random.randn(),
                    lna + 0.1 * np.random.randn(),
                    tau + 1 * np.random.randn()]
        if not np.isfinite(lnprior(p0_trial)):
            p0.append(p0_trial)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, args=data,
                                    threads=3)

    print("Running burn-in")
    p0, lnp, _ = sampler.run_mcmc(p0, 1000)
    sampler.reset()

    print("Running second burn-in")
    p = p0[np.argmax(lnp)]
    p0 = [p + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]
    p0, _, _ = sampler.run_mcmc(p0, 1000)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, 1000)

    return sampler


init_guess = [0.4, 0.275, 30*365, 2571990.5, -7.2, 45]
args = (x, y, yerr)

# Set up the Gaussian process.
s0, alpha, per, t0, lna, tau = init_guess

kernel = np.exp(lna) * Matern32Kernel(tau)
gp = george.GP(kernel)

# Pre-compute the factorization of the matrix.
gp.compute(x, yerr)

# Compute the log likelihood.

t = np.linspace(x.min(), x.max(), 5000)
mu, cov = gp.predict(y - model(init_guess, x), t)
mu += model(init_guess, t)
std = np.sqrt(np.diag(cov))

# plt.figure()
# plt.plot(t, mu)
# plt.errorbar(x, y, yerr=yerr, fmt='.', color='k', ecolor='gray')
# plt.show()

sampler = fit_gp(init_guess, args)

samples = sampler.flatchain

fig, ax = plt.subplots(samples.shape[1], samples.shape[1], figsize=(8, 8))
corner(samples, fig=fig,
       labels=['s0', 'alpha', 'per', 't0', 'lna', 'tau'])
fig.savefig('plots/corner.png', bbox_inches='tight', dpi=200)
plt.show()

t = np.concatenate((x, np.linspace(x.min(), x.max()+x.ptp(), 300)))
t = np.sort(t)

plt.figure()
for s in samples[np.random.randint(len(samples), size=250)]:
    s0, alpha, per, t0, lna, tau = s

    gp = george.GP(np.exp(lna) * Matern32Kernel(tau))
    gp.compute(x, yerr)
    m = gp.sample_conditional(y - model(s, x), t) + model(s, t)
    plt.plot_date(Time(t, format='jd').plot_date, m, '-',
                  color="#4682b4", alpha=0.05)
plt.errorbar(Time(x, format='jd').plot_date, y, yerr=yerr, fmt=".k", capsize=0,
             zorder=10)
plt.ylabel(r"$S$-index")
plt.title("Gaussian process model")
plt.savefig('plots/model.png', bbox_inches='tight', dpi=200)
plt.show()

rescaled_samples = np.copy(samples)
rescaled_samples[:, 2] /= 365
rescaled_samples[:, 3] -= rescaled_samples[:, 3].mean()
rescaled_samples[:, 3] /= 365

fig, ax = plt.subplots(rescaled_samples.shape[1],
                       rescaled_samples.shape[1], figsize=(8, 8))
corner(rescaled_samples, fig=fig,
       labels=['$S_0$', r'$\alpha$', '$P$ [years]', '$t_0$ [years]',
               r'$\ln{a}$', r'$\tau$'])
suptitle = (r'$S(t) = S_0 + \alpha \, \sin\left(\frac{2\pi}{P} (t - t_0)\right),$' +
            '\n' + r'$k(x_i, x_j) = a\,$Matern3/2($\tau$)')
fig.suptitle(suptitle, fontsize=15)
fig.savefig('plots/corner_scaled.png', bbox_inches='tight', dpi=200)
plt.show()

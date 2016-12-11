import george
from george.kernels import CosineKernel
import numpy as np
from toolkit import json_to_stars
import matplotlib.pyplot as plt
import emcee

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

x = np.array(x)[sort]
y = np.array(y)[sort] #- np.mean(y)
yerr = np.array(yerr)[sort]


def model(p, x, y, yerr):
    return p[0] * np.ones_like(x)


def lnlike_gp(p, x, y, yerr):
    mean, a, per1, per2, lns = p
    gp = george.GP(a * CosineKernel(per1) + CosineKernel(per2))
    gp.compute(x, np.sqrt(yerr ** 2 + np.exp(2 * lns)))
    return gp.lnlikelihood(y - model(p, x, y, yerr))


def lnprior(p):
    mean, a, per1, per2, lns = p
    if not ((0 < mean < 1) and (20 < per1 < 40) and (8*365 < per2 < 50*365)
            and (lns < 1) and (0 < a)):
        return -np.inf
    else:
        return 0


def lnprob_gp(p, x, y, yerr):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(p, x, y, yerr)


def fit_gp(initial, data, nwalkers=10):
    ndim = len(initial)

    p0 = []

    while len(p0) < nwalkers:

        p0_trial = [initial[0] + 0.05 * np.random.randn(),
                    initial[1] + 0.05 * np.random.randn(),
                    initial[2] + 1 * np.random.randn(),
                    initial[3] + 200 * np.random.randn(),
                    initial[4] + 1 * np.random.randn()]
        if not np.isfinite(lnprior(p0_trial)):
            p0.append(p0_trial)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, args=data)

    print("Running burn-in")
    p0, lnp, _ = sampler.run_mcmc(p0, 2000)
    sampler.reset()

    print("Running second burn-in")
    p = p0[np.argmax(lnp)]
    p0 = [p + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]
    p0, _, _ = sampler.run_mcmc(p0, 2000)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, 5000)

    return sampler


init_guess = [0.48, 0.1, 29.2, 20*365, -4.0]
args = (x, y, yerr)

# Set up the Gaussian process.
kernel = 0.5 * CosineKernel(29.2) + CosineKernel(20 * 365)
gp = george.GP(kernel)

# Pre-compute the factorization of the matrix.
gp.compute(x, yerr)

# Compute the log likelihood.

t = np.linspace(x.min(), x.max(), 5000)
mu, cov = gp.predict(y - init_guess[0], t)
mu += init_guess[0]
std = np.sqrt(np.diag(cov))

plt.figure()
plt.plot(t, mu)
plt.errorbar(x, y, yerr=yerr, fmt='.', color='k', ecolor='gray')
plt.show()

sampler = fit_gp(init_guess, args)

samples = sampler.flatchain
t = np.linspace(x.min(), x.max(), 1000)

from corner import corner
corner(samples, labels=['mean', 'a', 'per1', 'per2', 'lns'])

plt.figure()
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
for s in samples[np.random.randint(len(samples), size=10)]:
    mean, a, per1, per2, lns = s

    gp = george.GP(a * CosineKernel(per1) + CosineKernel(per2))
    gp.compute(x, np.sqrt(yerr**2 + np.exp(2*lns)))
    m = gp.sample_conditional(y - model(s, x, y, yerr), t) + model(s, t, y, yerr)
    plt.plot(t, m, color="#4682b4", alpha=0.8)
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.xlim(x.min(), x.max())
plt.title("results with Gaussian process noise model")
plt.show()
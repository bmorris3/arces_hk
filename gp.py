import george
from george.kernels import CosineKernel, ExpSine2Kernel
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
    mean, lna, lngamma, per1, per2, lns = p
    gp = george.GP(np.exp(lna) * ExpSine2Kernel(np.exp(lngamma), per1) +
                   CosineKernel(per2))
    gp.compute(x, np.sqrt(yerr ** 2 + np.exp(2 * lns)))
    return gp.lnlikelihood(y - model(p, x, y, yerr))


def lnprior(p):
    mean, lna, lngamma, per1, per2, lns = p
    if not ((0 < mean < y.max()) and (25 < per1 < 35) and (5*365 < per2 < 40*365)
            and (lns < 1) and (-50 < lna < 10) and (3 < lngamma < 10)):
        return -np.inf
    else:
        return 0


def lnprob_gp(p, x, y, yerr):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(p, x, y, yerr)


def fit_gp(initial, data, nwalkers=12):
    ndim = len(initial)

    p0 = []

    mean, lna, lngamma, per1, per2, lns = initial

    while len(p0) < nwalkers:

        p0_trial = [mean + 0.01 * np.random.randn(),
                    lna + 0.05 * np.random.randn(),
                    lngamma + 0.1 * np.random.randn(),
                    per1 + 1 * np.random.randn(),
                    per2 + 200 * np.random.randn(),
                    lns + 0.1 * np.random.randn()]
        if not np.isfinite(lnprior(p0_trial)):
            p0.append(p0_trial)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, args=data,
                                    threads=3)

    print("Running burn-in")
    p0, lnp, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running second burn-in")
    p = p0[np.argmax(lnp)]
    p0 = [p + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]
    p0, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, 1000)

    return sampler


init_guess = [0.45, -8, 4.5, 29.2, 20*365, -4.5]
args = (x, y, yerr)

# Set up the Gaussian process.
mean, lna, lngamma, per1, per2, lns = init_guess

kernel = (np.exp(lna) * ExpSine2Kernel(np.exp(lngamma), per1) +
          CosineKernel(per2))
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
t = np.linspace(x.min(), x.max(), 10000)

from corner import corner
fig, ax = plt.subplots(samples.shape[1], samples.shape[1], figsize=(8, 8))
corner(samples, fig=fig,
       labels=['mean', 'lna', 'lngamma', 'per1', 'per2', 'lns'])
plt.show()

# plt.figure()
# plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
# for s in samples[np.random.randint(len(samples), size=10)]:
#     mean, lna, lngamma, per1, per2, lns = s
#
#     gp = george.GP(np.exp(lna) * ExpSine2Kernel(np.exp(lngamma), per1) +
#                    CosineKernel(per2))
#     gp.compute(x, np.sqrt(yerr**2 + np.exp(2*lns)))
#     m = gp.sample_conditional(y - model(s, x, y, yerr), t) + model(s, t, y, yerr)
#     plt.plot(t, m, color="#4682b4", alpha=0.8)
# plt.ylabel(r"$y$")
# plt.xlabel(r"$t$")
# plt.xlim(x.min(), x.max())
# plt.title("results with Gaussian process noise model")
# plt.show()
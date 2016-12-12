import george
from george.kernels import CosineKernel, Matern32Kernel
import numpy as np
from toolkit import json_to_stars
import matplotlib.pyplot as plt
import emcee
from corner import corner

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
    s0, alpha, per, t0, lna, tau, lns = p
    return s0 + alpha * np.sin(2*np.pi/per * (x - t0))


def lnlike_gp(p, x, y, yerr):
    s0, alpha, per, t0, lna, tau, lns = p
    gp = george.GP(np.exp(lna) * Matern32Kernel(tau))
    gp.compute(x, np.sqrt(yerr ** 2 + np.exp(2 * lns)))
    return gp.lnlikelihood(y - model(p, x, y, yerr))


def lnprior(p):
    s0, alpha, per, t0, lna, tau, lns = p
    if not ((alpha < s0 < y.max()) and (5*365 < per < 50*365) and (lns < 5)
            and (-50 < lna < 10) and (1 < tau < 200) and (alpha > 0)):
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

    s0, alpha, per, t0, lna, tau, lns = initial

    while len(p0) < nwalkers:

        p0_trial = [s0 + 0.05 * np.random.randn(),
                    alpha + 0.01 * np.random.randn(),
                    per + 300 * np.random.randn(),
                    t0 + 50 * np.random.randn(),
                    lna + 0.1 * np.random.randn(),
                    tau + 1 * np.random.randn(),
                    lns + 0.1 * np.random.randn()]
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
    p0, _, _ = sampler.run_mcmc(p0, 2000)

    return sampler


init_guess = [0.4, 0.275, 30*365, 2571990.5, -7.2, 45, -4.5]
args = (x, y, yerr)

# Set up the Gaussian process.
s0, alpha, per, t0, lna, tau, lns = init_guess

kernel = np.exp(lna) * Matern32Kernel(tau)
gp = george.GP(kernel)

# Pre-compute the factorization of the matrix.
gp.compute(x, yerr)

# Compute the log likelihood.

t = np.linspace(x.min(), x.max(), 5000)
mu, cov = gp.predict(y - model(init_guess, x, y, yerr), t)
mu += model(init_guess, t, y, yerr)
std = np.sqrt(np.diag(cov))

plt.figure()
plt.plot(t, mu)
plt.errorbar(x, y, yerr=yerr, fmt='.', color='k', ecolor='gray')
plt.show()

sampler = fit_gp(init_guess, args)

samples = sampler.flatchain

fig, ax = plt.subplots(samples.shape[1], samples.shape[1], figsize=(8, 8))
corner(samples, fig=fig,
       labels=['s0', 'alpha', 'per', 't0', 'lna', 'tau', 'lns'])
plt.show()
t = np.linspace(x.min(), x.max()+x.ptp(), 1500)

plt.figure()
for s in samples[np.random.randint(len(samples), size=50)]:
    s0, alpha, per, t0, lna, tau, lns = s

    gp = george.GP(np.exp(lna) * Matern32Kernel(tau))
    gp.compute(x, np.sqrt(yerr**2 + np.exp(2*lns)))
    m = gp.sample_conditional(y - model(s, x, y, yerr), t) + model(s, t, y, yerr)
    plt.plot(t, m, color="#4682b4", alpha=0.15)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.xlim(t.min(), t.max())
plt.title("results with Gaussian process noise model")
plt.show()
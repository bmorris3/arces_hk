from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import george
from george.kernels import Matern32Kernel, CosineKernel, ExpSquaredKernel
import numpy as np
import matplotlib.pyplot as plt
import emcee
from corner import corner


__all__ = ['fit_gp', 'plot_corner', 'plot_draws', 'trap_model']


def trap_model(p, times):
    high, low, period, duration_low, duration_slope, phase = p
    period = period
    low_duration = duration_low
    slope_dur = duration_slope
    a = low
    b = high

    low_half_dur = low_duration/2
    slope_half_dur = (slope_dur + low_duration)/2
    half_period = period/2
    t0 = period/2

    x = (times - phase) % period

    f = np.zeros_like(x)
    slope = (a - b) / (low_half_dur - slope_half_dur)
    bottom = (x - t0 >= -low_half_dur) & (x - t0 <= low_half_dur)
    up_slope = (low_half_dur < x - t0) & (slope_half_dur >= x - t0)
    down_slope = (-low_half_dur > x - t0) & (-slope_half_dur <= x - t0)
    top = (((slope_half_dur < x - t0) & (half_period >= x - t0)) |
           ((-slope_half_dur > x - t0) & (-half_period <= x - t0)))

    f[bottom] = a
    f[up_slope] = slope * (x[up_slope] - t0 - low_half_dur) + a
    f[down_slope] = -slope * (x[down_slope] - t0 + low_half_dur) + a
    f[top] = b

    return f


def model(p, x):
#    return trap_model(p[:-2], x)
    c, lntau, period, var = p
    return np.zeros_like(x) + c


def lnlike_gp(p, x, y, yerr):
    c, lntau, period, var = p
    gp = george.GP(Matern32Kernel(np.exp(lntau)) * CosineKernel(period))
    gp.compute(x, np.sqrt(yerr**2 + var))
    return gp.lnlikelihood(y - model(p, x))


def lnprior(p, x, y, yerr):
    c, lntau, period, var = p
    if not ((5 < period < 15) and (0 < lntau < 10) and (0 < c) and (0 <= var)):
        return -np.inf
    else:
        return 0


def lnprob_gp(p, x, y, yerr):
    lp = lnprior(p, x, y, yerr)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(p, x, y, yerr)


def fit_gp(initial, data, nwalkers=16, nsteps=1000):
    """

    Parameters
    ----------
    initial : list
        initial parameters
    data : tuple
        arguments: (x, y, yerr)
    nwalkers : int
        number of MCMC walkers

    Returns
    -------

    """
    ndim = len(initial)

    p0 = []

    x, y, yerr = data

    c, lntau, period, var = initial

    while len(p0) < nwalkers:

        p0_trial = [c + 0.01 * np.random.randn(),
                    lntau + 1 * np.random.randn(),
                    period + 1 * np.random.randn(),
                    var + 0.1 * np.random.randn()]

        if np.isfinite(lnprior(p0_trial, x, y, yerr)):
            p0.append(p0_trial)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, args=data,
                                    threads=3)

    print("Running burn-in")
    p0, lnp, _ = sampler.run_mcmc(p0, nsteps)
    sampler.reset()

    print("Running second burn-in")
    p = p0[np.argmax(lnp)]
    p0 = [p + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]
    p0, _, _ = sampler.run_mcmc(p0, nsteps)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, nsteps)

    return sampler


def plot_corner(samples):
    fig, ax = plt.subplots(samples.shape[1], samples.shape[1], figsize=(8, 8))
    corner(samples, fig=fig,
           labels=['c', 'lntau', 'period', 'var'])
    return fig, ax


def plot_draws(samples, x, y, yerr, n_draws=150):
    t = np.concatenate((x, np.linspace(x.min()-x.ptp()/2,
                                       x.max()+x.ptp()/2, 300)))
    t = np.sort(t)

    fig, ax = plt.subplots()
    for s in samples[np.random.randint(len(samples), size=n_draws)]:
        c, lntau, period, var = s

        kernel = (Matern32Kernel(np.exp(lntau)) +
                  CosineKernel(period))
        gp = george.GP(kernel)
        gp.compute(x, np.sqrt(yerr**2 + var))
        m = gp.sample_conditional(y - model(s, x), t) + model(s, t)
        ax.plot(t, m, '-', color="#4682b4", alpha=0.05)
    ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0,
                zorder=10)
    ax.set_ylabel(r"$S$-index")
    ax.set_title("GP")
    return fig, ax

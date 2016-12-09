from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.optimize import fmin
import emcee

__all__ = ['mcmc_fit', 'initial_odr_fit']


def v_vector(theta):
    """
    Hogg+ 2010, Eqn 29.
    """
    return [[-np.sin(theta)], [np.cos(theta)]]


def ln_likelihood(p, x, y, x_err, y_err, max_theta=np.pi/2, min_theta=0,
                  min_lnf=0):
    """
    Hogg+ 2010, Eqn 30., with an additional parameter that scales up the
    uncertainty in the x dimension, ``x_err``, by a constant factor.

    The likelihood has been written assuming x and y uncertainties are
    uncorrelated.
    """
    # theta, b, lnf = p
    theta, b, lnf, V = p

    # Assert prior:
    if (theta < min_theta or theta > max_theta or lnf < min_lnf or V < 0
        or b < -0.5 or b > 0.5):
        return -np.inf

    v = v_vector(theta)
    f = np.exp(lnf)

    delta = v[0][0] * x + v[1][0] * y - b * np.cos(theta)
    sigma_sq = v[0][0]**2 * (f * x_err)**2 + v[1][0]**2 * y_err**2
    # ln_like = np.sum(-0.5 * (delta**2 / sigma_sq + np.log(sigma_sq) +
    #                  np.log(2*np.pi)))

    ln_like = np.sum(-0.5 * (delta**2 / (sigma_sq + V) + np.log(sigma_sq + V) +
                     np.log(2*np.pi)))

    return ln_like


def initial_odr_fit(s_apo, s_mwo, init_guess):
    """
    Use `~scipy.optimize.fmin` to minimize the chi^2 for initial parameters.

    Parameters
    ----------
    s_apo : `Measurement`
    s_mwo : `Measurement`

    init_guess : list or `~numpy.ndarray`

    Returns
    -------
    initial_params : `~numpy.ndarray`
    """
    initial_params = fmin(lambda *args, **kwargs: -ln_likelihood(*args, **kwargs),
                          init_guess, args=(s_apo.value, s_mwo.value,
                                            s_apo.err, s_mwo.err))
    return initial_params


def mcmc_fit(s_apo, s_mwo, init_guess, nwalkers, n_steps_burnin=1500,
             n_steps_postburnin=4000, ln_likelihood=ln_likelihood):
    ndim = len(init_guess)
    p0 = [[init_guess[0] + 0.05 * np.random.randn(),
           init_guess[1] + 0.01 * np.random.randn(),
           # init_guess[2] + 0.001 * np.random.randn()]
           init_guess[2] + 0.001 * np.random.randn(),
           init_guess[3] + 0.001 * np.random.randn()]
          for i in range(nwalkers)]

    args = (s_apo.value, s_mwo.value, s_apo.err, s_mwo.err)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_likelihood, args=args,
                                threads=2)

    # Burn in for this many steps:
    p1 = sampler.run_mcmc(p0, n_steps_burnin)[0]
    sampler.reset()

    # Now run for this many more steps:
    sampler.run_mcmc(p1, n_steps_postburnin)
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    return samples

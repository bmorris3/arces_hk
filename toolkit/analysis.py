from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np


def v_vector(theta):
    """
    Hogg+ 2010, Eqn 29.
    """
    return np.array([-np.sin(theta), np.cos(theta)]).T


def ln_likelihood(p, x, y, x_err, y_err):
    """
    Hogg+ 2010, Eqn 30., modified to allow x_err to float
    """
    theta, b, lnf = p

    if theta < 1.4 or theta > 1.6 or lnf < -5:
        return -np.inf

    z = np.array([x, y])
    v = v_vector(theta)
    f = np.exp(lnf)

    ln_like = 0
    for i in range(len(x)):
        z_i = np.atleast_2d(z[:, i]).T
        delta = np.dot(v.T, z_i) - b * np.cos(theta)
        s = np.array([[(f * x_err[i])**2, 0], [0, y_err[i]**2]])
        sigma_sq = np.dot(np.dot(v.T, s), v)
        ln_like -= 0.5 * ((delta**2 / sigma_sq) + np.log(sigma_sq) +
                          np.log(2*np.pi))

    return ln_like[0]


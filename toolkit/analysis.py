from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt


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
    z = np.array([x, y])
    v = v_vector(theta)

    ln_like = 0
    for i in range(len(x)):
        z_i = np.atleast_2d(z[:, i]).T
        delta = np.dot(v.T, z_i) - b * np.cos(theta)
        s = np.array([[x_err[i]**2, 0], [0, y_err[i]**2]])
        sigma_sq = np.dot(np.dot(v.T, s), v)

        # model = np.dot(v, np.array([x[i], y[i]]).T) - b * np.cos(theta)
        # inv_sigma2 = 1.0/(sigma_sq + model**2 * np.exp(2*lnf))
        # ln_like -= (delta**2 / 2 * inv_sigma2 - np.log(inv_sigma2))
        # model = np.dot(v, np.array([x[i], y[i]]).T) - b * np.cos(theta)

        ln_like -= delta**2 / 2 / sigma_sq

    return ln_like[0]


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
    Hogg+ 2010, Eqn 30.
    """
    theta, b = p
    z = np.array([x, y])
    v = v_vector(theta)

    ln_like = 0
    for i in range(len(x)):
        z_i = np.atleast_2d(z[:, i]).T
        delta = np.dot(v.T, z_i) - b * np.cos(theta)
        s = np.array([[x_err[i]**2, 0], [0, y_err[i]**2]])
        sigma_sq = np.dot(np.dot(v.T, s), v)
        ln_like -= delta**2 / 2 / sigma_sq

    return ln_like[0]


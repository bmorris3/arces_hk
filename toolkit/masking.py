import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def gaussian(x, a, x0, sigma):
    return a * np.exp(-0.5 * (x - x0)**2 / sigma**2)

# def derivative_gaussian(p, x, y):
#     a, x0, sigma = p
#     return -(x-x0)/sigma**2 * gaussian(x, a, x0, sigma)

def chi2(p, x, y):
    return np.sum((gaussian(x, *p) - y)**2)


def get_spectrum_mask(spectrum, cutoff=1.5):
    """
    Fit the raw, pre-normalized spectrum's blaze function
    with a Gaussian. Use a

    Parameters
    ----------
    spectrum : `~specutils.Spectrum1D`
        Spectrum to fit
    cutoff : float
        Mask channels greater than ``cuttoff``-sigma away from
        the peak flux.

    Returns
    -------
    mask : `~numpy.ndarray`
        Mask that excludes channels greater than ``cuttoff``-sigma away from
        the peak flux.
    """
    initp = np.array([spectrum.flux.max().value,
                      spectrum.wavelength.mean().value,
                      spectrum.wavelength.ptp().value/4])

    bestp = fmin_l_bfgs_b(chi2, initp[:],
                          approx_grad=True,
                          args=(spectrum.wavelength.value,
                                spectrum.flux.value),
                          bounds=[(0, np.inf),
                                  (spectrum.wavelength.min().value,
                                   spectrum.wavelength.max().value),
                                  (spectrum.wavelength.ptp().value/8,
                                   spectrum.wavelength.ptp().value/2)])[0]

    best_a, best_x0, best_sigma = bestp

    mask = ((spectrum.wavelength.value > best_x0 - cutoff * best_sigma) &
            (spectrum.wavelength.value < best_x0 + cutoff * best_sigma))

    return mask
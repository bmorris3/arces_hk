from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
from glob import glob

__all__ = ['glob_spectra_paths']


def glob_spectra_paths(data_dir, target_names):
    """
    Collect paths to spectrum FITS files.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing spectrum FITS files
    target_names : list
        String patterns that match the beginning of files with targets to
        collect.

    Returns
    -------
    spectra_paths : list
        List of paths to spectrum FITS files
    """
    # Collect files for each target:
    spectra_paths_lists = [glob(os.path.join(data_dir,
                                             '{0}*.wfrmcpc.fits'.format(name)))
                           for name in target_names]

    # Reduce to one list:
    spectra_paths = reduce(list.__add__, spectra_paths_lists)

    return spectra_paths

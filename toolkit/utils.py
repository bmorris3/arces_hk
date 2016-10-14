from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
from glob import glob
from astroquery.simbad import Simbad
from astropy.table import Table
from astropy.io import ascii

from .catalog import query_catalog_for_object

__all__ = ['glob_spectra_paths']


results_dir = '/astro/users/bmmorris/Dropbox/Apps/ShareLaTeX/CaII_HAT-P-11/results/'


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

def construct_standard_star_table(star_list, write_to=results_dir):

    names = []
    sp_types = []
    s_mwo = []
    sigma_mwo = []

    for star in star_list:
        names.append(star.upper())
        customSimbad = Simbad()
        customSimbad.add_votable_fields('sptype')
        sp_type = customSimbad.query_object(star)['SP_TYPE'][0]
        sp_types.append(sp_type)

        star_mwo_tbl = query_catalog_for_object(star)
        s_mwo.append(star_mwo_tbl['Smean'])
        sigma_mwo.append(star_mwo_tbl['e_Smean'])

    standard_table = Table([names, sp_types, s_mwo, sigma_mwo],
                           names=['Star', 'Sp.~Type', '$S_{MWO}$', '$\sigma_{MWO}$'])

    latexdict = dict(col_align='l l c c', preamble=r'\begin{center}',
                     tablefoot=r'\end{center}',
                     caption=r'Stars observed to calibrate the $S$-index '
                             r'(see Section~\ref{sec:def_s_index}). \label{tab:cals}',
                     data_start=r'\hline')

    output_path = os.path.join(results_dir, 'cal_stars.tex')

    # output_path,
    ascii.write(standard_table, format='latex', latexdict=latexdict)

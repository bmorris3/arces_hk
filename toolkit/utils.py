from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import functools
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.8f')

from glob import glob
from astroquery.simbad import Simbad
from astropy.table import Table
from astropy.io import ascii

from .catalog import query_catalog_for_object
from .activity import Measurement, SIndex, StarProps

__all__ = ['glob_spectra_paths', 'stars_to_json']


results_dir = '/astro/users/bmmorris/Dropbox/Apps/ShareLaTeX/CaII_HAT-P-11/results/'


def glob_spectra_paths(data_dir, target_names):
    """
    Collect paths to spectrum FITS files.

    Parameters
    ----------
    data_dir : str or list
        Paths to the directories containing spectrum FITS files
    target_names : list
        String patterns that match the beginning of files with targets to
        collect.

    Returns
    -------
    spectra_paths : list
        List of paths to spectrum FITS files
    """
    if type(data_dir) != list:
        data_dir = [data_dir]

    all_spectra_paths = []

    for d_dir in data_dir:

        # Collect files for each target:
        spectra_paths_lists = [glob(os.path.join(d_dir,
                                                 '{0}*.wfrmcpc.fits'.format(name)))
                               for name in target_names]

        # Reduce to one list:
        spectra_paths = functools.reduce(list.__add__, spectra_paths_lists)

        all_spectra_paths.extend(spectra_paths)

    return all_spectra_paths


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


def list_attrs(obj):
    attrs = [i for i in dir(obj) if not i.startswith('_')]
    return attrs


def floats_to_strings(d):
    dictionary = d.copy()
    for key in dictionary:
        dictionary[key] = str(dictionary[key])
    return dictionary


def stars_to_json(star_list, output_path='star_data.json'):
    """
    Save list of stellar properties to a JSON file.

    Parameters
    ----------
    star_list : list of `StarProps`
        Star properties to save to json
    output_path : str
        File path to output
    """
    stars_attrs = list_attrs(star_list[0])
    all_data = dict()

    for star in star_list:
        star_data = dict()

        for attr in stars_attrs:
            value = getattr(star, attr)

            if isinstance(value, Measurement) or isinstance(value, SIndex):
                value = floats_to_strings(value.__dict__)
            else:
                value = str(value)

            star_data[attr] = value

        all_data[star.name] = star_data

    with open(output_path, 'w') as w:
        json.dump(all_data, w, indent=4, sort_keys=True)


def json_to_stars(json_path):
    """
    Loads JSON archive into list of `StarProps` objects.

    Parameters
    ----------
    json_path : str
        Path to saved stellar properties

    Returns
    -------
    stars : list of `StarProps`
        List of stellar properties.
    """
    with open(json_path, 'r') as w:
        dictionary = json.load(w)

    stars = [StarProps.from_dict(dictionary[star]) for star in dictionary]
    return stars

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import functools
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.8f')
from glob import glob

import numpy as np
from astroquery.simbad import Simbad
from astropy.table import Table
from astropy.io import ascii
from astropy.time import Time

from .catalog import query_catalog_for_object
from .activity import Measurement, SIndex, StarProps

__all__ = ['glob_spectra_paths', 'stars_to_json', 'json_to_stars',
           'parse_hires']


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


# def construct_standard_star_table(stars, write_to=results_dir):
#
#     names = []
#     sp_types = []
#     s_mwo = []
#     s_apo = []
#
#     for star in stars:
#         names.append(star.name.upper())
#         customSimbad = Simbad()
#         customSimbad.add_votable_fields('sptype')
#         sp_type = customSimbad.query_object(star.name)['SP_TYPE'][0]
#         sp_types.append(sp_type)
#
#         s_mwo.append(star.s_mwo.to_latex())
#         s_apo.append(star.s_apo.to_latex())
#
#     standard_table = Table([names, sp_types, s_mwo, s_apo],
#                            names=['Star', 'Sp.~Type', '$S_{MWO}$', '$S_{APO}$'])
#
#     standard_table.sort(keys='$S_{MWO}$')
#
#     latexdict = dict(col_align='l l c c', preamble=r'\begin{center}',
#                      tablefoot=r'\end{center}',
#                      caption=r'Stars observed to calibrate the $S$-index '
#                              r'(see Section~\ref{sec:def_s_index}). \label{tab:cals}',
#                      data_start=r'\hline')
#
#     # output_path,
#     ascii.write(standard_table, format='latex', latexdict=latexdict,
#                 output='cal_stars.tex')
#

def combine_measurements(measurement_list):
    mean = np.mean([m.value for m in measurement_list])
    err = np.sqrt(np.sum(np.array([m.err for m in measurement_list])**2))
    return Measurement(value=mean, err=err, meta=len(measurement_list))


def construct_standard_star_table(stars, write_to=results_dir):

    mwo_dict = dict()
    apo_dict = dict()
    for star in stars:
        mwo_dict[star.name.upper()] = []
        apo_dict[star.name.upper()] = []

    names = list(mwo_dict.keys())

    for star_name in names:
        all_obs_this_star = [star for star in stars if star.name.upper() == star_name]
        mwo_dict[star_name] = combine_measurements([s.s_mwo for s in all_obs_this_star])
        apo_dict[star_name] = combine_measurements([s.s_apo for s in all_obs_this_star])

    sp_types = []
    s_mwo = []
    s_apo = []
    n_obs = []

    for star in names:
        s_mwo.append(mwo_dict[star].to_latex())
        s_apo.append(apo_dict[star].to_latex())

        n_obs.append(apo_dict[star].meta)

        customSimbad = Simbad()
        customSimbad.add_votable_fields('sptype')
        sp_type = customSimbad.query_object(star)['SP_TYPE'][0]
        sp_types.append(sp_type)

    # for star in stars:
    #     names.append(star.name.upper())
    #     customSimbad = Simbad()
    #     customSimbad.add_votable_fields('sptype')
    #     sp_type = customSimbad.query_object(star.name)['SP_TYPE'][0]
    #     sp_types.append(sp_type)
    #
    #     s_mwo.append(star.s_mwo.to_latex())
    #     s_apo.append(star.s_apo.to_latex())

    standard_table = Table([names, sp_types, s_mwo, s_apo, n_obs],
                           names=['Star', 'Sp.~Type', '$S_{MWO}$', '$S_{APO}$', '$N$'])

    standard_table.sort(keys='$S_{MWO}$')

    latexdict = dict(col_align='l l c c c', preamble=r'\begin{center}',
                     tablefoot=r'\end{center}',
                     caption=r'Stars observed to calibrate the $S$-index '
                             r'(see Section~\ref{sec:def_s_index}). \label{tab:cals}',
                     data_start=r'\hline')

    # output_path,
    ascii.write(standard_table, format='latex', latexdict=latexdict,
                output='cal_stars.tex')



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
    stars_attrs = star_list[0].__dict__.keys()
    all_data = dict()

    for star in star_list:
        star_data = dict()

        for attr in stars_attrs:
            value = getattr(star, attr)

            if isinstance(value, Measurement):
                value = floats_to_strings(value.__dict__)
            elif isinstance(value, SIndex):
                value = value.to_dict()
            else:
                value = str(value)

            star_data[attr] = value

        all_data[star.name + '; ' + str(star.time.datetime)] = star_data

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


def parse_hires(path):
    text_file = open(path, 'r').read().splitlines()

    header_line = text_file[0].split()
    data = {header: [] for header in header_line}

    for line in text_file[1:]:
        split_line = line.split()

        for i, header in enumerate(header_line):
            if header in ['Signal/Noise', 'ModJD', 'S-value']:
                data[header].append(float(split_line[i]))
            else:
                j = 1 if len(split_line) > len(header_line) else 0
                data[header].append(split_line[i+j] + split_line[i])

    table = Table(data)

    floats = np.array(table['ModJD'].data) + 2440000.0 #+ 4e4 - 0.5

    table['time'] = Time(floats, format='jd')

    return table


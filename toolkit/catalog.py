"""
Tools for measuring equivalent widths, S-indices.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = 1e10   # Otherwise would only show first 50 values

__all__ = ['get_duncan_catalog', 'sindex_catalog', 'query_catalog_for_object']

sindex_catalog = None
duncan1991 = 'III/159A'


def get_duncan_catalog():
    """
    Parameters
    ----------

    Returns
    -------
    """
    global sindex_catalog

    if sindex_catalog is None:
        catalogs = Vizier.get_catalogs(duncan1991)
        catalog_table = catalogs[0]  # This is the table with the data
        sindex_catalog = catalog_table

    return sindex_catalog


def query_catalog_for_object(identifier, catalog=duncan1991):
    """
    Parameters
    ----------
    identifier : str

    catalog : str (optional)

    Returns
    -------

    """
    query = Vizier.query_object(identifier, catalog=catalog)

    if len(query) > 0:
        return query[0][0]
    else:
        return dict(Smean=np.nan, Smin=np.nan, Smax=np.nan)

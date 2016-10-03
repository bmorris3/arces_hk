"""
Tools for measuring equivalent widths, S-indices.
"""
import numpy as np
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier

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
        Vizier.ROW_LIMIT = -1   # Otherwise would only show first 50 values
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
    return Vizier.query_object(identifier, catalog=catalog)[0][0]
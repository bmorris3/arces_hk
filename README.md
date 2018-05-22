# arces_hk
Measuring chromospheric activity with APO ARC 3.5 m + ARCES. 

[![DOI](https://zenodo.org/badge/69041786.svg)](https://zenodo.org/badge/latestdoi/69041786) [![](http://img.shields.io/badge/arXiv-1709.03913-orange.svg?style=flat)](https://arxiv.org/abs/1709.03913)


Workflow
========

* Add star directories and names to `mwo.py`, then run it
* Add dates and names to `hat11.py`, then run it
* Run `calibrate.py` to determine the APO-MWO calibration constants
* Run `plot_calibrated_hat11.py` to see the calibrated HAT-P-11 S-indices

Citation
--------

If you make use of this code, please cite [Morris et al 2017](http://adsabs.harvard.edu/abs/2017ApJ...848...58M):
```
@ARTICLE{Morris2017,
   author = {{Morris}, B.~M. and {Hawley}, S.~L. and {Hebb}, L. and {Sakari}, C. and 
	{Davenport}, J.~R.~A. and {Isaacson}, H. and {Howard}, A.~W. and 
	{Montet}, B.~T. and {Agol}, E.},
    title = "{Chromospheric Activity of HAT-P-11: An Unusually Active Planet-hosting K Star}",
  journal = {\apj},
archivePrefix = "arXiv",
   eprint = {1709.03913},
 primaryClass = "astro-ph.SR",
 keywords = {methods: observational, planet{\ndash}star interactions, stars: activity, stars: chromospheres, starspots},
     year = 2017,
    month = oct,
   volume = 848,
      eid = {58},
    pages = {58},
      doi = {10.3847/1538-4357/aa8cca},
   adsurl = {http://adsabs.harvard.edu/abs/2017ApJ...848...58M},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```

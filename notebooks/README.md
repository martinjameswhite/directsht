# Notebooks

This directory contains notebooks that illustrate how to use the code
and give some examples of typical calculations.  It also contains
demonstrations that the code behaves as expected and comparisons
with other approaches.

In addition to the notebooks there are a few scripts that can be run.
These were designed to allow the simple use of NERSC compute nodes for
larger jobs, rather than running them on a shared Jupyter server.

making_maps.ipynb generates spherical harmonic coefficients for a set
of delta functions and then uses Healpy maps to visualize the maps
generated from these coefficients.  It illustrates the ringing from a
truncated SHT, and shows that our coefficients are returned in a format
compatible with Healpy's conventions for manipulating maps and SHT
coefficients.

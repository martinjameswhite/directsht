# Notebooks

This directory contains notebooks that illustrate how to use the code
and give some examples of typical calculations.  It also contains
demonstrations that the code behaves as expected and comparisons
with other approaches.

In addition to the notebooks there are a few scripts that can be run.
These were designed to allow the simple use of NERSC compute nodes for
larger jobs, rather than running them on a shared Jupyter server.

***

`analyzing_mocks.ipynb` is the main notebook, and it compares different
techniques for performing a harmonic-space two-point analysis (i.e. an
angular power spectrum) on a point set.

`golden_spiral.ipynb` generates a set of points laid out in a Golden
spiral (a.k.a. Fibonacci spiral) pattern and then analyzes their
angular clustering in harmonic space.  This somewhat artificial example
demonstrates point sets with significant small-scale power can be tricky
to analyze with pixelized maps (unless higher-order charge assignment
schemes are used, which we do not consider here).

`making_maps.ipynb` generates spherical harmonic coefficients for a set
of delta functions and then uses Healpy maps to visualize the maps
generated from these coefficients.  It illustrates the ringing from a
truncated SHT, and shows that our coefficients are returned in a format
compatible with Healpy's conventions for manipulating maps and SHT
coefficients.

`making_mocks.ipynb` gives an example of how to make a set of points whose
clustering can be analyzed.  This uses a crude version of a lognormal mock
method, code for which is distributed with the repo.  We use these mocks
primarily as an example dataset to analyze to show how different approaches
compare and how well they agree.

`mode_coupling_mat.ipynb` demonstrates that the "helper code" we provide
to compute the mode-coupling matrix and the window matrix for the
pseudo-spectrum method works as advertized.

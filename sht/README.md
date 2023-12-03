# Direct spherical harmonic transform code.

Given a set of points on the sphere, with weights, computes the harmonic
transform coefficients by direct summation.  See the notebooks directory
in the repo for examples of how to use the code.

In addition there is some "helper" code to compute the mode-coupling
and window matrices that are used in the pseudo-spectrum method.  These
are provided for convenience and since they do not need to be recomputed
very often the methods were not written in an optimized manner.  Finally
some utilities are used as part of the SHT code.

As part of the testing of the code we use point sets generated from a
lognormal mock catalog.  A (somewhat crude) code for producing such a
set of points is provided as well.

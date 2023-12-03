# direct_sht
Direct spherical harmonic transform code for point sets on the sphere.

The code can be installed with

python3 -m pip install -v git+https://github.com/martinjameswhite/direct_sht

It requires numpy, scipy and numba.  If JAX is available it can be used to
speed up the computation, but the code will automatically fall back to numpy
if JAX is not present.

***

## Usage

The code is relatively straightforward to use.  Given a set of points
defined by arrays of theta and phi (in radians) and weights the DirectSHT
class can be called to provide the spherical harmonic transform coefficients
alm.

First import the main class and create an instance:
```
from sht.sht import DirectSHT

# The class takes a number of multipoles to cmopute (lmax=Nl-1)
# and the number of spline points for interpolation.
# Typically Nx ~ 2 Nl is a good choice.
Nl = 500
Nx = 1024 
# Generate an instance of the class.
sht= DirectSHT(Nl,Nx)
```
If Nl and Nx are large then creating an instance can take a few seconds
because the code computes a table of spherical harmonics during initialization.

Then the transform can be done with
```
alms = sht(thetas,phis,weights)
```
The code returns a complex array stored in the same convention as Healpy
uses.  You can get the index of a given ell,m mode using the "indx" method
of the class.

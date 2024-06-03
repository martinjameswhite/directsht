# directsht

Direct spherical harmonic transform code for point sets on the sphere.
More details can be found in

http://arxiv.org/abs/2312.12285 \
Harmonic analysis of discrete tracers of large-scale structure

The code can be installed with
```
python3 -m pip install -v git+https://github.com/martinjameswhite/directsht
```
It requires numpy, scipy, healpy and numba.  If JAX is available, it can be used to
speed up the computation, but the code will automatically fall back to numpy
if JAX is not present.  Some of the notebooks use healpy for visualization, the
main code uses healpy only for coordinate rotations.

The code is much faster when run on GPUs. When they are available and JAX is installed, the code automatically distributes computation and memory across them.

***

## Usage

The code is relatively straightforward to use.  Given a set of points,
defined by arrays of theta and phi (in radians) and weights, the DirectSHT
class can be called to provide the spherical harmonic transform coefficients
alm.

First import the main class and create an instance:
```
from sht.sht import DirectSHT

# The class takes a number of multipoles to compute (lmax=Nl-1)
# and the number of spline points for interpolation.
# Typically Nx ~ Nl and Nx ~ 2 Nl is a conservative choice.
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

We give several examples of how to compute alms for different sets of points,
do a pseudo-spectrum calculation for mock galaxies (generated by the LogNormalMocks
class and using the MaskDeconvolution class to handle the mode-coupling matrices
and window functions) and look at how the code performs in Jupyter notebooks within
the `notebooks` directory.  Please look there for further information.

# Attribution
If you find this code useful, please cite the original paper
```
@ARTICLE{2024JCAP...05..010B,
       author = {{Baleato Lizancos}, Ant{\'o}n and {White}, Martin},
        title = "{Harmonic analysis of discrete tracers of large-scale structure}",
      journal = {\jcap},
     keywords = {galaxy surveys, power spectrum, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2024,
        month = may,
       volume = {2024},
       number = {5},
          eid = {010},
        pages = {010},
          doi = {10.1088/1475-7516/2024/05/010},
archivePrefix = {arXiv},
       eprint = {2312.12285},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024JCAP...05..010B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
'''
and code
```
@software{2024ascl.soft05011B,
       author = {{Baleato Lizancos}, Ant{\'o}n and {White}, Martin},
        title = "{DirectSHT: Direct spherical harmonic transform}",
 howpublished = {Astrophysics Source Code Library, record ascl:2405.011},
         year = 2024,
        month = may,
          eid = {ascl:2405.011},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ascl.soft05011B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
'''



# Code to generate a lognormal catalog of approximately Npnt
# points, assuming a Gaussian angular power spectrum, clg.

import numpy as np
import healpy as hp


def lognormal_catalog(Npnt,clg,nside=256, verbose=False):
    """Returns (theta,phi,wt) for about Npnt points.  This
    should be run with Npnt<<Npix=12*nside^2."""
    # Get a random number generator.
    rng = np.random.default_rng()
    gmap = hp.synfast(clg,nside,alm=False,pol=False)
    if verbose:
        print("gmap in range ({:e},{:e})".format(np.min(gmap),np.max(gmap)))
    emap = np.exp(gmap)
    emap*= Npnt/np.sum(emap)
    if verbose:
        print("emap in range ({:e},{:e})".format(np.min(emap),np.max(emap)))
    ngal = rng.poisson(lam=emap,size=emap.size)
    ipix = np.nonzero(ngal>0)[0]
    thta,phi = hp.pix2ang(nside,ipix,lonlat=False)
    wt   = np.ones_like(thta)
    # Now very slightly perturb the positions away from
    # the pixel centers.
    blur = np.sqrt( hp.pixelfunc.nside2pixarea(nside) )
    thta+= blur*rng.uniform(low=-0.5,high=0.5,size=thta.size)
    phi += blur*rng.uniform(low=-0.5,high=0.5,size=thta.size)
    return( (thta,phi,wt) )
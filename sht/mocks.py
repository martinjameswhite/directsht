# Code to generate a lognormal catalog of approximately Npnt
# points, assuming a Gaussian angular power spectrum, clg.

import numpy as np
import healpy as hp




class LogNormalMocks:
    def __init__(self, Npnt, nside=2048, clg=None, verbose=False,
                 alpha=1.1, lmax=1000, ell0 = 10.,
                 theta_range=(0, np.pi), phi_range=(0, 2 * np.pi)):
        """
        Generate a lognormal catalog of sources with some masking
        :param Npnt: int. Number of points in the catalog
        :param nside: int. Healpix nside parameter
        :param clg: 1D numpy array. Angular power spectrum of the Gaussian field.
        :param verbose: bool. Whether to print out information about the catalog
        :param alpha: float. Power law index of the (Gaussian) angular power spectrum
        :param lmax: int. Maximum ell for the angular power spectrum
        :param ell0: float. The knee of the default Gaussian angular power spectrum
        :param theta_range: tuple of floats btw (0, np.pi). The range of theta to allow
        :param phi_range: tuple of floats btw (0, 2 * np.pi). The range of theta to allow
        """
        self.Npnt        = Npnt
        self.nside       = nside
        self.verbose     = verbose
        self.theta_range = theta_range
        self.phi_range   = phi_range
        #
        if clg is None:
            ell = np.arange(lmax)
            self.clg = 0.001 * (ell0 / (ell + ell0))**alpha
        else:
            self.clg = clg
        # 
    def __call__(self, seed=None, verbose=True):
        """
        Generate the catalog and mask it
        :param seed: int. Random seed to allow for reproducibility of mock catalog
        :return:
        """
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)
        thta, phi, wt = self.lognormal_catalog(verbose)
        mask = self.make_mask(thta, phi)
        return( (thta[mask],phi[mask],wt[mask]) )
        #
    def lognormal_catalog(self, verbose=False):
        """
        Returns (theta,phi,wt) for about Npnt points.  This
        should be run with Npnt<<Npix=12*nside^2.
        """
        gmap = hp.synfast(self.clg, self.nside, alm=False, pol=False)
        if verbose:
            print("gmap in range ({:e},{:e})".format(np.min(gmap), np.max(gmap)))
        emap = np.exp(gmap)
        emap*= self.Npnt / np.sum(emap)
        if verbose:
            print("emap in range ({:e},{:e})".format(np.min(emap), np.max(emap)))
        ngal = self.rng.poisson(lam=emap, size=emap.size)
        # Since, by assumption, Npnt<<Npix we have 0 or 1 objects per pixel.
        ipix     = np.nonzero(ngal > 0)[0]
        thta,phi = hp.pix2ang(self.nside,ipix,lonlat=False)
        wt       = np.ones_like(thta)
        # Now very slightly perturb the positions away from
        # the pixel centers.
        blur = np.sqrt(hp.pixelfunc.nside2pixarea(self.nside))
        thta += blur * self.rng.uniform(low=-0.5,high=0.5,size=thta.size)
        phi  += blur * self.rng.uniform(low=-0.5,high=0.5,size=thta.size)
        return ((thta,phi,wt))
        #
    def make_mask(self, thetas, phis):
        """
        Make a mask for the catalog
        :param thetas: Full-sky array of theta values
        :param phis:   Full-sky array of phi values
        :return: The indices of the catalog that are in the observed region
        """
        # Cut to the "observed region"
        return (np.nonzero((thetas > self.theta_range[0]) & (thetas < self.theta_range[1]) & \
                        (phis >  self.phi_range[0]) & (phis < self.phi_range[1]))[0])

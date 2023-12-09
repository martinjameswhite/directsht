# Code to generate a lognormal catalog of approximately Npnt
# points, assuming a Gaussian angular power spectrum, clg.

import numpy as np
import healpy as hp
from scipy.special import roots_legendre, eval_legendre



class LogNormalMocks:
    def __init__(self, Npnt, nside=2048, lmax=1000, 
                 clg=None, cl_ln=None, norm=0.01, alpha=2., ell0=10.,
                 theta_range=(0, np.pi), phi_range=(0, 2 * np.pi), verbose=False):
        """
        Generate a lognormal catalog of sources with some masking
        :param Npnt: int. Number of points in the catalog
        :param nside: int. Healpix nside parameter
        :param lmax: int. Maximum ell for the angular power spectrum
        :param clg: 1D numpy array. Angular power spectrum of the Gaussian field.
        :param cl_ln: 1D numpy array. Angular power spectrum of the lognormal field.
        :param norm: float. Normalization of the default (Gaussian) power spectrum.
        :param alpha: float. Power law index of the (Gaussian) angular power spectrum
        :param ell0: float. The knee of the default Gaussian angular power spectrum
        :param theta_range: tuple of floats btw (0, np.pi). The range of theta to allow
        :param phi_range: tuple of floats btw (0, 2 * np.pi). The range of theta to allow
        :param verbose: bool. Whether to print out information about the catalog
        """
        assert (cl_ln is None) or (clg is None), "Cannot specify both cl_ln and clg"
        self.Npnt        = Npnt
        self.nside       = nside
        self.verbose     = verbose
        self.theta_range = theta_range
        self.phi_range   = phi_range
        #
        if cl_ln is not None:
            self.clg = get_gauss_cl_from_ln_cl(cl_ln)
        elif clg is not None:
            self.clg = clg
        else:
            ell = np.arange(lmax)
            self.clg = norm * (ell0/(ell+ell0))**alpha
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
            print("gmap in range ({:e},{:e})".format(np.min(gmap),np.max(gmap)))
        emap = np.exp(gmap)
        smap = np.sum(emap)
        mmax = np.max(emap)
        idone= 0
        imax = int(2e-4 * 12*self.nside**2) # Most points in any iteration.
        tlst,plst,wlst = np.array([]),np.array([]),np.array([])
        while idone<self.Npnt:
            itry = min(imax,self.Npnt-idone)
            fact = float(itry)/smap
            if verbose:
                print("scaling factor {:e}, maxprob={:e}".format(fact,fact*mmax))
            ngal = self.rng.poisson(lam=fact*emap,size=emap.size)
            # Since, by assumption, Npnt<<Npix we have 0 or 1 objects per pixel.
            ipix     = np.nonzero(ngal > 0)[0]
            thta,phi = hp.pix2ang(self.nside,ipix,lonlat=False)
            wt       = np.ones_like(thta)
            # Now very slightly perturb the positions away from
            # the pixel centers.
            blur   = np.sqrt(hp.pixelfunc.nside2pixarea(self.nside))
            thta  += blur * self.rng.uniform(low=-0.5,high=0.5,size=thta.size)
            phi   += blur * self.rng.uniform(low=-0.5,high=0.5,size=phi.size)
            tlst   = np.append(tlst,thta)
            plst   = np.append(plst,phi)
            wlst   = np.append(wlst,wt)
            idone += itry
        return( (tlst,plst,wlst) )
        #
    def make_mask(self,thetas,phis):
        """
        Make a mask for the catalog
        :param thetas: Full-sky array of theta values
        :param phis:   Full-sky array of phi values
        :return: The indices of the catalog that are in the observed region
        """
        # Cut to the "observed region"
        return( np.nonzero((thetas > self.theta_range[0]) & (thetas < self.theta_range[1]) & \
                        (phis >  self.phi_range[0]) & (phis < self.phi_range[1]))[0] )

    def get_theory_Cl(self, lmax_out=None, shot_noise=True, gauss_order=1000):
        """
        Get the theory Cl's for the log-normal mocks
        :param lmax_out: int.
            Maximum ell to compute the Cl's for. Defaults to lmax of Gaussian field in exponent
        :param shot_noise: bool.
            Whether to include shot noise in the returned Cls. Default is True/yes.
        :param gauss_order: int.
            Order of Gauss-Legendre quadrature to use in integration.
        :return: np.ndarray.
            1D numpy array of shape (lmax+1) containing the lognormal Cl's
        """
        if lmax_out is None:
            lmax_out = len(self.clg)-1
        # Get nodes and weights for Gauss-Legendre quadrature
        xs, weights = roots_legendre(gauss_order)
        # Compute the correlation function of the Gaussian field
        gauss_corrfunc = get_corrfunc_from_Cl(self.clg, xs)
        # Compute the correlation function of the lognormal field
        ln_corrfunc = np.exp(gauss_corrfunc) - 1
        # Compute the Cl's of the lognormal field
        ln_cls = get_Cl_from_corrfunc(ln_corrfunc, xs, weights, lmax_out)
        if shot_noise:
            sn = 1/float(self.Npnt) * (4*np.pi) # TODO: implement sth that allows for non-uniform weighting
        else:
            sn = 0
        return( ln_cls + sn )

def get_corrfunc_from_Cl(cls, xs):
    """
    Compute correlation function from Cl's
    :param cls: np.ndarray.
        1D numpy array containing Cl's at consecutive multipoles
    :param xs: np.ndarray
        1D numpy array with values [-1,1] at which to evaluate the correlation
         function ( note that x=cos\theta ).
    :return: np.ndarray.
        correlation function C(\theta=arccos(xs))
    """
    corr_func = np.zeros_like(xs)
    for ell, cl in enumerate(cls):
        corr_func += (2*ell+1) * cl * eval_legendre(ell, xs)
    return( corr_func/(4*np.pi) )

def get_Cl_from_corrfunc(cf_at_xs, xs, weights, lmax):
    """
    Compute Cl's from correlation function using Gauss-Legendre quadrature
    :param cf_at_xs: np.ndarray.
        1D array containing the correlation function evaluated at xs i.e. C(\theta=arccos(xs))
    :param xs: np.ndarray.
        1D numpy array with values [-1,1]. The roots of the Legendre polynomial of order equal
        to the order of our quadrature rule.
    :param weights: np.ndarray.
        1D numpy array with weights for the Gauss-Legendre quadrature
    :param lmax: int.
        Maximum ell to compute the Cl's for.
    :return: np.ndarray.
        1D numpy array of shape (lmax+1) containing the Cl's
    """
    cls = np.zeros(lmax+1)
    for ell in range(lmax+1):
        cls[ell] = np.dot(weights, cf_at_xs*eval_legendre(ell,xs) )
    return( 2*np.pi*cls )

def get_gauss_cl_from_ln_cl(ln_cl, gauss_order=1000):
    """
    Compute the Cl's of the Gaussian field from the Cl's of the lognormal field
    :param ln_cl: np.ndarray.
        1D numpy array containing the Cl's of the lognormal field
    :param gauss_order: int.
        Order of Gauss-Legendre quadrature to use in integration.
    :return: np.ndarray.
        1D numpy array of same length as ln_cl with the Cl's of the Gaussian field
    """
    xs, weights = roots_legendre(gauss_order)
    ln_corrfunc = get_corrfunc_from_Cl(ln_cl, xs)
    gauss_corrfunc = np.log(ln_corrfunc + 1)
    return get_Cl_from_corrfunc(gauss_corrfunc, xs, weights, len(ln_cl)-1)

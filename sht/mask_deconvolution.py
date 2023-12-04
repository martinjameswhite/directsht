#!/usr/bin/env python
#
# Code to deconvolve the mode-coupling of pseudo-Cls
# induced by a mask.  Since this code is run only
# rarely we have not attempted any optimizations.
#

import numpy  as np
import sys

sys.path.append('../sht')
from  threej000 import Wigner3j




class MaskDeconvolution:
    def __init__(self,Nl,W_l,verbose=True):
        """
        Class to deconvolve the mode-coupling of pseudo-Cls.

        It computes the necessary Wigner 3j symbols and mode-coupling matrix on
        initialization so that they do not have to be recomputed on successive calls
        to mode-decouple the pseudo-Cls of noise-debiased bandpowers.

        :param Nl: int. The number of multipoles to compute the mode-coupling matrix for.
                        The maximum ell, lmax, is Nl-1.
        :param W_l: 1D numpy array. Window function. Must be provided at every ell.
                    If shorter than 2*lmax will be right-padded with zeros.
        :param verbose: bool. Whether to print out information about progress
        """
        self.Nl   = Nl
        self.lmax = Nl-1
        pad       = max(0,2*Nl-1-W_l.size)
        self.W_l  = np.pad(W_l,(0,pad),'constant',constant_values=0)
        # 
        # Precompute the expensive stuff
        if verbose:
            print("Precomputing Wigner 3j symbols...")
        # Precompute the required Wigner 3js
        self.w3j000 = Wigner3j(2*Nl-1)
        #
        if verbose:
            print("Computing the mode-coupling matrix...")
        # Compute the mode-coupling matrix
        self.Mll = self.get_M()
        #
    def __call__(self,Cl,bins):
        """
        Compute the mode-decoupled bandpowers given some binning scheme.
        :param C_l: 1D numpy array of length self.lmax+1.
                    Per-ell angular power spectrum of the signal.
        :param bins: An Nbin x Nell matrix to perform the binning.
                    It helps to have an additional bin on either end that is
                    computed but not used, just to avoid edge effects with
                    the mode decoupling, etc.
        :return: tuple of (1D numpy array, 1D numpy array). The first array contains
                    the ells at which the bandpowers are computed. The second array
                    contains the mode-decoupled bandpowers.
        """
        # Adjust the length of the input Cl.
        if len(Cl)<self.Nl:
            Clx = np.pad(Cl,(0,self.Nl-len(Cl)),'constant',constant_values=0)
        elif len(Cl)>self.Nl:
            Clx = Cl[:self.Nl]
        else:
            Clx = Cl
        # Get the inverse bin-bin mode-coupling matrix.
        Mbb_inv = self.get_Mbb_inv(bins)
        # Bin the Cls.
        Cb = np.dot(bins,Clx)
        # Mode-decouple the bandpowers
        Cb_decoupled = self.decouple_Cls(Mbb_inv,Cb)
        # Compute the binned ells.
        binned_ells = np.dot(bins,np.arange(self.Nl))/np.sum(bins,axis=1)
        return( (binned_ells,Cb_decoupled) )
        #
    def convolve_theory_Cls(self,Clt,bins):
        """
        Convolve some theory Cls with the bandpower window function
        :param Clt: 1D numpy array of length self.lmax+1. Theory Cls
        :param bins: An Nbin x Nell matrix to perform the binning.
        :return: 1D numpy array
        """
        assert (len(Clt)>=self.Nl), ("Clt must be provided up to lmax")
        # Get the window matrix.
        Mbl = self.window_matrix(bins)
        Cb_decoupled = np.dot(Mbl,Clt[:self.Nl])
        # Compute the binned ells.
        binned_ells = np.dot(bins,np.arange(self.Nl))/np.sum(bins,axis=1)
        return( (binned_ells,Cb_decoupled) )
        #
    def window_matrix(self,bins):
        """
        Returns the window matrix that converts a theory C_l into the
        binned and decoupled pseudo-spectra.
        :param bins: An Nbin x Nell matrix to perform the binning.
        :return 2D numpy array
        """
        # Get the inverse binned matrix
        Mbb_inv = self.get_Mbb_inv(bins)
        # Bin the theory Cl's.
        Mbl = np.matmul(Mbb_inv,np.matmul(bins,self.Mll))
        return(Mbl)
        #
    def W(self,l,debug=False):
        """
        Window function for a given multipole l.
        :param l: int. Multipole to evaluate the window function at.
        :param debug: Bool. If True, check the mode-coupling matrix becomes 1 in the full-sky.
        :return: float. Value of the window function at multipole l.
        """
        if debug:
            # In the full sky, the mode-coupling matrix should become the identity matrix
            if l == 0:
                return(4*np.pi)  # [\int d\hat{n} Y^*_{00}(\hat{n})]^2
            else:
                return(0)
        else:
            return self.W_l[l]
        #
    def get_M(self,debug=False):
        """
        Compute the per-multipole mode-coupling matrix M_{l1,l2} for a given lmax.
        :param lmax: int. Maximum multipole to compute the mode-coupling matrix for.
        :param debug: Bool. If True, check the matrix becomes the identity in the full-sky limit.
        :return: 2D array of shape (lmax+1,lmax+1) containing the mode-coupling matrix.
        """
        M = np.zeros((self.lmax+1, self.lmax+1))
        for l1 in range(self.lmax+1):
            for l2 in range(self.lmax+1):
                for l3 in range(abs(l1-l2),l1+l2+1):
                    if (l1+l2+l3)%2==0:
                        M[l1,l2] += (2*l3+1)*self.W(l3,debug) *\
                                    self.w3j000(l1,l2,l3)**2
                                    
                M[l1,l2] *= 2*l2+1
        M /= 4*np.pi
        return(M)
        #
    def get_Mbb_inv(self,bins):
        """
        Returns the bin-bin mode-coupling matrix.
        This is meant largely for internal use.
        """
        # Use where the binning_matrix is non-zero to define the ells for which
        # our bandpowers would be assumed to be constants.
        bins_no_wt = np.zeros_like(bins)
        bins_no_wt[bins>0] = 1.0
        # Bin the mode-coupling matrix into those bins.
        Mbb = np.matmul(np.matmul(bins,self.Mll),bins_no_wt.T)
        # Invert the binned matrix
        Mbb_inv = np.linalg.inv(Mbb)
        return(Mbb_inv)
        #
    def binning_matrix(self,type='linear',start=0,step=16):
        """
        Returns a 'binning matrix', B, such that B.vec is a binned
        version of vec.  This is a "helper" function.
        :param type: Type of binning.
                     'linear' (default) gives linear binning.
        :param step: size of linear step.
        :return 2D array of shape (Nbins,lmax).
        """
        Nl   = self.lmax+1
        bins = np.zeros( (Nl,Nl) )
        if type=='linear':
            dell = lambda ell: step
        elif type=='sqrt':
            dell = lambda ell: int(np.ceil(np.sqrt(4.*ell)+step))
        else:
            raise RuntimeError("Unknown step type.")
        ii = 0
        l0 = start
        l1 = l0 + dell(l0)
        while l1<=Nl:
            bins[ii,l0:min(l1,Nl)] = 1/float(l1-l0)
            l0,l1 = l1,l1+dell(l1)
            ii   += 1
        bins = bins[:ii,:]
        return(bins)
        #
    def decouple_Cls(self,Minv,Cb):
        """
        Mode-decouple some bandpowers
        :param Minv: Inverse of the binned mode-coupling matrix
        :param Cb: 1D array containing the binned Cls
        :return: 1D array containing the mode-decoupled bandpowers
        """
        return( np.matmul(Cb,Minv) )
        #

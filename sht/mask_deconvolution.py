#!/usr/bin/env python
#
# Code to deconvolve the mode-coupling of pseudo-Cls
# induced by a mask
#

import numpy  as np
import sys
sys.path.append('../sht')
from  threej000 import Wigner3j
from scipy.interpolate import interp1d

class MaskDeconvolution:
    def __init__(self, lmax, C_l, N_l, W_l, W_l_ells, lperBin=2 ** 4, verbose=True):
        """
        Class to deconvolve the mode-coupling of pseudo-Cls
        :param lmax: int. Maximum multipole to compute the mode-coupling matrix for.
        :param C_l: 1D numpy array. Per-ell angular power spectrum of the signal.
        :param N_l: 1D numpy array. Per-ell angular power spectrum of the noise.
        :param W_l: 1D numpy array. Window function.
        :param W_l_ells: 1D numpy array. Ells at which the window function is provided.
               The maximum ell must be >= than 2*lmax for deconvolution to work.
        :param lperBin: int. Number of ells per bin.
        :param verbose: bool. Whether to print out information about progress
        """
        assert ((lmax+1) % lperBin == 0), "lmax+1 must be a multiple of lperBin"
        assert (W_l_ells[-1] >= 2*lmax), "W_l must be provided up to at least 2*lmax"
        self.lmax = lmax
        self.lperBin = lperBin

        if verbose:
            print("Precomputing Wigner 3j symbols...")
        # Precompute the required Wigner 3js
        self.w3j000 = Wigner3j(2 * lmax + 1)
        # Interpolate the window function (in case it's not provided at every ell)
        self.W_l = interp1d(W_l_ells, W_l, kind='cubic')
        if verbose:
            print("Computing the mode-coupling matrix...")
        # Compute the mode-coupling matrix
        self.Mll = self.get_M()
        if verbose:
            print("Binning, inverting and decoupling...")
        # Bin the matrix
        self.init_binning()
        self.Mbb = self.bin_matrix(self.Mll)
        # Invert the binned matrix
        self.Mbb_inv = np.linalg.inv(self.Mbb)
        # Bin the Cls and Nls
        self.Cb = self.bin_Cls(C_l); self.Nb = self.bin_Cls(N_l)
        # Mode-decouple the noise-debiased bandpowers
        self.Cb_decoupled = self.decouple_Cls(self.Mbb_inv, self.Cb, self.Nb)

    def W(self, l, debug=False):
        """
        Window function for a given multipole l.
        :param l: int. Multipole to evaluate the window function at.
        :param debug: Bool. If True, check the mode-coupling matrix becomes 1 in the full-sky.
        :return: float. Value of the window function at multipole l.
        """
        if debug:
            # In the full sky, the mode-coupling matrix should become the identity matrix
            if l == 0:
                return 4 * np.pi  # [\int d\hat{n} Y^*_{00}(\hat{n})]^2
            else:
                return 0
        else:
            return self.W_l(l)

    def get_M(self, debug=False):
        """
        Compute the per-multipole mode-coupling matrix M_{l1,l2} for a given lmax.
        :param lmax: int. Maximum multipole to compute the mode-coupling matrix for.
        :param debug: Bool. If True, check the matrix becomes the identity in the full-sky limit.
        :return: 2D array of shape (lmax+1,lmax+1) containing the mode-coupling matrix.
        """
        M = np.zeros((self.lmax + 1, self.lmax + 1))
        for l1 in range(self.lmax + 1):
            for l2 in range(l1, self.lmax + 1):
                for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                    if (l1 + l2 + l3) % 2 == 0:
                        M[l1, l2] += ((2 * l3 + 1) * (2 * l2 + 1)
                                      * self.w3j000(l1, l2, l3) ** 2
                                      * self.W(l3, debug) / (4 * np.pi))
                        M[l2, l1] = M[l1, l2]
        return M

    def init_binning(self):
        """
        Set up the binning matrix to combine Cls and mode-coupling
        matrix into coarses ell bins.
        """
        self.bins = np.zeros(((self.lmax+1) // self.lperBin, self.lmax + 1))
        for i in range(0, self.lmax + 1, self.lperBin):
            self.bins[i // self.lperBin, i:i + self.lperBin] = 1 / float(self.lperBin)
        # Also set it such that we drop the ell=0 bin
        # when we do our average(s).
        self.bins[0, 0] = 0.0
        self.binned_ells = np.dot(self.bins, np.arange(self.lmax + 1))

    def bin_matrix(self, M):
        """
        Bin the mode-coupling matrix into bandpowers
        :param M: 2D array of shape (lmax+1,lmax+1) containing
                  the mode-coupling matrix.
        :return: 2D array of shape (lmax+1//lperBin,lmax+1//lperBin)
                  containing the binned mode-coupling matrix
        """
        return np.matmul(np.matmul(self.bins, M), self.bins.T)

    def bin_Cls(self, Cl):
        """
        Bin the Cls into bandpowers
        :param Cl: 1D array of shape (lmax+1) containing the Cls
        :return: 1D array of shape (lmax+1//lperBin) containing the binned Cls
        """
        return np.dot(self.bins, Cl)

    def decouple_Cls(self, Minv, Cb, Nb):
        """
        Noise-debias and bode-decouple some bandpowers
        :param Minv: 2D array of shape (lmax+1//lperBin,lmax+1//lperBin)
                     containing the inverse of the binned mode-coupling matrix
        :param Cb: 1D array of shape (lmax+1//lperBin) containing the binned Cls
        :param Nb: 1D array of shape (lmax+1//lperBin) containing the binned noise Nls
        :return: 1D array of shape (lmax+1//lperBin) containing the noise-debiased and
                    mode-decoupled bandpowers
        """
        return np.matmul(Cb-Nb, Minv)


import numpy as np
import numba as nb

import sht.shared_interp_funcs as shared_interp_funcs


def get_vs(mmax, phi_data_reshaped, reshaped_inputs, loop_in_JAX=True, N_chunks=None,
           pad=False, verbose=False):
    """
    Wrapper function to call get_vs from shared_interp_funcs
    """
    return shared_interp_funcs.get_vs(mmax, phi_data_reshaped, reshaped_inputs,
                                      loop_in_JAX=loop_in_JAX, N_chunks=N_chunks,
                                      pad=pad, verbose=verbose)

def get_vs_np(mmax, phi_data_reshaped, reshaped_inputs):
    """
    Calculate the v's for each m using numpy.
    :param mmax: int. Maximum m value in the calculation
    :param phi_data_reshaped: 2D numpy array of shape (bin_num, bin_len) with data phi values,
            zero-padded to length bin_len in bins with fewer points
    :param reshaped_inputs: 2D numpy array of shape (4, bin_num, bin_len) with zero padding as
            in phi_data_reshaped. The 1st dimension corresponds to the four auxiliary arrays in
            the calculation of the v's.
    :return: a tuple of two 3D numpy arrays of shape (mmax+1, 4, bin_num) with the real and
            imaginary parts of the v's at each m.

    NOTE: This function can be trivially parallelized, but we don't do that here.
    """
    vs_r = np.zeros((mmax+1, 4, phi_data_reshaped.shape[0]))
    vs_i=vs_r.copy()
    for m in range(mmax+1):
        vs_r[m,:,:], vs_i[m,:,:] = shared_interp_funcs.get_vs_at_m(m, phi_data_reshaped,
                                                                   reshaped_inputs)
    return vs_r, vs_i

def accumulate(arr):
    '''
    Sum over the second axis of a 2D array -- i.e., the key binning operation!
    :param arr: 2D numpy array, where the axis 0 contains the different bins
                and the axis 1 (last axis) contains the data in each bin
    :return: 1D numpy array of the sums of the data in each bin
    '''
    return np.sum(arr, axis=-1)

def get_alm_np(Ylm_i, Ylm_ip1, dYlm_i, dYlm_ip1, vs, m):
    """
    The key function: get alm by summing over all interpolated, weighted
    Y_lm's using numpy. Interpolation uses cubic Hermite splines
    :param Ylm_i: 1d numpy array of Ylm at sample indices i
    :param dYlm_i: 1d numpy array of first derivatives of Ylm at sample indices i
    :param Ylm_ip1: 1d numpy array of Ylm at sample indicesi+1
    :param dYlm_ip1: 1d numpy array of first derivatives of Ylm at sample indices i+1
    :param vs: np array of size (len(ms),4,Nsamples_theta) with the v_{i,j}(m) \
               (at a fixed m) used in the direct_SHT algorithm
    :param m: m value of the alm we want to calculate
    :return: a 1D numpy array with the alm value
    """
    return(np.sum(Ylm_i*vs[m,0] + dYlm_i*vs[m,1]
                  + Ylm_ip1*vs[m,2] + dYlm_ip1*vs[m,3]))

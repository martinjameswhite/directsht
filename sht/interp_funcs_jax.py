import numpy as np
import psutil
import utils
#from functools import partial


jax_present = True
from jax import jit, vmap
import jax.numpy as jnp
from functools import partial


default_dtype = None # Replace if you want to use a different dtype from the env default

#@partial(jit, donate_argnums=(1,2))
def get_vs(mmax, phi_data_reshaped, reshaped_inputs, loop_in_JAX=True, N_chunks=None,
           pad=False, verbose=False):
    """
    Wrapper function for get_vs_np and get_vs_jax. Defaults to JAX version when JAX is present.
    :param mmax: int. Maximum m value in the calculation
    :param phi_data_reshaped: 2D numpy array of shape (bin_num, bin_len) with data phi values,
        zero-padded to length bin_len in bins with fewer points
    :param reshaped_inputs: 2D numpy array of shape (4, bin_num, bin_len) with zero padding as
        in phi_data_reshaped. The 1st dimension corresponds to the four auxiliary arrays in
        the calculation of the v's.
    :param loop_in_JAX: bool. Whether to loop over m in JAX or in NumPy. Defaults to False,
        because JAX doesn't support in-place operations, so it's quite a bit slower
    :param N_chunks: int (optional). Number of chunks to break the vmap into if using JAX.
        This helps avoid memory issues. Must be a divisor of the number of (nonnegative) ms.
        Default is None, in which case the code will choose the highest value that won't
        exhaust the available memory.
    :param pad: bool. Whether to pad the vectorized dimension to make it divisible by N_chunks.
        By default (pad=True) we do this. The alternative is to split into the nearest number
        of equal divisible chunks.
    :return: a tuple of two 3D numpy arrays of shape (mmax+1, 4, bin_num) with the real and
        imaginary parts of the v's at each m.
    """
    if not jax_present or not loop_in_JAX:
        # Run loop in numpy and possibly move to GPU later
        return get_vs_np(mmax, phi_data_reshaped, reshaped_inputs)
    else:
        if verbose: print('\nSome info on the computation of the vs:')
        if verbose: print('This is how much memory we have available: ', psutil.virtual_memory().available/1e9, 'GB')
        if N_chunks is None:
            # How much memory does vmap ideally want to calculate the vs_real and vs_imag?
            tot_memory = 2*utils.predict_memory_usage((mmax+1)*reshaped_inputs.size,
                                                    reshaped_inputs.dtype)
            if verbose: print('Ideally, vmap would want',tot_memory/1e9 ,'GB of memory')
            # What fraction of the available memory do we want to use?
            max_mem_frac = 0.9
            if verbose: print('We will be using at most',100*max_mem_frac,'% of the available memory')
            rough_N_chunks = tot_memory/(max_mem_frac*psutil.virtual_memory().available)
            #
            if pad:
                N_chunks = int(np.ceil(rough_N_chunks))
            else:
                # Find the divisors of the number of ms
                divisors = np.arange(1, mmax+1 + 1)[(mmax+1) % np.arange(1, (mmax+1) + 1) == 0]
                # Find the divisor that is closest to rough_N_chunks but larger
                N_chunks = divisors[np.argmax(divisors >= rough_N_chunks)]
        # Calculate the padding we will need to make the vectorized dimension divisible by chunks
        if pad and (mmax+1)%N_chunks != 0:
            padding = (N_chunks+1)*((mmax+1)//N_chunks) - (mmax+1)
            N_chunks += 1
        else:
            padding = 0
        #
        if verbose: print('We will be breaking the computation into ',N_chunks,' chunks')
        if verbose: print('Note: because of our padding, ', 100*padding/(mmax+1+padding),
                          '% of our calculations are useless\n')
        # Vectorize and JIT-compile the function
        get_vs_at_m_mapped = vmap(jit(get_vs_at_m), in_axes=(0,None,None))
        # Loop over batches to avoid memory issues
        f = lambda ms: get_vs_at_m_mapped(ms, phi_data_reshaped, reshaped_inputs)
        m_array = jnp.arange(mmax+1 + padding)
        # Split the array into chunks and apply the vmapped function to each chunk
        chunked_vs = [f(chunk) for chunk in jnp.split(m_array, N_chunks)]
        # Unzip the batches
        vs_real_stacked, vs_imag_stacked = tuple(zip(*chunked_vs))
        # Concatenate the batches
        vs_real, vs_imag = [jnp.concatenate(vs_stacked) for vs_stacked in [vs_real_stacked, vs_imag_stacked]]
        #
        '''
        if pad:
            # Remove the padding introduced in the current function
            vs_real, vs_imag = vs_real[np.arange(mmax+1, dtype=int),:,:], vs_imag[np.arange(mmax+1, dtype=int),:,:]
        '''
        return vs_real, vs_imag

#@jit
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
    vs_r = np.zeros((mmax+1, 4, phi_data_reshaped.shape[0]), dtype=default_dtype)
    vs_i=vs_r.copy()
    for m in range(mmax+1):
        vs_r[m,:,:], vs_i[m,:,:] = get_vs_at_m(m, phi_data_reshaped, reshaped_inputs)
    return vs_r, vs_i

#@partial(jit, static_argnums=(0,))
def get_vs_at_m(m, phi_data_reshaped, reshaped_inputs):
    """
    Calculate the v's for a given m.
    :param m: int. m value at which to calculate the v's
    :param phi_data_reshaped: 2D numpy array of shape (bin_num, bin_len) with data phi values,
            zero-padded to length bin_len in bins with fewer points
    :param reshaped_inputs: 2D numpy array of shape (4, bin_num, bin_len) with zero padding as
            in phi_data_reshaped. The 1st dimension corresponds to the four auxiliary arrays in
            the calculation of the v's.
    :return: a tuple of two 2D numpy arrays of shape (4, bin_num) with the real and
            imaginary parts of the v's at this m.

    NOTE: Naively, this function should be jitted. However, the overhead from JIT compilation
    is longer than the time it takes to run the function, so we don't jit it.
    """
    vs_r, vs_i = [collapse(reshaped_inputs * phi_dep) for phi_dep in
                  [jnp.cos(m * phi_data_reshaped), jnp.sin(m * phi_data_reshaped)]]
    return vs_r, vs_i

@partial(jit, donate_argnums=(0,))
def collapse(arr):
    '''
    Sum over the second axis of a 2D array -- i.e., the key binning operation!
    :param arr: 2D numpy array, where the axis 0 contains the different bins
                and the axis 1 (last axis) contains the data in each bin
    :return: 1D numpy array of the sums of the data in each bin
    '''
    return jnp.sum(arr, axis=-1)

@partial(jit, donate_argnums=(4,))
def get_alm_jax(Ylm_i, Ylm_ip1, dYlm_i, dYlm_ip1, vs):
    """
    The key function: get alm by summing over all interpolated, weighted
    Y_lm's using JAX. Interpolation uses cubic Hermite splines
    :param Ylm_i: 1d numpy array of Ylm at sample indices i
            IMPORTANT NOTE: the zeroth element of this array is the value of m
            this is a hack to be able to pass the value of m through the vmap
    :param dYlm_i: 1d numpy array of first derivatives of Ylm at sample indices i
    :param Ylm_ip1: 1d numpy array of Ylm at sample indicesi+1
            IMPORTANT NOTE: the zeroth element of this array is the value of m
            this is a hack to be able to pass the value of m through the vmap
    :param dYlm_ip1: 1d numpy array of first derivatives of Ylm at sample indices i+1
    :param vs: jnp array of size (mmax+1,4,Nx) with the v_{i,j}(m)
    :return: a 1D numpy array with the alm value
    """

    return(jnp.sum(Ylm_i*vs[0] + dYlm_i*vs[1] + Ylm_ip1*vs[2] + dYlm_ip1*vs[3]))

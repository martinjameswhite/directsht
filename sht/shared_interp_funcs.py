import numpy as np
import psutil

import sht.interp_funcs_py as interp_funcs_py
import sht.shared_utils    as shared_utils

try:
    jax_present = True
    from jax import vmap, jit, devices
    import jax.numpy as jnp
    import sht.interp_funcs_jax as interp_funcs
except ImportError:
    jax_present = False
    import numpy as jnp
    import sht.interp_funcs_py as interp_funcs
    print("JAX not found. Falling back to NumPy.")


def get_vs(mmax, phi_data_reshaped, reshaped_inputs, loop_in_JAX=True, N_chunks=None,
           pad=False, verbose=False):
    """
    Helper function for get_vs_np and get_vs_jax. Defaults to JAX version when JAX is present.
    :param mmax: int. Maximum m value in the calculation
    :param phi_data_reshaped: 2D numpy array of shape (bin_num, bin_len) with data phi values,
        zero-padded to length bin_len in bins with fewer points
    :param reshaped_inputs: 2D numpy array of shape (4, bin_num, bin_len) with zero padding as
        in phi_data_reshaped. The 1st dimension corresponds to the four auxiliary arrays in
        the calculation of the v's.
    :param loop_in_JAX: bool. Whether to loop over m in JAX (when available) or in NumPy.
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
        return interp_funcs_py.get_vs_np(mmax, phi_data_reshaped, reshaped_inputs)
    else:
        if verbose: print('\nSome info on the computation of the vs:')
        if verbose: print('This is how much memory we have available: ', psutil.virtual_memory().available/1e9, 'GB')
        if N_chunks is None:
            # How much memory does vmap ideally want to calculate the vs_real and vs_imag?
            tot_memory = 2*shared_utils.predict_memory_usage((mmax+1)*reshaped_inputs.size,
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
        return vs_real, vs_imag

# @partial(jit, static_argnums=(0,))
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
    vs_r, vs_i = [interp_funcs.accumulate(reshaped_inputs * phi_dep) for phi_dep in
                  [jnp.cos(m * phi_data_reshaped), jnp.sin(m * phi_data_reshaped)]]
    return vs_r, vs_i

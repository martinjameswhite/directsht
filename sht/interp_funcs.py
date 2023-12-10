import numpy as np
#from functools import partial

try:
    jax_present = True
    from jax import jit
    import jax.numpy as jnp
except ImportError:
    jax_present = False
    print("JAX not found. Falling back to NumPy.")
    import numpy as jnp
    jit = lambda x: x  # Define a dummy jit decorator for fallback

default_dtype = None # Replace if you want to use a different dtype from the env default

def get_vs(mmax, phi_data_reshaped, reshaped_inputs, loop_in_JAX=False):
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
    :return: a tuple of two 3D numpy arrays of shape (mmax+1, 4, bin_num) with the real and
            imaginary parts of the v's at each m.
    """
    if jax_present and loop_in_JAX:
        return get_vs_jax(mmax, phi_data_reshaped, reshaped_inputs)
    else:
        # Run loop in numpy and possibly move to GPU later
        return get_vs_np(mmax, phi_data_reshaped, reshaped_inputs)
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

def get_vs_jax(mmax, phi_data_reshaped, reshaped_inputs):
    """
    Calculate the v's for each m using JAX.
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
    vs_r = jnp.zeros((mmax+1, 4, phi_data_reshaped.shape[0]), dtype=default_dtype)
    vs_i=vs_r.copy()
    for m in range(mmax+1):
        vs_r_at_m, vs_i_at_m = get_vs_at_m(m, phi_data_reshaped, reshaped_inputs)
        #TODO: Doing this with JAX is very inefficient!
        vs_r = vs_r.at[m,:,:].set(vs_r_at_m)
        vs_i = vs_i.at[m,:,:].set(vs_i_at_m)
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

@jit
def collapse(arr):
    '''
    Sum over the second axis of a 2D array -- i.e., the key binning operation!
    :param arr: 2D numpy array, where the axis 0 contains the different bins
                and the axis 1 (last axis) contains the data in each bin
    :return: 1D numpy array of the sums of the data in each bin
    '''
    return jnp.sum(arr, axis=-1)

def get_alm_jax(Ylm_i, Ylm_ip1, dYlm_i, dYlm_ip1, vs):
    """
    The key function: get alm by summing over all interpolated, weighted
    Y_lm's using JAX. Interpolation uses cubic Hermite splines
    :param Ylm_i: 1d numpy array of Ylm at sample indices i
    :param dYlm_i: 1d numpy array of first derivatives of Ylm at sample indices i
    :param Ylm_ip1: 1d numpy array of Ylm at sample indicesi+1
    :param dYlm_ip1: 1d numpy array of first derivatives of Ylm at sample indices i+1
    :param vs: np array of size (Nsamples_theta,4) with the v_{i,j}(m) \
               (at a fixed m) used in the direct_SHT algorithm
    :return: a 1D numpy array with the alm value
    """
    return(jnp.sum(Ylm_i*vs[0] + dYlm_i*vs[1]
                   + Ylm_ip1*vs[2] + dYlm_ip1*vs[3]))

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

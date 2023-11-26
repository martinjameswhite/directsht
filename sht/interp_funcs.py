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

default_dtype = 'float32'

@jit
def collapse(arr):
    '''
    Sum over the second axis of a 2D array -- i.e., the key binning operation!
    :param arr: 2D numpy array, where the 1st axis contains the different bins
                and the 2nd axis contains the data in each bin
    :return: 1D numpy array of the sums of the data in each bin
    '''
    return jnp.sum(arr, axis=-1)


def get_vs(mmax, phi_data_reshaped, reshaped_inputs):
    """
    Wrapper function for get_vs_np and get_vs_jax
    :param mmax: int. Maximum m value in the calculation
    :param phi_data_reshaped:
    :param reshaped_inputs:
    :return:
    """
    if jax_present:
        return get_vs_jax(mmax, phi_data_reshaped, reshaped_inputs)
    else:
        return get_vs_np(mmax, phi_data_reshaped, reshaped_inputs)
#@jit
def get_vs_np(mmax, phi_data_reshaped, reshaped_inputs):
    vs_r = np.zeros((mmax+1, 4, phi_data_reshaped.shape[0])); vs_i=vs_r.copy()
    for m in range(mmax+1):
        vs_r[m,:,:], vs_i[m,:,:] = get_vs_at_m(m, phi_data_reshaped, reshaped_inputs)
    return vs_r, vs_i

def get_vs_jax(mmax, phi_data_reshaped, reshaped_inputs):
    vs_r = jnp.zeros((mmax+1, 4, phi_data_reshaped.shape[0])); vs_i=vs_r.copy()
    for m in range(mmax+1):
        vs_r_at_m, vs_i_at_m = get_vs_at_m(m, phi_data_reshaped, reshaped_inputs)
        vs_r = vs_r.at[m,:,:].set(vs_r_at_m)
        vs_i = vs_i.at[m,:,:].set(vs_i_at_m)
    return vs_r, vs_i

#@partial(jit, static_argnums=(0,))
#Note: overhead from JIT compilation is longer than the time it takes to run the function
def get_vs_at_m(m, phi_data_reshaped, reshaped_inputs):
    vs_r, vs_i = [collapse(reshaped_inputs * phi_dep) for phi_dep in
                  [jnp.cos(m * phi_data_reshaped), jnp.sin(m * phi_data_reshaped)]]
    return vs_r, vs_i

def get_alm_jax(Ylm, dYlm, vs):
    """
    The key function: get alm by summing over all interpolated weighted
    Y_lm's using JAX. Interpolation uses cubic Hermite splines
    :param Ylm: 1d numpy array of Ylm samples. Ideally, in device memory already
    :param dYlm: 1d numpy array of first derivatives of Ylm at sample points.\
                 Ideally, in device memory already
    :param vs: np array of size (Nsamples_theta,4) with the v_{i,j}(m) \
               (at a fixed m) used in the direct_SHT algorithm
    :param m: m value of the alm we want to calculate
    :return: a 1D numpy array with the alm value
    """
    return(jnp.sum(Ylm[:-1] * vs[0,:-1] + dYlm[:-1] * vs[1,:-1] +\
                   Ylm[ 1:] * vs[2,1:] + dYlm[ 1:] * vs[3,1:]))

def get_alm_np(Ylm, dYlm, vs, m):
    """
    The key function: get alm by summing over all interpolated weighted
    Y_lm's using Numpy. Interpolation uses cubic Hermite splines
    :param Ylm: 1d numpy array of Ylm samples. Ideally, in device memory already
    :param dYlm: 1d numpy array of first derivatives of Ylm at sample points.\
                 Ideally, in device memory already
    :param vs: np array of size (len(ms),Nsamples_theta,4) with the \
               v_{i,j}(m) in the direct_SHT algorithm
    :param m: m value of the alm we want to calculate
    :return: a 1D numpy array with the alm value
    """
    return(np.sum(Ylm[:-1] * vs[m,0,:-1] + dYlm[:-1] * vs[m,1,:-1] +\
                   Ylm[ 1:] * vs[m,2,1:] + dYlm[ 1:] * vs[m,3,1:]))

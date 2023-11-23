import numpy as np

try:
    jax_present = True
    from jax import device_put, vmap, jit
    import jax.numpy as jnp
except ImportError:
    jax_present = False
    print("JAX not found. Falling back to NumPy.")
    import numpy as jnp

default_dtype = 'float32'

@jit
def collapse(arr):
    '''
    Sum over the second axis of a 2D array -- i.e., the key binning operation!
    :param arr: 2D numpy array, where the 1st axis contains the different bins
                and the 2nd axis contains the data in each bin
    :return: 1D numpy array of the sums of the data in each bin
    '''
    return jnp.sum(arr, axis=1)

#@jit
def get_vs(ms, phi_data_reshaped, reshaped_inputs):
    vs_r = np.zeros((len(ms), reshaped_inputs[0].shape[0], 4), dtype=default_dtype)
    vs_i = np.zeros((len(ms), reshaped_inputs[0].shape[0], 4), dtype=default_dtype)
    for m in ms:
        phi_dep_real,phi_dep_imag = [fn(m*phi_data_reshaped) for fn in [jnp.cos, jnp.sin]]
        for j, input_ in enumerate(reshaped_inputs):
            vs_r[m,:,j] = np.array(collapse(input_*phi_dep_real))
            vs_i[m,:,j] = np.array(collapse(input_*phi_dep_imag))
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
    return(jnp.sum(Ylm[:-1] * vs[:-1,0] + dYlm[:-1] * vs[:-1,1] +\
                   Ylm[ 1:] * vs[1:,2] + dYlm[ 1:] * vs[1:,3]))

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
    return(np.sum(Ylm[:-1] * vs[m,:-1,0] + dYlm[:-1] * vs[m,:-1,1] +\
                   Ylm[ 1:] * vs[m, 1:,2] + dYlm[ 1:] * vs[m, 1:,3]))

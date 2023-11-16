import numpy as np
import utils

try:
    jax_present = True
    from jax import device_put, vmap, jit
    import jax.numpy as jnp
except ImportError:
    jax_present = False
    print("JAX not found. Falling back to NumPy.")
    import numpy as jnp



def precompute_vs(Nsamples_theta,bin_indices,\
                  phi_data_sorted,w_i_sorted,t,ms,which_part):
    '''
    Calculate the v_{i,j}(m) in the direct_SHT algorithm and move them
    to the device where JAX will operate (e.g. GPU)
    :param Nsamples_theta : number of theta samples
    :param bin_indices: a 1d numpy array with indices of what bin each \
           data point belongs to
    :param phi_data_sorted: a 1d numpy array of phi data points \
           (same length as theta_data_sorted)
    :param w_i_sorted: a 1d numpy array of weights for each theta data point
    :param t: a 1d numpy array of t = theta_data-theta_sample[i]
    :param ms: a 1d numpy array of m indices of the Ylm's
    :param which_part: 'cos' or 'sin' for the real or imaginary part.
    :return: 3D numpy arrays of the v_{i,j}(m) in the direct_SHT algorithm
    '''
    Nbins = Nsamples_theta
    #
    input_1 = w_i_sorted * (2*t + 1) * (1-t)**2
    input_2 = w_i_sorted * t * (1-t)**2
    input_3 = w_i_sorted * t**2 * (3-2*t)
    input_4 = w_i_sorted * t**2 * (t-1)
    #
    vs = np.zeros((len(ms), Nsamples_theta, 4))
    for i, m in enumerate(ms):
        phi_dep = utils.get_phi_dep(phi_data_sorted,m,which_part)
        for j, input in enumerate([input_1,input_2,input_3,input_4]):
            vs[i,:,j] = utils.bin_data(input*phi_dep,bin_indices,Nbins)
    if jax_present:
        # Move arrays to GPU memory
        vs = device_put(vs)
    return(vs)
    #


def get_alm(Ylm, dYlm, vs, m):
    """
    The key function: get alm by summing over all interpolated weighted
    Y_lm's. Interpolation uses cubic Hermite splines
    :param Ylm: 1d numpy array of Ylm samples. Ideally, in device memory already
    :param dYlm: 1d numpy array of first derivatives of Ylm at sample points.\
                 Ideally, in device memory already
    :param vs: np array of size (len(ms),Nsamples_theta,4) with the \
               v_{i,j}(m) in the direct_SHT algorithm
    :param m: m value of the alm we want to calculate
    :return: a 1D numpy array with the alm value
    """
    return(jnp.sum(Ylm[:-1] * vs[m,:-1,0] + dYlm[:-1] * vs[m,:-1,1] +\
                   Ylm[ 1:] * vs[m, 1:,2] + dYlm[ 1:] * vs[m, 1:,3]))

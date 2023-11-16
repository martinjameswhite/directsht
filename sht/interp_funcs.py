import numpy as np
import sht.utils as utils

try:
    jax_present = True
    from jax import device_put, vmap, jit
    import jax.numpy as jnp
except ImportError:
    jax_present = False
    print("JAX not found. Falling back to NumPy.")
    import numpy as jnp

def precompute_vs(Nsamples_theta, bin_indices, phi_data_sorted, w_i_sorted, t, ms, which_part):
    '''
    Calculate the v_{i,j} in the direct_SHT algorithm and move them to the device where JAX will operate (e.g. GPU)
    :param Nsamples_theta : number of theta samples
    :param bin_indices: a 1d numpy array with indices of what bin each data point belongs to
    :param phi_data_sorted: a 1d numpy array of phi data points (same length as theta_data_sorted)
    :param w_i_sorted: a 1d numpy array of weights for each theta data point
    :param t: a 1d numpy array of t = theta_data-theta_sample[i] for each theta data point
    :param ms: a 1d numpy array of m indices of the Ylm's
    :param which_part: 'cos' or 'sin' for the real or imaginary part of the phi dependence
    :return: a list of four 1D numpy arrays of the v_{i,j} in the direct_SHT algorithm
    '''
    # We now sum up all the w_p f(t) in each spline region i
    Nbins = Nsamples_theta
    v_0 = np.zeros((len(ms), Nsamples_theta)); v_1 = v_0.copy(); v_2 = v_0.copy(); v_3 = v_0.copy()
    input_1 = w_i_sorted * (2*t + 1) * (1-t)**2
    input_2 = w_i_sorted * t * (1-t)**2
    input_3 = w_i_sorted * t**2 * (3-2*t)
    input_4 = w_i_sorted * t**2 * (t-1)
    for i, m in enumerate(ms):
        phi_dep = utils.get_phi_dep(phi_data_sorted, m, which_part)
        v_0[i, :] = utils.bin_data(input_1 * phi_dep, bin_indices, Nbins)
        v_1[i, :] = utils.bin_data(input_2 * phi_dep, bin_indices, Nbins)
        v_2[i, :] = utils.bin_data(input_3 * phi_dep, bin_indices, Nbins)
        v_3[i, :] = utils.bin_data(input_4 * phi_dep, bin_indices, Nbins)

    if jax_present:
        # Move arrays to GPU memory
        v_0 = device_put(v_0)
        v_1 = device_put(v_1)
        v_2 = device_put(v_2)
        v_3 = device_put(v_3)
    return [v_0, v_1, v_2, v_3]


def get_alm(Ylm, dYlm, vs):
    """
    The key function: get alm by summing over all interpolated weighted Y_lm's. Interpolation uses cubic Hermite splines
    :param Ylm: 1d numpy array of Ylm samples. Ideally, in device memory already
    :param dYlm: 1d numpy array of first derivatives of Ylm at sample points. Ideally, in device memory already
    :param vs: a list of four 1D numpy arrays of the v_{i,j} in the direct_SHT algorithm
    :return: a 1D numpy array with the alm value
    """
    return jnp.sum(Ylm[:-1] * vs[0][:-1] + dYlm[:-1] * vs[1][:-1] + Ylm[1:] * vs[2][1:] + dYlm[1:] * vs[3][1:])

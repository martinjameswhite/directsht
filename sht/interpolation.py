import numpy as np
from numba import njit
from jax import device_put
import jax.numpy as jnp

@njit
def get_sum(x_samples, x_data, y_data):
    """
    Calculate the v_{i,j} in the direct_SHT algorithm. Essentially, this entails summing all data points y_data
    with x_data in the range [x_samples[i], x_samples[i+1])
    :param x_samples: a 1d numpy array of x samples
    :param x_data: a 1d numpy array of x data points
    :param y_data: a 1d numpy array of y data points
    :return: a 1D numpy array of the sum of y_data points in each bin. This has same length as x_samples
    """
    sum = np.zeros_like(x_samples)
    j=0
    for x_d, y_d in zip(x_data, y_data):
        if x_d<x_samples[j+1]:
            sum[j] += y_d
        else:
            j+=1
    return sum

def precompute_t(theta_samples, theta_data_sorted):
    """
    Calculate t = theta_data-theta_sample[i] for each theta data point
    :param theta_samples: a 1d numpy array of theta samples
    :param theta_data_sorted: a 1d numpy array of theta data points
    :return: a 1D numpy array the size of theta_data_sorted
    """
    which_spline_idx = np.digitize(theta_data_sorted, theta_samples) - 1
    return (theta_data_sorted - theta_samples[which_spline_idx])

def precompute_vs(theta_samples, theta_data_sorted, w_i_sorted, t):
    '''
    Calculate the v_{i,j} in the direct_SHT algorithm and move them to the device where JAX will operate (e.g. GPU)
    :param theta_samples: a 1d numpy array of theta samples
    :param theta_data_sorted: a 1d numpy array of theta data points
    :param w_i_sorted: a 1d numpy array of weights for each theta data point
    :param t: a 1d numpy array of t = theta_data-theta_sample[i] for each theta data point
    :return: a list of four 1D numpy arrays of the v_{i,j} in the direct_SHT algorithm
    '''
    # We now sum up all the w_p f(t) in each spline region i, where f(t) = (2t+1)(1-t)^2, t(1-t)^2, t^2(3-2t), t^2(t-1)
    v_0 = device_put(get_sum(theta_samples, theta_data_sorted, w_i_sorted * (2*t + 1) * (1-t)**2))
    v_1 = device_put(get_sum(theta_samples, theta_data_sorted, w_i_sorted * t * (1-t)**2))
    v_2 = device_put(get_sum(theta_samples, theta_data_sorted, w_i_sorted * t**2 * (3-2*t)))
    v_3 = device_put(get_sum(theta_samples, theta_data_sorted, w_i_sorted * t**2 * (t-1)))
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

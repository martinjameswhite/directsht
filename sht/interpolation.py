import numpy as np
from numba import njit

@njit
def cubic_spline(x, y, ind, t_0, t_1, t_2, t_3, endpoints="natural"):
    """
    Cubic spline interpolation routine (inspired by that in JAX_COSMO) adapted to the needs of direct_SHT
    :param x: 1d numpy array of x samples
    :param y: 1d numpy array of y samples
    :param ind: 1d numpy array of sample indices -- almost trivial np.arange(len(x))
    :param t_0: 1d numpy array with (data-sample[i])^0 for each spline bin
    :param t_1: 1d numpy array with (data-sample[i])^1 for each spline bin
    :param t_2: 1d numpy array with (data-sample[i])^2 for each spline bin
    :param t_3: 1d numpy array with (data-sample[i])^3 for each spline bin
    :param endpoints: str. "natural" or "not-a-knot"
    :return: 1d numpy array of interpolated values
    """

    n_data = len(x)
    # Difference vectors
    h = np.diff(x)  # x[i+1] - x[i] for i=0,...,n-1
    p = np.diff(y)  # y[i+1] - y[i]

    # Special values for the first and last equations
    zero = np.array([0.0])
    one = np.array([1.0])
    A00 = one if endpoints == "natural" else np.array([h[1]])
    A01 = zero if endpoints == "natural" else np.array([-(h[0] + h[1])])
    A02 = zero if endpoints == "natural" else np.array([h[0]])
    ANN = one if endpoints == "natural" else np.array([h[-2]])
    AN1 = (
        -one if endpoints == "natural" else np.array([-(h[-2] + h[-1])])
    )  # A[N, N-1]
    AN2 = zero if endpoints == "natural" else np.array([h[-1]])  # A[N, N-2]

    # Construct the tri-diagonal matrix A
    A = np.diag(np.concatenate((A00, 2 * (h[:-1] + h[1:]), ANN)))
    upper_diag1 = np.diag(np.concatenate((A01, h[1:])), k=1)
    upper_diag2 = np.diag(np.concatenate((A02, np.zeros(n_data - 3))), k=2)
    lower_diag1 = np.diag(np.concatenate((h[:-1], AN1)), k=-1)
    lower_diag2 = np.diag(np.concatenate((np.zeros(n_data - 3), AN2)), k=-2)
    A += upper_diag1 + upper_diag2 + lower_diag1 + lower_diag2

    # Construct RHS vector s
    center = 3 * (p[1:] / h[1:] - p[:-1] / h[:-1])
    s = np.concatenate((zero, center, zero))
    # Compute spline coefficients by solving the system
    coefficients = np.linalg.solve(A, s)

    # Compute the spline coefficients for a given x
    knots = x

    # Include the right endpoint in spline piece C[m-1]
    ind = np.clip(ind, 0, len(knots) - 2)
    h = np.diff(knots)[ind]

    c = coefficients[ind]
    c1 = coefficients[ind + 1]
    a = y[ind]
    a1 = y[ind + 1]
    b = (a1 - a) / h - (2 * c + c1) * h / 3.0
    d = (c1 - c) / (3 * h)

    # Evaluation of the spline.
    return a * t_0 + b * t_1 + c * t_2 + d * t_3

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
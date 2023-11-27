import numpy as np
import numba as nb

try:
    jax_present = True
    from jax import device_put, vmap, jit
    import jax.numpy as jnp
except ImportError:
    jax_present = False
    print("JAX not found. Falling back to NumPy.")
    import numpy as jnp

default_dtype = 'float32'
def find_bin_edges(arr):
    '''
    Find the indices of transitions between different values in an array
    :param arr: 1D numpy array indicating what bin each element belongs to (must be sorted)
    :return: 1D numpy array of indices where the value in arr changes (includes 0)
    '''
    # Find the differences between consecutive elements
    differences = np.diff(arr)
    # Find the indices where differences are non-zero
    transition_indices = np.nonzero(differences)[0] + 1
    # Prepend beginning index
    transition_indices = np.insert(transition_indices, 0, 0, axis=0)
    # For indexing convenience, append None to the end
    transition_indices = np.append(transition_indices, None)
    return transition_indices

def reshape_array(data, bin_edges, bin_num, bin_len):
    '''
    Reshape a 1D array into a 2D array to facilitate binning
    :param data: 1D numpy array of data to be binned
    :param bin_edges: 1D numpy array of indices where the value in data changes (includes 0)
    :param bin_num: int. Number of bins where there is data
    :param bin_len: int. Maximum number of points in a bin
    :return: 2D numpy array of reshaped data, zero padded in bins with fewer points
    '''
    data_reshaped = np.zeros((bin_num, bin_len))
    for i in range(bin_num-1):
        fill_in = data[bin_edges[i]:bin_edges[i+1]]
        data_reshaped[i,:len(fill_in)] = fill_in
    return data_reshaped

def reshape_vs_array(inputs, bin_edges, bin_num, bin_len):
    '''
    #TODO: document better
    :param inputs: list of four 1D numpy array of data to be binned
    :param bin_edges: 1D numpy array of indices where the value in data changes (includes 0)
    :param bin_num: int. Number of bins where there is data
    :param bin_len: int. Maximum number of points in a bin
    :return: 2D numpy array of reshaped data, zero padded in bins with fewer points
    '''
    # Dimensions: vs label, bin_num, bin_len
    data_reshaped = np.zeros((4,bin_num, bin_len))
    for i in range(bin_num-1):
        for j, input_ in enumerate(inputs):
            fill_in = input_[bin_edges[i]:bin_edges[i+1]]
            data_reshaped[j,i,:len(fill_in)] = fill_in
    return data_reshaped

def insert_next_integer(arr):
    result = []
    for num in arr:
        result.append(num)
        next_num = num + 1
        if next_num not in arr:
            result.append(next_num)
    return result

def getlm(lmax, szalm, i=None):
    """Get the l and m from index and lmax. From Healpy

    Parameters
    ----------
    lmax : int
      The maximum l defining the alm layout
    szalm : int
      The size of the alm array
    i : int or None
      The index for which to compute the l and m.
      If None, the function return l and m for i=0..Alm.getsize(lmax)
    """
    if i is None:
        i = np.arange(szalm)
    assert (
        np.max(i) < szalm
    ), "Invalid index, it should less than the max alm array length of {}".format(
        szalm
    )

    with np.errstate(all="raise"):
        m = (
            np.ceil(
                ((2 * lmax + 1) - np.sqrt((2 * lmax + 1) ** 2 - 8 * (i - lmax))) / 2
            )
        ).astype(int)
        l = i - m * (2 * lmax + 1 - m) // 2
    return (l, m)

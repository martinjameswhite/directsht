import numpy as np

def find_transitions(arr):
    '''
    Find the indices where the data transitions from one bin/spline
    interval to the next
    :param arr: 1D numpy array indicating what bin/spline each element
        belongs to (must be sorted)
    :return: 1D numpy array of indices where the value in arr changes,
    '''
    # Find the differences between consecutive elements
    differences = np.diff(arr)
    # Find the indices where differences are non-zero
    transition_indices = np.nonzero(differences)[0] + 1
    return transition_indices

def reshape_phi(data, bin_edges):
    '''
    Reshape a 1D array into a 2D array to facilitate accumulation within
    bins in the computation of the v's
    :param data: 1D numpy array of data to be binned
    :param bin_edges: 1D numpy array of indices where the values in data go
        from one bin to the next.
    :return: 2D numpy array of shape (bin_num, max_bin_length), zero padded
        where bins have fewer than bin_len points
    '''
    # Split the data into bins bounded by interpolation nodes
    split_arr = np.split(data, bin_edges, axis=0)
    # Find the maximum length of the subarrays
    max_bin_length = max(subarray.shape[0] for subarray in split_arr)
    # Zero-pad the subarrays to the maximum length
    padded_arrs = [np.pad(subarray, (0, max_bin_length-subarray.shape[0]),
                                 mode='constant') for subarray in split_arr]
    # Stack the padded subarrays vertically
    return np.stack(padded_arrs, axis=0)


def reshape_aux(inputs, bin_edges):
    '''
    Reshape the four auxiliary 1D arrays into a 2D array shaped in a way
    that facilitates accumulation within bins during computation of the v's.
    :param inputs: list of four 1D numpy array of data to be binned
    :param bin_edges: 1D numpy array of indices where the values in data go
        from one bin to the next.
    :return: 2D numpy array of shape (4, bin_num, max_bin_length), zero padded
        in bins with fewer than bin_len points
    '''
    # Stack list of inputs into a (4, Npnt) array
    inputs_arr = np.vstack(inputs)
    # Split the data into bins bounded by interpolation nodes
    split_arr = np.split(inputs_arr, bin_edges, axis=1)
    # Find the maximum length of the subarrays
    max_bin_length = max(subarray.shape[1] for subarray in split_arr)
    # Zero-pad the subarrays along axis=1 to the maximum length
    padded_arrs = [np.pad(subarray, ((0, 0), (0,max_bin_length-subarray.shape[1])),
                                 mode='constant') for subarray in split_arr]
    # Stack the padded subarrays vertically
    return np.stack(padded_arrs, axis=1)

def predict_memory_usage(num_elements, dtype):
    '''
    Predict the memory usage of an array of a given size and dtype
    :param num_elements: int.
        The number of elements in the array
    :param dtype: np.dtype
        The dtype of the array
    :return: float
        The predicted memory usage in bytes
    '''
    # Get the size of each element in bytes
    element_size = dtype.itemsize
    # Calculate total memory usage in bytes
    total_memory = element_size * num_elements
    return total_memory
import numpy as np
try:
    jax_present = True
    import jax
    from jax.sharding import PositionalSharding
    from jax.experimental import mesh_utils
    N_devices = 1#len(jax.devices())
except ImportError:
    jax_present = False
    print("JAX not found. Falling back to NumPy.")

default_dtype = None # Replace if you want to use a different dtype from the env default

def find_transitions(arr):
    '''
    Find the indices where the data transitions from one bin/spline to the next
    :param arr: 1D numpy array indicating what bin/spline each element belongs to (must be sorted)
    :return: 1D numpy array of indices where the value in arr changes,
                including a 0 at the beginning and None at the end for indexing convenience
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

def reshape_phi_array(data, bin_edges, bin_num, bin_len):
    '''
    Reshape a 1D array into a 2D array to facilitate binning in computation of v's
    :param data: 1D numpy array of data to be binned
    :param bin_edges: 1D numpy array of indices where the values in data go
            from one bin to the next. Must include 0 and a None at the end.
    :param bin_num: int. Number of bins where there is data
    :param bin_len: int. Maximum number of points in a bin
    :return: 2D numpy array of shape (bin_num, bin_len), zero padded in bins
            with fewer than bin_len points
    '''
    data_reshaped = np.zeros((bin_num, bin_len), dtype=default_dtype)
    for i in range(bin_num):
        fill_in = data[bin_edges[i]:bin_edges[i+1]]
        data_reshaped[i,:len(fill_in)] = fill_in
    return data_reshaped

def reshape_aux_array(inputs, bin_edges, bin_num, bin_len):
    '''
    Reshape the four auxiliary 1D arrays into a 2D array shaped in such a way
    to facilitate binning during computation of the v's.
    :param inputs: list of four 1D numpy array of data to be binned
    :param bin_edges:  1D numpy array of indices where the values in data go
            from one bin to the next. Must include 0 and a None at the end.
    :param bin_num: int. Number of bins where there is data
    :param bin_len: int. Maximum number of points in a bin
    :return: 2D numpy array of shape (4, bin_num, bin_len), zero padded in bins
            with fewer than bin_len points
    '''
    # Dimensions: vs label, bin_num, bin_len
    data_reshaped = np.zeros((4,bin_num, bin_len), dtype=default_dtype)
    for i in range(bin_num):
        for j, input_ in enumerate(inputs):
            fill_in = input_[bin_edges[i]:bin_edges[i+1]]
            data_reshaped[j,i,:len(fill_in)] = fill_in
    return data_reshaped

def getlm(lmax, szalm, i=None):
    '''
    Get the l and m from index and lmax. From Healpy.
    :param lmax: int. The maximum l defining the alm layout
    :param szalm: int. The size of the alm array
    :param i: int or None. The index for which to compute the l and m.
            If None, the function returns l and m for i=0..Alm.getsize(lmax)
    '''
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

def move_to_device(arr, verbose=False, axis=0):
    '''
    Helper function to distribute an array across devices.
    :param arr: np.ndarray
        The array to be distributed. The splitting distribution is done
         across its zeroth dimension.
    :param verbose: bool
        Whether to visualize the sharding scheme. Only works for !d2D arrays.
    :return:
    '''
    if axis==1:
        assert len(arr.shape) == 3, "Can only split along axis=1 for 3D arrays"
    # Initialize sharding scheme
    sharding = PositionalSharding(mesh_utils.create_device_mesh(N_devices))
    # Is the dimension divisible by the number of devices?
    remainder = arr.shape[0] % N_devices
    if remainder != 0:
        if axis==0:
            # Zero-pad the zeroth dimension of the array to be divisible by len(jax.devices())
            arr = np.pad(arr,((0, N_devices-remainder), (0,0)), mode='constant', constant_values=0)
        elif axis==1 and len(arr.shape) == 3:
            # Zero-pad the first dimension of the array to be divisible by len(jax.devices())
            arr = np.pad(arr,((0, 0), (0,N_devices-remainder), (0, 0)), mode='constant', constant_values=0)
    # Initialize the sharding scheme with as many devices as there are available
    if len(arr.shape) == 3:
        if axis == 0:
            sharding_reshaped = sharding.reshape(N_devices, 1, 1)
        elif axis == 1:
            sharding_reshaped = sharding.reshape(1, N_devices, 1)
        # Visualizing the sharding is not supported for 3D arrays
        verbose=False
    elif len(arr.shape) == 2:
        sharding_reshaped = sharding.reshape(N_devices, 1)
    else:
        sharding_reshaped = sharding
    # Distribute the array across devices
    arr = jax.device_put(arr, sharding_reshaped)
    if verbose:
        # Visualize the sharding
        jax.debug.visualize_array_sharding(arr)
    return arr
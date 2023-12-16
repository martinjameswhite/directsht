import numpy as np
try:
    jax_present = True
    import jax
    from jax import jit, lax
    from jax.sharding import PositionalSharding
    from functools import partial
    from jax.experimental import mesh_utils
    # Choose the number of devices we'll be parallelizing across
    N_devices = len(jax.devices())
except ImportError:
    jax_present = False
    N_devices = 1
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

def move_to_device(arr, axis=0, pad_axes=None, verbose=False):
    '''
    Helper function to shard (i.e. distribute) an array across devices.
    :param arr: np.ndarray
        The array to be sharded
    :param axis: int
        The axis along which to shard the array
    :param pad_axes: int or list of int. (optional)
        The axes along which to pad the array. By default, padding is only along sharded axis
    :param verbose: bool
        Whether to visualize the sharding scheme. Only works for !d2D arrays.
    :return:
    '''
    if pad_axes is None:
        pad_axes = axis
    if axis==1:
        assert len(arr.shape) == 3, "Only sharding along axis=1 is supported for 3D arrays"
    # Initialize sharding scheme
    sharding = PositionalSharding(mesh_utils.create_device_mesh(N_devices))
    # If needed, zero-pad the array so that its length along the sharded dimension
    # is divisible by the number of devices we're distributing across
    arr = pad_to_shard(arr, pad_axes)
    # Reshape sharding scheme based on array dimensions
    if len(arr.shape) == 3:
        sharding_reshaped = sharding.reshape((N_devices, 1, 1) if axis == 0 else (1, N_devices, 1))
        verbose = False  # Visualizing sharding is not supported for 3D arrays
    elif len(arr.shape) == 2:
        sharding_reshaped = sharding.reshape((N_devices, 1))
    else:
        sharding_reshaped = sharding
    arr = jax.device_put(arr, sharding_reshaped)
    if verbose:
        # Visualize the sharding
        jax.debug.visualize_array_sharding(arr)
    return arr

def init_array(Nl, Nx, Ndevices, axes=[0,1]):
    '''
    Helper function to initialize empty array with the appropriate sharding
    structure, as opposed to generating it on a single device and moving it
    Pads the input if necessary to have length along sharded dim that's
    divisible by the number of devices
    '''
    # Initialize sharding scheme
    sharding = PositionalSharding(mesh_utils.create_device_mesh(N_devices))
    sharding = sharding.reshape((N_devices, 1, 1))
    # This is a trick to shard the array at instantiation
    @partial(jax.jit, static_argnums=(0,1,2), out_shardings=sharding)
    def f(Nl,Nx,axes=[0,1]):
        return pad_to_shard(jax.numpy.zeros((Nl, Nl, Nx)), axes)
    return f(Nl, Nx)


def pad_to_shard(arr, axes=0):
    '''
    Pad an array with zeros so that its length along the sharded dimension is
    divisible by the number of devices we're distributing across
    :param arr: np.ndarray
        The array to be padded (if necessary)
    :param axis: int or list of int. (optional)
        The axis along which to pad the array. Defaults to 0.
    :return: np.ndarray
        The array with the padding added (if necessary)
    '''
    if not isinstance(axes, list):
        axes = [axes]
    #
    for axis in axes:
        # Check whether the length of the original array is divisible by the number
        remainder = arr.shape[axis] % N_devices
        if remainder != 0:
            pad_width = [(0, 0)] * arr.ndim
            pad_width[axis] = (0, N_devices - remainder)
            if isinstance(arr, np.ndarray):
                arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
            elif isinstance(arr, jax.numpy.ndarray):
                arr = jax.numpy.pad(arr, pad_width, mode='constant', constant_values=0)
            else:
                raise TypeError("Input array must be a numpy or JAX array")
    return arr

def unpad(arr, unpadded_len, axis=0):
    '''
    Remove the padding we applied to enable sharding, first checking whether
    it's nececessary
    :param arr: np.ndarray
        The array to be unpadded (if necessary)
    :param unpadded_len: int.
        The length of the dimension prior to padding
    :param axis: int. (optional)
        The axis along which to unpad the array. Defaults to 0.
    :return: np.ndarray
        The array with the padding removed (if necessary)
    '''
    # Check whether the length of the original array was divisible by the number
    # of devices we're using. This tells us whether we had the need to zero-pad
    # in order to shard the array, and therefore whether we need to unpad now.
    remainder = unpadded_len % N_devices
    if remainder != 0:
        # Remove the padding
        return arr.take(range(unpadded_len), axis=axis)
    else:
        # No need for unpadding
        return arr

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
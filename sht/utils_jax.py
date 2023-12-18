import numpy as np
import jax
from jax.sharding import PositionalSharding
from functools import partial
from jax.experimental import mesh_utils
import shared_utils

# Choose the number of devices we'll be parallelizing across
N_devices = len(jax.devices())


def find_transitions(arr):
    '''
    Wrapper function to call find_transitions from shared_utils
    '''
    return shared_utils.find_transitions(arr)

def reshape_phi(data, bin_edges):
    '''
    Wrapper function to call reshape_phi_array from shared_utils
    '''
    return shared_utils.reshape_phi(data, bin_edges)

def reshape_aux(inputs, bin_edges):
    '''
    Wrapper function to call reshape_phi_array from shared_utils
    '''
    return shared_utils.reshape_aux(inputs, bin_edges)

def getlm(lmax, szalm, i=None):
    '''
    Wrapper around shared_utils's getlm
    '''
    return shared_utils.getlm(lmax, szalm, i)


def move_to_device(arr, axis=0, pad_axes=None, verbose=False):
    '''
    Helper function to shard (i.e. distribute) an array across devices (typically GPUs).
    :param arr: np.ndarray
        The array to be moved to the devices and sharded
    :param axis: int
        The axis along which to shard the array
    :param pad_axes: int or list of int. (optional)
        The axes along which to pad the array. By default, padding is only along sharded axis
    :param verbose: bool
        Whether to visualize the sharding scheme. Only works for 2D arrays.
    :return:
    '''
    if pad_axes is None:
        pad_axes = axis
    if axis == 1:
        assert len(arr.shape) == 3, "Only sharding along axis=1 is supported for 3D arrays"
    # Initialize sharding scheme
    sharding = PositionalSharding(mesh_utils.create_device_mesh(N_devices))
    # If needed, zero-pad the array so that its length along the sharded dimension(s)
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
    # Move to the GPU with the desired sharding scheme
    arr = jax.device_put(arr, sharding_reshaped)
    if verbose:
        # Visualize the sharding
        jax.debug.visualize_array_sharding(arr)
    return arr


def init_array(Nl, Nx, N_devices, axes=[0, 1]):
    '''
    Helper function to initialize an empty array with the appropriate sharding
    structure, as opposed to generating it on a single device and moving it.
    Pads the input if necessary to have a length along sharded dim that's
    divisible by the number of devices
    '''
    # Initialize sharding scheme
    sharding = PositionalSharding(mesh_utils.create_device_mesh(N_devices))
    sharding = sharding.reshape((N_devices, 1, 1))
    # This is a trick to shard the array at instantiation
    @partial(jax.jit, static_argnums=(0, 1, 2), out_shardings=sharding)
    def f(Nl, Nx, axes=[0, 1]):
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

def to_hp_convention(alm_grid_real, alm_grid_imag):
    '''
    Get a 1D vector of alms in the healpy convention from an (Nl, Nl)
    array in the indexing scheme indx = lambda ell, m: (m, ell - m)
    :param alm: np.ndarray
        2D array of size (Nl, Nl) with alms in the indexing convention where the ith
        row corresponds to i=m and the jth column corresponds to j=ell-m
    :return: np.ndarray
        1D array of alms in Healpy convention
    '''
    def flatten_mat(mat, flat_idx):
        # First roll so it looks like an upper triangular matrix. Then extract the
        # upper triangular indices
        return np.array([np.roll(mat[i, :], i) for i in range(len(mat))])[flat_idx]

    assert alm_grid_real.shape == alm_grid_imag.shape, "Input arrays must have the same shape"
    # Get the size of the alm array
    Nl = alm_grid_real.shape[0]
    # Initialize the output array
    alm_hp = np.zeros((Nl * (Nl + 1)) // 2, dtype='complex128')
    # Fill in the output array
    flat_idx = np.triu_indices(alm_grid_real.shape[0])
    alm_real, alm_imag = [flatten_mat(mat, flat_idx) for mat in [alm_grid_real, alm_grid_imag]]
    alm_hp[:] = alm_real - 1j * alm_imag
    return alm_hp
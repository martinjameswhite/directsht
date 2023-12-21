import numpy as np
import sht.shared_utils

N_devices = 1

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

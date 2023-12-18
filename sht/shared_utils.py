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
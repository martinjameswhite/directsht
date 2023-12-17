import numpy as np

def null_unphys(a, b):
    # Dummy function. It just returns the inputs.
    return (a, b)

def norm_ext(Y, Nl):
    '''
    Normalize the Ylm's by the (2ell+1)/4pi factor relating Plm to Ylm
    :param Yv: np.ndarray of shape ((Nl*(Nl+1))//2, Nx) containing the Plm's (or dPlm's)
    :return: np.ndarray of shape ((Nl*(Nl+1))//2, Nx) containing  (2ell+1)/4pi * Plm's (or dPlm's)
    '''
    #TODO: Make this faster by jitting. Could also normalize Yv and Yd at the same time.
    #
    # The healpy indexing scheme -- and ours
    indx = lambda m, ell: (m * (2 * Nl - 1 - m)) // 2 + ell
    for ell in range(Nl):
        fact = np.sqrt((2 * ell + 1) / 4. / np.pi)
        for m in range(ell + 1):
            ii = indx(ell, m)
            Y[ii, :] *= fact
    return (Y)
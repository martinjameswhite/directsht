#!/usr/bin/env python3
#
import numpy as np
import time
import numba


@numba.jit(nopython=True)
def threej000(ell1,ell2,ell3,store):
    """Returns the Wigner 3j symbol for integer ell's and m1=m2=m3=0."""
    J = ell1+ell2+ell3
    # Order ell1, ell2 and ell3 such that j1>=j2>=j3.
    ells = [ell1,ell2,ell3]
    j1   = max(ells)
    j3   = min(ells)
    j2   = J-j1-j3
    # Work out the index.
    ii   = (j1*(j1+1)*(j1+2))//6 + (j2*(j2+1))//2 + j3
    if ii<store.size:
        if store[ii]<1e41:
            return(store[ii])
    if (J%2>0):
        if (ii<store.size):
            store[ii] = 0.0
        return(0.0)
    if (j1==j2)&(j3==0):
        res = (-1.)**j1 / np.sqrt(2*j1 + 1.0)
        if (ii<store.size):
            store[ii] = res
        return( res )
    elif (j1!=j2)&(j3==0):
        if (ii<store.size):
            store[ii] = 0.0
        return(0.0)
    else:
        num = (J-2*j2-1)*(J-2*j3+2)
        den = (J-2*j2  )*(J-2*j3+1)
        fac = np.sqrt(float(num)/float(den))
        res = fac*threej000(j1,j2+1,j3-1,store)
        if (ii<store.size):
            store[ii] = res
        return(res)
    #

@numba.jit(nopython=True)
def fill_simple3j(Nl,store):
    """Fill in the 'easy' values of 3j-000."""
    for j1 in range(Nl):
        ii = (j1*(j1+1)*(j1+2))//6 + (j1*(j1+1))//2 + 0
        store[ii] = (-1.)**j1/np.sqrt(2*j1+1.)
    #



@numba.jit(nopython=True)
def get_index_jitted(l1,l2,l3):
    """
    Get the index of the Wigner 3j symbol (with m1=m2=m3=0) in the table.
    :param l1: int. l1
    :param l2: int. l2
    :param l3: int. l3
    :return: int. The index of the Wigner 3j symbol in the table.
    """
    # Order ell1, ell2 and ell3 such that j1>=j2>=j3.
    ells = [l1, l2, l3]
    j1 = max(ells)
    j3 = min(ells)
    j2 = l1 + l2 + l3 - j1 - j3
    # Work out the index.
    return(  (j1*(j1+1)*(j1+2))//6 + (j2*(j2+1))//2 + j3 )

class Wigner3j:
    def __init__(self,Nl):
        """
        Class to compute and save/load Wigner 3j symbols.
        :param Nl: int. (One plus) the maximum l for which to compute the 3j's
        """
        self.Nl = Nl
        self.store = self.get_3js()
        #
    def __call__(self,l1,l2,l3):
        """
        Compute the Wigner 3j symbol for integer l's and m1=m2=m3=0.
        :param l1: int. l1
        :param l2: int. l2
        :param l3: int. l3
        :return: float. The Wigner 3j symbol
        """
        return self.store[self.get_index(l1, l2, l3)]
        #
    def get_index(self,l1,l2,l3):
        """
        Wrapper function to get the index of the Wigner 3j symbol
        (with m1=m2=m3=0) in the table.
        :param l1: int. l1
        :param l2: int. l2
        :param l3: int. l3
        :return: int. The index of the Wigner 3j symbol in the table.
        """
        return (get_index_jitted(l1,l2,l3))
        #
    def get_3js(self):
        """A loop to fill the 'store' array.  Can also be used for
        timing tests."""
        store = np.zeros((self.Nl*(self.Nl+1)*(self.Nl+2))//6,dtype='float64')+1e42
        fill_simple3j(self.Nl,store)
        for j1 in range(self.Nl):
            for j2 in range(j1 + 1):
                for j3 in range(j2 + 1):
                    threej000(j1, j2, j3, store)
        return(store)
        #



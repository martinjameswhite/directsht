#!/usr/bin/env python3
#
import numpy as np
import time
import numba
from sympy.physics.wigner import wigner_3j


@numba.jit(nopython=True)
def threej000(ell1,ell2,ell3,store):
    """Returns the Wigner 3j symbol for integer ell's and m1=m2=m3=0."""
    J = ell1+ell2+ell3
    if (J%2>0):
        return(0.0)
    # Order ell1, ell2 and ell3 such that j1>=j2>=j3.
    ells = [ell1,ell2,ell3]
    j1   = max(ells)
    j3   = min(ells)
    j2   = J-j1-j3
    # Work out the index.
    ii   = (j1*(j1+1)*(j1+2))//6 + (j2*(j2+1))//2 + j3
    if ii<store.size:
        if store[ii]<1e39:
            return(store[ii])
    if (j1==j2)&(j3==0):
        return( (-1.)**j1/np.sqrt(2*j1+1.0) )
    elif (j1!=j2)&(j3==0):
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


class Wigner3j:
    def __init__(self, Nl):
        """
        Class to compute and save/load Wigner 3j symbols.
        :param Nl: int. (One plus) the maximum l for which to compute the 3j's
        """
        self.Nl = Nl
        self.store = self.get_3js()

    def __call__(self, l1, l2, l3):
        """
        Compute the Wigner 3j symbol for integer l's and m1=m2=m3=0.
        :param l1: int. l1
        :param l2: int. l2
        :param l3: int. l3
        :return: float. The Wigner 3j symbol
        """
        return self.store[self.get_index(l1, l2, l3)]

    def get_index(self, l1, l2, l3):
        """
        Get the index of the Wigner 3j symbol in the table.
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
        return  (j1 * (j1 + 1) * (j1 + 2)) // 6 + (j2 * (j2 + 1)) // 2 + j3

    def get_3js(self):
        """A loop to do timing tests on."""
        store = np.zeros((self.Nl*(self.Nl+1)*(self.Nl+2))//6,dtype='float64')+1e42
        for j1 in range(self.Nl):
            for j2 in range(j1 + 1):
                for j3 in range(j2 + 1):
                    threej000(j1, j2, j3, store)
        return(store)

if __name__=="__main__":
    for Nl in [1000]:
        t0 = time.time()
        temp_3js = Wigner3j(Nl)
        print("Nl=",Nl," took ",time.time()-t0," seconds.")

        # Compare to sympy
        l1=Nl-1
        l2=Nl-2
        l3=1
        our_result = temp_3js(l1,l2,l3)
        sympy_result = wigner_3j(l1,l2,l3,0,0,0).n(32)
        print('Testing l1={}, l2={}, l3={}'.format(l1,l2,l3))
        print('Fractional difference with sympy: ',
              (our_result-sympy_result)/sympy_result)
        print('\n')
    #

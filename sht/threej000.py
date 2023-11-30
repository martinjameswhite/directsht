#!/usr/bin/env python3
#
import numpy as np
import time
import numba
from sympy.physics.wigner import wigner_3j

#
#
# This code should be modified to generate an astropy
# table of the Wigner-3j coefficients, and save it.
#
# We can "order" an input list of ells via:
# L  = l1+l2+l3
# j1 = max([l1,l2,l3])
# j3 = min([l1,l2,l3])
# j2 = L-j1-j2
# then the index is
# ii=(j1*(j1+1)*(j1+2))//6+(j2*(j2+1))//2+j3
#

#from sympy.physics.wigner import wigner_3j




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


def do_test(Nl):
    """A loop to do timing tests on."""
    store = np.zeros((Nl*(Nl+1)*(Nl+2))//6,dtype='float64') + 1e42
    for j1 in range(Nl):
        for j2 in range(j1+1):
            for j3 in range(j2+1):
                #w3j = float(wigner_3j(j1,j2,j3,0,0,0))
                m3j = threej000(j1,j2,j3,store)
                #J   = j1+j2+j3
                #if J%2==0:
                #    err = (m3j-w3j)/(np.abs(w3j)+0.1)
                #    print("{:2d} {:2d} {:2d} ".format(j1,j2,j3)+\
                #          "{:10.5} {:10.5f} {:10.5f}".format(w3j,m3j,err))
    return(store)


# Note we should really store (j1,j2,j3) as:
# index = j1(j1+1)(j1+2)/6 + j2(j2+1)/2 + j3


if __name__=="__main__":
    for Nl in [200]:
        t0 = time.time()
        output = do_test(Nl)
        print("Nl=",Nl," took ",time.time()-t0," seconds.")

        # Compare to sympy
        l1=Nl-1
        l2=Nl-2
        l3=1
        ii= (l1*(l1+1)*(l1+2))//6 + (l2*(l2+1))//2 + l3
        our_result = output[ii]
        sympy_result = wigner_3j(l1,l2,l3,0,0,0).n(32)
        print('Testing l1={}, l2={}, l3={}'.format(l1,l2,l3))
        print('Fractional difference with sympy: ',
              (our_result-sympy_result)/sympy_result)
        print('\n')
    #

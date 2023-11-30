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
#

#from sympy.physics.wigner import wigner_3j


# For timing I'm remaking this array each time
# (see __main__ below).
# It should all be in some class and set up like
# this when the class is instantiated.
Nlmax = 512
###store = np.zeros( (Nlmax,Nlmax,Nlmax) ) + 1e40

@numba.jit(nopython=True)
def threej000(j1,j2,j3, store):
    """Returns the Wigner 3j symbol for integer j's and m1=m2=m3=0."""
    J = j1+j2+j3
    if (J%2>0):
        return(0)
    if store[j1,j2,j3]<1e39:
        return(store[j1,j2,j3])
    if (j1>=j2)&(j1>=j3)&(j2>=j3):
        if (j1==j2)&(j3==0):
            return( (-1)**j1/np.sqrt(2*j1+1.0) )
        elif (j1!=j2)&(j3==0):
            return( 0.0 )
        else:
            num = (J-2*j2-1)*(J-2*j3+2)
            den = (J-2*j2  )*(J-2*j3+1)
            fac = np.sqrt(float(num)/float(den))
            return(fac*threej000(j1,j2+1,j3-1, store))
    elif (j1>=j2)&(j1>=j3)&(j2< j3):
        return(threej000(j1,j3,j2,store))	# No minus sign since J even.
    elif (j1>=j2)&(j1< j3)&(j2< j3):
        return(threej000(j3,j1,j2,store))
    elif (j1< j2)&(j1>=j3)&(j2>=j3):
        return(threej000(j2,j1,j3,store))
    elif (j1< j2)&(j1< j3)&(j2>=j3):
        return(threej000(j2,j3,j1,store))
    elif (j1< j2)&(j1< j3)&(j2< j3):
        return(threej000(j3,j2,j1,store))
    else:
        raise RuntimeError
    #


def do_test(Nl):
    """A loop to do timing tests on."""
    store = np.zeros((Nl, Nl, Nl)) + 1e40
    for j1 in range(Nl):
        for j2 in range(j1+1):
            for j3 in range(j2+1):
                #w3j = float(wigner_3j(j1,j2,j3,0,0,0))
                m3j = threej000(j1,j2,j3, store)
                #J   = j1+j2+j3
                #if J%2==0:
                #    err = (m3j-w3j)/(np.abs(w3j)+0.1)
                #    print("{:2d} {:2d} {:2d} ".format(j1,j2,j3)+\
                #          "{:10.5} {:10.5f} {:10.5f}".format(w3j,m3j,err))
                store[j1,j2,j3] = m3j
                store[j2,j3,j1] = m3j
                store[j3,j1,j2] = m3j
                store[j1,j3,j2] = m3j
                store[j2,j1,j3] = m3j
                store[j3,j2,j1] = m3j
    return store


# Note we should really store (j1,j2,j3) as:
# index = j1(j1+1)(j1+2)/6 + j2(j2+1)/2 + j3


if __name__=="__main__":
    for Nl in [50,100,250,500]:
        t0 = time.time()
        output = do_test(Nl)
        print("Nl=",Nl," took ",time.time()-t0," seconds.")

        # Compare to sympy
        l1=Nl-1
        l2=Nl-2
        l3=1
        our_result = output[l1,l2,l3]
        sympy_result = wigner_3j(l1,l2,l3,0,0,0).n(32)
        print('Testing l1={}, l2={}, l3={}'.format(l1,l2,l3))
        print('Fractional difference with sympy: ',
              (our_result-sympy_result)/sympy_result)
        print('\n')
    #

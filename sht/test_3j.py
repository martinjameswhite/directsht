#!/usr/bin/env python3
#
import numpy as np
import time

from threej000            import Wigner3j
from sympy.physics.wigner import wigner_3j




if __name__=="__main__":
    # Do some tests and timing of the class.
    for Nl in [1000]:
        t0 = time.time()
        temp_3js = Wigner3j(Nl)
        print("Generating Nl=",Nl," took ",time.time()-t0," seconds.",flush=True)
        #
        # Compare to sympy
        l1,l2,l3=Nl-1,Nl-2,1
        our_result   = temp_3js(l1,l2,l3)
        sympy_result = wigner_3j(l1,l2,l3,0,0,0).n(32)
        print('Testing l1={}, l2={}, l3={}'.format(l1,l2,l3))
        print('Fractional difference with sympy: ',
              (our_result-sympy_result)/(sympy_result+1e-40))
        print('\n')
        #
        l1,l2,l3=Nl-2,Nl-3,5
        our_result   = temp_3js(l1,l2,l3)
        sympy_result = wigner_3j(l1,l2,l3,0,0,0).n(32)
        print('Testing l1={}, l2={}, l3={}'.format(l1,l2,l3))
        print('Fractional difference with sympy: ',
              (our_result-sympy_result)/(sympy_result+1e-40))
        print('\n')
        #
        l1,l2,l3=2,4,6
        our_result   = temp_3js(l1,l2,l3)
        sympy_result = wigner_3j(l1,l2,l3,0,0,0).n(32)
        print('Testing l1={}, l2={}, l3={}'.format(l1,l2,l3))
        print('Fractional difference with sympy: ',
              (our_result-sympy_result)/(sympy_result+1e-40))
        print('\n')
    #

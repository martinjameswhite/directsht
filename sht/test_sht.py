#!/usr/bin/env python
#
# Test the SHT class against the SciPy versions.
# Checks the function values and first derivatives
# for Ylm(x=cos[theta],0).
#
#
import numpy as np
import time

from   sht import DirectSHT






if __name__=="__main__":
    # Compare Ylm against the SciPy library version.
    # Generate a grid of Ylm using our class.
    Nl,Nx = 500,1024
    now   = time.time()
    sht   = DirectSHT(Nl,Nx)
    print("Computation took ",time.time()-now," seconds.",flush=True)
    # Now let's compare to the library version.  Compute a
    # table in the same format from SciPy.  Note SciPy has
    # the ell,m indices and theta,phi arguments reversed!
    from scipy.special import sph_harm
    #
    print("Generating scipy library table for ell<45.",flush=True)
    Ylb = np.zeros( ((Nl*(Nl+1))//2,Nx) )
    for ell in range(45):
        for m in range(ell+1):
            ii = sht.indx(ell,m)
            for i,x in enumerate(sht.x):
                theta     = np.arccos(x)
                Ylb[ii,i] = np.real( sph_harm(m,ell,0,theta) )
    # Pick some values of x=cos(theta) to compare.
    ix = np.array([0,Nx//3,2*Nx//3,Nx-1])
    print("\nCompare our class to SciPy for x=cos(theta)=",sht.x[ix])
    # First cross-check some Ylm values for sanity.
    print("\nCheck Ylm values.")
    print("Compare pairs of lines labeled by (ell,m):")
    for ell in [1,10,25,40]:
        for m in [0,5,15,25,40]:
            if m<=ell:
                lbs,mys,j = "","",sht.indx(ell,m)
                for i in ix: lbs+= " {:12.4e}".format(sht.get_Ylm(ell,m)[i])
                for i in ix: mys+= " {:12.4e}".format(Ylb[j,i])
                print("({:2d},{:2d}):".format(ell,m))
                print("\t"+lbs)
                print("\t"+mys)
    #
    # Now print some values for the derivative (wrt Cos[theta]).
    print("\nCheck Ylm derivatives.")
    print("Compare pairs of lines labeled by (ell,m):")
    Ym = []
    Ym.append(3*sht.x * np.sqrt(5/4./np.pi))
    Ym.append(-5.42630291944221461*sht.x*(1-sht.x**2)**4)
    for j, (ell,m) in enumerate(zip([2,10],[0,10])):
        lbs,mys = "",""
        for i in ix: lbs+= " {:12.4e}".format(Ym[j][i])
        for i in ix: mys+= " {:12.4e}".format(sht.get_dYlm(ell,m)[i])
        print("({:2d},{:2d}):".format(ell,m))
        print("\t"+lbs)
        print("\t"+mys)

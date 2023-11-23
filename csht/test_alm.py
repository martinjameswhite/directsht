#!/usr/bin/env python
#
# Test the SHT class against the SciPy versions.
# Checks the function values and first derivatives
# for Ylm(x=cos[theta],0).
# Also does a sum of Ylm^star for some points as a
# test.
#
#
import numpy as np
import time


from   sht import DirectSHT






if __name__=="__main__":
    for Nl in [128,256]:
        now = time.time()
        sht = DirectSHT(Nl,2*Nl)
        print("Nl=",Nl," computation took ",time.time()-now," seconds.",flush=True)
    # Compare Ylm.
    Nl,Nx = 500,1024
    now   = time.time()
    sht   = DirectSHT(Nl,Nx)
    print("Computation took ",time.time()-now," seconds.",flush=True)
    # Now let's compare to the library version.  Compute a
    # table in the same format from SciPy.  Note SciPy has
    # the ell,m indices and theta,phi arguments reversed!
    from scipy.special import sph_harm
    #
    Ylb = np.zeros( ((Nl*(Nl+1))//2,Nx) )
    for ell in range(41):
        for m in range(ell+1):
            ii = sht.indx(ell,m)
            for i,x in enumerate(sht.x):
                theta     = np.arccos(x)
                Ylb[ii,i] = np.real( sph_harm(m,ell,0,theta) )
    # Just cross-check some Ylm values for sanity.
    print("Values:")
    ii = [0,Nx//3,2*Nx//3,Nx-1]
    Ylm= np.array(sht.Yv[:]).reshape(((Nl*(Nl+1))//2,Nx))
    for ell in [1,10,25,40]:
        for m in [0,5,15,25,40]:
            if m<=ell:
                lbs,mys,j = "","",sht.indx(ell,m)
                for i in ii: lbs+= " {:12.4e}".format(Ylm[j,i])
                for i in ii: mys+= " {:12.4e}".format(Ylb[j,i])
                print("({:2d},{:2d}):".format(ell,m))
                print("\t"+lbs)
                print("\t"+mys)
    # Now print some values for the derivative (wrt Cos[theta]).
    print("\nDerivatives:")
    Ym = np.zeros( ((Nl*(Nl+1))//2,Nx) )
    Ym[sht.indx( 2, 0),:] = 3*sht.x * np.sqrt(5/4./np.pi)
    Ym[sht.indx(10,10),:] = -4.19758*sht.x*(1-sht.x**2)**4*np.sqrt(21/4./np.pi)
    Yd = np.array(sht.Yd[:]).reshape(((Nl*(Nl+1))//2,Nx))
    for ell,m in zip([2,10],[0,10]):
        lbs,mys,j = "","",sht.indx(ell,m)
        for i in ii: lbs+= " {:12.4e}".format(Ym[j,i])
        for i in ii: mys+= " {:12.4e}".format(Yd[j,i])
        print("({:2d},{:2d}):".format(ell,m))
        print("\t"+lbs)
        print("\t"+mys)
    #
    # Now test the interpolation code.
    print("\nTest sum of Ylms.")
    tt = np.linspace(0.6,1.4,1000)
    pp = np.linspace(0.0,3.1,1000)
    wt = np.ones_like(tt)
    now= time.time()
    res= sht(tt,pp,wt)
    print("sht computation took ",time.time()-now," seconds for ",tt.size," points.\n",flush=True)
    for ell in range(4):
        for m in range(ell+1):
            ii = sht.indx(ell,m)
            print("({:3d},{:3d}) - ".format(ell,m)+str(res[ii]))
    #

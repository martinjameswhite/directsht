#!/usr/bin/env python3
#
import numpy as np

class ThreeJ000:
    """Returns the Wigner 3j symbol for integer j's and m1=m2=m3=0.
    Uses https://arxiv.org/pdf/2602.15605 .
    The sign is (-1)**pp where pp=(j1+j2+j3)/2."""
    def __init__(self,Lmax=512):
        """Initialize the class."""
        # Pre-compute the "g" function.
        lng = np.zeros(2*Lmax)
        for p in range(1,lng.size):
            lng[p] = lng[p-1]+np.log( 1.-0.5/p )
        self.gg = np.exp(lng)
        #
    def one3j(self,j1,j2,j3):
        """Evaluate a single 3-j symbol for integer js and m1=m2=m3=0."""
        J  = j1+j2+j3
        if (J%2>0): return(0)
        if ((j3<-np.abs(j1-j2))|(j3>j1+j2)): return(0)
        if ((j2<-np.abs(j1-j3))|(j2>j1+j3)): return(0)
        if ((j1<-np.abs(j2-j3))|(j1>j2+j3)): return(0)
        pp = J//2
        p1 = (-j1+j2+j3)//2
        p2 = ( j1-j2+j3)//2
        p3 = ( j1+j2-j3)//2
        gfac= self.gg[p1]*self.gg[p2]*self.gg[p3]/self.gg[pp]
        tj  = np.sqrt( gfac/(J+1.) ) # Sign ignored.
        tj *= (-1)**pp
        return(tj)
        #





if __name__=="__main__":
    from sympy.physics.wigner import wigner_3j
    Lmax= 25
    tj  = ThreeJ000(Lmax)
    ok  = True
    eps = 1e-4
    for j1 in range(Lmax):
        for j2 in range(Lmax):
            for j3 in range(Lmax):
                w3j = float(wigner_3j(j1,j2,j3,0,0,0))
                m3j = tj.one3j(j1,j2,j3)
                ok &= np.abs(m3j-w3j)/(np.abs(w3j)+eps)<eps
    print("All 3j in agreement: ",ok,flush=True)

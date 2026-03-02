#!/usr/bin/env python3
#
import numpy as np
from sympy.physics.wigner import wigner_3j


def threej000(j1,j2,j3):
    """Returns the absolute value of the Wigner 3j symbol for
    integer j's and m1=m2=m3=0.
    Uses https://arxiv.org/pdf/2602.15605 ."""
    J  = j1+j2+j3
    if (J%2>0): return(0)
    # Check the triange condition:
    tri = (j3<-abs(j1-j2))|(j3>j1+j2)| \
          (j2<-abs(j1-j3))|(j2>j1+j3)| \
          (j1<-abs(j2-j3))|(j1>j2+j3)
    if (tri): return(0)
    pp = J//2
    p1 = (-j1+j2+j3)//2
    p2 = ( j1-j2+j3)//2
    p3 = ( j1+j2-j3)//2
    # This part should really be pre-computed and stored, say using
    # a class with the 3j's returned via a __call__ method.
    lng = np.zeros(pp+1)
    for p in range(1,pp+1):
        lng[p] = lng[p-1]+np.log( 1. - 0.5/p )
    gg  = np.exp(lng)
    gfac= gg[p1]*gg[p2]*gg[p3]/gg[pp]
    tj  = np.sqrt( gfac/(J+1.) ) # Sign ignored.
    return(tj)
    #


def threej000_recurse(j1,j2,j3):
    """Returns the Wigner 3j symbol for integer j's and m1=m2=m3=0."""
    J = j1+j2+j3
    if (J%2>0):
        return(0)
    if (j1>=j2)&(j1>=j3)&(j2>=j3):
        if (j1==j2)&(j3==0):
            return( (-1)**j1/np.sqrt(2*j1+1.0) )
        elif (j1!=j2)&(j3==0):
            return( 0.0 )
        else:
            num = (J-2*j2-1)*(J-2*j3+2)
            den = (J-2*j2  )*(J-2*j3+1)
            fac = np.sqrt(float(num)/float(den))
            return(fac*threej000(j1,j2+1,j3-1))
    elif (j1>=j2)&(j1>=j3)&(j2< j3):
        return(threej000(j1,j3,j2))	# No minus sign since J even.
    elif (j1>=j2)&(j1< j3)&(j2< j3):
        return(threej000(j3,j1,j2))
    elif (j1< j2)&(j1>=j3)&(j2>=j3):
        return(threej000(j2,j1,j3))
    elif (j1< j2)&(j1< j3)&(j2>=j3):
        return(threej000(j2,j3,j1))
    elif (j1< j2)&(j1< j3)&(j2< j3):
        return(threej000(j3,j2,j1))
    else:
        raise RuntimeError
    #


if __name__=="__main__":
    for j1 in range(10):
        for j2 in range(10):
            for j3 in range(10):
                w3j = float(wigner_3j(j1,j2,j3,0,0,0))
                w3j = np.abs(w3j)
                m3j = threej000(j1,j2,j3)
                J   = j1+j2+j3
                if J%2==0:
                    err = (m3j-w3j)/(np.abs(w3j)+0.1)
                    print("{:2d} {:2d} {:2d} ".format(j1,j2,j3)+\
                          "{:10.5} {:10.5f} {:10.5f}".format(w3j,m3j,err))

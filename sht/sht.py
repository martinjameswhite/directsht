#!/usr/bin/env python
#
# A Python class to handle "direct" harmonic transforms of
# point sets on the sphere.
# Uses Numba.
#
#
import numpy as np
import numba as nb


class DirectSHT:
    """Brute-force spherical harmonic transforms."""
    def __init__(self,Nell,Nx):
        """Initialize the class, build the interpolation tables."""
        xx,Yv = self.compute_ylm_table(Nell,Nx)
        Yd    = self.compute_der_table()
        self.Nell,self.Nx      = Nell,Nx
        self.x,self.Yv,self.Yd = xx,Yv,Yd
        #
    def __call__(self,theta,phi,wt):
        """Returns alm for a collection of points at (theta,phi), in
        radians, with weights wt."""
        return(0)
        #
    def indx(self,ell,m):
        """The index in the grid storing Ylm for ell>=0, 0<=m<=ell."""
        ii = ( ell*(ell+1) )//2 + m
        return(ii)
    #@nb.njit
    def slow_recurrence(self,Nl,xx,Ylm):
        """Pull out the slow, multi-loop piece of the recurrence.  Have
        Nl,x,Ylm passed as arguments to allow a "dummy" call to compile
        and use of JIT."""
        for m in range(0,Nl-1):
            for ell in range(m+2,Nl):
                i0,i1,i2    = self.indx(ell  ,m),\
                              self.indx(ell-1,m),\
                              self.indx(ell-2,m)
                fact1,fact2 = np.sqrt( (ell-m)*1./(ell+m) ),\
                               np.sqrt( (ell-m-1.)/(ell+m-1.) )
                Ylm[i0,:]   = (2*ell-1)*xx*Ylm[i1,:]-\
                              (ell+m-1)   *Ylm[i2,:]*fact2
                Ylm[i0,:]  *= fact1/(ell-m)
        return(Ylm)
        #
    def compute_ylm_table(self,Nl,Nx):
        """Use recurrence relations to compute a table of Ylm[cos(theta),0]
        for ell>=0, m>=0, x=>0.  Can use symmetries to get m<0 and/or x<0,
        viz. (-1)^m for m<0 and (-1)^(ell-m) for x<0.
        Returns x,Y[ell,m,x=Cos[theta],0]"""
        xx = np.arange(Nx)/float(Nx)
        sx = np.sqrt(1-xx**2)
        Ylm= np.zeros( ((Nl*(Nl+1))//2,Nx) )
        #
        # First we do the m=0 case.
        Ylm[self.indx(0,0),:] = np.ones_like(xx)
        Ylm[self.indx(1,0),:] = xx.copy()
        for ell in range(2,Nl):
            i0,i1,i2  = self.indx(ell,0),self.indx(ell-1,0),self.indx(ell-2,0)
            Ylm[i0,:] = (2*ell-1)*xx*Ylm[i1,:]-(ell-1)*Ylm[i2,:]
            Ylm[i0,:]/= float(ell)
        # Now we fill in m>0.
        # To keep the recurrences stable, we treat "high m" and "low m"
        # separately.  Start with the highest value of m allowed:
        for m in range(1,Nl):
            i0,i1     = self.indx(m,m),self.indx(m-1,m-1)
            Ylm[i0,:] = -np.sqrt(1.0-1./(2*m))*sx*Ylm[i1,:]
        # Now do m=ell-1
        for m in range(1,Nl-1):
            i0,i1     = self.indx(m,m),self.indx(m+1,m)
            Ylm[i1,:] = np.sqrt(2*m+1.)*xx*Ylm[i0,:]
        # Finally fill in ell>m+1:
        # First a dummy, warmup run to JIT compile, then the real thing.
        _   = self.slow_recurrence( 1,xx,Ylm)
        Ylm = self.slow_recurrence(Nl,xx,Ylm)
        # And finally put in the (2ell+1)/4pi normalization:
        for ell in range(Nl):
            fact = np.sqrt( (2*ell+1)/4./np.pi )
            for m in range(ell+1):
                ii         = self.indx(ell,m)
                Ylm[ii,:] *= fact
        return( (xx,Ylm) )
        #
    def compute_der_table(self):
        """Use recurrence relations to compute a table of derivatives of
        Ylm[cos(theta),0] for ell>=0, m>=0, x=>0.  Assumes the Ylm table
        has already been built."""
        return(0)
        #

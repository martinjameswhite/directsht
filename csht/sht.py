#!/usr/bin/env python
#
# A Python class to handle "direct" harmonic transforms of
# point sets on the sphere.
# Uses Numba.
#
#
import numpy  as np
import ctypes as ct
import time
import os


# Find out where we are.
fullpath = os.path.dirname(__file__) + "/"



def ext_slow_recurrence(Nl,xx,Ylm):
    """Pull out the slow, multi-loop piece of the recurrence.  In
    order to use JIT this can not be part of the class, and we need
    to pass Nl,x,Ylm as arguments."""
    # This should match the convention used in the SHT class below.
    indx = lambda ell,m:  int(m * (2 * (Nl-1) + 1 - m) / 2 + ell)
    for m in range(0,Nl-1):
        for ell in range(m+2,Nl):
            i0,i1,i2    = indx(ell  ,m),\
                          indx(ell-1,m),\
                          indx(ell-2,m)
            fact1,fact2 = np.sqrt( (ell-m)*1./(ell+m) ),\
                          np.sqrt( (ell-m-1.)/(ell+m-1.) )
            Ylm[i0,:]   = (2*ell-1)*xx*Ylm[i1,:]-\
                          (ell+m-1)   *Ylm[i2,:]*fact2
            Ylm[i0,:]  *= fact1/(ell-m)
    return(Ylm)
    #



def ext_der_slow_recurrence(Nl,xx,Yv,Yd):
    """Pull out the slow, multi-loop piece of the recurrence for the
    derivatives."""
    # This should match the convention used in the SHT class below.
    indx = lambda ell,m:  int(m * (2 * (Nl-1) + 1 - m) / 2 + ell)
    omx2 = 1.0-xx**2
    for ell in range(Nl):
        for m in range(1,ell+1):
            i0,i1     = indx(ell,m),indx(ell-1,m)
            fact      = np.sqrt( float(ell-m)/(ell+m) )
            Yd[i0,:]  = (ell+m)*fact*Yv[i1,:]-ell*xx*Yv[i0,:]
            Yd[i0,:] /= omx2
    return(Yd)
    #



class DirectSHT:
    """Brute-force spherical harmonic transforms."""
    def __init__(self,Nell,Nx,xmax=0.9):
        """Initialize the class, build the interpolation tables.
        :param  Nell: Number of ells, and hence ms.
        :param  Nx:   Number of x grid points.
        :param xmax:  Maximum value of |cos(theta)| to compute.
        """
        self.Nell, self.Nx, self.xmax = Nell, Nx, xmax
        self.Nlm   = (Nell*(Nell+1))//2
        self.Nsize = self.Nlm*Nx
        xx = np.arange(Nx,dtype='float64')/float(Nx-1) * xmax
        Yv = np.ascontiguousarray(np.zeros(self.Nsize,dtype='f8'))
        Yd = np.ascontiguousarray(np.zeros(self.Nsize,dtype='f8'))
        # Convert to c_double_Array object.
        Yv = (ct.c_double*Yv.size)(*(Yv))
        Yd = (ct.c_double*Yd.size)(*(Yv))
        self.mylib = ct.CDLL(fullpath+"sht_helper.so")
        self.x,self.Yv,self.Yd = xx,Yv,Yd
        #
        self.mylib.make_table(ct.c_int(Nell),ct.c_int(Nx),ct.c_double(xmax),\
                              ct.byref(Yv),ct.byref(Yd))
        #
    def __call__(self,theta,phi,wt,reg_factor=1.,verbose=True):
        """
        Returns alm for a collection of real-valued points at (theta,phi),
        in radians, with weights wt.
        :param theta: 1D numpy array of theta values for each point.
         Must be between [0,pi].
        :param phi: 1D numpy array of phi values for each point.
         Must be between [0,2pi].
        :param wt: 1D numpy array of weights for each point.
        :return: alm in the Healpix indexing convention,
                 i.e., complex coefficients alm[m*(2*lmax+1-m)/2+l]
                 with l in [0, lmax] and m in [0, l]
        """
        assert len(theta)==len(phi) and \
               len(phi)==len(wt),"theta,phi,wt must be the same length."
        assert np.all( (theta>=0) & (theta>np.arccos(self.xmax)) & (theta<np.arccos(-self.xmax))),\
               "theta must be in [ACos[xmax],ACos[-xmax])."
        Npnt = len(theta)
        # Convert theta, phi and wt to c_double_Arrays.
        tt = np.ascontiguousarray(theta)
        tt = (ct.c_double*tt.size)(*(tt))
        pp = np.ascontiguousarray(phi)
        pp = (ct.c_double*pp.size)(*(pp))
        ww = np.ascontiguousarray(wt)
        ww = (ct.c_double*ww.size)(*(ww))
        # Make space for the cosine and sine components.
        carr = np.ascontiguousarray(np.zeros(self.Nlm,dtype='f8'))
        sarr = np.ascontiguousarray(np.zeros(self.Nlm,dtype='f8'))
        # Convert to c_double_Array object.
        carr = (ct.c_double*carr.size)(*(carr))
        sarr = (ct.c_double*sarr.size)(*(sarr))
        self.mylib.do_transform(ct.c_int(self.Nell),ct.c_int(self.Nx),ct.c_double(self.xmax),\
                                ct.byref(self.Yv),ct.byref(self.Yd),\
                                ct.c_int(Npnt),ct.byref(tt),ct.byref(pp),ct.byref(ww),\
                                ct.byref(carr),ct.byref(sarr))
        carr,sarr = np.array(carr[:]),np.array(sarr[:])
        return( (carr,sarr) )
    def indx(self,ell,m):
        """
        The index in the grid storing Ylm for ell>=0, 0<=m<=ell.
        Matches the Healpix convention.
        Note: this should match the indexing in ext_slow_recurrence
        and ext_der_slow_recurrence.
        :param  ell: ell value to return.
        :param  m:   m value to return.
        :return ii:  Index value in the value and derivatives grids.
        """
        lmax = self.Nell-1
        ii = int(m * (2*lmax+1-m)/2 + ell)
        return(ii)
    def slow_recurrence(self,Nl,xx,Ylm):
        """Pull out the slow, multi-loop piece of the recurrence.
        THIS IS CURRENTLY NOT USED."""
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
    def interp_test(self,ell,m,xx):
        """Interpolates Ylm(acos(x),0) at positions xx. Using a
        cubic Hermite spline.  Assumes xx is sorted.
        Used during development for code and convergence tests.
        :param  ell: ell value to return.
        :param  m:   m value to return.
        :param  xx:  Array of cos(theta) values, assumed sorted.
        :return yx:  Interpolated Y[ell,m,x=Cos[theta],0].
        """
        jj = self.indx(ell,m)
        i1 = np.digitize(xx,self.x)
        i0 = i1-1
        dx = 0.5*(self.x[2]-self.x[0])
        tt = (xx - self.x[i0])/dx
        t1 = (tt-1.0)**2
        t2 = tt**2
        s0 = (1+2*tt)*t1
        s1 = tt*t1
        s2 = t2*(3-2*tt)
        s3 = t2*(tt-1.0)
        yx = self.Yv[jj,i0]*s0+self.Yd[jj,i0]*s1*dx +\
             self.Yv[jj,i1]*s2+self.Yd[jj,i1]*s3*dx
        return(yx)
    def compute_Plm_table(self,Nl,xx):
        """Use recurrence relations to compute a table of Ylm[cos(theta),0]
        for ell>=0, m>=0, x>=0.  Can use symmetries to get m<0 and/or x<0,
        viz. (-1)^m for m<0 and (-1)^(ell-m) for x<0.
        :param  Nl: Number of ells (and hence m's) in the grid.
        :param  xx: Array of x points (non-negative and increasing).
        :return Y[ell,m,x=Cos[theta],0] without the sqrt{(2ell+1)/4pi}
        normalization (that is applied in __init__)
        """
        # Set up a regular grid of x values.
        Nx = xx.size
        sx = np.sqrt(1-xx**2)
        Plm= np.zeros( ((Nl*(Nl+1))//2,Nx), dtype='float64')
        #
        # First we do the m=0 case.
        Plm[self.indx(0,0),:] = np.ones_like(xx)
        Plm[self.indx(1,0),:] = xx.copy()
        for ell in range(2,Nl):
            i0,i1,i2  = self.indx(ell,0),self.indx(ell-1,0),self.indx(ell-2,0)
            Plm[i0,:] = (2*ell-1)*xx*Plm[i1,:]-(ell-1)*Plm[i2,:]
            Plm[i0,:]/= float(ell)
        # Now we fill in m>0.
        # To keep the recurrences stable, we treat "high m" and "low m"
        # separately.  Start with the highest value of m allowed:
        for m in range(1,Nl):
            i0,i1     = self.indx(m,m),self.indx(m-1,m-1)
            Plm[i0,:] = -np.sqrt(1.0-1./(2*m))*sx*Plm[i1,:]
        # Now do m=ell-1
        for m in range(1,Nl-1):
            i0,i1     = self.indx(m,m),self.indx(m+1,m)
            Plm[i1,:] = np.sqrt(2*m+1.)*xx*Plm[i0,:]
        # Finally fill in ell>m+1:
        # First a dummy, warmup run to JIT compile, then the real thing.
        _   = ext_slow_recurrence( 1,xx,Plm)
        Plm = ext_slow_recurrence(Nl,xx,Plm)
        return(Plm)
        #
    def compute_der_table(self,Nl,xx,Yv):
        """Use recurrence relations to compute a table of derivatives of
        Ylm[cos(theta),0] for ell>=0, m>=0, x=>0.
        Assumes the Ylm table has already been built (passed as Yv).
        :param  Nl: Number of ells in the derivative grid.
        :param  xx: Values of cos(theta) at which to evaluate derivs.
        :param  Yv: Already computed Ylm values.
        :return Yd: The table of first derivatives.
        """
        Yd = np.zeros( ((Nl*(Nl+1))//2,xx.size), dtype='float64')
        Yd[self.indx(1,0),:] = np.ones_like(xx)
        # Do the case m=0 separately.
        for ell in range(2,Nl):
            i0,i1    = self.indx(ell,0),self.indx(ell-1,0)
            Yd[i0,:] = ell/(1-xx**2)*(Yv[i1,:]-xx*Yv[i0,:])
        # then build the m>0 tables.
        _  = ext_der_slow_recurrence( 1,xx,Yv,Yd)
        Yd = ext_der_slow_recurrence(Nl,xx,Yv,Yd)
        return(Yd)
        #

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






class DirectSHT:
    """Brute-force spherical harmonic transforms."""
    def __init__(self,Nell,Nx,xmax=0.75):
        """Initialize the class, build the interpolation tables.
        :param  Nell: Number of ells, and hence ms.
        :param  Nx:   Number of x grid points.
        :param xmax:  Maximum value of |cos(theta)| to compute.
        """
        self.Nell, self.Nx, self.xmax = Nell, Nx, xmax
        self.Nlm   = (Nell*(Nell+1))//2 # Total (l,m) size.
        self.Nsize = self.Nlm*Nx        # Size of tables.
        xx = np.arange(Nx,dtype='float64')/float(Nx-1) * xmax
        self.x = xx
        # Set up c_double_Array objects for storing Ylm and its
        # first derivative.
        self.Yv    = (ct.c_double*self.Nsize)()
        self.Yd    = (ct.c_double*self.Nsize)()
        self.mylib =  ct.CDLL(fullpath+"sht_helper.so")
        #
        self.mylib.make_table(ct.c_int(Nell),ct.c_int(Nx),ct.c_double(xmax),\
                              ct.byref(self.Yv),ct.byref(self.Yd))
        #
    def old_call(self,theta,phi,wt):
        """
        KEPT FOR ARCHIVAL PURPOSES/DEBUGGING.
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
        # Convert theta, phi and wt to c_double_Arrays and take cos(theta).
        tt = (ct.c_double*Npnt)()
        pp = (ct.c_double*Npnt)()
        ww = (ct.c_double*Npnt)()
        tt[:],pp[:],ww[:] = np.cos(theta),phi,wt
        # Make space for the cosine and sine components, as c_double_Array objects.
        carr = (ct.c_double*self.Nlm)()
        sarr = (ct.c_double*self.Nlm)()
        self.mylib.do_transform(ct.c_int(self.Nell),ct.c_int(self.Nx),ct.c_double(self.xmax),\
                                ct.byref(self.Yv),ct.byref(self.Yd),\
                                ct.c_int(Npnt),ct.byref(tt),ct.byref(pp),ct.byref(ww),\
                                ct.byref(carr),ct.byref(sarr))
        carr,sarr = np.array(carr[:]),np.array(sarr[:])
        res = carr - 1j*sarr
        return( res )
    def __call__(self,theta,phi,wt):
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
        # Make space for the cosine and sine components, as c_double_Array objects.
        carr = (ct.c_double*self.Nlm)()
        sarr = (ct.c_double*self.Nlm)()
        alm  = np.zeros(self.Nlm,dtype='complex128')
        # We need to break theta into positive and negative cos(theta) and
        # treat them separately.
        cost = np.cos(theta)
        ww   = np.nonzero(cost<0)[0]
        if len(ww)>0:
            Npnt = len(ww)
            ii   = ww[np.argsort(-cost[ww])]
            # Convert theta, phi and wt to c_double_Arrays and take cos(theta).
            tt = (ct.c_double*Npnt)()
            pp = (ct.c_double*Npnt)()
            ww = (ct.c_double*Npnt)()
            tt[:],pp[:],ww[:] = cost[ii],phi[ii],wt[ii]
            self.mylib.do_transform(\
                ct.c_int(self.Nell),ct.c_int(self.Nx),ct.c_double(self.xmax),\
                ct.byref(self.Yv),ct.byref(self.Yd),\
                ct.c_int(Npnt),ct.byref(tt),ct.byref(pp),ct.byref(ww),\
                ct.byref(carr),ct.byref(sarr))
            # Generate complex a_{lm} from sine and cosine components.
            alm += np.array(carr[:])-1j*np.array(sarr[:])
            # Insert phase factor (-1)^(ell-m).  This is slow but, ok.
            for ell in range(self.Nell):
                for m in range(1-(ell%2),ell+1,2):
                    alm[self.indx(ell,m)] *= -1.0
        #
        # Now treat the non-negative cos(theta).
        ww = np.nonzero(cost>=0)[0]
        if len(ww)>0:
            Npnt = len(ww)
            ii   = ww[np.argsort(cost[ww])]
            # Convert theta, phi and wt to c_double_Arrays and take cos(theta).
            tt = (ct.c_double*Npnt)()
            pp = (ct.c_double*Npnt)()
            ww = (ct.c_double*Npnt)()
            tt[:],pp[:],ww[:] = cost[ii],phi[ii],wt[ii]
            self.mylib.do_transform(\
                ct.c_int(self.Nell),ct.c_int(self.Nx),ct.c_double(self.xmax),\
                ct.byref(self.Yv),ct.byref(self.Yd),\
                ct.c_int(Npnt),ct.byref(tt),ct.byref(pp),ct.byref(ww),\
                ct.byref(carr),ct.byref(sarr))
            # No phase factor to worry about.
            alm += np.array(carr[:])-1j*np.array(sarr[:])
        return(alm)
        #
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
        ii= (m*(2*self.Nell-1-m))//2 + ell
        return(ii)

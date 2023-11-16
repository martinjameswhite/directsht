#!/usr/bin/env python
#
# A Python class to handle "direct" harmonic transforms of
# point sets on the sphere.
# Uses Numba.
#
#
import numpy as np
import numba as nb
import interp_funcs as interp
import utils
import time

try:
    jax_present = True
    from   jax import device_put, vmap, jit
except ImportError:
    jax_present = False
    print("JAX not found. Falling back to NumPy.")


@nb.njit
def ext_slow_recurrence(Nl,xx,Ylm):
    """Pull out the slow, multi-loop piece of the recurrence.  In
    order to use JIT this can not be part of the class, and we need
    to pass Nl,x,Ylm as arguments."""
    # This should match the convention used in the SHT class below.
    indx = lambda ell,m:  ( ell*(ell+1) )//2 + m
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



@nb.njit
def ext_der_slow_recurrence(Nl,xx,Yv,Yd):
    """Pull out the slow, multi-loop piece of the recurrence for the
    derivatives."""
    # This should match the convention used in the SHT class below.
    indx = lambda ell,m:  ( ell*(ell+1) )//2 + m
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
    def __init__(self,Nell,Nx):
        """Initialize the class, build the interpolation tables."""
        xx,Yv = self.compute_Plm_table(Nell,Nx)
        Yd    = self.compute_der_table(Nell,xx,Yv)
        # And finally put in the (2ell+1)/4pi normalization:
        for ell in range(Nell):
            fact = np.sqrt( (2*ell+1)/4./np.pi )
            for m in range(ell+1):
                ii        = self.indx(ell,m)
                Yv[ii,:] *= fact
                Yd[ii,:] *= fact
        self.Nell,self.Nx      = Nell,Nx
        self.x,self.Yv,self.Yd = xx,Yv,Yd
        #
    def __call__(self,theta,phi,wt):
        """Returns alm for a collection of points at (theta,phi), in
        radians, with weights wt."""
        assert len(theta)==len(phi) and len(phi)==len(wt), "theta,phi,wt must all be the same length."
        assert np.all( (theta>=0) & (theta<=np.pi) ), "theta must be in [0,pi]." #TODO: do we include the endpoints?
        assert np.all( (phi>=0) & (phi<=2*np.pi) ), "phi must be in [0,2pi]." #TODO: do we include the endpoints?
        t0= time.time()

        # Sort the data in ascending order of theta
        sorted_indices = np.argsort(theta)
        theta_data_sorted = theta[sorted_indices]
        # TODO: Convert from [0,1] to theta!
        theta_samples = self.x
        w_i_sorted = wt[sorted_indices]
        phi_data_sorted = phi[sorted_indices]
        t1 = time.time()
        print("Sorting took ",t1-t0," seconds.",flush=True)

        # Find which spline region each point falls into
        spline_idx = np.digitize(theta_data_sorted, theta_samples) - 1
        t = theta_data_sorted - theta_samples[spline_idx]

        # We now sum up all w_p f(t) in each spline region i
        ms = np.arange(self.Nell, dtype=int)
        vs_real, vs_imag = [interp.precompute_vs(len(theta_samples), spline_idx, phi_data_sorted,
                                                 w_i_sorted, t, ms, which_part) for which_part in ['cos', 'sin']]

        t2 = time.time()
        print("Precomputing vs took ",t2-t1," seconds.",flush=True)

        # TODO: CAREFUL! This does not match the convention used in the SHT class above!!!
        # Get the ordering of alm's in the array
        ell_ordering, m_ordering = utils.getlm(self.Nell, len(self.Yv[:, 0]))

        if jax_present:
            # Get a grid of all alm's -- best run on a GPU!
            get_all_alms_w_jax = vmap(jit(interp.get_alm), in_axes=(0, 0, None, 0))
            # Notice we put the Ylm and dYlm tables in device memory for a speed boost
            alm_grid_realpart = get_all_alms_w_jax(device_put(self.Yv), device_put(self.Yd), vs_real, m_ordering)
            alm_grid_imagpart = get_all_alms_w_jax(device_put(self.Yv), device_put(self.Yd), vs_imag, m_ordering)
            alm_grid = np.array(alm_grid_realpart) + 1j * np.array(alm_grid_imagpart)

        else:
            # JIT compile the get_alm function and vectorize it
            get_alm_jitted = nb.jit(nopython=True)(interp.get_alm)

            alm_grid = np.zeros(len(self.Yv[:,0]), dtype=complex)
            vs_tot = vs_real + 1j * vs_imag
            #TODO: parallelize this
            for i, (Ylm, dYlm, m) in enumerate(zip(self.Yv, self.Yd, m_ordering)):
                alm_grid[i] = get_alm_jitted(Ylm, dYlm, vs_tot, m)
        t3 = time.time()
        print("Computing alm's took ",t3-t2," seconds.",flush=True)
        return alm_grid

        #
    def indx(self,ell,m):
        """The index in the grid storing Ylm for ell>=0, 0<=m<=ell."""
        ii = ( ell*(ell+1) )//2 + m
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
    def compute_Plm_table(self,Nl,Nx):
        """Use recurrence relations to compute a table of Ylm[cos(theta),0]
        for ell>=0, m>=0, x=>0.  Can use symmetries to get m<0 and/or x<0,
        viz. (-1)^m for m<0 and (-1)^(ell-m) for x<0.
        Returns x,Y[ell,m,x=Cos[theta],0] without the sqrt{(2ell+1)/4pi}
        normalization (that is applied in __init__"""
        # Set up a regular grid of x values.
        xx = np.arange(Nx)/float(Nx)
        sx = np.sqrt(1-xx**2)
        Plm= np.zeros( ((Nl*(Nl+1))//2,Nx) )
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
        return( (xx,Plm) )
        #
    def compute_der_table(self,Nl,xx,Yv):
        """Use recurrence relations to compute a table of derivatives of
        Ylm[cos(theta),0] for ell>=0, m>=0, x=>0.  Assumes the Ylm table
        has already been built (passed as Yv)."""
        Yd = np.zeros( ((Nl*(Nl+1))//2,xx.size) )
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

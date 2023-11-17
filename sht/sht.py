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

default_dtype = 'float64'
@nb.njit
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
        self.Nell, self.Nx = Nell, Nx
        xx,Yv = self.compute_Plm_table(Nell,Nx)
        Yd    = self.compute_der_table(Nell,xx,Yv)
        # And finally put in the (2ell+1)/4pi normalization:
        for ell in range(Nell):
            fact = np.sqrt( (2*ell+1)/4./np.pi )
            for m in range(ell+1):
                ii        = self.indx(ell,m)
                Yv[ii,:] *= fact
                Yd[ii,:] *= fact
        self.x,self.Yv,self.Yd = xx,Yv,Yd
        #
    def __call__(self,theta,phi,wt):
        """
        Returns alm for a collection of real-valued points at (theta,phi), in radians, with weights wt.
        :param theta: 1D numpy array of theta values for each point. Must be between [0,pi].
        :param phi: 1D numpy array of phi values for each point. Must be between [0,2pi].
        :param wt: 1D numpy array of weights for each point.
        :return: alm in the Healpix indexing convention,
                i.e., complex coefficients alm[m * (2*lmax+1-m)/2 + l] with l in [0, lmax] and m in [0, l]
        """
        assert len(theta)==len(phi) and len(phi)==len(wt), "theta,phi,wt must all be the same length."
        assert np.all( (theta>=0) & (theta<=np.pi) ), "theta must be in [0,pi)."

        # Get the indexing of ell and m in the Healpix convention for later use
        ell_ordering, m_ordering = utils.getlm(self.Nell, len(self.Yv[:, 0]))
        # Eventually, we will need to multiply the alm's by (-1)^{ell-m} for x=cos\theta<0
        parity_factor = (-1) ** (ell_ordering - m_ordering)
        # Initialize storage array in dtype compatible with Healpy
        alm_grid_tot = np.zeros(len(ell_ordering), dtype='complex128')

        # We've precomputed P_{\ell m}(x=cos(theta)), so let's work with x thus defined
        x_full = np.cos(theta)

        # Split into positive and negative values of x. We'll treat them separately and rejoin them ar the very end
        pos_idx, neg_idx = ([i for i, value in enumerate(x_full) if value >= 0],
                            [i for i, value in enumerate(x_full) if value < 0])
        if pos_idx and neg_idx:
            which_case = zip([x_full[pos_idx], np.abs(x_full[neg_idx])], [1., parity_factor], [pos_idx, neg_idx])
        elif pos_idx:
            which_case = zip([x_full[pos_idx]], [1.], [pos_idx])
        elif neg_idx:
            which_case = zip([np.abs(x_full[neg_idx])], [parity_factor], [neg_idx])
        else:
            raise ValueError("The theta array seems to be empty!")

        if jax_present:
            # We'll want to move big arrays to GPU memory only once
            Yv_jax = device_put(self.Yv)
            dYv_jax = device_put(self.Yd)

        for x, par_fact, idx in which_case:
            t0 = time.time()
            # Sort the data in ascending order of theta
            sorted_idx = np.argsort(x)
            x_data_sorted = x[sorted_idx]; w_i_sorted = wt[idx][sorted_idx]; phi_data_sorted = phi[idx][sorted_idx]
            x_samples = self.x

            t1 = time.time()
            print("Sorting took ",t1-t0," seconds.",flush=True)

            # Find which spline region each point falls into
            spline_idx = np.digitize(x_data_sorted, x_samples) - 1
            t = x_data_sorted - x_samples[spline_idx]

            t15 = time.time()
            print("Digitizing took ",t15-t1," seconds.",flush=True)

            # We now sum up all w_p f(t) in each spline region i
            ms = np.arange(self.Nell, dtype=int)
            vs_real, vs_imag = [interp.precompute_vs(len(x_samples), spline_idx, phi_data_sorted,
                                                     w_i_sorted, t, ms, which_part) for which_part in ['cos', 'sin']]

            t2 = time.time()
            print("Precomputing vs took ",t2-t1," seconds.",flush=True)

            if jax_present:
                # Get a grid of all alm's -- best run on a GPU!
                get_all_alms_w_jax = vmap(jit(interp.get_alm), in_axes=(0, 0, None, 0))
                # Notice we put the Ylm and dYlm tables in device memory for a speed boost
                alm_grid_real = get_all_alms_w_jax(Yv_jax, dYv_jax, vs_real, m_ordering)
                alm_grid_imag = get_all_alms_w_jax(Yv_jax, dYv_jax, vs_imag, m_ordering)
                alm_grid = np.array(alm_grid_real, dtype='complex128') - 1j *np.array(alm_grid_imag, dtype='complex128')

            else:
                # JIT compile the get_alm function and vectorize it
                get_alm_jitted = nb.jit(nopython=True)(interp.get_alm)

                alm_grid = np.zeros(len(self.Yv[:,0]), dtype='complex128')
                vs_tot = vs_real - 1j * vs_imag
                #TODO: parallelize this
                for i, (Ylm, dYlm, m) in enumerate(zip(self.Yv, self.Yd, m_ordering)):
                    alm_grid[i] = get_alm_jitted(Ylm, dYlm, vs_tot, m)

            # For x<0, we need to multiply by (-1)^{ell-m}
            alm_grid_tot += par_fact * alm_grid
            t3 = time.time()
            print("Computing alm's took ",t3-t2," seconds.",flush=True)

        return alm_grid_tot

        #
    def indx(self,ell,m):
        """
        The index in the grid storing Ylm for ell>=0, 0<=m<=ell. Matches the Healpix convention.
        Note: this should match the indexing in ext_slow_recurrence()
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
    def compute_Plm_table(self,Nl,Nx):
        """Use recurrence relations to compute a table of Ylm[cos(theta),0]
        for ell>=0, m>=0, x=>0.  Can use symmetries to get m<0 and/or x<0,
        viz. (-1)^m for m<0 and (-1)^(ell-m) for x<0.
        Returns x,Y[ell,m,x=Cos[theta],0] without the sqrt{(2ell+1)/4pi}
        normalization (that is applied in __init__"""
        # Set up a regular grid of x values.
        xx = np.arange(Nx, dtype=default_dtype)/float(Nx)
        sx = np.sqrt(1-xx**2)
        Plm= np.zeros( ((Nl*(Nl+1))//2,Nx), dtype=default_dtype)
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
        Yd = np.zeros( ((Nl*(Nl+1))//2,xx.size), dtype=default_dtype)
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

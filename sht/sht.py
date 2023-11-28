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
from scipy.stats import mode

try:
    jax_present = True
    from   jax import device_put, vmap, jit
except ImportError:
    jax_present = False
    device_put = lambda x: x  # Dummy definition for fallback
    print("JAX not found. Falling back to NumPy.")



@nb.njit
def ext_slow_recurrence(Nl,xx,Ylm):
    """Pull out the slow, multi-loop piece of the recurrence.  In
    order to use JIT this can not be part of the class, and we need
    to pass Nl,x,Ylm as arguments."""
    # This should match the convention used in the SHT class below.
    indx = lambda ell,m:  (m*(2*Nl-1-m))//2 + ell
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
    indx = lambda ell,m:  (m*(2*Nl-1-m))//2 + ell
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
    def __init__(self,Nell,Nx,xmax=0.875):
        """Initialize the class, build the interpolation tables.
        :param  Nell: Number of ells, and hence ms.
        :param  Nx:   Number of x grid points.
        :param xmax:  Maximum value of |cos(theta)| to compute.
        """
        self.Nell, self.Nx, self.xmax = Nell, Nx, xmax
        xx = np.arange(Nx,dtype='float64')/float(Nx-1) * xmax
        Yv = self.compute_Plm_table(Nell,xx)
        Yd = self.compute_der_table(Nell,xx,Yv)
        # And finally put in the (2ell+1)/4pi normalization:
        for ell in range(Nell):
            fact = np.sqrt( (2*ell+1)/4./np.pi )
            for m in range(ell+1):
                ii        = self.indx(ell,m)
                Yv[ii,:] *= fact
                Yd[ii,:] *= fact
        self.x,self.Yv,self.Yd = xx,Yv,Yd
        #
    def __call__(self,theta,phi,wt,reg_factor=1.,verbose=True):
        """
        Returns alm for a collection of real-valued points at (theta,phi),
        in radians, with weights wt.
        :param theta: 1D numpy array of theta values for each point.
         Must be between [0,pi], and also satisfy [ACos[xmax],ACos[-xmax]
        :param phi: 1D numpy array of phi values for each point.
         Must be between [0,2pi].
        :param wt: 1D numpy array of weights for each point.
        :param reg_factor: Scaling to apply to weights to avoid numerical
         over/underflow. It gets removed at the end.
        :param verbose: if True, print out timing information.
        :return: alm in the Healpix indexing convention,
                 i.e., complex coefficients alm[m*(2*lmax+1-m)/2+l]
                 with l in [0, lmax] and m in [0, l]
        """
        assert len(theta)==len(phi) and \
               len(phi)==len(wt),"theta,phi,wt must be the same length."
        assert np.all( (theta>=0) & (theta>np.arccos(self.xmax)) & (theta<np.arccos(-self.xmax))),\
               "theta must be in [ACos[xmax],ACos[-xmax])."

        x_samples = self.x
        # TODO: this way of calculating dx assumes the points are evenly spaced in x
        dx = x_samples[1] - x_samples[0]
        # Multiply the weights by a regularization factor to avoid numerical
        # under/overflow
        wt*= reg_factor
        # Get the indexing of ell and m in the Healpix convention for
        # later use
        ell_ordering,m_ordering = utils.getlm(self.Nell-1,len(self.Yv[:, 0]))
        # Eventually, we will need to multiply the alm's by (-1)^{ell-m}
        # for x=cos\theta<0
        parity_factor = (-1)**(ell_ordering - m_ordering)
        # Initialize storage array in dtype compatible with Healpy
        alm_grid_tot = np.zeros(len(ell_ordering), dtype='complex128')
        # We've precomputed P_{\ell m}(x=cos(theta)), so let's work
        # with x thus defined
        x_full = np.cos(theta)
        # Split into positive and negative values of x.
        # We'll treat them separately and rejoin them ar the very end
        pos_idx,neg_idx = ([i for i, value in enumerate(x_full) if value >= 0],\
                           [i for i, value in enumerate(x_full) if value <  0])
        if pos_idx and neg_idx:
            # The case where there's both +ve and -ve values of x
            which_case = zip([x_full[pos_idx],np.abs(x_full[neg_idx])],\
                             [1.,parity_factor],[pos_idx,neg_idx])
        elif pos_idx:
            # The case where there's only +ve values of x
            which_case = zip([x_full[pos_idx]],[1.],[pos_idx])
        elif neg_idx:
            # The case where there's only -ve values of x
            which_case = zip([np.abs(x_full[neg_idx])],[parity_factor],\
                             [neg_idx])
        else:
            raise ValueError("The theta array seems to be empty!")
        #
        # Treat +ve and -ve x separately
        for x, par_fact, idx in which_case:
            t0 = time.time()
            # Sort the data in ascending order of theta
            sorted_idx = np.argsort(x)
            x_data_sorted = x[sorted_idx]; w_i_sorted = wt[idx][sorted_idx];
            phi_data_sorted = phi[idx][sorted_idx]
            #
            t1 = time.time()
            if verbose: print("Sorting took ",t1-t0," seconds.",flush=True)
            #
            # Find which spline region each point falls into
            spline_idx = np.digitize(x_data_sorted, x_samples) - 1
            t = (x_data_sorted - x_samples[spline_idx]) / dx
            #
            # Find which bins (bounded by the elements of x_samples) are populated
            occupied_bins = np.unique(spline_idx)
            bin_num = len(occupied_bins)
            # Then, we find the maximum number of points in a bin
            bin_len = mode(spline_idx).count
            # Find the data indices where transitions btw splines/bins happen
            transitions = utils.find_transitions(spline_idx)
            # Reshape the inputs into a 2D array for fast binning during
            # computation of the v's. Our binning scheme involves zero-padding bins with fewer
            # than bin_len points, but cos(0)=1!=0, so we need a mask to discard spurious zeros!
            mask = utils.reshape_phi_array(np.ones_like(phi_data_sorted), transitions, bin_num, bin_len)
            # Mask and put in GPU memory
            reshaped_phi_data = device_put(mask * utils.reshape_phi_array(phi_data_sorted,
                                                                          transitions, bin_num, bin_len))
            # Repeat the process for  the other required inputs
            reshaped_inputs = utils.reshape_aux_array([w_i_sorted*input_ for input_ in
                                                      [(2*t+1)*(1-t)**2,t*(1-t)**2,t**2*(3-2*t)
                                                          ,t**2*(t-1)]], transitions, bin_num, bin_len)
            reshaped_inputs = device_put(mask * reshaped_inputs)

            # Query only theta bins that have data. While we're at it, scale derivatives by dx
            Yv_i_short = self.Yv[:, occupied_bins]; Yv_ip1_short = self.Yv[:, occupied_bins+1]
            dYv_i_short = dx*self.Yd[:, occupied_bins]; dYv_ip1_short = dx*self.Yd[:, occupied_bins+1]
            # If JAX is available, move big arrays to GPU
            Yv_i_short = device_put(Yv_i_short); Yv_ip1_short = device_put(Yv_ip1_short)
            dYv_i_short = device_put(dYv_i_short); dYv_ip1_short = device_put(dYv_ip1_short)

            #
            t15 = time.time()
            if verbose: print("Digitizing & reshaping took ",t15-t1," seconds.",flush=True)
            #
            # Precompute the v's
            vs_real, vs_imag = interp.get_vs(self.Nell-1, reshaped_phi_data, reshaped_inputs)

            #
            t2 = time.time()
            if verbose:
                print("Precomputing vs took ",t2-t1," seconds.",flush=True)
            #
            if jax_present:
                # Rearrange by m value of every a_lm index.
                # This is rather memory-inefficient, but it makes it very easy to
                # batch over with JAX's vmap. For lmax=500, each vs_* is O(1GB). For
                # lmax=1000, each vs_* is O(4GB). We might want to consider alternatives
                vs_real, vs_imag = [vs[m_ordering, :, :] for vs in [vs_real, vs_imag]]
                # Get a grid of all alm's by batching over (ell,m) -- best run on a GPU!
                get_all_alms_w_jax = vmap(jit(interp.get_alm_jax),in_axes=(0,0,0,0,0))
                # Notice we've put the Ylm and dYlm tables in device memory for a speed boost
                alm_grid_real = get_all_alms_w_jax(Yv_i_short, Yv_ip1_short,
                                                   dYv_i_short, dYv_ip1_short, vs_real)
                alm_grid_imag = get_all_alms_w_jax(Yv_i_short, Yv_ip1_short,
                                                   dYv_i_short, dYv_ip1_short, vs_imag)
                alm_grid = (np.array(alm_grid_real, dtype='complex128')
                            - 1j *np.array(alm_grid_imag, dtype='complex128'))
            else:
                # JIT compile the get_alm function
                get_alm_jitted = nb.jit(nopython=True)(interp.get_alm_np)
                #
                alm_grid = np.zeros(len(Yv_i_short[:,0]), dtype='complex128')
                vs_tot = vs_real - 1j * vs_imag
                #TODO: parallelize this
                for i, (Ylm_i, Ylm_ip1, dYlm_i, dYlm_ip1, m) in \
                  enumerate(zip(Yv_i_short, Yv_ip1_short, dYv_i_short, dYv_ip1_short,m_ordering)):
                    alm_grid[i] = get_alm_jitted(Ylm_i, Ylm_ip1, dYlm_i, dYlm_ip1, vs_tot, m)
            # For x<0, we need to multiply by (-1)^{ell-m}
            alm_grid_tot += par_fact * alm_grid
            t3 = time.time()
            if verbose:
                print("Computing alm's took ",t3-t2," seconds.",flush=True)
        return(alm_grid_tot/reg_factor)
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

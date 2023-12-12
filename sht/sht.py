#!/usr/bin/env python
#
# A Python class to handle "direct" harmonic transforms of
# point sets on the sphere.
# Uses Numba.
#
#
import numpy as np
import interp_funcs as interp
import utils
import time
from scipy.stats import mode

try:
    jax_present = True
    from   jax import vmap, jit, devices
    from jax.sharding import PositionalSharding
    from jax.experimental import mesh_utils
    from utils import move_to_device
    import jax.numpy as jnp
    from functools import partial
    from jax.lax import fori_loop
    # Choose the number of devices we'll be parallelizing across
    N_devices = len(devices())
except ImportError:
    jax_present = False
    move_to_device = lambda x, **kwargs: x  # Dummy definition for fallback
    print("JAX not found. Falling back to NumPy.")
    import numpy as jnp
    from numba import njit as jit
    N_devices = 1

@partial(jit, static_argnums=(0,1,2))
def compute_Plm_table(Nl,Nx,xmax):
    """Use recurrence relations to compute a table of Ylm[cos(theta),0]
    for ell>=0, m>=0, x>=0.  Can use symmetries to get m<0 and/or x<0,
    viz. (-1)^m for m<0 and (-1)^(ell-m) for x<0.
    :param  Nl: Number of ells (and hence m's) in the grid.
    :param  xx: Array of x points (non-negative and increasing).
    :return Y[ell,m,x=Cos[theta],0] without the sqrt{(2ell+1)/4pi}
    normalization (that is applied in __init__)
    """

    @partial(jit, static_argnums=(0, 2), donate_argnums=(1,))
    def get_mzeros(ell, Plm, indx, xx):
        i0, i1, i2 = indx(ell, 0), indx(ell - 1, 0), indx(ell - 2, 0)
        Plm = Plm.at[i0, :].set((2 * ell - 1) * xx * Plm[i1, :] - (ell - 1) * Plm[i2, :])
        return Plm.at[i0, :].divide(1.0 * ell)
    #
    @partial(jit, static_argnums=(0, 2), donate_argnums=(1,))
    def get_mhigh(m, Plm, indx, sx):
        i0, i1 = indx(m, m), indx(m - 1, m - 1)
        return Plm.at[i0, :].set(-jnp.sqrt(1.0 - 1. / (2 * m)) * sx * Plm[i1, :])
    #
    @partial(jit, static_argnums=(0, 2), donate_argnums=(1,))
    def get_misellm1(m, Plm, indx, xx):
        i0, i1 = indx(m, m), indx(m + 1, m)
        return Plm.at[i1, :].set(jnp.sqrt(2 * m + 1.) * xx * Plm[i0, :])
    #
    @partial(jit, static_argnums=(0, 3), donate_argnums=(2,))
    def ext_slow_recurrence(Nl, xx, Ylm, indx):
        body_fun = lambda m, Ylms: partial_fun_Ylm(m, Ylm, indx, xx, Nl)
        return fori_loop(0, Nl - 1, body_fun, Ylm)
    #
    @partial(jit, static_argnums=(0, 2, 4), donate_argnums=(1,))
    def partial_fun_Ylm(m, Ylm, indx, xx, Nl):
        body_fun = lambda ell, Ylms: full_fun_Ylm(ell, m, Ylms, indx, xx)
        return fori_loop(m + 2, Nl, body_fun, Ylm)
    #
    @partial(jit, static_argnums=(0, 1, 3), donate_argnums=(2,))
    def full_fun_Ylm(ell, m, Ylm, indx, xx):
        i0, i1, i2 = indx(ell, m), \
            indx(ell - 1, m), \
            indx(ell - 2, m)
        fact1, fact2 = jnp.sqrt((ell - m) * 1. / (ell + m)), \
            jnp.sqrt((ell - m - 1.) / (ell + m - 1.))
        Ylm = Ylm.at[i0, :].set((2 * ell - 1) * xx * Ylm[i1, :] - \
                                (ell + m - 1) * Ylm[i2, :] * fact2)
        return Ylm.at[i0, :].multiply(fact1 / (ell - m))
    #
    # This should match the convention used in the SHT class below.
    indx = lambda ell, m: (m * (2 * Nl - 1 - m)) // 2 + ell
    # Set up a regular grid of x values.
    xx = jnp.arange(Nx)/float(Nx-1) * xmax
    sx = jnp.sqrt(1-xx**2)
    Plm= jnp.zeros( ((Nl*(Nl+1))//2,Nx))
    # Distribute the grid across devices if possible
    Plm = move_to_device(Plm)
    #
    # First we do the m=0 case.
    Plm= Plm.at[indx(0,0),:].set(jnp.ones_like(xx))
    Plm = Plm.at[indx(1, 0), :].set(xx.copy())
    Plm = fori_loop(2, Nl, lambda ell, Plms: get_mzeros(ell, Plms, indx, xx), Plm)
    # Now we fill in m>0.
    # To keep the recurrences stable, we treat "high m" and "low m"
    # separately.  Start with the highest value of m allowed:
    Plm = fori_loop(1, Nl, lambda m, Plms: get_mhigh(m, Plms, indx, sx), Plm)
    # Now do m=ell-1
    Plm = fori_loop(1, Nl-1, lambda m, Plms: get_misellm1(m, Plms, indx, xx), Plm)
    # Finally fill in ell>m+1:
    Plm = ext_slow_recurrence(Nl,xx,Plm,indx)
    return(Plm)

@partial(jit, static_argnums=(0,1,2))
def compute_der_table(Nl,Nx,xmax,Yv):
    """Use recurrence relations to compute a table of derivatives of
    Ylm[cos(theta),0] for ell>=0, m>=0, x=>0.
    Assumes the Ylm table has already been built (passed as Yv).
    :param  Nl: Number of ells in the derivative grid.
    :param  xx: Values of cos(theta) at which to evaluate derivs.
    :param  Yv: Already computed Ylm values.
    :return Yd: The table of first derivatives.
    """

    @partial(jit, static_argnums=(0, 2), donate_argnums=(1,))
    def get_m0der(ell, Yd, indx, xx, Yv):
        i0, i1 = indx(ell, 0), indx(ell - 1, 0)
        return Yd.at[i0, :].set(ell / (1 - xx ** 2) * (Yv[i1, :] - xx * Yv[i0, :]))
    #
    @partial(jit, static_argnums=(0,4), donate_argnums=(3,))
    def ext_der_slow_recurrence(Nl, xx, Yv, Yd, indx):
        omx2 = 1.0 - xx ** 2
        body_fun = lambda ell, dYlm: partial_fun_dYlm(ell, Yv, dYlm, indx, xx, omx2)
        return fori_loop(0, Nl, body_fun, Yd)
    #
    @partial(jit, static_argnums=(0, 3), donate_argnums=(2,))
    def partial_fun_dYlm(ell, Yv, Yd, indx, xx, omx2):
        body_fun = lambda m, dYlm: full_fun_dYlm(ell, m, Yv, dYlm, indx, xx, omx2)
        return fori_loop(1, ell + 1, body_fun, Yd)
    #
    @partial(jit, static_argnums=(0, 1, 4), donate_argnums=(3,))
    def full_fun_dYlm(ell, m, Yv, Yd, indx, xx, omx2):
        i0, i1 = indx(ell, m), indx(ell - 1, m)
        fact = jnp.sqrt(1.0 * (ell - m) / (ell + m))
        Yd = Yd.at[i0, :].set((ell + m) * fact * Yv[i1, :] - ell * xx * Yv[i0, :])
        return Yd.at[i0, :].divide(omx2)
    #
    xx = jnp.arange(Nx) / float(Nx - 1) * xmax
    # This should match the convention used in the SHT class below.
    indx = lambda ell, m: (m * (2 * Nl - 1 - m)) // 2 + ell
    Yd = jnp.zeros( ((Nl*(Nl+1))//2,xx.size))
    # Distribute the grid across devices if possible
    Yd = move_to_device(Yd)
    Yd = Yd.at[indx(1,0),:].set(jnp.ones_like(xx))
    # Do the case m=0 separately.
    Yd = fori_loop(2, Nl, lambda ell, Yds: get_m0der(ell, Yds, indx, xx, Yv), Yd)
    # then build the m>0 tables.
    _  = ext_der_slow_recurrence( 1,xx,Yv,Yd,indx)
    Yd = ext_der_slow_recurrence(Nl,xx,Yv,Yd,indx)
    return(Yd)

def full_norm(ell, m, Y, indx, fact):
    ii = indx(ell, m)
    return Y.at[ii, :].multiply(fact)

def partial_norm_func(ell, Yv, indx):
    fact = jnp.sqrt((2 * ell + 1) / 4. / np.pi)
    body_fun_Y = lambda m, Ylms: full_norm(ell, m, Ylms, indx, fact)
    return fori_loop(0, ell + 1, body_fun_Y, Yv)



class DirectSHT:
    """Brute-force spherical harmonic transforms."""
    def __init__(self,Nell,Nx,xmax=0.875):
        """Initialize the class, build the interpolation tables.
        :param  Nell: Number of ells, and hence ms.
        :param  Nx:   Number of x grid points.
        :param xmax:  Maximum value of |cos(theta)| to compute.
        """
        self.Nell, self.Nx, self.xmax = Nell, Nx, xmax
        xx = jnp.arange(Nx)/float(Nx-1) * xmax
        print('here')
        Yv = compute_Plm_table(Nell,Nx,xmax)
        print('done with Yv')
        Yd = compute_der_table(Nell,Nx,xmax,Yv)
        print('done with Yd')
        # And finally put in the (2ell+1)/4pi normalization:
        body_fun = lambda ell, Ylm: partial_norm_func(ell, Ylm, self.indx)
        Yv, Yd = [fori_loop(0, Nell, body_fun, Y) for Y in [Yv, Yd]]
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

        # If working on GPU, move arrays to device.
        tm2 = time.time()
        if jax_present:
            # This is a hack to be able to pass the m value through vmap later on
            self.Yv = move_to_device(np.insert(self.Yv, 0, m_ordering, axis=1))
        else:
            self.Yv = move_to_device(self.Yv)
        self.Yd = move_to_device(self.Yd)
        tm1 = time.time()
        if verbose: print("Moving to GPU took ", tm1 - tm2, " seconds.", flush=True)

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
            bin_len = mode(spline_idx, keepdims=False).count
            # Find the data indices where transitions btw splines/bins happen
            transitions = utils.find_transitions(spline_idx)
            # Reshape the inputs into a 2D array for fast binning during
            # computation of the v's. Our binning scheme involves zero-padding bins with fewer
            # than bin_len points, but cos(0)=1!=0, so we need a mask to discard spurious zeros!
            mask = utils.reshape_phi_array(np.ones_like(phi_data_sorted), transitions, bin_num, bin_len)
            # Mask and put in GPU memory, distributing across devices if possible
            reshaped_phi_data = move_to_device(mask * utils.reshape_phi_array(phi_data_sorted,
                                                                          transitions, bin_num, bin_len))
            # Repeat the process for the other required inputs
            reshaped_inputs = utils.reshape_aux_array([w_i_sorted*input_ for input_ in
                                                      [(2*t+1)*(1-t)**2,t*(1-t)**2,t**2*(3-2*t)
                                                          ,t**2*(t-1)]], transitions, bin_num, bin_len)
            reshaped_inputs = move_to_device(mask * reshaped_inputs, axis=1)
            #
            t15 = time.time()
            if verbose: print("Digitizing & reshaping took ",t15-t1," seconds.",flush=True)
            #
            # Precompute the v's
            vs_real, vs_imag = interp.get_vs(self.Nell-1, reshaped_phi_data, reshaped_inputs)
            #
            t2 = time.time()
            if verbose: print("Precomputing vs took ",t2-t1," seconds.",flush=True)
            #
            if jax_present:
                # Remove zero-padding introduced when sharding to calculate v's
                vs_real, vs_imag = [vs[:, :, np.arange(bin_num, dtype=int)] for vs in [vs_real, vs_imag]]
                # Get a grid of all alm's by batching over (ell,m) -- best run on a GPU!
                get_all_alms_w_jax = vmap(jit(interp.get_alm_jax),in_axes=(0,0,0,0,None))
                # Note that we use a hack to pass the m value through vmap as the first element of every row of Yv
                # We also scale derivatives by dx
                alm_grid_real = get_all_alms_w_jax(self.Yv[:, np.insert(occupied_bins+1,0,0)],
                                                   self.Yv[:, np.insert(occupied_bins+2,0,0)],
                                                   dx*self.Yd[:, occupied_bins], dx*self.Yd[:, occupied_bins+1], vs_real)
                alm_grid_imag = get_all_alms_w_jax(self.Yv[:, np.insert(occupied_bins+1,0,0)],
                                                   self.Yv[:, np.insert(occupied_bins+2,0,0)],
                                                   dx*self.Yd[:, occupied_bins], dx*self.Yd[:, occupied_bins+1], vs_imag)
                # Combine real and imaginary parts of the alms
                alm_grid = (np.array(alm_grid_real, dtype='complex128')
                            - 1j *np.array(alm_grid_imag, dtype='complex128'))
                # If we introduced padding when sharding, remove it
                alm_grid = utils.unpad(alm_grid, len(ell_ordering))
            else:
                # JIT compile the get_alm function
                get_alm_jitted = jit(nopython=True)(interp.get_alm_np)
                #
                alm_grid = np.zeros(len(self.Yv[:, occupied_bins][:,0]), dtype='complex128')
                vs_tot = vs_real - 1j * vs_imag
                #TODO: parallelize this
                #Note that we scale derivatives by dx
                for i, (Ylm_i, Ylm_ip1, dYlm_i, dYlm_ip1, m) in \
                  enumerate(zip(self.Yv[:, occupied_bins], self.Yv[:, occupied_bins+1],
                                dx*self.Yd[:, occupied_bins], dx*self.Yd[:, occupied_bins+1],m_ordering)):
                    alm_grid[i] = get_alm_jitted(Ylm_i, Ylm_ip1, dYlm_i, dYlm_ip1, vs_tot, m)
            # For x<0, we need to multiply by (-1)^{ell-m}
            alm_grid_tot += par_fact * alm_grid
            t3 = time.time()
            if verbose:
                print("Computing alm's took ",t3-t2," seconds.",flush=True)
        # Undo the hack
        self.Yv = self.Yv[:, 1:]
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
        #

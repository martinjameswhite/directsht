#!/usr/bin/env python
#
# A Python class to handle "direct" harmonic transforms of
# point sets on the sphere.
# Uses Numba.
#
#
import numpy as np
import time

try:
    jax_present = True
    from jax import vmap, jit, devices
    import jax.numpy as jnp
    import legendre_jax as legendre
    import interp_funcs_jax as interp
    import utils_jax as utils
    from utils_jax import move_to_device
except ImportError:
    jax_present = False
    move_to_device = lambda x, **kwargs: x  # Dummy definition for fallback
    print("JAX not found. Falling back to NumPy.")
    from numba import njit as jit
    import legendre_py as legendre
    import interp_funcs_py as interp
    import utils_py as utils


class DirectSHT:
    """Brute-force spherical harmonic transforms."""
    def __init__(self, Nell, Nx, xmax=0.875, dflt_type='float64', null_unphysical=True):
        """Initialize the class, build the interpolation tables.
        :param  Nell: int. Number of ells, and hence ms.
        :param  Nx:   int. Number of x grid points.
        :param xmax:  float. Maximum value of |cos(theta)| to compute.
        :param dflt_type: str. Default dtype to use for all arrays. Defaults to 'float64'.
            Double precision is strongly recommended.
        :param null_unphysical: bool. Only has an effect if jax_present=True. If True,
            set all Ylm's with ell<m to zero. Otherwise, these entries will return junk
            when queried (the normal algorithm does not care about this, but setting
            null_unphysical=False is marginally faster).
        """
        self.Nell, self.Nx, self.xmax = Nell, Nx, xmax
        # Set up a regular grid of x values.
        xx = jnp.arange(Nx, dtype=dflt_type) if jax_present else np.arange(Nx, dtype=dflt_type)
        xx *= xmax / float(Nx - 1)
        # Compute Plm and its derivate w.r.t. x
        Yv = legendre.compute_Plm_table(Nell, xx)
        Yd = legendre.compute_der_table(Nell, xx, Yv)
        # Multiply by the (2ell+1)/4pi normalization factor relating Plm to Ylm
        Yv, Yd = [legendre.norm_ext(Y, self.Nell) for Y in [Yv, Yd]]
        # Null unphysical entries (id needed)
        Yv, Yd = legendre.null_unphys(Yv, Yd) if null_unphysical else (Yv, Yd)
        self.x, self.Yv, self.Yd = xx, Yv, Yd
        #
    def indx(self,ell,m):
        """
        The index of a given (ell,m) in the alm array produced when calling
        and instance of the DirectSHT class. Matches the Healpix convention.
        :param  ell: ell value to return.
        :param  m:   m value to return.
        :return ii:  Index value in the output alm array
        """
        ii= (m*(2*self.Nell-1-m))//2 + ell
        return(ii)
        #
    def get_Ylm(self, ell, m):
        '''
        Return the spherical harmonics Ylm(x_i, 0) for a given ell and m,
        where x_i = cos\theta_i are the sampled grid points.
        :param ell: int.
        :param m:   int.
        :return:   1D numpy array of length Nx.
        '''
        if jax_present:
            return(self.Yv[m, ell-m, :])
        else:
            return(self.Yv[self.indx(ell,m), :])

    def get_dYlm(self, ell, m):
        '''
        Return the derivate w.r.t. x of Ylm(x_i, 0) for a given ell and m,
        where x_i = cos\theta_i are the sampled grid points.
        :param ell: int.
        :param m:   int.
        :return:   1D numpy array of length Nx.
        '''
        if jax_present:
            return(self.Yd[m, ell-m, :])
        else:
            return(self.Yd[self.indx(ell,m), :])

    def alm2cl(self,alm):
        """
        Returns the pseudo-spectrum given alms.
        :param alm: Values of the SHT coefficients,
                    assumed compatible with this class instance.
        :return cl: 1D numpy array containing 1/(2l+1) Sum_m |alm|^2.
        """
        cl = np.zeros(self.Nell)
        for ell in range(self.Nell):
            cl[ell] = alm[self.indx(ell,0)].real**2
            for m in range(1,ell+1):
                ii       = self.indx(ell,m)
                cl[ell] += 2*(alm[ii].real**2+alm[ii].imag**2)
            cl[ell] /= 2*ell + 1.
        return(cl)
        #

    def __call__(self, theta, phi, wt, reg_factor=1., verbose=True):
        """
        Returns alm for a collection of real-valued points at (theta,phi),
        in radians, with weights wt.
        :param theta: 1D numpy array of theta values for each point.
            Must be between [0,pi], and also satisfy [ACos[xmax],ACos[-xmax]
        :param phi: 1D numpy array of phi values for each point.
            Must be between [0,2pi].
        :param wt: 1D numpy array of weights for each point.
        :param reg_factor: Scaling to apply to weights to avoid numerical
            over/underflow. It gets removed at the end. Typically not used.
        :param verbose: if True, print out timing information.
        :return: alm in the Healpix indexing convention,
                 i.e., complex coefficients alm[m*(2*lmax+1-m)/2+l]
                 with l in [0, lmax] and m in [0, l]
        """
        assert len(theta) == len(phi) and \
               len(phi) == len(wt), "theta,phi,wt must be the same length."
        assert np.all((theta >= 0) & (theta > np.arccos(self.xmax)) & (theta < np.arccos(-self.xmax))), \
            "theta must be in [ACos[xmax],ACos[-xmax])."
        #
        x_samples = self.x
        # TODO: this way of calculating dx assumes the points are evenly spaced in x
        dx = x_samples[1] - x_samples[0]
        # Multiply the weights by a regularization factor to avoid numerical
        # under/overflow
        wt *= reg_factor
        # Get the indexing of ell and m in the Healpix convention for
        # later use
        ell_ordering, m_ordering = utils.getlm(self.Nell-1, (self.Nell*(self.Nell+1))//2)
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
        pos_idx, neg_idx =  np.where(x_full >= 0)[0], np.where(x_full < 0)[0]
        if (len(pos_idx)>0) and (len(neg_idx)>0):
            # The case where there's both +ve and -ve values of x
            which_case = zip([x_full[pos_idx], np.abs(x_full[neg_idx])],
                             [1., parity_factor], [pos_idx, neg_idx])
        elif (len(pos_idx)>0):
            # The case where there's only +ve values of x
            which_case = zip([x_full[pos_idx]], [1.], [pos_idx])
        elif (len(neg_idx)>0):
            # The case where there's only -ve values of x
            which_case = zip([np.abs(x_full[neg_idx])], [parity_factor],
                             [neg_idx])
        else:
            raise ValueError("The theta array seems to be empty!")
        #
        # Treat +ve and -ve x separately
        for x, par_fact, idx in which_case:
            t0 = time.time()
            # Find which spline interval each point falls into
            spline_idx = np.digitize(x, x_samples) - 1
            # Reorder the data in bins. We don't care about specific order inside bin, so
            # this sorting is faster than if we had to actually sort all the data
            sorted_idx = np.argsort(spline_idx)
            spline_idx = spline_idx[sorted_idx]
            x_data_sorted = x[sorted_idx]
            w_i_sorted = wt[idx][sorted_idx]
            phi_data_sorted = phi[idx][sorted_idx]
            # Calculate the t's. We'll need when interpolating
            t = (x_data_sorted - x_samples[spline_idx]) / dx
            #
            t1 = time.time()
            if verbose: print("Sorting & digitizing took ", t1 - t0, " seconds.", flush=True)
            #
            # Find which bins (bounded by the elements of x_samples) are populated.
            # The point is to avoid unnecessary storage and computation
            occupied_bins = np.unique(spline_idx)
            # Find the data indices where transitions btw splines/bins happen
            transitions = utils.find_transitions(spline_idx)
            # Reshape the inputs into a 2D array for fast accumulation within bins
            # during computation of the v's. Our binning scheme involves zero-padding
            # bins with fewer than the maximum number of points in a bin, but
            # cos(0)=1!=0, so we need a mask to discard spurious zeros!
            mask = utils.reshape_phi(np.ones_like(phi_data_sorted), transitions)
            # Mask and put in GPU memory distributing across devices (if possible)
            reshaped_phi_data = move_to_device(mask * utils.reshape_phi(phi_data_sorted, transitions))
            # Repeat the process for the other required inputs
            reshaped_inputs = utils.reshape_aux([w_i_sorted * input_ for input_ in
                                                 [(2*t+1)*(1-t)**2, t*(1-t)**2, t**2*(3-2*t), t**2*(t-1)]],
                                                transitions)
            reshaped_inputs = move_to_device(mask * reshaped_inputs, axis=1)
            #
            t15 = time.time()
            if verbose: print("Reshaping took ", t15 - t1, " seconds.", flush=True)
            #
            # Precompute the v's
            vs_real, vs_imag = interp.get_vs(self.Nell-1, reshaped_phi_data, reshaped_inputs)
            #
            t2 = time.time()
            if verbose: print("Precomputing vs took ", t2 - t1, " seconds.", flush=True)
            #
            if jax_present:
                # Remove zero-padding introduced when sharding to calculate v's
                vs_real, vs_imag = [vs[:,:,np.arange(len(occupied_bins), dtype=int)] for vs in [vs_real, vs_imag]]
                # Get a grid of all alm's by batching over (ell,m) -- best run on a GPU!
                get_all_alms = vmap(vmap(interp.get_alm_jax, (0,0,0,0,None)), (0,0,0,0,0))
                # Note that we scale the derivatives by dx, as required by Hermite interpolation
                alm_real, alm_imag = [get_all_alms(self.Yv[:,:,occupied_bins], self.Yv[:,:,occupied_bins+1],
                                                   dx*self.Yd[:,:,occupied_bins], dx*self.Yd[:,:,occupied_bins+1], vs)
                                      for vs in [vs_real, vs_imag]]
                # Combine real and imaginary parts of the alms, and adapt to healpy convention
                alm_grid = utils.to_hp_convention(alm_real, alm_imag)
                # If we introduced padding when sharding, remove it
                alm_grid = utils.unpad(alm_grid, len(ell_ordering))
            else:
                # JIT compile the get_alm function
                get_alm_jitted = jit(nopython=True)(interp.get_alm_np)
                # Temporary storage for alms
                alm_grid = np.zeros((self.Nell*(self.Nell+1))//2, dtype='complex128')
                vs_tot = vs_real - 1j * vs_imag
                # TODO: parallelize this
                # Note that we scale the derivatives by dx, as required by Hermite interpolation
                for i, (Ylm_i, Ylm_ip1, dYlm_i, dYlm_ip1, m) in \
                        enumerate(zip(self.Yv[:, occupied_bins], self.Yv[:, occupied_bins+1],
                                      dx*self.Yd[:, occupied_bins], dx*self.Yd[:, occupied_bins+1], m_ordering)):
                    alm_grid[i] = get_alm_jitted(Ylm_i, Ylm_ip1, dYlm_i, dYlm_ip1, vs_tot, m)
            # For x<0, we need to multiply by (-1)^{ell-m}
            alm_grid_tot += par_fact * alm_grid
            t3 = time.time()
            if verbose:
                print("Computing alm's took ", t3 - t2, " seconds.", flush=True)
        return (alm_grid_tot / reg_factor)
        #

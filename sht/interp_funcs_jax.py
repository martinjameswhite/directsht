from jax import jit
import jax.numpy as jnp
from functools import partial
import shared_interp_funcs
jax_present = True


default_dtype = None # Replace if you want to use a different dtype from the env default

#@partial(jit, donate_argnums=(1,2))
def get_vs(mmax, phi_data_reshaped, reshaped_inputs, loop_in_JAX=True, N_chunks=None,
           pad=False, verbose=False):
    """
    Wrapper function to call get_vs from shared_interp_funcs
    """
    return shared_interp_funcs.get_vs(mmax, phi_data_reshaped, reshaped_inputs,
                                      loop_in_JAX=loop_in_JAX, N_chunks=N_chunks,
                                      pad=pad, verbose=verbose)

@partial(jit, donate_argnums=(0,))
def accumulate(arr):
    '''
    Sum over the second axis of a 2D array -- i.e., the key binning operation!
    :param arr: 2D numpy array, where the axis 0 contains the different bins
                and the axis 1 (last axis) contains the data in each bin
    :return: 1D numpy array of the sums of the data in each bin
    '''
    return jnp.sum(arr, axis=-1)

@partial(jit, donate_argnums=(4,))
def get_alm_jax(Ylm_i, Ylm_ip1, dYlm_i, dYlm_ip1, vs):
    """
    The key function: get alm by summing over all interpolated, weighted
    Y_lm's using JAX. Interpolation uses cubic Hermite splines
    :param Ylm_i: 1d numpy array of Ylm at sample indices i
            IMPORTANT NOTE: the zeroth element of this array is the value of m
            this is a hack to be able to pass the value of m through the vmap
    :param dYlm_i: 1d numpy array of first derivatives of Ylm at sample indices i
    :param Ylm_ip1: 1d numpy array of Ylm at sample indicesi+1
            IMPORTANT NOTE: the zeroth element of this array is the value of m
            this is a hack to be able to pass the value of m through the vmap
    :param dYlm_ip1: 1d numpy array of first derivatives of Ylm at sample indices i+1
    :param vs: jnp array of size (mmax+1,4,Nx) with the v_{i,j}(m)
    :return: a 1D numpy array with the alm value
    """

    return(jnp.sum(Ylm_i*vs[0] + dYlm_i*vs[1] + Ylm_ip1*vs[2] + dYlm_ip1*vs[3]))
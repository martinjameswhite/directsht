import numpy as np
from jax import vmap, jit, devices
from utils_jax import move_to_device
import utils_jax as utils
import jax.numpy as jnp
from functools import partial
from jax.lax import fori_loop

# Choose the number of devices we'll be parallelizing across
N_devices = len(devices())

def null_unphys(Yv, Yd):
    '''
    Zero-out spurious entries (artefacts of our implementation)
    :param Yv: jnp array of size (Nell, Nell) with the Ylms
    :param Yd: jnp array of size (Nell, Nell) with the derivatives of Ylms
    :return: tuple (Yv, Yd) where we've zeroed out the unphysical entries
    '''
    # TODO: Make this act in place by donating buffers
    mask = jnp.triu(jnp.ones((Yv.shape[0], Yv.shape[1])))
    mask = jnp.array([jnp.roll(mask[i, :], -i) for i in range(len(mask))])
    Yv, Yd = [jnp.nan_to_num(Y) * mask[:, :, None] for Y in [Yv, Yd]]
    return (Yv, Yd)

def compute_Plm_table(Nl, xx):
    """Use recurrence relations to compute a table of Ylm[cos(theta),0]
    for ell>=0, m>=0, x>=0.  Can use symmetries to get m<0 and/or x<0,
    viz. (-1)^m for m<0 and (-1)^(ell-m) for x<0.
    :param  Nl: Number of ells (and hence m's) in the grid.
    :param  xx: Array of x points (non-negative and increasing).
    :return Y[ell,m,x=Cos[theta],0] without the sqrt{(2ell+1)/4pi}
    normalization (that is applied in __init__)
    """

    # We donate argnums to enforce in-place array updates
    @partial(jit, donate_argnums=(1,))
    def get_mhigh(m, Plm, sx):
        indx = lambda ell, m: (m, ell - m)
        i0, i1 = indx(m, m), indx(m - 1, m - 1)
        return Plm.at[i0[0], i0[1], :].set(-jnp.sqrt(1.0 - 1. / (2 * m)) * sx * Plm[i1[0], i1[1], :])

    #
    @partial(jit, donate_argnums=(1,))
    def get_misellm1(m, Plm, xx):
        indx = lambda ell, m: (m, ell - m)
        i0, i1 = indx(m, m), indx(m + 1, m)
        return Plm.at[i1[0], i1[1], :].set(jnp.sqrt(2 * m + 1.) * xx * Plm[i0[0], i0[1], :])

    #
    @jit
    def ext_slow_recurrence(xx, Plm):
        # Note we use Plm.shape[0] instead of Nl to allow padding
        return vmap(partial_fun_Ylm, (0, 0, None))(jnp.arange(0, Plm.shape[0], dtype='int32'), Plm, xx)

    #
    @partial(jit, donate_argnums=(1,))
    def partial_fun_Ylm(m, Ylm_row, xx):
        body_fun = lambda ell, Ylm_at_m: full_fun_Ylm(ell, m, Ylm_at_m, xx)
        return fori_loop(0, len(Ylm_row) - 2, body_fun, Ylm_row)

    #
    @partial(jit, donate_argnums=(2,))
    def full_fun_Ylm(i, m, Ylm_at_m, xx):
        # Our indexing scheme is (m, ell-m), so since the loops start at the third column
        # (i.e. ell=m+2) we can get ell from the loop index as
        ell = m + i + 2
        # The recursion relies on the previous two elements on this row
        i0, i1, i2 = i + 2, i + 1, i
        fact1, fact2 = jnp.sqrt((ell - m) * 1. / (ell + m)), \
            jnp.sqrt((ell - m - 1.) / (ell + m - 1.))
        Ylm_at_m = Ylm_at_m.at[i0, :].set(((2 * ell - 1) * xx * Ylm_at_m[i1, :]
                                           - (ell + m - 1) * Ylm_at_m[i2, :] * fact2)
                                          * fact1 / (ell - m))
        return Ylm_at_m

    #
    # This should match the convention used in the SHT class below.
    # We shift all the entries so that rows start at ell=m. This helps recursion.
    indx = lambda ell, m: (m, ell - m)
    Nx = len(xx)
    sx = jnp.sqrt(1 - xx ** 2)
    # Distribute the grid across devices if possible
    Plm = jnp.zeros((Nl, Nl, Nx))
    # First we do the l=m=0 and l=1, m=0 cases
    Plm = Plm.at[indx(0, 0)[0], indx(0, 0)[1], :].set(jnp.ones_like(xx))
    Plm = Plm.at[indx(1, 0)[0], indx(1, 0)[1], :].set(xx.copy())
    # Now we fill in m>0.
    # To keep the recurrences stable, we treat "high m" and "low m"
    # separately.  Start with the highest value of m allowed:
    Plm = fori_loop(1, Nl, lambda m, Plms: get_mhigh(m, Plms, sx), Plm)
    # Now do m=ell-1
    Plm = fori_loop(1, Nl - 1, lambda m, Plms: get_misellm1(m, Plms, xx), Plm)
    # Now we distribute/shard it across GPUS. Note that we should only do this
    # once we've computed the diagonals, which happens in a direction orthogonal
    # to our row-based sharding!
    Plm = move_to_device(Plm, pad_axes=[0, 1])
    # Finally fill in ell>m+1:
    Plm = ext_slow_recurrence(xx, Plm)
    return (Plm)


def compute_der_table(Nl, xx, Yv):
    """Use recurrence relations to compute a table of derivatives of
    Ylm[cos(theta),0] for ell>=0, m>=0, x=>0.
    Assumes the Ylm table has already been built (passed as Yv).
    :param  Nl: Number of ells in the derivative grid.
    :param  xx: Values of cos(theta) at which to evaluate derivs.
    :param  Yv: Already computed Ylm values.
    :return Yd: The table of first derivatives.
    """

    #
    def ext_der_slow_recurrence(xx, Yv, Yd):
        omx2 = 1.0 - xx ** 2
        # Note we use Yv.shape[0] instead of Nl to allow padding
        rows = jnp.arange(0, Yv.shape[0], dtype='int32')
        cols = jnp.arange(0, Yv.shape[1], dtype='int32')
        return vmap(vmap(full_fun_dYlm, (0, None, 0, 0, None, None)),
                    (None, 0, None, 0, None, None), 1)(rows, cols, Yv, Yd, xx, omx2)

    @partial(jit, donate_argnums=(3,))
    def full_fun_dYlm(m, i, Yv_at_m, Yd_at_ell_m, xx, omx2):
        # Our indexing scheme is (m, ell-m), so we can get ell from the loop index as
        ell = m + i
        i0, i1 = i, i - 1
        # indx = lambda ell, m: (m, ell - m)
        # i0, i1 = indx(ell, m), indx(ell - 1, m)
        fact = jnp.sqrt(1.0 * (ell - m) / (ell + m))
        Yd_at_ell_m = Yd_at_ell_m.at[:].set(((ell + m) * fact * Yv_at_m[i1, :] - ell * xx * Yv_at_m[i0, :]) / omx2)
        return Yd_at_ell_m

    @partial(jit, donate_argnums=(3,))
    def fill_dYmm(m, i, Yv_at_m, Yd_at_m, xx, omx2):
        # Our indexing scheme is (m, ell-m), so we can get ell from the loop index as
        ell = m + i
        return Yd_at_m.at[i, :].set((- ell * xx * Yv_at_m[i, :]) / omx2)

    def fill_dYmm_ext(Yv, Yd, xx):
        omx2 = 1.0 - xx ** 2
        # Note we use Yv.shape[0] instead of Nl to allow padding
        rows = jnp.arange(0, Yv.shape[0], dtype='int32')
        return vmap(fill_dYmm, (0, None, 0, 0, None, None))(rows, 0, Yv, Yd, xx, omx2)

    #
    Nx = len(xx)
    # This should match the convention used in the SHT class below.
    indx = lambda ell, m: (m, ell - m)
    # Distribute the grid across devices if possible
    Yd = utils.init_array(Nl, Nx, N_devices)
    # then build the m>0 tables.
    Yd = ext_der_slow_recurrence(xx, Yv, Yd)
    # Do ell=1, m=0 and ell=0, m=0, which the recursion can't give us.
    Yd = Yd.at[indx(1, 0)[0], indx(1, 0)[1], :].set(jnp.ones_like(xx))
    Yd = Yd.at[indx(0, 0)[0], indx(0, 0)[1], :].set(jnp.zeros_like(xx))
    # The zeroth column (i.e. ell=m) is pathological in our implementation
    # so do it again
    Yd = fill_dYmm_ext(Yv, Yd, xx)
    return (Yd)


@partial(jit, donate_argnums=(2,))
def norm(m, i, Ylm_at_m_ell):
    # Get ell in our indexing scheme where indx = lambda ell, m: (m, ell - m)
    ell = m + i
    return Ylm_at_m_ell.at[:].multiply(jnp.sqrt((2 * ell + 1) / 4. / np.pi))


def norm_ext(Y, Nl):
    '''
    Normalize the Plm's by the (2ell+1)/4pi factor relating Plm to Ylm
    :param Yv: jnp.ndarray of shape (Nl, Nl, Nx) containing the Plm's
    :return: jnp.ndarray of shape (Nl, Nl, Nx) containing the (2ell+1)/4pi Ylm
    '''
    # Note we use Yv.shape[0] instead of Nl to allow padding
    rows = jnp.arange(0, Y.shape[0], dtype='int32')
    cols = jnp.arange(0, Y.shape[1], dtype='int32')
    # This is effectively a loop over ms and ells, multiplying each entry by the norm
    return vmap(vmap(norm, (0, None, 0)),
                (None, 0, 0))(rows, cols, Y)
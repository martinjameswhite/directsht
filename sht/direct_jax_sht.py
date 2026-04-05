#
# Code to compute the "direct" SHT transform of a weighted
# set of points on the sphere.
# This was written by Google Gemini, and the prompt can
# be found at the end of the code.
#
#

import jax
import jax.numpy as jnp
from functools import partial

# Enforce double precision globally for XLA/JAX
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnums=(3,))
def compute_sht(x, phi, wt, Nell):
    """
    Computes the Spherical Harmonic Transform (SHT) using JAX.
    
    Args:
        x: Array of shape (Npnt,) containing cos(theta) in [-1, 1].
        phi: Array of shape (Npnt,) containing azimuthal angles in [0, 2pi).
        wt: Array of shape (Npnt,) containing integration weights.
        Nell: Integer specifying the maximum degree/size of SHT.
        
    Returns:
        out: 1D complex128 array of SHT coefficients.
    """
    # Ensure inputs are float64 to maintain double precision constraints
    x = jnp.asarray(x, dtype=jnp.float64)
    phi = jnp.asarray(phi, dtype=jnp.float64)
    wt = jnp.asarray(wt, dtype=jnp.float64)

    abs_x = jnp.abs(x)
    
    # Calculate the total size required for the 1D output array
    out_size = (Nell * (Nell + 1)) // 2
    out = jnp.zeros(out_size, dtype=jnp.complex128)

    # Outer loop condition (m < Nell)
    def m_loop_cond(state_m):
        m, P_mm, out = state_m
        return m < Nell

    # Outer loop body
    def m_loop_body(state_m):
        m, P_mm, out = state_m
        m_float = m.astype(jnp.float64)

        # Precompute the exponential (taking complex conjugate equivalent)
        exp_neg_imphi = jnp.exp(-1j * m_float * phi)

        # Inner loop condition (ell < Nell)
        def ell_loop_cond(state_ell):
            ell, P_prev2, P_prev1, out = state_ell
            return ell < Nell

        # Inner loop body
        def ell_loop_body(state_ell):
            ell, P_prev2, P_prev1, out = state_ell
            ell_float = ell.astype(jnp.float64)

            # --- Recurrence Evaluation (Stable & NaN-Proof) ---
            # Using jnp.maximum limits to safely compute terms even when
            # they are ignored by the branchless selection later (avoiding division by zero).
            divisor1 = jnp.maximum(ell_float - m_float, 1.0)
            term1 = (2.0 * ell_float - 1.0) / divisor1 * abs_x * P_prev1

            safe_epm_1 = jnp.maximum(ell_float + m_float - 1.0, 1.0)
            safe_em_1 = jnp.maximum(ell_float - m_float - 1.0, 0.0)
            term2_sqrt = jnp.sqrt(safe_em_1 / safe_epm_1)
            term2 = term2_sqrt * (safe_epm_1 / divisor1) * P_prev2

            P_rec = jnp.sqrt((ell_float - m_float) / jnp.maximum(ell_float + m_float, 1.0)) * (term1 - term2)

            # Branchless selection based on the specific degree sequence state
            P_curr = jnp.where(
                ell == m, P_mm,
                jnp.where(
                    ell == m + 1, jnp.sqrt(2.0 * m_float + 1.0) * abs_x * P_mm,
                    P_rec
                )
            )

            # Flip the sign for negative x when (ell - m) is odd
            sign_flip = jnp.where((x < 0.0) & ((ell - m) % 2 == 1), -1.0, 1.0)

            # Compute conjugate of the Spherical Harmonic (Y_lm*)
            Y_lm_star = jnp.sqrt((2.0 * ell_float + 1.0) / (4.0 * jnp.pi)) * P_curr * sign_flip * exp_neg_imphi

            # Compute inner product over points
            sum_val = jnp.sum(wt * Y_lm_star)

            # Calculate proper storage index and assign value
            ii = (m * (2 * Nell - 1 - m)) // 2 + ell
            out = out.at[ii].set(sum_val)

            # Advance state: P_prev2 <- P_prev1, P_prev1 <- P_curr
            return (ell + 1, P_prev1, P_curr, out)

        # Initialize and execute the inner `ell` loop
        # Note: P_prev2 and P_prev1 initialize as zeros; they are ignored for ell == m and ell == m+1
        init_state_ell = (m, jnp.zeros_like(x), jnp.zeros_like(x), out)
        _, _, _, out = jax.lax.while_loop(ell_loop_cond, ell_loop_body, init_state_ell)

        # Advance state to prepare P_mm for the next `m` iteration
        next_m = m + 1
        next_m_float = next_m.astype(jnp.float64)
        
        # P_mm_next relation using maximum boundaries to prevent NaNs/Complex casts in real bounds
        P_mm_next = -jnp.sqrt(jnp.maximum(1.0 - x**2, 0.0)) * jnp.sqrt(1.0 - 1.0 / jnp.maximum(2.0 * next_m_float, 1.0)) * P_mm

        return (next_m, P_mm_next, out)


    # Initialize and execute the outer `m` loop
    # m starts at 0, P_0^0 is initialized to an array of ones
    init_state_m = (jnp.int32(0), jnp.ones_like(x), out)
    _, _, final_out = jax.lax.while_loop(m_loop_cond, m_loop_body, init_state_m)

    return final_out


# I would like you to write a Python code to perform a spherical-harmonic
# transform (SHT) on a set of input points.  The code should use JIT
# compilation and JAX whenever possible to speed up the computation and
# enable it to run on a GPU.  Double precision (x64) should be enforced throughout the calculation steps.
#
# The code will take as input three arrays, containing x=cos(theta), phi and a floating point weight for Npnt points, and an integer (Nell) specifying the size of the SHT.  The values of x will be in the range [-1,1] and those of phi in the range [0,2 pi).  The calling convention should look like
# compute_sht(x, phi, wt, Nell)
#
# You will need to use Python's built-in functools.partial to "bind" the static_argnums keyword argument to jax.jit before it wraps your function.
#
# The spherical harmonic transform involves computing the values of several spherical harmonics at each of the input points, taking their complex conjugate and summing over the points each weighted by the value of "wt".  The spherical harmonics are indexed by two integers (ell,m) which satisfy 0<=ell<Nell and 0<=m<=ell.  The spherical harmonic values should be stored in a 1D array where the mapping between ell and m and the array index is
#
# ii = (m*(2*Nell-1-m))//2 + ell
#
# The maximum value of the transform, Nell, is known in advance and fixed and will be passed to the function.  However Nell can be quite large (over 1000) so you should write the inner blocks using jax.lax.while_loop.
#
# There may be a large number of points, so we should be careful about
# allocating very large arrays.  Summing the points inside the loop will
# help keep the memory footprint small.
# You will need to treat x>=0 and x<0 separately.  For x>0 you can use the method below.  For x<0 you need to use the absolute value of x but flip the sign of the spherical harmonic if ell-m is odd.  Because of this, you only need to explicitly compute the spherical harmonics for the absolute value of x (multiplying the result by -1 if ell-m is odd and x<0).
#
# The spherical harmonic for a given x and phi can be written
# \begin{equation}
#     Y_{\ell m}(\theta,\phi) = \sqrt{\frac{(2\ell+1)}{4\pi}}\ \bar{P}_\ell^m(x=\cos\theta)\, e^{im\phi}
# \end{equation}
# where
# \begin{equation}
#     \bar{P}_\ell^m(x) \equiv \sqrt{\frac{(\ell-m)!}{(\ell+m)!}}\ P_\ell^m(x)
#     \quad , \quad m\ge 0
# \end{equation}
# The $\bar{P}$ can be computed by recurrence starting from
# \begin{equation}
#     \bar{P}_m^m = -\sqrt{1-x^2} \sqrt{ 1-(2m)^{-1} } \ \bar{P}_{m-1}^{m-1}
#     \quad \text{and} \quad
#     \bar{P}_{m+1}^m = \sqrt{2m+1}\, x\ \bar{P}_m^m\,,
# \end{equation}
# with the special cases $\bar{P}_0^0(x)=P_0(x)=1$ and
# $\bar{P}_1^0(x)=P_1(x)=x$.  Now, $\bar{P}$ is dominant on the degree
# ($\ell$) and minimal on the order ($m$), so for all other $\ell$ and $m$
# \begin{equation}
#     \bar{P}_\ell^m
#     = \sqrt{\frac{(\ell-m)}{(\ell+m)}}\, \left[ \frac{2\ell-1}{\ell-m}\ x\,  \bar{P}_{\ell-1}^m(x) -  \sqrt{\frac{(\ell-m-1)}{(\ell+m-1)}} \, \frac{\ell+m-1}{\ell-m}\  \bar{P}_{\ell-2}^m(x) \right]
# \end{equation}
# is stable with $m$ increasing from $0$ and $\ell$ increasing from $m+2$ for each $m$.
#
# You should be sure to use safe quantities for the recurrence to avoid division by zero or sqrt of negative numbers.
#
# Please include this prompt at the end of the code, as a comment using the # symbol at the beginning of each line to indicate to Python that the line is a comment (do not use a triple quoted string).

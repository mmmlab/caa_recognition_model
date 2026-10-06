# configure for compatibility with Python 3
from __future__ import absolute_import, division, print_function

# standard library imports
from typing import NamedTuple

# scientific library imports
import pylab as pl
from scipy import stats

# local imports
from .base_model import BaseDDM
from ..config import EPS, INF_PROXY
from ..utils.multinomial_funcs import get_rect_prob


class SISPVarCBoundDDM(BaseDDM):
    """
    SISP model with a variable confidence boundary (C_High) across trials.

    Instead of varying the remember/know boundary (as in SISPVarRBoundDDM),
    this model varies the confidence boundary C_High ~ N(c_mean, sigma_c).
    The remember/know boundary remains fixed.

    Motivation: tests whether trial-to-trial variability lives in the
    confidence judgment (how much evidence is needed for high confidence)
    rather than the recollection criterion (what counts as remembering).
    Comparing fit against SISPVarRBoundDDM tells you which source of
    variability is more cognitively meaningful.

    Parameters
    ----------
    c              : mean of the C_High confidence boundary distribution
    mu_t           : target drift rate
    mu_l           : lure drift rate
    d              : diffusion constant
    tc_bound       : boundary collapse rate
    r_bound_offset : fixed distance from z0 to remember criterion
    sigma_c        : trial-to-trial variability of the confidence boundary
    z0             : shared starting point (targets and lures)
    t_post         : post-decision accumulation interval
    t0             : non-decision time
    """

    class Params(NamedTuple):
        c: float              # Mean of the high confidence boundary
        mu_t: float           # Target drift
        mu_l: float           # Lure drift
        d: float              # Diffusion constant
        tc_bound: float       # Boundary collapse rate
        r_bound_offset: float # Remember boundary offset (fixed)
        sigma_c: float        # Variability of the confidence boundary across trials
        z0: float             # Shared starting point
        t_post: float         # Post-decision accumulation time
        t0: float             # Non-decision time

    @staticmethod
    def split_params(model_params):
        c_val, mu_t, mu_l, d, tc, r_off, s_c, z0, dT, t0 = model_params
        c_list = [c_val, 0]

        target_params = (c_list, mu_t, d, tc, r_off, s_c, z0, dT, t0)
        lure_params   = (c_list, mu_l, d, tc, r_off, s_c, z0, dT, t0)

        return target_params, lure_params

    def predicted_proportions(self, params):
        c, mu_r, d, tc_bound, r_bound_offset, sigma_c, z0, t_post, t0 = params
        delta_t   = self.config.delta_t
        max_t     = self.config.max_t
        nr_tsteps = self.config.nr_tsteps
        nr_ssteps = self.config.nr_ssteps

        # Create confidence bins
        c = pl.array(c, ndmin=1)
        n = len(c)
        clims = pl.hstack(([INF_PROXY], c, [-INF_PROXY]))

        # Fixed remember criterion
        r_bound = z0 + r_bound_offset

        # Standard deviation and mean drift of the accumulator per-step
        sigma = pl.sqrt(2 * d * delta_t)
        mu    = mu_r * delta_t

        # Time axis
        t      = pl.linspace(delta_t, max_t, nr_tsteps)
        to_idx = pl.argmin((t - t0) ** 2)

        # Collapsing boundary
        bound = pl.exp(-tc_bound * pl.clip(t - t0, 0, None))

        # Spatial grid
        space_lim = max(bound) + 3 * sigma
        delta_s   = 2 * space_lim / nr_ssteps
        x         = pl.linspace(-space_lim, space_lim, nr_ssteps)

        # Diffusion kernel and FFT
        kernel    = stats.norm.pdf(x, mu, sigma) * delta_s
        ft_kernel = self._fft(kernel)

        # Output arrays
        tx          = pl.zeros((len(t), len(x)))
        p_old       = pl.zeros(pl.shape(t))
        p_new       = pl.zeros(pl.shape(t))
        p_rem_conf  = pl.zeros((n + 1, pl.size(t)))
        p_know_conf = pl.zeros((n + 1, pl.size(t)))

        # Initialize the probability mass distribution of the first time step
        tx[to_idx] = stats.norm.pdf(x, mu + z0, sigma) * delta_s

        for i in range(to_idx, len(t)):
            # Only convolve for steps AFTER the first one
            if i > to_idx:
                tx[i] = abs(pl.ifftshift(self._ifft(self._fft(tx[i - 1]) * ft_kernel)))

            # Extract particles that crossed boundaries
            p_pos    = tx[i][x >= bound[i]]
            p_old[i] = pl.sum(p_pos)
            p_new[i] = pl.sum(tx[i][x <= -bound[i]])

            # Zero out particles that already crossed the boundary
            tx[i] *= abs(x) < bound[i]

            p_sum = pl.sum(p_pos)
            if p_sum <= EPS:
                continue

            # Find the expected location of the mass that crossed
            x_pos        = x[x >= bound[i]]
            crossing_val = pl.dot(p_pos, x_pos) / p_sum

            # --- POST-DECISION BIVARIATE MATH ---
            # 1. Final Accumulated Evidence (X)
            mu_X = crossing_val + mu_r * t_post
            s2_X = (2 * d * t_post) + EPS

            # 2. Variable Confidence Boundary (C ~ N(c_mean, sigma_c))
            #    Z = X - C. If Z > 0, response is "High Confidence"
            #    c_mean here is clims[j-1] (the high confidence boundary)
            s2_C = (sigma_c ** 2) + EPS

            # 3. Fixed remember/know split
            #    Y = X - r_bound. If Y > 0, response is "Remember"
            mu_Y  = mu_X - r_bound
            s2_Y  = s2_X           # r_bound is fixed so no extra variance
            cov_XY = s2_X          # cov(X, X - const) = var(X)

            for j in range(1, len(clims)):
                c_upper = clims[j - 1]   # upper edge of this confidence bin
                c_lower = clims[j]       # lower edge of this confidence bin

                # With variable C_High ~ N(c_mean, sigma_c), the effective
                # confidence bin edges are also fuzzy. Model the observed
                # evidence X relative to the variable boundary C:
                #   Z = X - C  where C ~ N(c_upper, sigma_c)
                # High conf: Z > 0  i.e. X > C
                # Low conf:  Z < 0  i.e. X < C
                # The medium confidence band (j=2) uses both c_upper and c_lower.

                mu_Z_upper  = mu_X - c_upper        # X - C_upper
                mu_Z_lower  = mu_X - c_lower        # X - C_lower
                s2_Z        = s2_X + s2_C           # var(X - C) = var(X) + var(C)
                cov_XZ      = s2_X                  # cov(X, X-C) = var(X)

                # Joint distribution of (X, Y) where Y = X - r_bound (fixed)
                # Confidence check is done on Z = X - C (variable)
                # We integrate P(Z in conf band AND Y > 0) for remember
                # and P(Z in conf band AND Y < 0) for know

                # Confidence band: c_lower < X < c_upper maps to
                # Z_lower < 0 < Z_upper roughly, but with fuzzy boundaries
                # we use the variable-C formulation:
                # P(c_lower < X < c_upper) ≈ P(Z_upper > 0) - P(Z_lower > 0)
                # combined with Y > 0 or Y < 0 for R/K split

                mu_mvn  = pl.array([mu_X,  mu_Y])
                cov_mvn = pl.array([[s2_X + s2_C, cov_XY],
                                    [cov_XY,       s2_Y]])

                mvn_dist = stats.multivariate_normal(
                    mean=mu_mvn, cov=cov_mvn, allow_singular=True)

                # Remember: Evidence in conf bin AND Y > 0
                RLL = pl.array([c_lower, 0.0])
                RUL = pl.array([c_upper, INF_PROXY])

                # Know: Evidence in conf bin AND Y < 0
                KLL = pl.array([c_lower, -INF_PROXY])
                KUL = pl.array([c_upper, 0.0])

                p_rem_conf [j - 1, i] = p_old[i] * get_rect_prob(mvn_dist, RLL, RUL)
                p_know_conf[j - 1, i] = p_old[i] * get_rect_prob(mvn_dist, KLL, KUL)

        return p_rem_conf, p_know_conf, p_new, t


param_bounds = (
    SISPVarCBoundDDM.Params(
        c=0.0,
        mu_t=-2.0,
        mu_l=-2.0,
        d=EPS,
        tc_bound=0.0,
        r_bound_offset=0.0,
        sigma_c=EPS,
        z0=-1.0,
        t_post=EPS,
        t0=0.0,
    ),
    SISPVarCBoundDDM.Params(
        c=3.0,
        mu_t=2.0,
        mu_l=2.0,
        d=2.0,
        tc_bound=2.0,
        r_bound_offset=3.0,
        sigma_c=5.0,
        z0=1.0,
        t_post=3.0,
        t0=1.0,
    ),
)

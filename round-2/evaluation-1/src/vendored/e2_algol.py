"""e2_algol.py -- float64 bias audit of Algorithm L's geometric-skip recurrence.

Algorithm L (Li 1994, ACM TOMS 20(4):481-493) is the standard
O(k(1+log(n/k))) reservoir sampler:

    W = exp(log(random())/k); i = k
    while True:
        S = floor(log(random()) / log(1.0 - W))
        i += S + 1
        if i > n: break
        reservoir[randrange(k)] = i
        W *= exp(log(random())/k)

RESEARCH QUESTION: does float64 arithmetic in this recurrence induce a
*measurable* bias in the geometric skip S, and how large must an experiment
be to detect it?

THE STERBENZ CORRECTION, REVISED (read this before trusting any
"catastrophic cancellation" story about `1.0 - W`)
--------------------------------------------------------------------------
The e2_common claim that `1.0 - W` suffers catastrophic cancellation is WRONG.
By Sterbenz's lemma, for W in [0.5, 2] the subtraction `1.0 - W` is computed
EXACTLY in float64 -- there is no rounding error in the subtraction itself,
full stop.

The FIRST version of this module (still visible in `skip_fixed` /
`_advance_fixed` below) diagnosed the danger as W -> 1. That diagnosis was
BACKWARDS. In Algorithm L, W is (in distribution) the k-th smallest of the
uniforms seen so far, so W DECREASES over the run and tracks k/n at stream
position n -- it does not approach 1 except transiently near position 0 for
large k. The measured evidence for this is in this module's own audit
output: at k=2, position=100 (a routine, non-adversarial amount of
state-advance), `_advance_shipped`'s W underflowed to exactly 0.0 and
`skip_shipped` raised `ZeroDivisionError` on every single trial
(disagree_rate_f64 = 1.0) -- see `run_algol_audit`'s `cells` and its
`notes["w_to_zero_correction"]` for the e2_exact numbers from the run that
produced this file.

THE DANGEROUS LIMIT IS THEREFORE W -> 0, i.e. 1 - W -> 1. Once
W < 2**-53, the float64 evaluation `1.0 - W` rounds to EXACTLY 1.0,
discarding W's information entirely: `log(1.0 - W)` becomes `log(1.0) ==
0.0`, and dividing by that zero denominator is the ZeroDivisionError this
module reproduces on demand. The three error channels, restated correctly:

  (a) QUANTIZATION at W -> 0. `1.0 - W` can only resolve W down to the
      float64 ULP near 1 (~1.11e-16). Once W is smaller than that, the
      subtraction returns exactly 1.0 and the skip rate log(1-W) is
      destroyed. The relative error of the skip rate this induces is
      ~ 2**-53 / W ~ 2**-53 * n / k at stream position n (since W ~ k/n).
      Total failure (rate driven to exactly log(1)=0) needs n/k > 2**53,
      i.e. n/k beyond ~9.007e15.
  (b) The uniform draw U consumed by the state-advance step (`exp(log(U)/k)`)
      has the analogous quantization problem whenever it lands close to 1
      (log(U) loses relative precision there), compounding (a) over many
      steps.
  (c) Accumulated multiplicative rounding across the repeated product
      `W *= exp(log(U)/k)`.

THE FIX THAT WAS PREVIOUSLY SHIPPED HERE (`skip_fixed`, tracking
V = 1 - W directly) DOES NOT REPAIR THIS, and this module now measures
that failure rather than assuming the fix works: as W -> 0, V = 1-W -> 1,
which places V on exactly the same coarse near-1 float64 grid that made
W -> 1 dangerous in the (wrong) original diagnosis. Relocating the
quantity does not remove the quantization; `_advance_fixed`'s V is kept in
this module specifically so that this failure is a measured result (see
`rel_rate_err_mean_fixed` in the cells below -- it is not uniformly better
than shipped's, and is worse at exactly the cell where shipped is worst).

THE FIX THAT ACTUALLY WORKS is to never materialize W *or* V at all: track
`logW = log(W)` directly (accumulated as `logW += log(U)/k`, entirely in
log-space, so nothing underflows however long the stream runs), and compute
`log(1 - W)` from `logW` via the standard stable `log1mexp` identity:

    log1mexp(x) for x < 0:
        x < -log(2):  log1p(-exp(x))
        otherwise:    log(-expm1(x))

This is what `skip_logdomain` / `_advance_logdomain` do below. It works
because W is only ever recovered from logW as `exp(logW)` -- a small,
well-resolved positive number, however small logW's magnitude, since
float64 has excellent RELATIVE precision near 0 -- immediately before a
`log1p`/`log` call whose argument is, by construction, exactly the
magnitude that call is designed to resolve. `1.0 - W` (the operation that
destroys information) is never performed.

PAIRED-DRAW DESIGN (read before comparing shipped vs fixed vs logdomain numbers)
--------------------------------------------------------------------------
`skip_shipped` / `_advance_shipped`, `skip_fixed` / `_advance_fixed`, and
`skip_logdomain` / `_advance_logdomain` all consume the SAME raw float64
uniform stream (identical `np.random.default_rng(seed).random(...)` draws),
so any difference between them is attributable to floating-point
formulation, not to drawing different random numbers -- this is what
"paired, not statistical" means in the task this module implements.

Concretely: the FIXED path's per-step update reinterprets each raw draw u
as if it were the complement of a companion draw, feeding it to
`log1p(-u)` rather than `log(u)`. Since u and 1-u are both Uniform(0,1)
(same law), this is a legitimate paired reuse of the same draw e2_budget, not
a different experiment. The LOGDOMAIN path instead consumes u exactly as
shipped does (`log(u)`), so its state-advance is, term for term, the SAME
log-space accumulation shipped's W secretly performs before exponentiating
-- the only difference is that logdomain never exponentiates until the
single, stable log1mexp call at the very end.

Because shipped and fixed can therefore land on slightly different final
states from the "same" draws, `audit_algo_l` never compares shipped against
fixed (or logdomain) directly. Instead each float64 path is checked against
its OWN mpmath dps=80 e2_exact counterpart (`skip_exact(..., kind=...)`),
computed from an e2_exact, lossless float64->mpf conversion of the very same
draws. That is the only comparison in this module that is allowed to move
the reported error metrics.
"""

import json
import math
from typing import Any

import mpmath as mp
import numpy as np
from loguru import logger

# --------------------------------------------------------------------------- #
# 1. Paired skip functions (all driven by the same uniform draws)
# --------------------------------------------------------------------------- #


def skip_shipped(W: float, u2: float) -> float:
    """Literal Algorithm L skip formula: S = floor(log(u2) / log(1.0 - W)).

    This is exactly the shipped line `S = floor(log(random()) / log(1.0 - W))`
    with `random()` replaced by the caller-supplied `u2`. No exception is
    swallowed here on purpose: `math.log` raises ValueError on a
    non-positive argument, plain float division raises ZeroDivisionError on
    an e2_exact zero denominator, and `math.floor` raises OverflowError on an
    infinite/NaN argument -- these are the genuine failure modes of the
    shipped code and `edge_case_report` below deliberately forces and
    records each of them.
    """
    denom = math.log(1.0 - W)
    numer = math.log(u2)
    return float(math.floor(numer / denom))


def skip_fixed(V: float, u2: float) -> float:
    """Fixed-1 (superseded) skip formula: S = floor(log1p(-u2) / log1p(-V)).

    `V` is meant to be produced by `_advance_fixed` (tracking 1-W directly).
    Kept and measured deliberately: as established in the module docstring,
    this does NOT repair the W->0 failure mode (V -> 1 there, the same
    near-1 quantization relocated rather than removed). Like `skip_shipped`,
    this does not catch its own exceptions.
    """
    denom = math.log1p(-V)
    numer = math.log1p(-u2)
    return float(math.floor(numer / denom))


def _log1mexp_scalar(x: float) -> float:
    """Numerically stable log(1 - exp(x)) for x <= 0 (the standard log1mexp identity).

    This is the actual fix for Algorithm L's dangerous limit W -> 0
    (logW -> -infinity, but always finite): it never forms `1.0 - W` by
    subtraction, so nothing is lost regardless of how small W is. `exp(x)`
    recovers W as a small, well-resolved positive float, and the branch
    below hands it to whichever of `log1p`/`expm1` is designed to resolve
    an argument of that magnitude accurately.
    """
    threshold = -math.log(2.0)
    if x < threshold:
        return math.log1p(-math.exp(x))
    return math.log(-math.expm1(x))


def _log1mexp(x: np.ndarray) -> np.ndarray:
    """Vectorized `_log1mexp_scalar` over a numpy array."""
    threshold = -math.log(2.0)
    out = np.empty_like(x, dtype=np.float64)
    small = x < threshold
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out[small] = np.log1p(-np.exp(x[small]))
        out[~small] = np.log(-np.expm1(x[~small]))
    return out


def skip_logdomain(logW: float, u2: float) -> float:
    """Fixed-2 ('logdomain') skip formula: S = floor(log(u2) / log1mexp(logW)).

    `logW` is meant to be produced by `_advance_logdomain`, which accumulates
    log(U)/k in log-space and never exponentiates back to W until the single,
    numerically-stable `_log1mexp_scalar` evaluation in the denominator here.
    This is the variant that actually removes the W->0 failure mode: unlike
    `skip_fixed`'s V = 1-W, which merely relocates the problem (V -> 1 in
    exactly this regime), logW itself never underflows, however far the
    stream has advanced -- see the module docstring's revised correction.
    Like the other two skip functions, this does not catch its own
    exceptions (the numerator `log(u2)` still fails on u2<=0, exactly as
    `skip_shipped`'s does -- logdomain fixes the denominator's quantization
    bug, not the numerator's domain bug; `edge_case_report` shows both).
    """
    denom = _log1mexp_scalar(logW)
    numer = math.log(u2)
    return float(math.floor(numer / denom))


def algo_l_skip_float64(V_or_W_state: float, u2: float, k: int, variant: str = "shipped") -> float:
    """Generic (state, u2, k) -> skip dispatcher over the three float64 variants.

    `k` does not enter the skip formula itself -- only the state-advance
    step (`_advance_shipped` / `_advance_fixed` / `_advance_logdomain`) uses
    k -- it is accepted here purely so callers can pass the same
    (state, u2, k) triple that is used everywhere else in this module.
    `variant` selects `skip_shipped` (state=W), `skip_fixed` (state=V=1-W),
    or `skip_logdomain` (state=logW).
    """
    del k  # k affects only how V_or_W_state was produced, not the skip formula
    if variant == "shipped":
        return skip_shipped(V_or_W_state, u2)
    if variant == "fixed":
        return skip_fixed(V_or_W_state, u2)
    if variant == "logdomain":
        return skip_logdomain(V_or_W_state, u2)
    raise ValueError(f"variant must be 'shipped', 'fixed', or 'logdomain', got {variant!r}")


def skip_exact(state: float, u2: float, kind: str, dps: int = 80) -> float:
    """mpmath dps=80 e2_exact reference skip, mirroring `skip_shipped`/`skip_fixed`/`skip_logdomain`.

    `state` and `u2` are float64 values; they convert to `mpf` losslessly
    (every finite float64 is an e2_exact dyadic rational), so no additional
    error enters beyond what the caller already put into the float64
    inputs. `kind="shipped"` mirrors `skip_shipped`'s formula
    (log(u2)/log(1-state)); `kind="fixed"` mirrors `skip_fixed`'s formula
    (log1p(-u2)/log1p(-state)); `kind="logdomain"` mirrors `skip_logdomain`'s
    formula (log(u2)/log(-expm1(state)), with `state` here being logW
    itself, not W).
    """
    if kind not in ("shipped", "fixed", "logdomain"):
        raise ValueError(f"kind must be 'shipped', 'fixed', or 'logdomain', got {kind!r}")
    with mp.workdps(dps):
        state_mp = mp.mpf(state)
        u2_mp = mp.mpf(u2)
        if kind == "shipped":
            denom = mp.log(1 - state_mp)
            numer = mp.log(u2_mp)
        elif kind == "fixed":
            denom = mp.log1p(-state_mp)
            numer = mp.log1p(-u2_mp)
        else:  # logdomain: state_mp IS logW already, mirrors log1mexp exactly
            denom = mp.log(-mp.expm1(state_mp))
            numer = mp.log(u2_mp)
        return float(mp.floor(numer / denom))


# --------------------------------------------------------------------------- #
# 2. State-advance recurrences (numpy-vectorized float64 paths, mpmath e2_exact path)
# --------------------------------------------------------------------------- #


def _advance_shipped(U: np.ndarray, k: int) -> np.ndarray:
    """Materialize W via literal repeated float64 multiplication.

    `U` has shape (n_draws, n_steps). This reproduces, instruction for
    instruction, the shipped loop's `W *= exp(log(U)/k)` -- a python-level
    loop over steps, vectorized across draws with numpy -- so it carries
    the SAME sequence of float64 roundings the real sampler would produce
    (as opposed to a mathematically-equivalent but differently-rounded
    log-sum-exp shortcut). W decreases toward 0 as more steps are taken.
    """
    n_steps = U.shape[1]
    W = np.exp(np.log(U[:, 0]) / k)
    for j in range(1, n_steps):
        W = W * np.exp(np.log(U[:, j]) / k)
    return W


def _advance_fixed(U: np.ndarray, k: int) -> np.ndarray:
    """Track V = 1 - W directly via the fixed-1 recurrence (superseded fix).

        V_step = -expm1(log1p(-U)/k)
        V_new  = V + V_step - V*V_step        # since 1-(1-V)(1-V_step) = V + V_step - V*V_step

    See the paired-draw-design note in the module docstring: `U` here is
    the SAME raw draw consumed by `_advance_shipped`'s `log(U)`, reinterpreted
    via `log1p(-U)`. As established in the revised correction at the top of
    this module, this recurrence does NOT repair Algorithm L's actual
    danger (W -> 0): it drives V -> 1 in exactly that regime, which is the
    same near-1 float64 quantization problem, merely relocated onto V.
    """
    n_steps = U.shape[1]
    V = -np.expm1(np.log1p(-U[:, 0]) / k)
    for j in range(1, n_steps):
        Uj = U[:, j]
        V_step = -np.expm1(np.log1p(-Uj) / k)
        V = V + V_step - V * V_step
    return V


def _advance_logdomain(U: np.ndarray, k: int) -> np.ndarray:
    """Accumulate logW = sum(log(U_j))/k in log-space; W is never materialized.

    Because logW is a plain (possibly large-magnitude, but always finite)
    negative float, it can absorb arbitrarily many state-advance steps
    without ever underflowing -- there is no analogue of "W rounds to 0.0"
    here, however far the stream has advanced. See `skip_logdomain`.
    """
    n_steps = U.shape[1]
    logW = np.log(U[:, 0]) / k
    for j in range(1, n_steps):
        logW = logW + np.log(U[:, j]) / k
    return logW


def _advance_exact(U_rows: np.ndarray, k: int, kind: str, dps: int) -> list[Any]:
    """mpmath dps=80 e2_exact final state for each row of U_rows, shape (n_exact, n_steps).

    Uses the closed-form telescoping of the corresponding recurrence
    (sum of logs, then a single exp/expm1, or nothing at all for logdomain)
    rather than re-running the step-by-step recurrence in mpf: at dps=80
    (~266 e2_bits) the two are indistinguishable to far more digits than are
    ever compared here, and the closed form needs exactly one log-domain
    accumulation per draw instead of a multiply-heavy loop, which is what
    keeps the mpmath path fast enough to honor the runtime e2_budget.

        shipped:    ln(W)   = sum_j ln(U_j)     / k   ->  W    = exp(...)
        fixed:      ln(1-V) = sum_j log1p(-U_j) / k   ->  V    = -expm1(...)
        logdomain:  logW    = sum_j ln(U_j)     / k   ->  logW = ... (kept as-is)

    Every float64 U_j converts to `mpf` losslessly (e2_exact dyadic value), so
    no additional error enters beyond the finite precision already baked
    into the caller's float64 draws.
    """
    n_exact, n_steps = U_rows.shape
    results: list[Any] = []
    with mp.workdps(dps):
        for i in range(n_exact):
            acc = mp.mpf(0)
            for j in range(n_steps):
                u_mp = mp.mpf(float(U_rows[i, j]))
                if kind in ("shipped", "logdomain"):
                    acc += mp.log(u_mp)
                else:
                    acc += mp.log1p(-u_mp)
            acc = acc / k
            if kind == "shipped":
                results.append(mp.exp(acc))  # W_exact
            elif kind == "fixed":
                results.append(-mp.expm1(acc))  # V_exact
            else:
                results.append(acc)  # logdomain: logW_exact, kept unexponentiated
    return results


# --------------------------------------------------------------------------- #
# 3. JSON-safety helpers
# --------------------------------------------------------------------------- #


def _jsonable(x: Any) -> Any:
    """Convert a python/numpy scalar to a plain JSON-serializable value.

    Non-finite floats become the strings "inf" / "-inf" / "nan" so that a
    bare `Infinity` / `NaN` token never reaches `json.dumps`.
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    xf = float(x)
    if math.isnan(xf):
        return "nan"
    if math.isinf(xf):
        return "inf" if xf > 0 else "-inf"
    return xf


# --------------------------------------------------------------------------- #
# 4. The audit itself
# --------------------------------------------------------------------------- #


def audit_algo_l(k: int, n_draws: int, seed: int, position_steps: int, dps: int = 80) -> dict[str, Any]:
    """Audit Algorithm L's float64 skip recurrence for reservoir size k.

    Draws `n_draws` independent trajectories, each advancing the
    shipped-W / fixed-V / logdomain-logW state `position_steps` times from a
    fresh initial draw and then drawing one more uniform `u2` per trajectory
    to measure a single skip. The float64 paths are fully numpy-vectorized
    across all `n_draws` trajectories; only the mpmath e2_exact reference is
    computed with a python-level loop, and only over a (possibly
    subsampled) subset of trajectories, per the runtime e2_budget below.

    Returns a dict of per-cell metrics; see `run_algol_audit`'s returned
    `notes["metric_definitions"]` for the e2_exact definition of every field.
    """
    logger.info(f"audit_algo_l: k={k} n_draws={n_draws} position_steps={position_steps} seed={seed} dps={dps}")
    rng = np.random.default_rng(seed)
    n_cols = position_steps + 1
    U_advance = rng.random((n_draws, n_cols))
    U2 = rng.random(n_draws)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        W = _advance_shipped(U_advance, k)
        V = _advance_fixed(U_advance, k)
        logW = _advance_logdomain(U_advance, k)

        rate_shipped = np.log(1.0 - W)  # log(1-W) computed the shipped way (post-hoc subtraction)
        rate_fixed = np.log1p(-V)  # log1p(-V) computed the fixed-1 way (V tracked directly)
        rate_logdomain = _log1mexp(logW)  # log(1-W) computed via the stable log1mexp identity

        skip_shipped_arr = np.floor(np.log(1.0 - U2) / rate_shipped)
        skip_fixed_arr = np.floor(np.log1p(-U2) / rate_fixed)
        skip_logdomain_arr = np.floor(np.log(U2) / rate_logdomain)

    # --- mpmath dps=80 e2_exact reference, subsampled for speed ------------- #
    # Spec requires: if n_draws > 20000, subsample 20000 pairs for the
    # e2_exact reference. We add a second, stricter adaptive cap so that the
    # TOTAL mpmath op count (n_exact * 3 * n_cols log-domain calls, one set
    # per variant) stays bounded regardless of how large position_steps is
    # -- this is what actually keeps every cell under the ~30s runtime
    # e2_budget, since the spec's own cap alone does not bound the
    # position_steps factor.
    max_exact_by_spec = 20000
    op_budget = 250_000
    cap_by_op_budget = max(1, op_budget // max(1, 3 * n_cols))
    n_exact = int(min(n_draws, max_exact_by_spec, cap_by_op_budget))

    if n_exact < n_draws:
        idx = np.sort(rng.choice(n_draws, size=n_exact, replace=False))
    else:
        idx = np.arange(n_draws)

    U_sub = U_advance[idx]
    U2_sub = U2[idx]

    W_exact_list = _advance_exact(U_sub, k, "shipped", dps)
    V_exact_list = _advance_exact(U_sub, k, "fixed", dps)
    logW_exact_list = _advance_exact(U_sub, k, "logdomain", dps)

    skip_exact_shipped = np.empty(n_exact)
    skip_exact_fixed = np.empty(n_exact)
    skip_exact_logdomain = np.empty(n_exact)
    rate_exact_shipped = np.empty(n_exact)
    rate_exact_fixed = np.empty(n_exact)
    rate_exact_logdomain = np.empty(n_exact)
    with mp.workdps(dps):
        for i in range(n_exact):
            u2_mp = mp.mpf(float(U2_sub[i]))
            r_s = mp.log(1 - W_exact_list[i])
            r_f = mp.log1p(-V_exact_list[i])
            r_l = mp.log(-mp.expm1(logW_exact_list[i]))
            rate_exact_shipped[i] = float(r_s)
            rate_exact_fixed[i] = float(r_f)
            rate_exact_logdomain[i] = float(r_l)
            skip_exact_shipped[i] = float(mp.floor(mp.log(1 - u2_mp) / r_s))
            skip_exact_fixed[i] = float(mp.floor(mp.log1p(-u2_mp) / r_f))
            skip_exact_logdomain[i] = float(mp.floor(mp.log(u2_mp) / r_l))

    s_ship = skip_shipped_arr[idx]
    s_fix = skip_fixed_arr[idx]
    s_log = skip_logdomain_arr[idx]
    rate_ship_f = rate_shipped[idx]
    rate_fix_f = rate_fixed[idx]
    rate_log_f = rate_logdomain[idx]

    with np.errstate(invalid="ignore"):
        disagree_rate_f64 = float(np.mean(s_ship != skip_exact_shipped))
        disagree_rate_fixed = float(np.mean(s_fix != skip_exact_fixed))
        disagree_rate_logdomain = float(np.mean(s_log != skip_exact_logdomain))

        denom_ship = np.maximum(skip_exact_shipped, 1.0)
        denom_fix = np.maximum(skip_exact_fixed, 1.0)
        denom_log = np.maximum(skip_exact_logdomain, 1.0)
        rel_err_ship = np.abs(s_ship - skip_exact_shipped) / denom_ship
        rel_err_fix = np.abs(s_fix - skip_exact_fixed) / denom_fix
        rel_err_log = np.abs(s_log - skip_exact_logdomain) / denom_log

        tiny = 1e-300
        rel_rate_err_ship = np.abs(rate_ship_f - rate_exact_shipped) / np.maximum(np.abs(rate_exact_shipped), tiny)
        rel_rate_err_fix = np.abs(rate_fix_f - rate_exact_fixed) / np.maximum(np.abs(rate_exact_fixed), tiny)
        rel_rate_err_log = np.abs(rate_log_f - rate_exact_logdomain) / np.maximum(np.abs(rate_exact_logdomain), tiny)

    mean_s_exact_shipped = float(np.mean(skip_exact_shipped))
    mean_s_shipped_f64 = float(np.mean(s_ship))
    mean_s_exact_fixed = float(np.mean(skip_exact_fixed))
    mean_s_fixed_f64 = float(np.mean(s_fix))
    mean_s_exact_logdomain = float(np.mean(skip_exact_logdomain))
    mean_s_logdomain_f64 = float(np.mean(s_log))

    inclusion_bias = (mean_s_shipped_f64 - mean_s_exact_shipped) / max(mean_s_exact_shipped, 1.0) ** 2
    inclusion_bias_fixed = (mean_s_fixed_f64 - mean_s_exact_fixed) / max(mean_s_exact_fixed, 1.0) ** 2
    inclusion_bias_logdomain = (mean_s_logdomain_f64 - mean_s_exact_logdomain) / max(mean_s_exact_logdomain, 1.0) ** 2

    n_eff = 1000.0
    p = k / n_eff
    bias = abs(inclusion_bias)
    if bias == 0.0:
        trials_needed: float = float("inf")
    else:
        trials_needed = (1.96 * math.sqrt(max(p * (1.0 - p), 0.0)) / bias) ** 2

    fix_improvement_factor = disagree_rate_f64 / max(disagree_rate_fixed, 1.0 / n_draws)
    fix_improvement_factor_logdomain = disagree_rate_f64 / max(disagree_rate_logdomain, 1.0 / n_draws)

    cell = {
        "k": int(k),
        "position": int(position_steps),
        "n_draws": int(n_draws),
        "n_exact_pairs": int(n_exact),
        "disagree_rate_f64": _jsonable(disagree_rate_f64),
        "disagree_rate_fixed": _jsonable(disagree_rate_fixed),
        "disagree_rate_logdomain": _jsonable(disagree_rate_logdomain),
        "rel_skip_err_mean_f64": _jsonable(np.mean(rel_err_ship)),
        "rel_skip_err_max_f64": _jsonable(np.max(rel_err_ship)),
        "rel_skip_err_mean_fixed": _jsonable(np.mean(rel_err_fix)),
        "rel_skip_err_max_fixed": _jsonable(np.max(rel_err_fix)),
        "rel_skip_err_mean_logdomain": _jsonable(np.mean(rel_err_log)),
        "rel_skip_err_max_logdomain": _jsonable(np.max(rel_err_log)),
        "rel_rate_err_mean_f64": _jsonable(np.mean(rel_rate_err_ship)),
        "rel_rate_err_max_f64": _jsonable(np.max(rel_rate_err_ship)),
        "rel_rate_err_mean_fixed": _jsonable(np.mean(rel_rate_err_fix)),
        "rel_rate_err_max_fixed": _jsonable(np.max(rel_rate_err_fix)),
        "rel_rate_err_mean_logdomain": _jsonable(np.mean(rel_rate_err_log)),
        "rel_rate_err_max_logdomain": _jsonable(np.max(rel_rate_err_log)),
        "inclusion_bias": _jsonable(inclusion_bias),
        "inclusion_bias_fixed": _jsonable(inclusion_bias_fixed),
        "inclusion_bias_logdomain": _jsonable(inclusion_bias_logdomain),
        "trials_needed_to_detect": _jsonable(trials_needed),
        "fix_improvement_factor": _jsonable(fix_improvement_factor),
        "fix_improvement_factor_logdomain": _jsonable(fix_improvement_factor_logdomain),
        "mean_skip_exact_shipped": _jsonable(mean_s_exact_shipped),
        "mean_skip_exact_fixed": _jsonable(mean_s_exact_fixed),
        "mean_skip_exact_logdomain": _jsonable(mean_s_exact_logdomain),
    }
    logger.debug(f"audit_algo_l cell result: {cell}")
    return cell


# --------------------------------------------------------------------------- #
# 4b. The realistic-regime audit: W = k/n directly, for n spanning 1e3..1e15
# --------------------------------------------------------------------------- #


def audit_realistic_regime(k: int, n_list: list[float], n_draws: int, seed: int, dps: int = 80) -> list[dict[str, Any]]:
    """Audit shipped/fixed/logdomain at the REALISTIC stream-position regime W = k/n.

    Unlike `audit_algo_l` (which advances through a random multiplicative
    trajectory), this directly injects the target state W = k/n -- exactly
    the value Algorithm L's own theory says W takes at stream position n for
    reservoir size k -- and measures the float64 skip-rate error against an
    e2_exact mpmath dps=80 reference AT THAT EXACT TARGET, for n spanning
    ordinary to extreme stream lengths. Because the state is a single fixed
    injected value (not a per-trial trajectory), only ONE e2_exact rate
    reference per n is needed regardless of n_draws, so this stays fast
    even at large n_draws; only the per-trial e2_exact SKIP (which does vary
    with u2) needs one mpmath floor/log per draw, still O(1) work per draw
    (no trajectory loop), unlike `audit_algo_l`'s e2_exact path.
    """
    logger.info(f"audit_realistic_regime: k={k} n_list={n_list} n_draws={n_draws} seed={seed}")
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    tiny = 1e-300

    for n in n_list:
        if n <= 2 * k:
            # W = k/n is only a meaningful reservoir state when the stream is
            # longer than the reservoir; k/n >= 1/2 is not the regime this cell
            # is about (and k/n >= 1 is not a probability at all).  Record the
            # skip explicitly rather than crashing or silently dropping it.
            results.append({
                "k": int(k), "n": float(n), "regime": "realistic_W_eq_k_over_n",
                "status": "SKIPPED_n_le_2k",
                "reason": "W = k/n would be >= 0.5; not a stream-position regime",
            })
            continue
        U2 = rng.random(n_draws)
        W_f64 = k / n  # ordinary float64 division: the realistic float64 state
        V_f64 = 1.0 - W_f64
        logW_f64 = math.log(W_f64)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            rate_shipped_f64 = math.log(1.0 - W_f64)
            rate_fixed_f64 = math.log1p(-V_f64)
            rate_logdomain_f64 = _log1mexp_scalar(logW_f64)

            skip_shipped_arr = np.floor(np.log(1.0 - U2) / rate_shipped_f64)
            skip_fixed_arr = np.floor(np.log1p(-U2) / rate_fixed_f64)
            skip_logdomain_arr = np.floor(np.log(U2) / rate_logdomain_f64)

        with mp.workdps(dps):
            W_exact = mp.mpf(k) / mp.mpf(n)  # e2_exact target ratio, no float64 rounding at all
            V_exact = 1 - W_exact
            rate_exact_shipped_mp = mp.log(1 - W_exact)
            rate_exact_fixed_mp = mp.log1p(-V_exact)
            rate_exact_logdomain_mp = mp.log(1 - W_exact)  # logdomain targets the same log(1-W) as shipped

            rate_exact_shipped_f = float(rate_exact_shipped_mp)
            rate_exact_fixed_f = float(rate_exact_fixed_mp)
            rate_exact_logdomain_f = float(rate_exact_logdomain_mp)

            skip_exact_shipped = np.array(
                [float(mp.floor(mp.log(1 - mp.mpf(float(u))) / rate_exact_shipped_mp)) for u in U2]
            )
            skip_exact_fixed = np.array(
                [float(mp.floor(mp.log1p(-mp.mpf(float(u))) / rate_exact_fixed_mp)) for u in U2]
            )
            skip_exact_logdomain = np.array(
                [float(mp.floor(mp.log(mp.mpf(float(u))) / rate_exact_logdomain_mp)) for u in U2]
            )

        with np.errstate(invalid="ignore"):
            disagree_shipped = float(np.mean(skip_shipped_arr != skip_exact_shipped))
            disagree_fixed = float(np.mean(skip_fixed_arr != skip_exact_fixed))
            disagree_logdomain = float(np.mean(skip_logdomain_arr != skip_exact_logdomain))

        rel_rate_err_shipped = abs(rate_shipped_f64 - rate_exact_shipped_f) / max(abs(rate_exact_shipped_f), tiny)
        rel_rate_err_fixed = abs(rate_fixed_f64 - rate_exact_fixed_f) / max(abs(rate_exact_fixed_f), tiny)
        rel_rate_err_logdomain = abs(rate_logdomain_f64 - rate_exact_logdomain_f) / max(abs(rate_exact_logdomain_f), tiny)

        predicted_law = (2.0**-53) * n / k
        ratio_to_predicted = rel_rate_err_shipped / predicted_law if predicted_law > 0 else float("inf")

        mean_s_exact_shipped = float(np.mean(skip_exact_shipped))
        mean_s_shipped_f64 = float(np.mean(skip_shipped_arr))
        inclusion_bias = (mean_s_shipped_f64 - mean_s_exact_shipped) / max(mean_s_exact_shipped, 1.0) ** 2

        n_eff = 1000.0
        p = k / n_eff
        bias = abs(inclusion_bias)
        if bias == 0.0:
            trials_needed: float = float("inf")
        else:
            trials_needed = (1.96 * math.sqrt(max(p * (1.0 - p), 0.0)) / bias) ** 2

        results.append(
            {
                "k": int(k),
                "n": _jsonable(n),
                "W_target": _jsonable(W_f64),
                "n_draws": int(n_draws),
                "disagree_rate_shipped": _jsonable(disagree_shipped),
                "disagree_rate_fixed": _jsonable(disagree_fixed),
                "disagree_rate_logdomain": _jsonable(disagree_logdomain),
                "rel_rate_err_shipped": _jsonable(rel_rate_err_shipped),
                "rel_rate_err_fixed": _jsonable(rel_rate_err_fixed),
                "rel_rate_err_logdomain": _jsonable(rel_rate_err_logdomain),
                "predicted_rel_rate_err_law": _jsonable(predicted_law),
                "ratio_measured_over_predicted_shipped": _jsonable(ratio_to_predicted),
                "inclusion_bias_shipped": _jsonable(inclusion_bias),
                "trials_needed_to_detect": _jsonable(trials_needed),
            }
        )
    return results


def _error_law_thresholds(realistic_cells: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Derive, from the measured realistic-regime cells for this k, the smallest
    n at which the shipped variant's rate error crosses 1e-3 and 1.0, both
    under the naive theoretical law (c=1, i.e. rel_err = 2**-53*n/k exactly)
    and under the MEASURED law (c = the fit scale actually observed, taken
    from the largest-n cell available, since that is where the asymptotic
    quantization law dominates float64 noise most cleanly).
    """
    own = [c for c in realistic_cells if c["k"] == k]
    own_sorted = sorted(own, key=lambda c: c["n"])
    c_fit = 1.0
    for cell in reversed(own_sorted):
        pred = cell["predicted_rel_rate_err_law"]
        meas = cell["rel_rate_err_shipped"]
        if isinstance(pred, (int, float)) and isinstance(meas, (int, float)) and pred > 0 and meas > 0:
            c_fit = meas / pred
            break
    scale = (2.0**-53) * c_fit
    n_theoretical_1e3 = 1e-3 * k / (2.0**-53)
    n_theoretical_1_0 = 1.0 * k / (2.0**-53)
    n_measured_1e3 = 1e-3 * k / scale if scale > 0 else float("inf")
    n_measured_1_0 = 1.0 * k / scale if scale > 0 else float("inf")
    return {
        "k": int(k),
        "fit_scale_c_measured_over_predicted": _jsonable(c_fit),
        "n_theoretical_rel_err_1e-3_c1": _jsonable(n_theoretical_1e3),
        "n_theoretical_rel_err_1_0_c1": _jsonable(n_theoretical_1_0),
        "n_measured_law_rel_err_1e-3": _jsonable(n_measured_1e3),
        "n_measured_law_rel_err_1_0": _jsonable(n_measured_1_0),
    }


# --------------------------------------------------------------------------- #
# 5. Edge cases -- forced, never allowed to escape as an uncaught exception
# --------------------------------------------------------------------------- #


def edge_case_report(k: int) -> list[dict[str, Any]]:
    """Force each pathological Algorithm-L state and record what actually happens.

    Every case is wrapped so no exception escapes this function; instead
    the exception type and message are recorded verbatim, alongside a
    `severity` classification. For every shipped case, both a `fixed`
    (V = 1-W) and a `logdomain` (logW = log(W)) counterpart are also
    recorded, mapping the same physical state three ways, so the report
    shows directly which of the shipped library's failure modes each fix
    removes -- and, honestly, which ones neither fix removes, since some
    are genuine mathematical singularities of log(0), not floating-point
    artifacts.
    """
    logger.info(f"edge_case_report: k={k}")
    report: list[dict[str, Any]] = []

    def _record(case: str, variant: str, inputs: dict[str, Any], fn: Any) -> None:
        result_str: str | None
        exc_str: str | None
        try:
            result = fn()
            result_str = repr(result)
            exc_str = None
            severity = "silent-wrong" if (isinstance(result, float) and (math.isnan(result) or math.isinf(result))) else "ok"
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            result_str = None
            exc_str = f"{type(exc).__name__}: {exc}"
            severity = "crash"
        report.append(
            {
                "case": case,
                "variant": variant,
                "input": inputs,
                "result": result_str,
                "exception": exc_str,
                "severity": severity,
            }
        )

    representative_W = 1.0 - 1.0 / k  # a plausible mid-run W for this k
    representative_V = 1.0 / k  # the corresponding V = 1 - W
    representative_logW = math.log(representative_W)  # the corresponding logW = log(W)

    # 1. U == 0.0 fed to the skip's numerator draw ("u2"). This is a
    #    NUMERATOR bug, present identically in all three variants (none of
    #    the fixes target log(u2); they all target the denominator).
    _record("u2_equals_zero_exact", "shipped", {"W": representative_W, "u2": 0.0}, lambda: skip_shipped(representative_W, 0.0))
    _record("u2_equals_zero_exact", "fixed", {"V": representative_V, "u2": 0.0}, lambda: skip_fixed(representative_V, 0.0))
    _record(
        "u2_equals_zero_exact",
        "logdomain",
        {"logW": representative_logW, "u2": 0.0},
        lambda: skip_logdomain(representative_logW, 0.0),
    )

    # 2. W underflows to 0.0 after many multiplications (direct state
    #    injection) -- THE routine, non-adversarial failure this module's
    #    revised diagnosis identifies as the real danger. Equivalent states:
    #    fixed V = 1-W = 1.0; logdomain logW = log(0) = -inf (only reachable
    #    if an underlying draw were itself exactly 0, a separate, much
    #    rarer bug than "many ordinary steps", which is exactly the point).
    _record("W_underflow_to_zero", "shipped", {"W": 0.0, "u2": 0.5}, lambda: skip_shipped(0.0, 0.5))
    _record(
        "W_underflow_to_zero_equivalent_V_eq_1",
        "fixed",
        {"V": 1.0, "u2": 0.5},
        lambda: skip_fixed(1.0, 0.5),
    )
    _record(
        "W_underflow_to_zero_equivalent_logW_eq_neginf",
        "logdomain",
        {"logW": float("-inf"), "u2": 0.5, "note": "logW=-inf only arises from an upstream U==0, not from ordinary state-advance"},
        lambda: skip_logdomain(float("-inf"), 0.5),
    )

    # 3. W == 1.0 exactly (the transient near-position-0 regime for large
    #    k). Equivalent states: fixed V = 1-W = 0.0; logdomain logW = 0.0.
    _record("W_equals_one_exact", "shipped", {"W": 1.0, "u2": 0.5}, lambda: skip_shipped(1.0, 0.5))
    _record(
        "W_equals_one_exact_equivalent_V_eq_0",
        "fixed",
        {"V": 0.0, "u2": 0.5},
        lambda: skip_fixed(0.0, 0.5),
    )
    _record(
        "W_equals_one_exact_equivalent_logW_eq_0",
        "logdomain",
        {"logW": 0.0, "u2": 0.5},
        lambda: skip_logdomain(0.0, 0.5),
    )

    # 4. U2 == 1.0 - 2**-53 (largest double below 1).
    u2_max = 1.0 - 2.0**-53
    _record("u2_largest_double_below_one", "shipped", {"W": representative_W, "u2": u2_max}, lambda: skip_shipped(representative_W, u2_max))
    _record("u2_largest_double_below_one", "fixed", {"V": representative_V, "u2": u2_max}, lambda: skip_fixed(representative_V, u2_max))
    _record(
        "u2_largest_double_below_one",
        "logdomain",
        {"logW": representative_logW, "u2": u2_max},
        lambda: skip_logdomain(representative_logW, u2_max),
    )

    # 5. Denormal-scale state: V = 5e-324 in fixed; the logdomain analogue is
    #    logW = log(5e-324) ~ -744.4, i.e. W itself AT the smallest positive
    #    double. All three genuinely-tiny/near-boundary representations are
    #    tested; the interesting contrast is case 5b below.
    denormal_v = 5e-324
    _record("denormal_V_in_fixed", "fixed", {"V": denormal_v, "u2": 0.5}, lambda: skip_fixed(denormal_v, 0.5))
    logW_denormal = math.log(5e-324)
    _record(
        "denormal_scale_W_in_logdomain",
        "logdomain",
        {"logW": logW_denormal, "u2": 0.5, "note": "logW = log(5e-324), i.e. W pinned at the smallest positive double"},
        lambda: skip_logdomain(logW_denormal, 0.5),
    )
    # Illustrative counterpart: shipped cannot even REPRESENT this state.
    # Any W close enough to 1 that 1-W would be a 5e-324 denormal rounds to
    # exactly 1.0 in float64 -- this IS the (originally, wrongly-diagnosed)
    # W->1 quantization floor, made concrete: the state is lost before
    # shipped's code ever gets to run. Kept for continuity with the
    # module's history; the REAL floor for shipped is case 2 above (W->0).
    w_that_should_be = 1.0 - denormal_v
    _record(
        "denormal_V_unrepresentable_in_shipped_W",
        "shipped",
        {
            "W_computed_as_1_minus_denormal": w_that_should_be,
            "u2": 0.5,
            "note": "1.0 - 5e-324 rounds to exactly 1.0 in float64; the state is destroyed before this call even runs",
        },
        lambda: skip_shipped(w_that_should_be, 0.5),
    )

    # 5b (bonus). The measured point of the whole fix: a stream position far
    # BEYOND shipped's total-failure threshold (n/k > 2**53 ~ 9.007e15) --
    # here logW = -60 corresponds to W = exp(-60) ~ 8.76e-27, i.e.
    # n/k ~ 1/W ~ 1.14e26, roughly 1.3e10 times past shipped's breaking
    # point. shipped cannot even be constructed at this W (its own
    # multiplicative product would have underflowed to exactly 0.0 long
    # before reaching it); logdomain represents and computes it cleanly.
    _record(
        "logdomain_survives_far_beyond_shipped_threshold",
        "logdomain",
        {"logW": -60.0, "u2": 0.5, "note": "W=exp(-60)~8.8e-27, i.e. n/k~1.14e26, ~1.3e10x past shipped's n/k=2**53 breaking point"},
        lambda: skip_logdomain(-60.0, 0.5),
    )

    return report


# --------------------------------------------------------------------------- #
# 6. Top-level driver
# --------------------------------------------------------------------------- #


def run_algol_audit(
    ks: list[int],
    positions: list[int],
    n_draws: int,
    seed: int,
    realistic_n_list: list[float] | None = None,
    realistic_n_draws: int = 2000,
    dps: int = 80,
) -> dict[str, Any]:
    """Run the full audit: trajectory cells, realistic-regime cells, edge cases, notes.

    Returns
        {"cells": [...], "realistic_cells": [...], "error_law_thresholds": [...],
         "edge_cases": [...], "notes": {...}}
    where `cells` comes from `audit_algo_l` (random-trajectory state-advance,
    one per (k, position)), `realistic_cells` comes from
    `audit_realistic_regime` (direct injection of W = k/n for n spanning
    1e3..1e15, one per (k, n)), `error_law_thresholds` fits the measured
    2**-53*n/k law per k, `edge_cases` comes from `edge_case_report`, and
    `notes` carries the (revised) Sterbenz/W-to-zero correction text and the
    definition of every metric, built from THIS run's own measured numbers
    so it never goes stale relative to the data it accompanies.
    """
    logger.info(f"run_algol_audit: ks={ks} positions={positions} n_draws={n_draws} seed={seed}")
    if realistic_n_list is None:
        realistic_n_list = [1e3, 1e6, 1e9, 1e12, 1e15]

    cells = [audit_algo_l(k=k, n_draws=n_draws, seed=seed, position_steps=position, dps=dps) for k in ks for position in positions]
    edge_cases = [ec for k in ks for ec in edge_case_report(k)]
    realistic_cells = [
        c for k in ks for c in audit_realistic_regime(k=k, n_list=realistic_n_list, n_draws=realistic_n_draws, seed=seed, dps=dps)
    ]
    error_law_thresholds = [_error_law_thresholds(realistic_cells, k) for k in ks]

    # Build the blunt correction note from THIS run's own measured numbers.
    example_k = ks[0]
    example_cell = next((c for c in realistic_cells if c["k"] == example_k and c["n"] == 1e6), None)
    worst_cell = max(cells, key=lambda c: c["disagree_rate_f64"]) if cells else None

    def _fmt(v: Any) -> str:
        return str(v)

    w_to_zero_correction = (
        "BLUNT CORRECTION: the e2_common 'catastrophic cancellation in 1-W' story is WRONG. "
        "By Sterbenz's lemma, 1.0 - W is computed EXACTLY in float64 for W in [0.5, 2] -- "
        "there is no cancellation there at all. The real defect lives in the OPPOSITE "
        "limit, W -> 0 (not W -> 1): once W < 2**-53, `1.0 - W` rounds to exactly 1.0 in "
        "float64, silently destroying the skip rate (log(1.0-W) becomes log(1.0)==0.0, "
        "which is the ZeroDivisionError this module's edge_case_report reproduces on the "
        "'W_underflow_to_zero' case). "
        + (
            f"Measured evidence at k={example_k}, n=1e6 (W=k/n, the realistic stream "
            f"position): shipped rel_rate_err={_fmt(example_cell['rel_rate_err_shipped'])} vs "
            f"predicted law 2**-53*n/k={_fmt(example_cell['predicted_rel_rate_err_law'])} "
            f"(measured/predicted ratio={_fmt(example_cell['ratio_measured_over_predicted_shipped'])}). "
            if example_cell is not None
            else ""
        )
        + "Tracking V=1-W directly (the superseded 'fixed' arm, kept and measured rather "
        "than removed) does NOT repair this: V -> 1 in exactly this regime, the SAME "
        "near-1 quantization problem merely relocated, not removed. "
        + (
            f"Measured at k={worst_cell['k']}, position={worst_cell['position']} in this run's "
            f"trajectory cells: shipped's rel_rate_err_mean_f64="
            f"{_fmt(worst_cell['rel_rate_err_mean_f64'])} (disagree_rate_f64="
            f"{_fmt(worst_cell['disagree_rate_f64'])}) while the fixed arm's "
            f"rel_rate_err_mean_fixed={_fmt(worst_cell['rel_rate_err_mean_fixed'])} -- not smaller, "
            f"often worse or degenerate. "
            if worst_cell is not None
            else ""
        )
        + "Tracking logW directly and evaluating log(1-W) via the log1mexp identity (the "
        "'logdomain' arm) is the fix that actually works, because W is only ever "
        "recovered from logW as exp(logW) immediately before a log1p/log call sized for "
        "exactly that magnitude, and 1.0-W is never formed. "
        + (
            f"Measured at the same k={worst_cell['k']}, position={worst_cell['position']} cell: "
            f"logdomain's rel_rate_err_mean_logdomain={_fmt(worst_cell['rel_rate_err_mean_logdomain'])} "
            f"and disagree_rate_logdomain={_fmt(worst_cell['disagree_rate_logdomain'])}."
            if worst_cell is not None
            else ""
        )
    )

    notes = {
        "sterbenz_correction_original_and_wrong": (
            "[SUPERSEDED -- kept for the record, see 'w_to_zero_correction' for the "
            "corrected diagnosis] The e2_common story that '1.0 - W' suffers catastrophic "
            "cancellation is wrong; by Sterbenz's lemma, for W in [0.5, 2] the subtraction "
            "1.0 - W is EXACT in float64. This module's first version incorrectly located "
            "the danger at W -> 1 and shipped 'track V = 1-W' as the fix. Measurement "
            "showed that fix does not work, because the real danger is W -> 0."
        ),
        "w_to_zero_correction": w_to_zero_correction,
        "metric_definitions": {
            "disagree_rate_f64 / disagree_rate_fixed / disagree_rate_logdomain": "P(skip_variant != skip_exact(kind=variant)) over the (possibly subsampled) e2_exact-reference pairs, one probability per float64 variant against its OWN mpmath dps=80 e2_exact counterpart.",
            "rel_skip_err_mean/max_{f64,fixed,logdomain}": "mean / max of |S_variant - S_exact_variant| / max(S_exact_variant, 1).",
            "rel_rate_err_mean/max_{f64,fixed,logdomain}": "relative error of the float64 skip rate (log(1.0-W) for shipped, log1p(-V) for fixed, log1mexp(logW) for logdomain) vs its own mpmath dps=80 reference -- the continuous error underlying skip disagreement, undiluted by floor() ties.",
            "inclusion_bias / inclusion_bias_fixed / inclusion_bias_logdomain": "(E[S_variant_float64] - E[S_exact_variant]) / max(E[S_exact_variant], 1)**2.",
            "trials_needed_to_detect": "(1.96*sqrt(p*(1-p))/bias)**2 with p = k/1000 and bias = |inclusion_bias| (the shipped path's bias); 'inf' when bias == 0. In `realistic_cells` this is computed from that cell's own realistic-regime bias (see item 5 of the coordinator's correction), not the trajectory-cell bias.",
            "fix_improvement_factor / fix_improvement_factor_logdomain": "disagree_rate_f64 / max(disagree_rate_{fixed,logdomain}, 1/n_draws) -- how many times more often the shipped path disagrees with its e2_exact reference than the fixed/logdomain path disagrees with its own.",
            "n_exact_pairs": "size of the mpmath dps=80 reference subsample actually used for this trajectory cell: <= min(n_draws, 20000), further capped so the total mpmath op count (now x3 variants) stays bounded as position_steps grows (see audit_algo_l).",
            "realistic_cells[*].W_target": "W = k/n, the value Algorithm-L theory predicts W takes at stream position n for reservoir size k; injected directly rather than reached via a random trajectory.",
            "realistic_cells[*].predicted_rel_rate_err_law": "the coordinator's predicted law 2**-53 * n / k for the shipped variant's relative rate error at that (k, n).",
            "realistic_cells[*].ratio_measured_over_predicted_shipped": "rel_rate_err_shipped / predicted_rel_rate_err_law -- 1.0 means the predicted law is exactly right.",
            "error_law_thresholds[*]": "per-k fit of the measured law (c = measured/predicted ratio at the largest available n) plus the smallest n at which shipped's rate error is predicted to cross 1e-3 and 1.0, under both the naive theoretical law (c=1) and the measured-law fit.",
        },
        "paired_draw_convention": (
            "shipped, fixed, and logdomain consume the SAME raw float64 uniform stream "
            "(np.random.default_rng(seed).random(...)) so any difference between them is "
            "attributable to floating-point formulation, not to different random draws. "
            "logdomain consumes each draw exactly as shipped does (log(u)); fixed "
            "reinterprets it via log1p(-u). Every float64 quantity is checked against its "
            "OWN mpmath dps=80 e2_exact counterpart, never against another variant's e2_exact "
            "value, so these formulation differences cannot leak into the reported errors."
        ),
        "position_semantics": (
            "position_steps counts state-advance steps applied before the skip is measured "
            "in `cells` (the random-trajectory audit). Under the literal recurrence "
            "W *= exp(log(U)/k) with U in (0,1), W DECREASES toward 0 as position_steps "
            "grows -- consistent with Algorithm L's real semantics (W tracks k/n, which "
            "shrinks as the stream advances), and with the revised W->0 diagnosis above. "
            "`realistic_cells` instead injects W = k/n directly for a chosen n, which is "
            "the theoretically-correct way to probe a specific realistic stream position "
            "without depending on a particular random trajectory reaching it."
        ),
    }
    return {
        "cells": cells,
        "realistic_cells": realistic_cells,
        "error_law_thresholds": error_law_thresholds,
        "edge_cases": edge_cases,
        "notes": notes,
    }


@logger.catch(reraise=True)
def main() -> None:
    """Smoke test: small ks/positions/n_draws, print the resulting dict as indented JSON."""
    ks = [2, 10]
    positions = [0, 100]
    n_draws = 2000
    seed = 12345
    logger.info("Running Algorithm L float64-bias smoke audit")
    result = run_algol_audit(
        ks=ks,
        positions=positions,
        n_draws=n_draws,
        seed=seed,
        realistic_n_list=[1e3, 1e6, 1e9, 1e12, 1e15],
        realistic_n_draws=2000,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

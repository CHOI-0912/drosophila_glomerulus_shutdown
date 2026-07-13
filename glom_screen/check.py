"""Invariant checks. Run before trusting any screen output.

    python -W error::RuntimeWarning -m glom_screen.check

Every check here is something that would otherwise fail silently: the
metrics would still come out as plausible numbers in [0, 1] and nothing
downstream would look wrong.
"""
import numpy as np

from . import config as cfg
from . import metrics as gm
from .engine import Screen


def main():
    screen = Screen()
    r0 = screen.cached_solve('control', 'control', [])
    universe = gm.build_universe(r0, screen.seed_idx,
                                 screen.silenced_union_idx)
    cohorts = {n: gm.top_n(r0, universe, n) for n in cfg.N_SWEEP}
    ok = True

    # 1. empty lesion == control.  Guards the whole solve/cache path.
    print('\n[1] empty silence set reproduces the control')
    r_empty = screen.solve([])
    same = np.array_equal(r_empty, r0)
    print(f"    identical: {same}")
    ok &= same

    # 2. a lesion must not be able to *raise* influence anywhere.
    #    W >= 0 in unsigned mode and r = sum_k (alpha W)^k s, so with alpha
    #    held fixed (engine.Screen), zeroing columns strictly removes terms
    #    from that series: r_lesion <= r_control, everywhere.
    #    This only holds because alpha is fixed.  Under the library's
    #    default -- alpha recomputed on the lesioned W -- surviving
    #    synapses get boosted and this check fails (measured: ratios up to
    #    1.012).  So this is also the guard that alpha did not drift.
    print('\n[2] lesions only reduce influence (fixed alpha => monotone)')
    variant = 'pn'
    glom = sorted(screen.variants[variant])[0]
    r = screen.cached_solve(variant, glom, screen.variants[variant][glom])
    coh = cohorts[cfg.N_HEADLINE]
    worst_cohort = (r[coh] / r0[coh]).max()
    pos = r0 > 0
    worst_all = (r[pos] / r0[pos]).max()
    print(f"    {variant}/{glom}: max r_lesion/r_control  "
          f"cohort={worst_cohort:.8f}  all={worst_all:.8f}")
    ok &= worst_all <= 1 + 1e-5

    # 3. monotonicity under union.  Silencing A and B together must
    #    disrupt at least as much as either alone.  This is the strongest
    #    self-check available for the combination analysis: a bug in set
    #    construction or condition bookkeeping breaks it immediately.
    print('\n[3] monotonicity: D(A u B) >= max(D(A), D(B))')
    gloms = sorted(screen.variants[variant])[:3]
    a, b = gloms[0], gloms[1]
    sa = screen.variants[variant][a]
    sb = screen.variants[variant][b]
    ra = screen.cached_solve(variant, a, sa)
    rb = screen.cached_solve(variant, b, sb)
    rab = screen.solve(sorted(set(sa) | set(sb)))

    def dmag(r):
        return gm.compute(r0, r, universe, {cfg.N_HEADLINE: coh})[0]

    da, db, dab = dmag(ra), dmag(rb), dmag(rab)
    for key in ('D_mag_sum', 'D_mag_mean'):
        lo = max(da[key], db[key])
        good = dab[key] >= lo - 1e-9
        print(f"    {key}: {a}={da[key]:.6f} {b}={db[key]:.6f} "
              f"union={dab[key]:.6f}  >= max: {good}")
        ok &= good

    # 4. a glomerulus with more silenced neurons than another is not
    #    automatically more disruptive -- but every metric must land in
    #    range.  (Out-of-range values are printed by metrics.compute.)
    print('\n[4] all metrics within [0, 1]')
    vals = [v for d in (da, db, dab)
            for k, v in d.items() if k != 'N']
    inrange = all(0.0 <= v <= 1.0 for v in vals)
    print(f"    in range: {inrange}")
    ok &= inrange

    print('\nOK' if ok else '\nFAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

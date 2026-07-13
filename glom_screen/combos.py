"""Pairwise glomerulus lesions and their interaction.

    python -m glom_screen.combos --variant pn

Every pair of the 54 glomeruli (1,431 conditions).  Metrics are computed
on the fly and only the rows are kept -- caching 1,431 full influence
vectors would be ~860MB for little benefit, whereas the singles cache
(109 vectors, ~65MB) is worth keeping so metric definitions can be
revised without re-solving.

Interaction is measured on retention, R = 1 - D_mag_sum, i.e. the fraction
of the control top-N influence mass that survives:

    R_expected(A,B) = R(A) * R(B)        # if the two lesions were independent
    synergy         = R_expected - R_observed

synergy > 0 means the pair destroys more than two independent lesions
would; < 0 means they are redundant (they cut the same pathways, so doing
both adds little).

Caveat that must be read alongside every synergy number: the target sets
overlap.  In the 'all' variant a neuron sits in 5.5 glomeruli on average,
so A and B frequently silence some of the same neurons.  n_overlap is
reported for exactly this reason -- apparent redundancy may be shared
membership rather than shared circuitry.
"""
import argparse
import itertools
import time

import numpy as np
import pandas as pd

from . import config as cfg
from . import metrics as gm
from .engine import Screen
from .run import control_and_universe


def run_pairs(screen, variant, n_headline=None):
    n_headline = n_headline or cfg.N_HEADLINE
    r0, universe, cohorts = control_and_universe(screen)
    cohorts = {n_headline: cohorts[n_headline]}

    groups = screen.variants[variant]
    gloms = sorted(groups)

    # singles first: needed as the baseline for the interaction term
    single = {}
    for g in gloms:
        r = screen.cached_solve(variant, g, groups[g])
        single[g] = gm.compute(r0, r, universe, cohorts,
                               cond=f'{variant}/{g}')[0]

    pairs = list(itertools.combinations(gloms, 2))
    print(f"[{variant}] {len(pairs)} pairs, N={n_headline}")
    rows = []
    t0 = time.time()
    for k, (a, b) in enumerate(pairs, 1):
        sa, sb = set(groups[a]), set(groups[b])
        union = sorted(sa | sb)
        r = screen.solve(union)
        m = gm.compute(r0, r, universe, cohorts, cond=f'{variant}/{a}+{b}')[0]

        r_obs = 1.0 - m['D_mag_sum']
        r_exp = (1.0 - single[a]['D_mag_sum']) * (1.0 - single[b]['D_mag_sum'])
        rows.append({
            'variant': variant,
            'glom_a': a,
            'glom_b': b,
            'n_a': len(sa),
            'n_b': len(sb),
            'n_overlap': len(sa & sb),
            'n_union': len(union),
            'N': n_headline,
            'D_set': m['D_set'],
            'D_mag_sum': m['D_mag_sum'],
            'D_mag_mean': m['D_mag_mean'],
            'importance_sum': m['importance_sum'],
            'importance_mean': m['importance_mean'],
            'D_mag_sum_a': single[a]['D_mag_sum'],
            'D_mag_sum_b': single[b]['D_mag_sum'],
            'synergy': r_exp - r_obs,
        })
        if k % 25 == 0 or k == len(pairs):
            rate = k / (time.time() - t0)
            print(f"  {k:>4}/{len(pairs)}  {rate:.1f}/s  "
                  f"eta {(len(pairs) - k) / rate / 60:.1f} min",
                  end='\r', flush=True)
    print()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='pn', choices=cfg.VARIANTS)
    args = ap.parse_args()

    screen = Screen()
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    df = run_pairs(screen, args.variant)

    out = cfg.RESULTS / f'pairs_{args.variant}.csv'
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows)")

    # monotonicity: a pair must disrupt at least as much as either half
    bad = df[df['D_mag_sum'] < np.maximum(df['D_mag_sum_a'],
                                          df['D_mag_sum_b']) - 1e-9]
    print(f"\nmonotonicity violations (D(AuB) < max(D(A),D(B))): {len(bad)}"
          f"  <- must be 0")

    print(f"\nmost disruptive pairs [{args.variant}]")
    print(df.nlargest(10, 'importance_sum')[
        ['glom_a', 'glom_b', 'n_union', 'n_overlap', 'D_set', 'D_mag_sum',
         'importance_sum']].to_string(index=False))

    print("\nmost synergistic pairs (destroy more than independent)")
    print(df.nlargest(10, 'synergy')[
        ['glom_a', 'glom_b', 'n_overlap', 'D_mag_sum_a', 'D_mag_sum_b',
         'D_mag_sum', 'synergy']].to_string(index=False))

    print("\nmost redundant pairs (destroy less than independent)")
    print(df.nsmallest(10, 'synergy')[
        ['glom_a', 'glom_b', 'n_overlap', 'D_mag_sum_a', 'D_mag_sum_b',
         'D_mag_sum', 'synergy']].to_string(index=False))


if __name__ == '__main__':
    main()

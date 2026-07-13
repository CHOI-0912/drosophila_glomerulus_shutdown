"""Single-glomerulus lesion screen.

    python -m glom_screen.run

Runs every glomerulus in neuron.csv, for both silence-target variants,
and writes glom_screen_results/singles.csv.
"""
import argparse
import time

import pandas as pd

from . import config as cfg
from . import metrics as gm
from .engine import Screen


def control_and_universe(screen):
    """Control solve, the fixed ranking universe, and the control top-N
    cohorts (which depend only on the control, so they are built once).
    """
    r0 = screen.cached_solve('control', 'control', [])
    universe = gm.build_universe(r0, screen.seed_idx,
                                 screen.silenced_union_idx)
    cohorts = {n: gm.top_n(r0, universe, n) for n in cfg.N_SWEEP}
    return r0, universe, cohorts


def run_singles(screen, variants):
    r0, universe, cohorts = control_and_universe(screen)
    print(f"universe: {universe.size} neurons "
          f"(of {screen.n}; seeds and all silence targets excluded)")

    rows = []
    for variant in variants:
        groups = screen.variants[variant]
        t0 = time.time()
        for k, (glom, silenced) in enumerate(sorted(groups.items()), 1):
            r = screen.cached_solve(variant, glom, silenced)
            aud = screen.audit.set_index('glomerulus').loc[glom]
            for m in gm.compute(r0, r, universe, cohorts,
                                cond=f'{variant}/{glom}'):
                rows.append({
                    'variant': variant,
                    'glomerulus': glom,
                    'n_listed': int(aud['n_listed']),
                    'n_seed_excluded': int(aud['n_seed_excluded']),
                    'n_effective': len(silenced),
                    **m,
                })
            print(f"  [{variant}] {k:>2}/{len(groups)} {glom:<5} "
                  f"({len(silenced):>3} silenced)", end='\r', flush=True)
        print(f"  [{variant}] {len(groups)} conditions in "
              f"{time.time() - t0:.0f}s" + ' ' * 20)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', action='append', choices=cfg.VARIANTS,
                    help='repeatable; default = all variants')
    args = ap.parse_args()
    variants = args.variant or list(cfg.VARIANTS)

    screen = Screen()
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    screen.audit.to_csv(cfg.RESULTS / 'target_audit.csv', index=False)

    df = run_singles(screen, variants)
    out = cfg.RESULTS / 'singles.csv'
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")

    head = df[df['N'] == cfg.N_HEADLINE]
    for variant in variants:
        sub = (head[head['variant'] == variant]
               .sort_values('importance_sum', ascending=False))
        print(f"\ntop 10 glomeruli  [{variant}, N={cfg.N_HEADLINE}, "
              f"importance_sum]")
        print(sub[['glomerulus', 'n_effective', 'D_set', 'D_mag_sum',
                   'D_mag_mean', 'importance_sum']]
              .head(10).to_string(index=False))


if __name__ == '__main__':
    main()

"""Does recomputing alpha on the lesioned matrix contaminate the metric?

_normalize_W rescales W so its largest real eigenvalue equals lambda_max.
Under a lesion that eigenvalue drops, so alpha = lambda_max / eig goes UP
and every surviving synapse is boosted.  That partially compensates the
lesion -- the "option 2" convention.

"Option 1" instead freezes alpha at the intact network's value, so the
only thing that changes is the removal of the silenced columns.

These lesions are tiny (1-35 neurons of 150,299), so the alpha drift is
tiny too -- but so is the signal (D ~ 1%).  This script measures both on
the same footing and reports whether the choice changes the ranking.
"""
import numpy as np

from . import config as cfg
from . import metrics as gm
from .engine import Screen, leading_eigenvalue, solve_steady_state


def solve_with_alpha(W_les, seed_vec, alpha):
    """(alpha * W_les - I) r = -seed_vec, with alpha supplied, not derived."""
    A = W_les.copy()
    A.scale(alpha)
    A.shift(-1.0)
    return np.abs(np.real(solve_steady_state(A, -seed_vec)))


def main():
    screen = Screen()
    ic = screen.ic
    seed_vec = screen.seed_vec
    alpha0 = screen.alpha
    print(f"\nintact: lambda_max(W)={screen.eig_intact:.4f}  "
          f"alpha_control={alpha0:.6e}")

    r0 = solve_with_alpha(ic.W.copy(), seed_vec, alpha0).astype(np.float32)
    universe = gm.build_universe(r0, screen.seed_idx,
                                 screen.silenced_union_idx)
    coh = {cfg.N_HEADLINE: gm.top_n(r0, universe, cfg.N_HEADLINE)}

    variant = 'pn'
    gloms = sorted(screen.variants[variant])
    print(f"\n{'glom':<6} {'n':>3} {'eig_lesion':>11} {'a_les/a_ctl':>11} "
          f"{'D_opt2':>9} {'D_opt1':>9} {'artifact':>9}")
    print('-' * 66)
    rows = []
    for glom in gloms:
        ids = screen.variants[variant][glom]
        idx = np.array([ic.id_to_index[i] for i in ids])
        W_les = ic._set_columns_to_zero(idx)

        eig_l = leading_eigenvalue(W_les)
        alpha_l = cfg.LAMBDA_MAX / eig_l

        r2 = solve_with_alpha(W_les, seed_vec,
                              alpha_l).astype(np.float32)   # option 2
        r1 = solve_with_alpha(W_les, seed_vec,
                              alpha0).astype(np.float32)    # option 1

        d2 = gm.compute(r0, r2, universe, coh, cond=f'opt2/{glom}')[0]
        d1 = gm.compute(r0, r1, universe, coh, cond=f'opt1/{glom}')[0]
        art = d1['D_mag_sum'] - d2['D_mag_sum']
        print(f"{glom:<6} {len(ids):>3} {eig_l:>11.4f} "
              f"{alpha_l / alpha0:>11.6f} {d2['D_mag_sum']:>9.5f} "
              f"{d1['D_mag_sum']:>9.5f} {art:>9.5f}", flush=True)
        rows.append((glom, d1['D_mag_sum'], d2['D_mag_sum']))

    import pandas as pd
    from scipy.stats import spearmanr
    df = pd.DataFrame(rows, columns=['glom', 'opt1', 'opt2'])
    rho = spearmanr(df['opt1'], df['opt2']).statistic
    print(f"\nSpearman(option1, option2) over {len(df)} glomeruli = {rho:.4f}")
    print(f"option2 underestimates D by "
          f"{(df['opt1'] - df['opt2']).mean():.5f} on average "
          f"({100 * (1 - df['opt2'] / df['opt1']).mean():.1f}% of the signal)")
    df.to_csv(cfg.RESULTS / 'alpha_probe.csv', index=False)


if __name__ == '__main__':
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    main()

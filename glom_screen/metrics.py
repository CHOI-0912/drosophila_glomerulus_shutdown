"""Disruption metrics for one lesion condition.

Both metrics are absolute fractions in [0, 1] -- 0 = the lesion changed
nothing, 1 = it destroyed everything -- rather than z-scores across the
glomerulus panel.  That matters because singles and combinations have to
be read on the same scale: a z-score is defined relative to whatever set
it was standardised over, so a single's z and a pair's z are not the same
ruler and cannot be compared.
"""
import numpy as np

from . import config as cfg


def build_universe(r_ctrl, seed_idx, silenced_union_idx):
    """Indices eligible for ranking, held fixed across every condition.

    Excludes seeds (driven directly by s, so trivially top-ranked), every
    neuron silenced by *any* condition (its score is an artefact of the
    lesion, not a response to it), and neurons with zero control influence
    (they cannot be in a top-N cohort and would divide by zero in
    D_mag_mean).

    The same universe is used for every variant so that 'pn' and 'all'
    rankings can be compared to each other.
    """
    mask = np.ones(r_ctrl.shape[0], dtype=bool)
    mask[list(seed_idx)] = False
    mask[list(silenced_union_idx)] = False
    mask &= r_ctrl > 0
    return np.flatnonzero(mask)


def top_n(values, universe, n):
    """Indices of the n largest `values` within `universe`, descending."""
    sub = values[universe]
    n = min(n, sub.size)
    part = np.argpartition(-sub, n - 1)[:n]
    return universe[part[np.argsort(-sub[part])]]


def _clip(x, label, cond):
    if not (-cfg.CLIP_TOL <= x <= 1 + cfg.CLIP_TOL):
        print(f"  ! {cond}: {label}={x:.6g} outside [0,1] beyond tolerance")
    return float(np.clip(x, 0.0, 1.0))


def compute(r_ctrl, r_les, universe, cohorts, cond=''):
    """Metrics for one lesion, for every N in `cohorts`.

    cohorts maps N -> the control top-N index array (precomputed once and
    reused, since it depends only on the control).

    The cohort is fixed on the control.  A neuron that falls out of the
    lesioned top-N still contributes its lesioned score to D_mag -- that
    is the point: ranking by the lesioned top-N instead would drop the
    worst-hit neurons from the average and systematically understate the
    damage.
    """
    out = []
    for n, cohort in cohorts.items():
        les_top = top_n(r_les, universe, n)
        overlap = np.intersect1d(cohort, les_top, assume_unique=True).size
        d_set = 1.0 - overlap / cohort.size

        c0, cl = r_ctrl[cohort], r_les[cohort]
        d_mag_sum = 1.0 - cl.sum() / c0.sum()
        d_mag_mean = float(np.mean(1.0 - cl / c0))

        tag = f"{cond} N={n}"
        d_set = _clip(d_set, 'D_set', tag)
        d_mag_sum = _clip(d_mag_sum, 'D_mag_sum', tag)
        d_mag_mean = _clip(d_mag_mean, 'D_mag_mean', tag)

        out.append({
            'N': n,
            'D_set': d_set,
            'D_mag_sum': d_mag_sum,
            'D_mag_mean': d_mag_mean,
            'importance_sum': (d_set + d_mag_sum) / 2,
            'importance_mean': (d_set + d_mag_mean) / 2,
        })
    return out

"""Edge-level (synapse) glomerulus screen.

    python -m glom_screen.run_edges

Runs both scopes ('orn', 'orn_local'), writes glom_screen_results/edges.csv,
and reports how the edge-level ranking compares to the neuron-level one.

Size is reported, not corrected for.  DP1l really does make 13,195 ORN
synapses, and cutting them really does disrupt more -- that is a fact about
the fly, not a nuisance.  The question is "which glomerulus, cut, disrupts
the network most", and the honest answer includes its size.

Read the result accordingly: the ranking correlates with synapse count at
rho = 0.85, so it says "this glomerulus sends the most olfactory input into
the brain", not "this glomerulus is a special bottleneck".
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as cfg
from . import metrics as gm
from .edges import EdgeLesion, orn_to_glomerulus
from .engine import Screen
from .run import control_and_universe

SCOPES = ('orn', 'orn_local')


def build_conditions(screen, el):
    """glomerulus -> {scope: CSR positions to zero}."""
    idx = screen.ic.id_to_index
    o2g = orn_to_glomerulus(screen.seeds)
    lesion = EdgeLesion(screen)

    glom_orn = {}
    for i, g in o2g.items():
        if i in screen.ids_in_W:
            glom_orn.setdefault(g, []).append(i)

    # co-arborisation edges: both endpoints listed under the same glomerulus
    in_W = screen.ids_in_W
    members = {g: [i for i in ids if i in in_W]
               for g, ids in screen.groups.items()}

    el = el[el['count'] >= cfg.COUNT_THRESH]
    stats, conds = {}, {}
    for g in sorted(glom_orn):
        orn_idx = np.array([idx[i] for i in glom_orn[g]])
        p_orn = lesion.columns_positions(orn_idx)

        mem = set(members.get(g, []))
        loc = el[el['pre'].isin(mem) & el['post'].isin(mem)]
        if len(loc):
            p_loc = lesion.positions_of(
                [idx[i] for i in loc['post']], [idx[i] for i in loc['pre']])
        else:
            p_loc = np.array([], dtype=int)

        conds[g] = {
            'orn': p_orn,
            'orn_local': np.union1d(p_orn, p_loc),
        }
        stats[g] = {
            'n_orn': len(orn_idx),
            'n_edges_orn': len(p_orn),
            'n_syn_orn': int(lesion.data[p_orn].sum()),
            'n_edges_local': len(p_loc),
            'n_edges_orn_local': len(conds[g]['orn_local']),
        }
    return lesion, conds, stats


def main():
    screen = Screen()
    el = pd.read_feather(cfg.EDGELIST)
    lesion, conds, stats = build_conditions(screen, el)
    gloms = sorted(conds)
    print(f"{len(gloms)} glomeruli; "
          f"{sum(s['n_edges_orn'] for s in stats.values())} ORN edges total")

    r0, universe, cohorts = control_and_universe(screen)

    # The edge machinery must reproduce the column-zeroing path exactly for
    # the 'orn' scope: cutting every edge out of ORN(g) IS zeroing those
    # columns.  If these disagree, the CSR position lookup is wrong.
    g0 = gloms[0]
    a = lesion.solve(conds[g0]['orn'])
    b = screen.solve(sorted(i for i, gg in
                            orn_to_glomerulus(screen.seeds).items()
                            if gg == g0 and i in screen.ids_in_W))
    same = np.allclose(a, b, rtol=1e-6, atol=1e-12)
    print(f"cross-check (edge-removal == column-zeroing for {g0}): {same}")
    if not same:
        raise RuntimeError('edge removal disagrees with column zeroing')

    rows = []
    for scope in SCOPES:
        t0 = time.time()
        for k, g in enumerate(gloms, 1):
            r = lesion.solve(conds[g][scope])
            for m in gm.compute(r0, r, universe, cohorts,
                                cond=f'{scope}/{g}'):
                rows.append({'scope': scope, 'glomerulus': g,
                             **stats[g], **m})
            print(f"  [{scope}] {k}/{len(gloms)} {g}", end='\r', flush=True)
        print(f"  [{scope}] {len(gloms)} conditions in "
              f"{time.time() - t0:.0f}s" + ' ' * 20)

    df = pd.DataFrame(rows)
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    out = cfg.RESULTS / 'edges.csv'
    df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(df)} rows)")

    head = df[df['N'] == cfg.N_HEADLINE]
    for scope in SCOPES:
        h = head[head['scope'] == scope].sort_values('importance_sum',
                                                     ascending=False)
        print(f"\ntop 12 [{scope}, N={cfg.N_HEADLINE}]")
        print(h[['glomerulus', 'n_orn', 'n_edges_orn', 'n_syn_orn',
                 'D_set', 'D_mag_sum', 'importance_sum']]
              .head(12).to_string(index=False))
        rho = spearmanr(h['importance_sum'], h['n_syn_orn']).statistic
        print(f"  Spearman(importance, synapse count) = {rho:+.3f}")

    a = (head[head['scope'] == 'orn']
         .set_index('glomerulus')['importance_sum'])
    b = (head[head['scope'] == 'orn_local']
         .set_index('glomerulus')['importance_sum'].reindex(a.index))
    print(f"\nSpearman(orn, orn_local) = {spearmanr(a, b).statistic:+.3f}"
          f"   <- does local circuitry add anything?")

    pn = pd.read_csv(cfg.RESULTS / 'singles.csv')
    pn = pn[(pn['variant'] == 'pn') & (pn['N'] == cfg.N_HEADLINE)]
    pn = pn.set_index('glomerulus')['importance_sum'].reindex(a.index)
    print(f"Spearman(edge 'orn', neuron 'pn') = "
          f"{spearmanr(a, pn).statistic:+.3f}"
          f"   <- low means the neuron-level screen was measuring "
          f"multiglomerular-PN contamination")


if __name__ == '__main__':
    main()

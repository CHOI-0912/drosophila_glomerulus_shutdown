"""Summary of the screen: ranking, and the things that could undermine it.

    python -m glom_screen.report

Four questions, each a way the ranking could be an artefact:

  1. Does it depend on the choice of N?  (N=500 was picked, not derived.)
  2. Does it depend on which neurons count as "the glomerulus"?  ('pn' vs
     'all' -- the 'all' targets sit in 5.5 glomeruli each on average, so
     those lesions are not glomerulus-specific.)
  3. Is it just glomerulus size?  Reported, not corrected for: a bigger
     glomerulus moves the network more, and that is part of the answer.
  4. Does the neuron-level ranking agree with the edge-level one?  It does
     not (rho ~ 0.33), and the edge-level screen is the one to trust: only
     it can attribute a lesion to a glomerulus without ambiguity.
"""
import pandas as pd
from scipy.stats import spearmanr

from . import config as cfg

METRIC = 'importance_sum'


def main():
    df = pd.read_csv(cfg.RESULTS / 'singles.csv')
    variants = sorted(df['variant'].unique())

    for v in variants:
        head = (df[(df['variant'] == v) & (df['N'] == cfg.N_HEADLINE)]
                .sort_values(METRIC, ascending=False))
        print(f"\n{'=' * 68}\nranking [{v}, N={cfg.N_HEADLINE}, {METRIC}]")
        print(head[['glomerulus', 'n_effective', 'D_set', 'D_mag_sum',
                    'D_mag_mean', 'importance_sum', 'importance_mean']]
              .to_string(index=False))

    print(f"\n{'=' * 68}\n[1] sensitivity to N")
    for v in variants:
        sub = df[df['variant'] == v]
        base = (sub[sub['N'] == cfg.N_HEADLINE]
                .set_index('glomerulus')[METRIC])
        line = [f"  {v:<4}"]
        for n in cfg.N_SWEEP:
            if n == cfg.N_HEADLINE:
                continue
            other = sub[sub['N'] == n].set_index('glomerulus')[METRIC]
            rho = spearmanr(base, other.reindex(base.index)).statistic
            line.append(f"N={n}: rho={rho:.3f}")
        print('  '.join(line))
    print("  (rho vs the N=500 ranking; near 1.0 means N is not driving it)")

    print("\n[2] sensitivity to target definition ('pn' vs 'all')")
    if len(variants) == 2:
        a, b = variants
        ha = (df[(df['variant'] == a) & (df['N'] == cfg.N_HEADLINE)]
              .set_index('glomerulus')[METRIC])
        hb = (df[(df['variant'] == b) & (df['N'] == cfg.N_HEADLINE)]
              .set_index('glomerulus')[METRIC].reindex(ha.index))
        rho = spearmanr(ha, hb).statistic
        print(f"  Spearman({a}, {b}) = {rho:.3f}")
        print("  high -> the ranking survives the choice of who counts as")
        print("  'the glomerulus'; low -> the two definitions disagree and")
        print("  the LN-driven 'all' lesions are not glomerulus-specific.")
    else:
        print("  (need both variants; run: python -m glom_screen.run)")

    print("\n[3] is it just size?")
    for v in variants:
        head = df[(df['variant'] == v) & (df['N'] == cfg.N_HEADLINE)]
        rho = spearmanr(head[METRIC], head['n_effective']).statistic
        print(f"  {v:<4} Spearman(importance, n_effective) = {rho:+.3f}")
    print("  Size is reported, not corrected for: a glomerulus with more")
    print("  neurons listed against it moves the network more, and that is")
    print("  part of the answer rather than a nuisance to be removed.")

    edges = cfg.RESULTS / 'edges.csv'
    if not edges.exists():
        print("\n  NOTE: the neuron-level ranking above is contaminated --")
        print("  most 'pn' targets are multiglomerular PNs or PNs of *other*")
        print("  glomeruli. Run `python -m glom_screen.run_edges` for the")
        print("  edge-level screen, which has no attribution ambiguity.")
        return

    e = pd.read_csv(edges)
    e = e[(e['scope'] == 'orn') & (e['N'] == cfg.N_HEADLINE)]
    e = e.set_index('glomerulus')

    print("\n[4] edge-level ('orn') vs neuron-level ('pn')")
    pn = df[(df['variant'] == 'pn') & (df['N'] == cfg.N_HEADLINE)]
    pn = pn.set_index('glomerulus')[METRIC].reindex(e.index)
    rho = spearmanr(e[METRIC], pn).statistic
    print(f"  Spearman(orn, pn) = {rho:+.3f}")
    print("  Low agreement is the point: cutting a glomerulus' ORN synapses")
    print("  and killing the PNs listed under it are different experiments,")
    print("  and only the first is unambiguously about that glomerulus.")

    print("\n  EDGE-LEVEL RANKING ('orn' -- no attribution ambiguity)")
    print(e.sort_values(METRIC, ascending=False).reset_index()[
        ['glomerulus', 'n_orn', 'n_edges_orn', 'n_syn_orn', 'D_set',
         'D_mag_sum', METRIC]].to_string(index=False))
    rho = spearmanr(e[METRIC], e['n_syn_orn']).statistic
    print(f"\n  Spearman(importance, synapse count) = {rho:+.3f}")
    print("  Read this as 'which glomerulus sends the most olfactory input")
    print("  into the brain', not as 'which glomerulus is a special hub'.")


if __name__ == '__main__':
    main()

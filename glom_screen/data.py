"""Loading and validation of the screen's inputs.

calculate_influence silently drops two classes of id: those absent from
the connectivity matrix (`if ii in self.id_to_index`) and those that are
also seeds (`np.setdiff1d(silenced_indices_temp, exclusion_indices)`).
Both are load-bearing here -- 828 of the 5,093 listed silence targets are
ORNs, i.e. seeds -- so every drop is counted and reported rather than left
to happen quietly.
"""
import pandas as pd

from . import config as cfg


def load_seeds(path=None):
    """ORN root_ids from orn.txt (one comma-separated line, no trailing
    newline).  Returns a set of str.
    """
    path = path or cfg.SEEDS_TXT
    raw = path.read_text().replace('\n', ',')
    return {tok.strip() for tok in raw.split(',') if tok.strip()}


def load_groups(path=None):
    """glomerulus -> [root_id] from neuron.csv.

    The file carries a UTF-8 BOM, so utf-8-sig is required or the first
    column name comes back as '﻿glomerulus'.
    """
    path = path or cfg.GROUPS_CSV
    df = pd.read_csv(path, encoding='utf-8-sig')
    missing = {'glomerulus', 'neurons'} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} must have columns 'glomerulus' and 'neurons'; "
            f"missing {sorted(missing)}. Found: {list(df.columns)}"
        )
    return {
        row.glomerulus: [t.strip() for t in str(row.neurons).split(',')
                         if t.strip()]
        for row in df.itertuples()
    }


def load_cell_class(path=None):
    """root_id -> cell_class as a plain dict.

    A dict, not a pandas Series: meta['root_id'] has duplicates, which
    makes Series.get() fall back to a linear scan and turns this lookup
    into minutes rather than milliseconds.
    """
    path = path or cfg.META
    meta = pd.read_feather(path, columns=['root_id', 'cell_class'])
    return dict(zip(meta['root_id'], meta['cell_class']))


def build_variants(groups, seeds, ids_in_W, cell_class):
    """Turn the raw neuron.csv lists into the silence sets actually used.

    Returns {variant: {glomerulus: [root_id]}} for the variants in
    cfg.VARIANTS, restricted to ids that exist in W and are not seeds --
    i.e. exactly the neurons calculate_influence will really silence.
    """
    variants = {}
    for name in cfg.VARIANTS:
        out = {}
        for glom, members in groups.items():
            eff = [i for i in members if i in ids_in_W and i not in seeds]
            if name == 'pn':
                eff = [i for i in eff
                       if cell_class.get(i) == cfg.PN_CELL_CLASS]
            out[glom] = sorted(set(eff))
        variants[name] = out
    return variants


def audit(groups, variants, seeds, ids_in_W):
    """Per-glomerulus accounting of what was listed vs what gets silenced.

    Returns a DataFrame; raises if any glomerulus would end up with an
    empty silence set (that condition would silently equal the control and
    score zero importance for the wrong reason).
    """
    rows = []
    for glom, members in groups.items():
        members = set(members)
        row = {
            'glomerulus': glom,
            'n_listed': len(members),
            'n_not_in_W': len(members - ids_in_W),
            'n_seed_excluded': len(members & seeds),
        }
        for name in cfg.VARIANTS:
            row[f'n_effective_{name}'] = len(variants[name][glom])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('glomerulus').reset_index(drop=True)

    for name in cfg.VARIANTS:
        empty = df.loc[df[f'n_effective_{name}'] == 0, 'glomerulus'].tolist()
        if empty:
            raise ValueError(
                f"variant {name!r}: these glomeruli have no neuron left to "
                f"silence, so their condition would be identical to the "
                f"control and score 0 importance for a purely bookkeeping "
                f"reason: {empty}"
            )
    return df


def seed_audit(seeds, ids_in_W):
    """How many seeds actually make it into the matrix."""
    present = seeds & ids_in_W
    return {
        'n_listed': len(seeds),
        'n_in_W': len(present),
        'n_dropped': len(seeds - ids_in_W),
    }

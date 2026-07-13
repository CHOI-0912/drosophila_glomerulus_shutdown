"""Synapse-level (edge) lesions, instead of removing whole neurons.

The neuron-level screen has an attribution problem it cannot escape: of the
862 PN slots in neuron.csv, only 171 are PNs that actually belong to the
glomerulus they are listed under, and 12 glomeruli have none at all.  The
rest are multiglomerular PNs (M_*) or PNs of *other* glomeruli that happen
to arborise there.  Killing them removes circuitry that serves many
glomeruli, so "silencing glomerulus g" was partly silencing its neighbours.

Cutting edges instead of neurons dissolves that problem, because the ORN
side of the connectome is unambiguous:

    every ORN has cell_type 'ORN_<glomerulus>', i.e. exactly one
    glomerulus.  So each of the 19,245 ORN->X edges belongs to exactly one
    glomerulus, and the edge sets of two glomeruli overlap in exactly zero
    edges -- by construction, not by luck.

A multiglomerular PN survives the lesion: it loses its input from
glomerulus g and keeps everything else, including its whole output.  That
is what "glomerulus g delivers nothing" actually means.

Two scopes:

  orn        remove every ORN(g) -> X edge.  Zero attribution ambiguity.
  orn_local  the above, plus edges between two neurons that *both* arborise
             in g (neuron.csv[g] x neuron.csv[g]) -- an approximation of
             "synapses located inside the glomerulus".  This is a heuristic:
             the edge list carries no synapse coordinates, so co-arborisation
             is the only available proxy, and it reintroduces some of the
             ambiguity that the 'orn' scope avoids.
"""
import numpy as np
import pandas as pd
from petsc4py import PETSc

from . import config as cfg
from .engine import solve_steady_state


def orn_to_glomerulus(seeds, meta_path=None):
    """ORN root_id -> glomerulus, parsed from cell_type 'ORN_<glom>'.

    Single-valued by construction: cell_type is one string per neuron, so
    no ORN can be claimed by two glomeruli.
    """
    meta = pd.read_feather(meta_path or cfg.META,
                           columns=['root_id', 'cell_type'])
    ct = dict(zip(meta['root_id'], meta['cell_type']))
    return {i: str(ct[i])[4:] for i in seeds
            if i in ct and str(ct[i]).startswith('ORN_')}


class EdgeLesion:
    """Removes individual edges from W and solves, with alpha held fixed.

    The CSR is taken straight off the assembled PETSc matrix rather than
    rebuilt from the edge list, so it is the same W the neuron-level screen
    used -- same count_thresh, same duplicate summing, same ordering.  No
    chance of the two screens silently disagreeing about what W is.
    """

    def __init__(self, screen):
        self.s = screen
        n = screen.n
        indptr, indices, data = screen.ic.W.getValuesCSR()
        self.n = n
        self.indptr = indptr.copy()
        self.indices = indices.copy()
        self.data = data.copy()

        # (row, col) of every stored entry, as one sorted int64 key, so an
        # arbitrary edge set can be located by searchsorted.  CSR is already
        # row-major with columns sorted within each row, so the key array is
        # globally sorted and no explicit sort is needed.
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        self.cols = indices.astype(np.int64)
        self.key = rows * np.int64(n) + self.cols

    def positions_of(self, post_idx, pre_idx):
        """CSR positions of the edges (pre_idx[k] -> post_idx[k]).

        W[post, pre] is the weight from pre to post, so post is the row.
        Raises if an edge is not present, rather than silently zeroing the
        wrong entry.
        """
        keys = (np.asarray(post_idx, dtype=np.int64) * np.int64(self.n)
                + np.asarray(pre_idx, dtype=np.int64))
        pos = np.searchsorted(self.key, keys)
        bad = (pos >= self.key.size) | (self.key[np.minimum(
            pos, self.key.size - 1)] != keys)
        if bad.any():
            raise KeyError(f'{bad.sum()} requested edges are not in W')
        return pos

    def columns_positions(self, pre_idx):
        """CSR positions of every edge leaving the neurons in pre_idx.

        Equivalent to zeroing those columns of W -- which, for ORNs, is
        exactly "cut every synapse this ORN makes", since an ORN's entire
        output is its axon terminal inside its own glomerulus.
        """
        return np.flatnonzero(np.isin(self.cols, np.asarray(pre_idx)))

    def solve(self, positions=()):
        """Steady-state influence with the edges at `positions` removed."""
        data = self.data
        if len(positions):
            data = data.copy()
            data[np.asarray(positions)] = 0.0

        A = PETSc.Mat().createAIJ(
            size=(self.n, self.n),
            csr=(self.indptr, self.indices, data),
            comm=PETSc.COMM_SELF,
        )
        # shift(-1) inserts diagonal entries that W may not carry
        A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        A.scale(self.s.alpha)      # alpha from the INTACT W; see engine.py
        A.shift(-1.0)
        x = solve_steady_state(A, -self.s.seed_vec)
        return np.abs(np.real(x)).astype(np.float32)

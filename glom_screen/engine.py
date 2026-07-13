"""Solve driver for the lesion screen.

This does NOT go through InfluenceCalculator.calculate_influence, and does
not modify the library.  It reuses the InfluenceCalculator object for what
it is good at -- building W from the edge list, and the id <-> matrix-index
mapping -- and then runs the steady-state solve itself, because the screen
needs one thing calculate_influence does not offer: **a fixed alpha**.

Why that matters (measured, see alpha_probe.py):

    _normalize_W rescales W so its leading real eigenvalue equals
    lambda_max.  Under a lesion that eigenvalue drops, so alpha =
    lambda_max / eig is recomputed UPWARD and every surviving synapse is
    boosted.  The boost is proportional to how much the lesion mattered,
    so it cancels almost exactly the effect being measured: across the 54
    glomeruli it absorbs 95.6% of the disruption, and what survives has
    Spearman -0.06 against the true effect.  A screen built on it would
    rank noise.

So alpha is computed once from the intact W and held fixed for every
condition.  Then the only thing that changes between control and lesion is
the removal of the silenced columns, which is the intended manipulation.
"""
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

from InfluenceCalculator import InfluenceCalculator

from . import config as cfg
from . import data as gdata


def leading_eigenvalue(W):
    eps = SLEPc.EPS().create()
    eps.setOperators(W)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setDimensions(1)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.setFromOptions()
    eps.solve()
    if eps.getConverged() < 1:
        raise RuntimeError('SLEPc eigensolver did not converge on W')
    return eps.getEigenvalue(0).real


def solve_steady_state(A, rhs):
    """(A) x = rhs by GMRES/ILU, refusing to return a non-converged iterate.

    A non-converged Krylov iterate has no NaNs, a plausible distribution,
    and error spread across every neuron -- it cannot be spotted anywhere
    downstream, so it has to be caught here.
    """
    b = PETSc.Vec().createWithArray(rhs, comm=PETSc.COMM_SELF)
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.getPC().setType(PETSc.PC.Type.ILU)
    x = A.createVecRight()
    ksp.solve(b, x)
    reason = ksp.getConvergedReason()
    if reason < 0:
        raise RuntimeError(
            f'GMRES did not converge (reason {reason}, '
            f'{ksp.getIterationNumber()} iterations)')
    return np.asarray(x)


class Screen:
    def __init__(self, verbose=True):
        if verbose:
            print('building W ...', flush=True)
        self.ic = InfluenceCalculator.from_feather(
            str(cfg.EDGELIST), str(cfg.META),
            signed=cfg.SIGNED, count_thresh=cfg.COUNT_THRESH,
            lambda_max=cfg.LAMBDA_MAX,
        )
        self.ids_in_W = set(self.ic.id_to_index)
        self.n = self.ic.n_neurons

        self.seeds = gdata.load_seeds()
        self.groups = gdata.load_groups()
        self.cell_class = gdata.load_cell_class()
        self.variants = gdata.build_variants(
            self.groups, self.seeds, self.ids_in_W, self.cell_class)
        self.audit = gdata.audit(
            self.groups, self.variants, self.seeds, self.ids_in_W)
        self.seed_info = gdata.seed_audit(self.seeds, self.ids_in_W)

        self.seed_idx = np.array(
            [self.ic.id_to_index[i] for i in sorted(self.seeds
                                                    & self.ids_in_W)])
        self.seed_vec = np.zeros(self.n)
        self.seed_vec[self.seed_idx] = 1.0

        union = {i for v in self.variants.values()
                 for ids in v.values() for i in ids}
        self.silenced_union_idx = np.array(
            [self.ic.id_to_index[i] for i in sorted(union)])

        # build_variants already drops seeds, so no silence target may be a
        # seed.  If one ever were, calculate_influence-style seed exclusion
        # would quietly un-silence it and the condition would score zero.
        assert not (union & self.seeds), 'silence target overlaps a seed'

        # alpha: derived once, from the INTACT matrix, and never again.
        self.eig_intact = leading_eigenvalue(self.ic.W)
        self.alpha = cfg.LAMBDA_MAX / self.eig_intact

        if verbose:
            print(f"  W: {self.n} neurons")
            print(f"  lambda_max(W) = {self.eig_intact:.4f} -> "
                  f"alpha = {self.alpha:.6e} (fixed for every condition)")
            print(f"  seeds: {self.seed_info['n_in_W']} in W, "
                  f"{self.seed_info['n_dropped']} not in W (dropped)")
            for name in cfg.VARIANTS:
                s = self.audit[f'n_effective_{name}']
                print(f"  variant {name!r}: {s.sum()} silence slots, "
                      f"per-glomerulus min={s.min()} "
                      f"median={int(s.median())} max={s.max()}")
            print(f"  listed targets that are seeds (never silenced): "
                  f"{self.audit['n_seed_excluded'].sum()}", flush=True)

    def solve(self, silenced=()):
        """Steady-state influence with `silenced` neurons' outputs removed.

        Returns a float32 vector aligned to matrix index.
        """
        if len(silenced):
            idx = np.array([self.ic.id_to_index[i] for i in silenced])
            A = self.ic._set_columns_to_zero(idx)
        else:
            A = self.ic.W.copy()
        A.scale(self.alpha)
        A.shift(-1.0)
        x = solve_steady_state(A, -self.seed_vec)
        return np.abs(np.real(x)).astype(np.float32)

    # -- cache ---------------------------------------------------------

    @staticmethod
    def _cache_path(variant, cond):
        return cfg.CACHE / variant / f'{cond}.npy'

    def cached_solve(self, variant, cond, silenced):
        p = self._cache_path(variant, cond)
        if p.exists():
            return np.load(p)
        r = self.solve(silenced)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, r)
        return r

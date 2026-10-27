"""A batched, device-resident relaxer: one optimizer step for the whole pool.

`ase.optimize.LBFGS` drives exactly one structure, so a generation of ~130
independent relaxations becomes ~13 000 serial single-point calls.  Here every
structure in the pool takes its step *simultaneously*: one forward+backward
over the concatenated graph advances all of them.

Design notes
------------
* **Two vectorized steppers.**  `lbfgs` reproduces `ase.optimize.LBFGS`
  exactly (fixed inverse-Hessian guess, no line search, `maxstep` clamp) with
  the per-structure history kept as stacked tensors and every inner product
  turned into a per-structure segment sum; `fire` is a cheaper first-order
  alternative that needs ~3x the steps.  L-BFGS is the default because it
  matches the reference path step for step *and* costs fewer forward passes.

* **Per-structure convergence + compaction.**  Converged structures freeze
  immediately and are physically dropped from the batch every
  `compact_every` steps, so the slow tail of a batch runs on a shrinking
  graph instead of dragging the whole pool along.

* **Cell degrees of freedom.**  Slab relaxation needs the in-plane lattice to
  relax (the original used `FrechetCellFilter(mask=[1,1,0,0,0,0])`).  We add
  the diagonal deformation gradient as extra DOFs exactly as
  `ase.filters.UnitCellFilter` does: DOF value `cell_factor * F_ii`, DOF force
  `virial_ii / F_ii / cell_factor`, atom DOFs held in the undeformed frame.
  Both filters drive forces *and* virial to zero, so they share minima.

* **One host sync per step.**  Convergence and neighbour-list staleness are
  tracked on the device and read back as a single 2-element tensor, so the
  Python loop never serializes on per-structure scalars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

if __package__:
    from .engine import Batch, Sys, to_np64
else:
    from engine import Batch, Sys, to_np64


# Largest per-atom force (eV/A) a structure may show before it is treated as a
# diverged relaxation rather than a valid geometry.  A physical structure never
# exceeds a few tens of eV/A even mid-step; a collapsed slab (atoms on top of
# their periodic images) sits at ~1e6, so this separates them by orders of
# magnitude.  Raise it if a legitimate precursor ever trips it.
FORCE_SANITY = 100.0


@dataclass
class RelaxResult:
    """Outcome for each input structure, in input order."""

    energy: np.ndarray            # (n,) eV, NaN if the structure failed
    positions: List[np.ndarray]   # relaxed geometries
    cells: np.ndarray             # (n,3,3)
    converged: np.ndarray         # (n,) bool, reached fmax
    steps: np.ndarray             # (n,) int, step at which it finished
    failed: np.ndarray            # (n,) bool, non-finite energy/forces
    n_steps_total: int = 0        # batched steps actually executed
    n_forward: int = 0            # forward passes (== batched steps)


# --------------------------------------------------------------------------- #
# steppers: given DOF forces, produce DOF displacements
# --------------------------------------------------------------------------- #
class _Stepper:
    """Common plumbing: a stepper sees atom DOFs (N,3) plus strain DOFs (n,3)
    and returns displacements for both.  All internal reductions are
    per-structure, so structures never interact."""

    def __init__(self, batch, maxstep: float):
        import torch
        self.torch = torch
        self.batch = batch
        self.maxstep = float(maxstep)

    def rebind(self, batch, keep_struct, keep_node):
        self.batch = batch

    def dot(self, ua, ue, va, ve):
        """Per-structure inner product over the full DOF vector."""
        d = self.batch.seg.sum((ua * va).sum(dim=1))
        if ue is not None:
            d = d + (ue * ve).sum(dim=1)
        return d

    def clamp(self, dr, dr_eps):
        raise NotImplementedError


class _Fire(_Stepper):
    """Vectorized FIRE (`ase.optimize.FIRE` semantics, per structure)."""

    def __init__(self, batch, maxstep=0.2, dt=0.1, dt_max=1.0, n_min=5,
                 f_inc=1.1, f_dec=0.5, a_start=0.1, f_alpha=0.99):
        super().__init__(batch, maxstep)
        torch = self.torch
        n, dev, dt_t = batch.n, batch.device, batch.dtype
        self.vel = torch.zeros_like(batch.positions)
        self.vel_eps = torch.zeros(n, 3, dtype=dt_t, device=dev)
        self.dt = torch.full((n,), float(dt), dtype=dt_t, device=dev)
        self.alpha = torch.full((n,), float(a_start), dtype=dt_t, device=dev)
        self.n_up = torch.zeros(n, dtype=torch.long, device=dev)
        self.p = dict(dt_max=dt_max, n_min=n_min, f_inc=f_inc, f_dec=f_dec,
                      a_start=a_start, f_alpha=f_alpha)

    def rebind(self, batch, keep_struct, keep_node):
        self.vel = self.vel[keep_node]
        self.vel_eps = self.vel_eps[keep_struct]
        self.dt = self.dt[keep_struct]
        self.alpha = self.alpha[keep_struct]
        self.n_up = self.n_up[keep_struct]
        self.batch = batch

    def step(self, f_dof, f_eps, active):
        torch, p = self.torch, self.p
        b = self.batch.batch_index
        eps = 1e-12
        vf = self.dot(self.vel, self.vel_eps, f_dof, f_eps)
        f_norm = torch.sqrt(self.dot(f_dof, f_eps, f_dof, f_eps) + eps)
        v_norm = torch.sqrt(self.dot(self.vel, self.vel_eps,
                                     self.vel, self.vel_eps))
        up = vf > 0.0

        mix_v = torch.where(up, 1.0 - self.alpha, torch.zeros_like(self.alpha))
        mix_f = torch.where(up, self.alpha * v_norm / f_norm,
                            torch.zeros_like(self.alpha))
        self.vel = mix_v[b].unsqueeze(1) * self.vel + mix_f[b].unsqueeze(1) * f_dof
        self.vel_eps = mix_v.unsqueeze(1) * self.vel_eps + mix_f.unsqueeze(1) * f_eps

        grow = up & (self.n_up > p["n_min"])
        self.dt = torch.where(grow, torch.clamp(self.dt * p["f_inc"], max=p["dt_max"]),
                              torch.where(up, self.dt, self.dt * p["f_dec"]))
        self.alpha = torch.where(
            grow, self.alpha * p["f_alpha"],
            torch.where(up, self.alpha, torch.full_like(self.alpha, p["a_start"])))
        self.n_up = torch.where(up, self.n_up + 1, torch.zeros_like(self.n_up))

        self.vel = self.vel + self.dt[b].unsqueeze(1) * f_dof
        self.vel_eps = self.vel_eps + self.dt.unsqueeze(1) * f_eps
        dr = self.dt[b].unsqueeze(1) * self.vel
        dr_eps = self.dt.unsqueeze(1) * self.vel_eps

        # FIRE clamps the whole-structure displacement norm, unlike L-BFGS.
        norm = torch.sqrt(self.dot(dr, dr_eps, dr, dr_eps) + eps)
        scale = torch.where(norm > self.maxstep, self.maxstep / norm,
                            torch.ones_like(norm)) * active
        self.vel = self.vel * active[b].unsqueeze(1)
        self.vel_eps = self.vel_eps * active.unsqueeze(1)
        return dr * scale[b].unsqueeze(1), dr_eps * scale.unsqueeze(1)


class _LBFGS(_Stepper):
    """Vectorized L-BFGS matching `ase.optimize.LBFGS` (no line search).

    The two-loop recursion is a sequence of inner products and axpy updates;
    both vectorize per structure, and the history is just a stack of tensors.
    Curvature pairs with non-positive `y.s` are skipped (rho = 0) instead of
    producing an infinity that would poison one structure's whole trajectory.
    """

    def __init__(self, batch, maxstep=0.2, memory=25, alpha=70.0, damping=1.0):
        super().__init__(batch, maxstep)
        self.H0 = 1.0 / float(alpha)
        self.memory = int(memory)
        self.damping = float(damping)
        self.s_a, self.s_e, self.y_a, self.y_e, self.rho = [], [], [], [], []
        self.r0_a = self.r0_e = self.f0_a = self.f0_e = None

    def rebind(self, batch, keep_struct, keep_node):
        self.s_a = [t[keep_node] for t in self.s_a]
        self.y_a = [t[keep_node] for t in self.y_a]
        self.s_e = [t[keep_struct] for t in self.s_e]
        self.y_e = [t[keep_struct] for t in self.y_e]
        self.rho = [t[keep_struct] for t in self.rho]
        for name in ("r0_a", "f0_a"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, v[keep_node])
        for name in ("r0_e", "f0_e"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, v[keep_struct])
        self.batch = batch

    def step(self, f_dof, f_eps, active, r_a=None, r_e=None):
        torch = self.torch
        b = self.batch.batch_index

        if self.r0_a is not None:
            s_a = r_a - self.r0_a
            s_e = r_e - self.r0_e
            y_a = self.f0_a - f_dof
            y_e = self.f0_e - f_eps
            ys = self.dot(y_a, y_e, s_a, s_e)
            rho = torch.where(ys > 1e-12, 1.0 / ys, torch.zeros_like(ys))
            self.s_a.append(s_a); self.s_e.append(s_e)
            self.y_a.append(y_a); self.y_e.append(y_e)
            self.rho.append(rho)
            if len(self.rho) > self.memory:
                for buf in (self.s_a, self.s_e, self.y_a, self.y_e, self.rho):
                    buf.pop(0)

        q_a, q_e = -f_dof, -f_eps
        m = len(self.rho)
        a_coef = [None] * m
        for i in range(m - 1, -1, -1):
            ai = self.rho[i] * self.dot(self.s_a[i], self.s_e[i], q_a, q_e)
            a_coef[i] = ai
            q_a = q_a - ai[b].unsqueeze(1) * self.y_a[i]
            q_e = q_e - ai.unsqueeze(1) * self.y_e[i]
        z_a, z_e = self.H0 * q_a, self.H0 * q_e
        for i in range(m):
            bi = self.rho[i] * self.dot(self.y_a[i], self.y_e[i], z_a, z_e)
            c = a_coef[i] - bi
            z_a = z_a + c[b].unsqueeze(1) * self.s_a[i]
            z_e = z_e + c.unsqueeze(1) * self.s_e[i]

        dr_a, dr_e = -z_a, -z_e
        # ASE clamps by the LONGEST single-row displacement, not the norm.
        row = self.batch.seg.amax(dr_a.norm(dim=1))
        row = torch.maximum(row, dr_e.abs().amax(dim=1))
        scale = torch.where(row >= self.maxstep, self.maxstep / (row + 1e-30),
                            torch.ones_like(row)) * self.damping * active
        dr_a = dr_a * scale[b].unsqueeze(1)
        dr_e = dr_e * scale.unsqueeze(1)

        self.r0_a, self.r0_e = r_a.clone(), r_e.clone()
        self.f0_a, self.f0_e = f_dof.clone(), f_eps.clone()
        return dr_a, dr_e


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #
def relax(engine, systems: Sequence[Sys], fmax: float = 0.05,
          steps: int = 200, cell_mask: Optional[Sequence[bool]] = None,
          skin: float = 0.6, compact_every: int = 5, algo: str = "lbfgs",
          maxstep: float = 0.2, memory: int = 25,
          on_step: Optional[Callable[[int, int, int], None]] = None,
          ) -> RelaxResult:
    """Relax every structure in `systems` together; results are per structure.

    `cell_mask` enables diagonal cell relaxation, e.g. `(True, True, False)`
    for in-plane-only slab relaxation (needs an engine that reports stress).
    """
    import torch

    n0 = len(systems)
    out_energy = np.full(n0, np.nan)
    out_pos: List[Optional[np.ndarray]] = [None] * n0
    out_cell = np.stack([s.cell for s in systems]) if n0 else np.zeros((0, 3, 3))
    out_conv = np.zeros(n0, dtype=bool)
    out_steps = np.zeros(n0, dtype=int)
    out_failed = np.zeros(n0, dtype=bool)
    if n0 == 0:
        return RelaxResult(out_energy, [], out_cell, out_conv, out_steps, out_failed)

    use_cell = cell_mask is not None and any(cell_mask)
    if use_cell and not getattr(engine, "supports_virial", False):
        raise RuntimeError("cell relaxation requested but the engine has no virial")

    batch = Batch(engine, systems, skin=skin)
    dev, dt_t = batch.device, batch.dtype
    b = batch.batch_index

    stepper = (_Fire(batch, maxstep=maxstep) if algo == "fire"
               else _LBFGS(batch, maxstep=maxstep, memory=memory))

    done = torch.zeros(batch.n, dtype=torch.bool, device=dev)
    failed = torch.zeros(batch.n, dtype=torch.bool, device=dev)
    fin_step = torch.zeros(batch.n, dtype=torch.long, device=dev)
    e_last = torch.full((batch.n,), float("nan"), dtype=dt_t, device=dev)
    drift = torch.zeros(batch.n, dtype=dt_t, device=dev)

    ref_pos = batch.positions.clone()             # atom DOFs (undeformed frame)
    cell0 = batch.cells.clone()
    Fd = torch.ones(batch.n, 3, dtype=dt_t, device=dev)
    cell_factor = batch.natoms.clone()
    dmask3 = torch.zeros(3, 3, dtype=dt_t, device=dev)
    if use_cell:
        dmask3 = torch.diag(torch.as_tensor(
            [1.0 if m else 0.0 for m in cell_mask], dtype=dt_t, device=dev))

    budget = batch.displacement_budget()
    need_rebuild = True
    total_steps = 0
    last_compact = 0

    def harvest(mask_np):
        en = to_np64(e_last)
        cv = (done & ~failed).detach().cpu().numpy()
        st = fin_step.detach().cpu().numpy()
        fl = failed.detach().cpu().numpy()
        pos = batch.positions_np()
        cells = batch.cells_np()
        for local in np.flatnonzero(mask_np):
            g = int(batch.index[local])
            a, z = batch.ptr_np[local], batch.ptr_np[local + 1]
            out_energy[g] = np.nan if fl[local] else en[local]
            out_pos[g] = pos[a:z].copy()
            out_cell[g] = cells[local]
            out_conv[g] = bool(cv[local])
            out_failed[g] = bool(fl[local])
            out_steps[g] = int(st[local]) or total_steps

    for step in range(1, steps + 1):
        total_steps = step
        if need_rebuild:
            batch.rebuild()
            drift.zero_()
            need_rebuild = False

        try:
            E, F, virial = batch.compute(need_virial=use_cell)
        except torch.OutOfMemoryError as e:
            raise RuntimeError(
                "CUDA out of memory during the batched relaxation "
                f"({batch.n} structures, max_batch_atoms={engine.max_batch_atoms}, "
                f"max_batch_edges={engine.max_batch_edges}, virial={use_cell}). "
                "Lower --max-batch-atoms / --max-batch-edges, add "
                "--no-relax-cell to skip the memory-heavy virial, and/or set "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.") from e
        e_last = E

        # A blown-up structure must not poison its neighbours in the batch:
        # every reduction here is per-graph, so we just freeze the offender.
        #
        # The force bound is the important one. A diverged relaxation is usually
        # still *finite* -- atoms land on top of their periodic images and the
        # energy comes out at 1e5-1e6 eV, not NaN -- so the isfinite tests alone
        # let it through, and every downstream number then inherits the wreck
        # (a slab that exploded yields E_ads ~ 1e6 while its barriers, being
        # differences, still look plausible). A physical structure never exceeds
        # a few tens of eV/A even mid-step, while these failures sit at 1e6, so
        # FORCE_SANITY separates them by ~6 orders of magnitude.
        fmag = batch.seg.amax(F.norm(dim=1))
        bad = ~torch.isfinite(E)
        bad = bad | ~torch.isfinite(fmag)
        bad = bad | (fmag > FORCE_SANITY)
        newly_bad = bad & ~failed
        failed = failed | bad

        # --- DOF forces --------------------------------------------------- #
        f_cart = F.masked_fill(batch.fixed.unsqueeze(1), 0.0)
        if use_cell:
            f_dof = f_cart * Fd[b]
            # UnitCellFilter's cell rows: mask, undo the deformation (column
            # scaling by 1/F for a diagonal F), then rescale to force units.
            cell_rows = (virial * dmask3.unsqueeze(0) / Fd.unsqueeze(1)) \
                / cell_factor.view(-1, 1, 1)
            f_eps = torch.diagonal(cell_rows, dim1=1, dim2=2).contiguous()
            row_fmax = cell_rows.norm(dim=2).amax(dim=1)
        else:
            f_dof = f_cart
            f_eps = torch.zeros(batch.n, 3, dtype=dt_t, device=dev)
            row_fmax = torch.zeros(batch.n, dtype=dt_t, device=dev)

        fmax_s = torch.maximum(batch.seg.amax(f_dof.norm(dim=1)), row_fmax)
        fmax_s = torch.nan_to_num(fmax_s, nan=float("inf"))

        newly_done = ((fmax_s < fmax) | bad) & ~done
        fin_step = torch.where(newly_done, torch.full_like(fin_step, step), fin_step)
        done = done | newly_done | bad
        active = (~done).to(dt_t)

        # --- take the step ------------------------------------------------- #
        f_dof = torch.nan_to_num(f_dof) * active[b].unsqueeze(1)
        f_eps = torch.nan_to_num(f_eps) * active.unsqueeze(1)
        if isinstance(stepper, _LBFGS):
            dr, dr_eps = stepper.step(f_dof, f_eps, active, r_a=ref_pos,
                                      r_e=Fd * cell_factor.unsqueeze(1))
        else:
            dr, dr_eps = stepper.step(f_dof, f_eps, active)

        ref_pos = ref_pos + dr
        if use_cell:
            Fd = Fd + dr_eps / cell_factor.unsqueeze(1)
            new_pos = ref_pos * Fd[b]
            batch.cells = cell0 * Fd.unsqueeze(1)
        else:
            new_pos = ref_pos
        drift = drift + batch.seg.amax((new_pos - batch.positions).norm(dim=1))
        batch.positions = new_pos

        # --- one host sync per step --------------------------------------- #
        flags = torch.stack([done.sum().to(dt_t),
                             (drift > budget).any().to(dt_t)])
        flags = to_np64(flags)
        n_done = int(round(flags[0]))
        need_rebuild = use_cell or bool(flags[1] > 0.5)
        if on_step is not None:
            on_step(step, int(batch.n) - n_done, n_done)
        if n_done >= batch.n:
            break

        # Compaction is adaptive rather than periodic: the point is to stop
        # paying for structures that already converged, and how quickly that
        # happens depends on the batch, not on the step number.
        if (compact_every and n_done >= max(1, batch.n // 8)
                and step - last_compact >= compact_every):
            last_compact = step
            done_np = done.detach().cpu().numpy()
            harvest(done_np)
            keep = np.flatnonzero(~done_np)
            node_keep = np.concatenate(
                [np.arange(batch.ptr_np[i], batch.ptr_np[i + 1]) for i in keep])
            ks = torch.as_tensor(keep, dtype=torch.long, device=dev)
            kn = torch.as_tensor(node_keep, dtype=torch.long, device=dev)
            done, failed = done[ks], failed[ks]
            fin_step, e_last, drift = fin_step[ks], e_last[ks], drift[ks]
            ref_pos, Fd, cell0 = ref_pos[kn], Fd[ks], cell0[ks]
            batch.select(keep)
            stepper.rebind(batch, ks, kn)
            b = batch.batch_index
            cell_factor = batch.natoms.clone()
            need_rebuild = True

    # Structures that ran out of steps had their positions advanced *after* the
    # last energy evaluation, so re-evaluate once to keep the reported energy
    # consistent with the reported geometry (E_ads subtracts these directly).
    n_forward = total_steps
    if bool((~done).any()):
        batch.rebuild()
        E, F, _ = batch.compute(need_virial=False)
        final_bad = ~torch.isfinite(E) | ~torch.isfinite(batch.seg.amax(F.norm(dim=1))) | (batch.seg.amax(F.norm(dim=1)) > FORCE_SANITY)
        failed = failed | ((~done) & final_bad)
        e_last = torch.where(done, e_last, E)
        n_forward += 1

    harvest(np.ones(batch.n, dtype=bool))
    for i in range(n0):
        if out_pos[i] is None:                       # never made it into a batch
            out_pos[i] = systems[i].positions.copy()
    return RelaxResult(out_energy, out_pos, out_cell, out_conv, out_steps,
                       out_failed, n_steps_total=total_steps,
                       n_forward=n_forward)

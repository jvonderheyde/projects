"""Batched energy/force engines: one device launch for *many* structures.

Why this module exists
----------------------
The expensive part of SAGA is thousands of *tiny* relaxations (27-90 atoms
each).  A single MACE forward+backward on 27 atoms is **latency**-bound, not
throughput-bound: the useful arithmetic is microseconds hiding behind
hundreds of microseconds of Python, ASE bookkeeping and kernel-launch
overhead.  Driving that from `ase.optimize.LBFGS`, one structure at a time,
leaves a GPU essentially idle -- measured ~80 ms per single-point on a
27-atom slab, i.e. ~1700 sequential calls (~2 min) *per candidate*.

The fix is not a faster kernel, it is a wider one.  Every relaxation in a
generation is independent, so we concatenate all of them into one
block-diagonal graph and take *one* forward+backward per optimizer step for
the whole population.  A batch of 128 slab+adsorbate systems costs barely
more than a single one, which is where the order-of-magnitude comes from.

Contents
--------
`Sys`            a calculator-agnostic snapshot of one structure.
`MaceEngine`     the raw MACE torch module, replicated across devices,
                 with chunking, dtype control and optional cuEquivariance.
`AseEngine`      any ASE calculator (EMT toy path), same interface.
`Batch`          the concatenated, device-resident state of many structures:
                 neighbour lists with a Verlet skin, per-chunk masking to the
                 exact cutoff, compaction as structures converge.

The batched relaxer in `relax.py` consumes `Batch` and never touches ASE or
torch-device details itself.
"""

from __future__ import annotations

import os
import time
import warnings

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:                                     # ~2x faster than ASE's python version
    from matscipy.neighbours import neighbour_list as _neighbour_list
    _NL_BACKEND = "matscipy"
except Exception:                        # pragma: no cover - fallback
    from ase.neighborlist import primitive_neighbor_list as _ase_nl
    _NL_BACKEND = "ase"

    def _neighbour_list(quantities, pbc, cell, positions, cutoff):
        return _ase_nl(quantities, pbc, cell, positions, cutoff)


# --------------------------------------------------------------------------- #
# structure snapshot
# --------------------------------------------------------------------------- #
@dataclass
class Sys:
    """One structure, flattened away from ASE so batches stay cheap to build.

    `fixed` is the FixAtoms mask (bottom slab layer); `tag` is opaque payload
    the caller uses to route results back (candidate index, adsorbate, ...).
    """

    numbers: np.ndarray          # (n,) int64
    positions: np.ndarray        # (n,3) float64
    cell: np.ndarray             # (3,3) float64
    pbc: np.ndarray              # (3,) bool
    fixed: np.ndarray            # (n,) bool
    tag: object = None

    @classmethod
    def from_atoms(cls, atoms, tag=None) -> "Sys":
        fixed = np.zeros(len(atoms), dtype=bool)
        for c in getattr(atoms, "constraints", []):
            # FixAtoms is the only constraint this pipeline uses; anything else
            # would silently change the physics, so it is ignored loudly.
            if type(c).__name__ == "FixAtoms":
                fixed[c.get_indices()] = True
            else:
                warnings.warn(f"ignoring unsupported constraint {type(c).__name__}")
        return cls(
            numbers=atoms.get_atomic_numbers().astype(np.int64),
            positions=atoms.get_positions().astype(np.float64),
            cell=np.asarray(atoms.get_cell(), dtype=np.float64).copy(),
            pbc=np.asarray(atoms.pbc, dtype=bool).copy(),
            fixed=fixed,
            tag=tag,
        )

    def to_atoms(self):
        from ase import Atoms
        from ase.constraints import FixAtoms
        at = Atoms(numbers=self.numbers, positions=self.positions,
                   cell=self.cell, pbc=self.pbc)
        if self.fixed.any():
            at.set_constraint(FixAtoms(mask=self.fixed.tolist()))
        return at


def pair_list(positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray,
              cutoff: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACE-compatible neighbour list for one structure.

    Mirrors `mace.data.neighborhood.get_neighborhood`: non-periodic axes are
    inflated so no spurious images appear, and true self-edges are dropped.
    Returns (sender, receiver, unit_shifts).
    """
    cell = np.array(cell, dtype=np.float64)
    if not cell.any():
        cell = np.eye(3)
    eye = np.eye(3)
    max_pos = float(np.max(np.abs(positions))) + 1.0
    for d in range(3):
        if not pbc[d]:
            cell[d, :] = max_pos * 5.0 * cutoff * eye[d, :]
    i, j, S = _neighbour_list("ijS", pbc=pbc, cell=cell,
                              positions=positions, cutoff=cutoff)
    keep = ~((i == j) & np.all(S == 0, axis=1))
    return (i[keep].astype(np.int64), j[keep].astype(np.int64),
            S[keep].astype(np.float64))


# --------------------------------------------------------------------------- #
# device / precision helpers
# --------------------------------------------------------------------------- #
def to_np64(t) -> np.ndarray:
    """Device tensor -> float64 numpy (move to host before casting)."""
    return t.detach().cpu().double().numpy()


def resolve_devices(spec: str) -> List["object"]:
    """Expand a device spec into a concrete list of torch devices.

    "auto"          -> every visible CUDA GPU, else CPU
    "cuda"          -> every visible CUDA GPU (data-parallel replicas)
    "cuda:0,cuda:1" -> exactly those
    "cpu"           -> the host
    """
    import torch

    spec = (spec or "auto").strip().lower()
    if spec in ("auto", ""):
        if torch.cuda.is_available():
            spec = "cuda"
        else:
            spec = "cpu"
    if spec == "cuda":
        n = torch.cuda.device_count()
        if n == 0:
            raise RuntimeError("device='cuda' requested but no CUDA GPU is visible")
        return [torch.device(f"cuda:{i}") for i in range(n)]
    return [torch.device(d.strip()) for d in spec.split(",") if d.strip()]


def resolve_dtype(precision: str, devices) -> "object":
    """float32 on the GPU (2-30x faster there), float64 on CPU where it is
    nearly free and matches the reference path."""
    import torch

    precision = (precision or "auto").strip().lower()
    kinds = {d.type for d in devices}
    if precision == "auto":
        precision = "float64" if kinds == {"cpu"} else "float32"
    return torch.float64 if precision == "float64" else torch.float32


def tune_backend(devices, dtype):
    """Global torch knobs that matter for many small graphs."""
    import torch

    torch.set_default_dtype(dtype)
    if any(d.type == "cuda" for d in devices):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# engines
# --------------------------------------------------------------------------- #
class MaceEngine:
    """The MACE model itself, one replica per device, driven by raw batches.

    Bypassing `MACECalculator` is deliberate: it rebuilds an `AtomicData`,
    a `DataLoader` and a `Batch` on *every* single-point call and only ever
    holds one structure.  Here the graph is built once per neighbour-list
    rebuild and reused across optimizer steps.
    """

    supports_virial = True

    def __init__(self, model_path: str, devices, dtype,
                 max_batch_atoms: int = 6144, max_batch_edges: int = 600_000,
                 enable_cueq: bool = True, keep_z: Optional[Sequence[int]] = None):
        import torch

        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        self.torch = torch
        self.devices = list(devices)
        self.dtype = dtype
        self.max_batch_atoms = int(max_batch_atoms)
        self.max_batch_edges = int(max_batch_edges)

        model = load_mace_module(model_path, dtype)

        self.n_elements_full = len(model.atomic_numbers)
        self.restricted_to = None
        if keep_z:
            try:
                model, kept = restrict_model_elements(model, keep_z)
                if len(kept) < self.n_elements_full:
                    self.restricted_to = kept
            except Exception as e:  # noqa: BLE001
                # A TorchScript-serialised model exposes no ``weight_views()``,
                # so its element channels cannot be trimmed.  Restriction is a
                # memory/speed optimisation, not physics -- run with the full
                # element set rather than refusing to start on a model that is
                # otherwise perfectly usable.
                warnings.warn(
                    f"cannot restrict the model to Z={sorted(set(int(z) for z in keep_z))} "
                    f"({type(e).__name__}: {e}); using all "
                    f"{self.n_elements_full} elements")

        self.r_max = float(model.r_max)
        zs = [int(z) for z in model.atomic_numbers.tolist()]
        self.n_elements = len(zs)
        self.z_index = np.full(128, -1, dtype=np.int64)
        for k, z in enumerate(zs):
            self.z_index[z] = k

        heads = getattr(model, "heads", None) or ["Default"]
        self.head_index = 0
        for k, h in enumerate(heads):
            if str(h).lower() == "default":
                self.head_index = k
        self.multi_head = len(heads) > 1

        self.replicas = []
        self.cueq_active = False
        for dev in self.devices:
            rep = model if dev == self.devices[0] else _deepcopy_model(model)
            rep, used_cueq = _maybe_cueq(rep, dev, enable_cueq)
            self.cueq_active = self.cueq_active or used_cueq
            self.replicas.append(rep.to(dev))
        self.pool = (ThreadPoolExecutor(max_workers=len(self.devices))
                     if len(self.devices) > 1 else None)

    # -- element one-hot ------------------------------------------------- #
    def node_attrs(self, numbers: np.ndarray):
        idx = self.z_index[numbers]
        if (idx < 0).any():
            bad = sorted({int(z) for z in numbers[idx < 0]})
            raise ValueError(f"elements {bad} are not in the MACE model")
        oh = np.zeros((len(numbers), self.n_elements), dtype=np.float64)
        oh[np.arange(len(numbers)), idx] = 1.0
        return oh

    # -- chunk planning -------------------------------------------------- #
    def plan(self, natoms: np.ndarray, nedges: np.ndarray) -> List[Tuple[int, int]]:
        """Contiguous structure ranges that each fit the per-launch budget."""
        chunks, start, atoms, edges = [], 0, 0, 0
        for i in range(len(natoms)):
            a, e = int(natoms[i]), int(nedges[i])
            if i > start and (atoms + a > self.max_batch_atoms
                              or edges + e > self.max_batch_edges):
                chunks.append((start, i))
                start, atoms, edges = i, 0, 0
            atoms += a
            edges += e
        chunks.append((start, len(natoms)))
        # Spread chunks over devices so every GPU has work in the same step.
        if len(self.devices) > 1 and len(chunks) < len(self.devices):
            chunks = _split_chunks(chunks, natoms, len(self.devices))
        return chunks

    def device_for(self, chunk_id: int):
        return self.devices[chunk_id % len(self.devices)]

    # -- the actual forward ---------------------------------------------- #
    def forward(self, chunk_id: int, data: dict, need_virial: bool):
        """Energy, forces and (optionally) the virial for one chunk.

        We ask MACE for `virials`, not `stress`: the virial is what a cell
        filter actually needs, and it avoids a redundant volume division.
        """
        torch = self.torch
        model = self.replicas[chunk_id % len(self.devices)]
        with torch.enable_grad():
            out = model(data, training=False, compute_force=True,
                        compute_virials=need_virial, compute_stress=False)
        e = out["energy"].detach()
        f = out["forces"].detach()
        v = out["virials"].detach() if need_virial and out.get("virials") is not None else None
        return e, f, v

    def _free_cuda(self):
        """Release cached device blocks after an OOM.

        Without this the allocator keeps the fragmented, near-full cache and the
        next allocation fails too -- which is why a single transient OOM would
        otherwise kill a run that would have fitted after a purge.
        """
        import gc
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def _run_job(self, job):
        """One chunk through the model, with a single OOM retry after a purge.

        The usual cause of an OOM here is allocator fragmentation rather than a
        genuinely oversized chunk, so purging and retrying once rescues runs that
        would otherwise die. If it survives the retry, the error names the
        chunk's atom and edge counts so it is obvious which limit to lower --
        the defaults suit a large foundation model and are far too high for a
        9M-parameter L=3 model on a consumer card.
        """
        try:
            return self.forward(*job)
        except Exception as e:
            if "out of memory" not in str(e).lower():
                raise
            self._free_cuda()
            try:
                return self.forward(*job)
            except Exception as e2:
                if "out of memory" not in str(e2).lower():
                    raise
                raise RuntimeError(
                    f"MACE ran out of memory. Lower max_batch_atoms (currently "
                    f"{self.max_batch_atoms}) and/or max_batch_edges "
                    f"(currently {self.max_batch_edges}); for this model "
                    f"~512-1024 atoms per launch is a realistic start."
                ) from e2

    def run_chunks(self, jobs):
        """jobs: list of (chunk_id, data, need_virial) -> list of (E, F, virial)."""
        if self.pool is None or len(jobs) == 1:
            return [self._run_job(j) for j in jobs]
        return list(self.pool.map(self._run_job, jobs))


def load_mace_module(model_path: str, dtype):
    """Load a MACE checkpoint onto the host, cast it, and freeze it.

    Two non-obvious details, both learned the hard way:

    * The model's own modules must be imported *before* `torch.load`.
      Unpickling a MACE checkpoint pulls in e3nn, whose Wigner-3j cache itself
      calls `torch.load`, and a nested `torch.load` clobbers the outer call's
      thread-local state on torch 2.x (it raises `AttributeError: map_location`).
    * The checkpoint must land on the CPU first, so it can be cast to the
      target dtype before it moves to the device.
    """
    import os

    import torch

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import mace.modules  # noqa: F401
        import e3nn.o3 as _o3
        _o3.wigner_3j(1, 1, 1)
        model = torch.load(model_path, map_location=torch.device("cpu"),
                           weights_only=False)
    model = model.to(dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _deepcopy_model(model):
    import copy
    return copy.deepcopy(model)


def restrict_model_elements(model, keep_z: Sequence[int]):
    """Shrink a MACE foundation model to the elements a run can actually make.

    A foundation checkpoint carries every element (89 in `small.model`), and the
    element axis is not free: the per-element product-basis weights and the
    `skip_tp` residual tensor product both contract over it, so `skip_tp` alone
    costs `n_atoms x 128 x 89 x 128` multiply-adds *per interaction block*.  A
    SAGA run only ever builds 6 metals plus H and N, so 81 of those 89 one-hot
    columns are structurally zero and every FLOP spent on them is wasted.

    The restriction is exact -- a one-hot column that is never set cannot
    contribute -- and it is where most of the remaining device time goes once
    the relaxations are batched.

    The subtlety is that e3nn bakes a normalization constant (`path_weight`)
    into each instruction that depends on input *multiplicity*, so rebuilding a
    layer as `8x0e` instead of `89x0e` would silently rescale it.  We undo that
    by scaling the copied weights by `old_path_weight / new_path_weight`.

    Returns `(restricted_model, kept_atomic_numbers)`.
    """
    import copy

    import torch
    from e3nn import o3

    zs = [int(z) for z in model.atomic_numbers.tolist()]
    keep = sorted({int(z) for z in keep_z})
    missing = [z for z in keep if z not in zs]
    if missing:
        raise ValueError(f"model has no parameters for elements Z={missing}")
    if len(keep) >= len(zs):
        return model, zs

    model = copy.deepcopy(model)
    idx = torch.as_tensor([zs.index(z) for z in keep], dtype=torch.long)
    k = len(keep)
    dtype = next(model.parameters()).dtype

    def copy_tp(old, new, dim: int):
        with torch.no_grad():
            for i, (io, inew) in enumerate(zip(old.instructions, new.instructions)):
                wo = old.weight_view_for_instruction(i)
                wn = new.weight_view_for_instruction(i)
                wn.copy_(wo.index_select(dim, idx) * (io.path_weight / inew.path_weight))
        return new

    # --- one-hot -> node feature embedding (89x0e -> 128x0e) --------------- #
    lin = model.node_embedding.linear
    new_lin = o3.Linear(o3.Irreps(f"{k}x0e"), lin.irreps_out).to(dtype)
    with torch.no_grad():
        wos, wns = list(lin.weight_views()), list(new_lin.weight_views())
        for i, (io, inew) in enumerate(zip(lin.instructions, new_lin.instructions)):
            wns[i].copy_(wos[i].index_select(0, idx)
                         * (io.path_weight / inew.path_weight))
    model.node_embedding.linear = new_lin
    model.node_embedding.irreps_in = o3.Irreps(f"{k}x0e")

    # --- per-element residual tensor product in every interaction block ---- #
    for inter in model.interactions:
        tp = getattr(inter, "skip_tp", None)
        if tp is None:
            continue
        new_tp = o3.FullyConnectedTensorProduct(
            tp.irreps_in1, o3.Irreps(f"{k}x0e"), tp.irreps_out).to(dtype)
        inter.skip_tp = copy_tp(tp, new_tp, dim=1)
        inter.node_attrs_irreps = o3.Irreps(f"{k}x0e")

    # --- product-basis weights (plain tensors, no e3nn normalization) ------ #
    for prod in model.products:
        for contraction in prod.symmetric_contractions.contractions:
            contraction.weights_max = torch.nn.Parameter(
                contraction.weights_max.data.index_select(0, idx).clone(),
                requires_grad=False)
            contraction.weights = torch.nn.ParameterList([
                torch.nn.Parameter(w.data.index_select(0, idx).clone(),
                                   requires_grad=False)
                for w in contraction.weights])

    # --- element bookkeeping ---------------------------------------------- #
    ae = model.atomic_energies_fn.atomic_energies
    model.atomic_energies_fn.register_buffer(
        "atomic_energies", ae.index_select(-1, idx).clone())
    model.register_buffer("atomic_numbers", torch.as_tensor(keep, dtype=torch.int64))
    for p in model.parameters():
        p.requires_grad_(False)
    return model, keep


def _maybe_cueq(model, device, enable: bool):
    """cuEquivariance rewrites MACE's tensor products into fused CUDA kernels.

    This matters more than it sounds: MACE's `SymmetricContraction` bakes an
    `opt_einsum` contraction order chosen for a *10-atom* example batch, so its
    plain e3nn path degrades badly for wide batches.  cuEq replaces those
    einsums with kernels that scale properly, which is exactly what a batched
    relaxer wants.  Silently skipped when unavailable or off-GPU.
    """
    if not enable or device.type != "cuda":
        return model, False
    try:
        from mace.cli.convert_e3nn_cueq import run as to_cueq
        return to_cueq(model), True
    except Exception:
        return model, False


def _split_chunks(chunks, natoms, want):
    """Break the largest chunks until there are at least `want` of them, so a
    multi-GPU run does not leave replicas idle on a small batch."""
    out = list(chunks)
    while len(out) < want:
        k = max(range(len(out)), key=lambda i: out[i][1] - out[i][0])
        a, b = out[k]
        if b - a < 2:
            break
        mid = a + (b - a) // 2
        out[k:k + 1] = [(a, mid), (mid, b)]
    return out


class AseEngine:
    """Fallback engine wrapping any ASE calculator (the EMT toy path).

    There is nothing to batch on the calculator side, but reusing the same
    `Batch`/`relax` machinery keeps a single code path -- and the structures
    still relax in lock-step, so the optimizer, bookkeeping and reporting are
    identical to the GPU run.
    """

    supports_virial = True

    def __init__(self, calc_factory, devices, dtype, n_workers: int = 0):
        import torch
        self.torch = torch
        self.devices = [torch.device("cpu")]
        self.dtype = dtype
        self.r_max = 0.0                     # no neighbour list needed
        self.calc_factory = calc_factory
        self.n_workers = int(n_workers)
        self._pool = (ThreadPoolExecutor(max_workers=self.n_workers)
                      if self.n_workers > 1 else None)
        self._calcs = [calc_factory() for _ in range(max(1, self.n_workers))]

    def compute(self, systems: Sequence[Sys], positions: np.ndarray,
                cells: np.ndarray, need_virial: bool):
        n = len(systems)
        E = np.zeros(n)
        F = np.zeros_like(positions)
        S = np.zeros((n, 3, 3)) if need_virial else None
        offs = np.concatenate([[0], np.cumsum([len(s.numbers) for s in systems])])

        def one(k):
            calc = self._calcs[k % len(self._calcs)]
            s = systems[k]
            at = s.to_atoms()
            at.set_cell(cells[k])
            at.set_positions(positions[offs[k]:offs[k + 1]])
            at.calc = calc
            e = at.get_potential_energy()
            f = at.get_forces(apply_constraint=False)
            # virial = -V * stress, matching what MaceEngine returns
            st = (-at.get_volume() * at.get_stress(voigt=False)
                  if need_virial else None)
            return k, e, f, st

        results = (self._pool.map(one, range(n)) if self._pool
                   else (one(k) for k in range(n)))
        for k, e, f, st in results:
            E[k] = e
            F[offs[k]:offs[k + 1]] = f
            if need_virial:
                S[k] = st
        return E, F, S


# --------------------------------------------------------------------------- #
# the concatenated batch
# --------------------------------------------------------------------------- #
class _Segments:
    """Per-structure reductions over a concatenated atom axis.

    Implemented with a small dense (n_struct, max_atoms) scratch buffer rather
    than `scatter_reduce`, because the dense form is supported identically on
    CPU and CUDA and the buffer is negligible at these sizes.
    """

    def __init__(self, natoms: np.ndarray, device, dtype):
        import torch
        self.torch = torch
        self.n = len(natoms)
        self.maxn = int(natoms.max()) if self.n else 0
        row = np.repeat(np.arange(self.n), natoms)
        col = np.concatenate([np.arange(k) for k in natoms]) if self.n else np.zeros(0)
        self.row = torch.as_tensor(row, dtype=torch.long, device=device)
        self.col = torch.as_tensor(col, dtype=torch.long, device=device)
        self.device = device
        self.dtype = dtype

    def _dense(self, vals, fill):
        d = self.torch.full((self.n, self.maxn), fill,
                            dtype=self.dtype, device=self.device)
        d[self.row, self.col] = vals
        return d

    def amax(self, vals):
        return self._dense(vals, float("-inf")).amax(dim=1)

    def sum(self, vals):
        return self._dense(vals, 0.0).sum(dim=1)


class Batch:
    """Many structures held as one block-diagonal graph on the device.

    Responsibilities
    ----------------
    * concatenated positions / cells / masks as device tensors;
    * neighbour lists with a Verlet **skin** so the (CPU-side) rebuild happens
      every ~10 steps instead of every step, and exact masking to `r_max`
      before the forward so the skin never changes the physics;
    * chunking into launches that fit the memory budget, round-robin across
      devices;
    * `select()` to drop converged structures mid-relaxation, which makes the
      long tail of a batch cheap.
    """

    def __init__(self, engine, systems: Sequence[Sys], skin: float = 0.6):
        import torch
        self.torch = torch
        self.engine = engine
        self.systems = list(systems)
        self.skin = float(skin)
        self.device = engine.devices[0]
        self.dtype = engine.dtype
        self.is_torch_engine = isinstance(engine, MaceEngine)

        self.index = np.arange(len(self.systems))     # -> caller's ordering
        self._natoms = np.array([len(s.numbers) for s in self.systems], dtype=np.int64)
        self._numbers = np.concatenate([s.numbers for s in self.systems]) \
            if self.systems else np.zeros(0, dtype=np.int64)
        self._pbc = np.stack([s.pbc for s in self.systems]) if self.systems \
            else np.zeros((0, 3), bool)

        pos = np.concatenate([s.positions for s in self.systems]) if self.systems \
            else np.zeros((0, 3))
        cells = np.stack([s.cell for s in self.systems]) if self.systems \
            else np.zeros((0, 3, 3))
        fixed = np.concatenate([s.fixed for s in self.systems]) if self.systems \
            else np.zeros(0, bool)

        self.positions = torch.as_tensor(pos, dtype=self.dtype, device=self.device).clone()
        self.cells = torch.as_tensor(cells, dtype=self.dtype, device=self.device).clone()
        self.fixed = torch.as_tensor(fixed, dtype=torch.bool, device=self.device)
        self._refresh_index_tensors()

        self._edges_valid = False
        self._nl_positions = None

    # -- membership bookkeeping ------------------------------------------ #
    def _refresh_index_tensors(self):
        torch = self.torch
        n = len(self._natoms)
        self.n = n
        self.natoms = torch.as_tensor(self._natoms, dtype=self.dtype, device=self.device)
        self.ptr_np = np.concatenate([[0], np.cumsum(self._natoms)]).astype(np.int64)
        bidx = np.repeat(np.arange(n), self._natoms)
        self.batch_index = torch.as_tensor(bidx, dtype=torch.long, device=self.device)
        self.seg = _Segments(self._natoms if n else np.zeros(1, dtype=np.int64),
                             self.device, self.dtype)
        self.n_atoms_total = int(self._natoms.sum())
        if self.is_torch_engine:
            self._node_attrs = self.torch.as_tensor(
                self.engine.node_attrs(self._numbers), dtype=self.dtype,
                device=self.device)

    def select(self, keep: np.ndarray):
        """Restrict the batch to `keep` (boolean or index array) in place."""
        keep = np.asarray(keep)
        if keep.dtype == bool:
            keep = np.flatnonzero(keep)
        if len(keep) == len(self._natoms):
            return
        node_keep = np.concatenate(
            [np.arange(self.ptr_np[i], self.ptr_np[i + 1]) for i in keep]
        ) if len(keep) else np.zeros(0, dtype=np.int64)
        torch = self.torch
        nk = torch.as_tensor(node_keep, dtype=torch.long, device=self.device)
        sk = torch.as_tensor(keep, dtype=torch.long, device=self.device)

        self.positions = self.positions[nk].contiguous()
        self.cells = self.cells[sk].contiguous()
        self.fixed = self.fixed[nk].contiguous()
        self.systems = [self.systems[i] for i in keep]
        self.index = self.index[keep]
        self._natoms = self._natoms[keep]
        self._numbers = self._numbers[node_keep]
        self._pbc = self._pbc[keep]
        self._refresh_index_tensors()
        self._edges_valid = False

    # -- geometry export -------------------------------------------------- #
    def positions_np(self) -> np.ndarray:
        return to_np64(self.positions)

    def cells_np(self) -> np.ndarray:
        return to_np64(self.cells)

    def structure_positions(self, k: int) -> np.ndarray:
        p = self.positions_np()
        return p[self.ptr_np[k]:self.ptr_np[k + 1]]

    # -- neighbour lists --------------------------------------------------- #
    def displacement_budget(self):
        """How far atoms may drift before the skinned list must be rebuilt."""
        return 0.5 * self.skin

    def rebuild(self):
        """(Re)build neighbour lists at r_max + skin and re-plan the chunks."""
        if not self.is_torch_engine:
            self._edges_valid = True
            return
        pos = self.positions_np()
        cells = self.cells_np()
        cutoff = self.engine.r_max + self.skin

        senders, receivers, shifts, counts = [], [], [], []
        for k in range(self.n):
            a, b = self.ptr_np[k], self.ptr_np[k + 1]
            i, j, S = pair_list(pos[a:b], cells[k], self._pbc[k], cutoff)
            senders.append(i + a)
            receivers.append(j + a)
            shifts.append(S)
            counts.append(len(i))
        counts = np.asarray(counts, dtype=np.int64)
        ei = np.stack([np.concatenate(senders), np.concatenate(receivers)]) \
            if self.n else np.zeros((2, 0), dtype=np.int64)
        us = np.concatenate(shifts) if self.n else np.zeros((0, 3))
        eptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

        self.chunks = self.engine.plan(self._natoms, counts)
        # Keep the neighbour list on the host (plus one small descriptor per
        # chunk).  The per-chunk *device* tensors are built lazily in
        # `_build_job` so `compute()` can stream them (build one chunk, run it,
        # free it) instead of holding the whole system's edge tensors on device
        # at once -- that is what lowered the GPU memory floor for wide NEB /
        # SAGA batches (see NEB.md, the OOM section).
        self._ei = ei
        self._us = us
        self._eptr = eptr
        self._counts = counts
        self._chunk_desc = []
        for cid, (s0, s1) in enumerate(self.chunks):
            self._chunk_desc.append(dict(
                cid=cid, dev=self.engine.device_for(cid), s0=s0, s1=s1,
                n0=int(self.ptr_np[s0]), n1=int(self.ptr_np[s1]),
                e0=int(eptr[s0]), e1=int(eptr[s1])))
        self._edges_valid = True
        self._nl_positions = self.positions.detach().clone()

    def _build_job(self, cd, need_virial):
        """Build one chunk's forward-pass inputs as device tensors.

        The streaming split: this method stays on ``Batch`` (it owns the
        concatenated positions/cells/node-attrs and the host-side neighbour
        list ``_ei``/``_us``/``_eptr``), while the *runner*
        (``MaceEngine._run_job`` / ``run_chunks``) stays on the engine.  A
        single call builds exactly one chunk, so ``compute()`` can build ->
        run -> free one chunk at a time instead of holding the whole system's
        edge tensors on device at once.
        """
        torch = self.torch
        s0, s1 = cd["s0"], cd["s1"]
        n0, n1 = cd["n0"], cd["n1"]
        e0, e1 = cd["e0"], cd["e1"]
        dev = cd["dev"]

        local_ei = torch.as_tensor(self._ei[:, e0:e1] - n0, dtype=torch.long,
                                   device=dev)
        local_us = torch.as_tensor(self._us[e0:e1], dtype=self.dtype, device=dev)
        local_eg = torch.as_tensor(
            np.repeat(np.arange(s1 - s0), self._counts[s0:s1]),
            dtype=torch.long, device=dev)
        local_batch = torch.as_tensor(
            np.repeat(np.arange(s1 - s0), self._natoms[s0:s1]),
            dtype=torch.long, device=dev)
        local_ptr = torch.as_tensor(
            np.concatenate([[0], np.cumsum(self._natoms[s0:s1])]),
            dtype=torch.long, device=dev)
        attrs = self._node_attrs[n0:n1].to(dev)
        head = None
        if self.engine.multi_head:
            head = torch.full((s1 - s0,), self.engine.head_index,
                              dtype=torch.long, device=dev)

        pos = self.positions[n0:n1].to(dev, copy=True).requires_grad_(True)
        cells = self.cells[s0:s1].to(dev)
        shifts = torch.einsum("ei,eij->ej", local_us, cells[local_eg])
        with torch.no_grad():
            d = pos[local_ei[1]] - pos[local_ei[0]] + shifts
            keep = d.pow(2).sum(-1) < self.engine.r_max * self.engine.r_max
        local_ei = local_ei[:, keep]
        local_us = local_us[keep]
        shifts = shifts[keep]
        data = {
            "positions": pos,
            "node_attrs": attrs,
            "edge_index": local_ei,
            "shifts": shifts,
            "unit_shifts": local_us,
            "cell": cells.reshape(-1, 3),
            "batch": local_batch,
            "ptr": local_ptr,
        }
        if head is not None:
            data["head"] = head
        return (cd["cid"], data, need_virial)

    # -- the forward pass -------------------------------------------------- #
    def compute(self, need_virial: bool = False):
        """Energies (n,), forces (N,3) and optionally the virial (n,3,3)."""
        if not self.is_torch_engine:
            E, F, S = self.engine.compute(
                self.systems, self.positions_np(), self.cells_np(), need_virial)
            torch = self.torch
            return (torch.as_tensor(E, dtype=self.dtype, device=self.device),
                    torch.as_tensor(F, dtype=self.dtype, device=self.device),
                    torch.as_tensor(S, dtype=self.dtype, device=self.device)
                    if need_virial else None)

        torch = self.torch
        if not self._edges_valid:
            self.rebuild()

        # Streaming.  With a single device, build -> run -> free one chunk at a
        # time so the resident inputs are bounded by the chunk, not by the whole
        # system (the previous chunk's tensors are released by refcounting
        # before the next is built).  With several devices the chunk tensors are
        # spread across them and holding them all is what lets the pool run
        # chunks concurrently, so there we build all jobs up front.
        outs = []
        if len(self.engine.devices) <= 1:
            for cd in self._chunk_desc:
                job = self._build_job(cd, need_virial)
                outs.append(self.engine.run_chunks([job])[0])
                del job
        else:
            jobs = [self._build_job(cd, need_virial) for cd in self._chunk_desc]
            outs = self.engine.run_chunks(jobs)

        E = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        F = torch.zeros_like(self.positions)
        S = (torch.zeros(self.n, 3, 3, dtype=self.dtype, device=self.device)
             if need_virial else None)
        for cd, (e, f, s) in zip(self._chunk_desc, outs):
            E[cd["s0"]:cd["s1"]] = e.to(self.device, self.dtype)
            F[cd["n0"]:cd["n1"]] = f.to(self.device, self.dtype)
            if need_virial and s is not None:
                S[cd["s0"]:cd["s1"]] = s.to(self.device, self.dtype)
        return E, F, S


def make_engine(config, keep_z: Optional[Sequence[int]] = None):
    """Build the engine described by `config` (MACE on device, or ASE/EMT)."""
    devices = resolve_devices(getattr(config, "device", "auto"))
    dtype = resolve_dtype(getattr(config, "precision", "auto"), devices)
    tune_backend(devices, dtype)
    if config.model_path:
        return MaceEngine(
            config.model_path, devices, dtype,
            max_batch_atoms=getattr(config, "max_batch_atoms", 6144),
            max_batch_edges=getattr(config, "max_batch_edges", 600_000),
            enable_cueq=getattr(config, "enable_cueq", True),
            keep_z=keep_z if getattr(config, "restrict_elements", True) else None,
        )

    def factory():
        from ase.calculators.emt import EMT
        return EMT()

    return AseEngine(factory, devices, dtype,
                     n_workers=getattr(config, "cpu_workers", 0))


def autotune_batch_atoms(engine, sample: Sequence[Sys],
                         widths=(1, 2, 4, 8, 16, 32, 64, 128, 256),
                         repeat: int = 3, tolerance: float = 0.05,
                         need_virial: bool = False,
                         verbose: bool = True) -> int:
    """Measure structures/second vs batch width and pick the best width.

    Wider is *usually* better, but not monotonically: MACE's
    `SymmetricContraction` bakes an `opt_einsum` contraction order chosen for a
    10-atom example, so some batch widths can be slower than smaller ones.
    Rather than hard-code a guess per device, we measure on the actual
    structures this run will produce and keep the widest width within
    `tolerance` of the best observed throughput.

    Two details matter for GPU memory:

    * The configured `engine.max_batch_atoms` is a hard **ceiling** — autotune
      only ever *lowers* it, never raises it above what the user asked for, so
      `--max-batch-atoms` is respected (previously it was overwritten).
    * When `need_virial` is set, throughput is measured with the virial, which
      is the memory-heavy slab-relaxation pass. Measuring without it (the
      probe pass) under-counts peak memory and used to pick a width that then
      OOM'd on the first slab relaxation.

    Returns the chosen `max_batch_atoms` (also written onto the engine).
    """
    import gc

    import torch

    if not isinstance(engine, MaceEngine) or not sample:
        return getattr(engine, "max_batch_atoms", 0)
    ceiling = int(engine.max_batch_atoms)      # user's memory budget (hard cap)
    per_struct = max(1, int(np.mean([len(s.numbers) for s in sample])))
    max_structs = max(1, ceiling // per_struct)
    pool = [sample[i % len(sample)] for i in range(max(widths))]

    def sync():
        for d in engine.devices:
            if d.type == "cuda":
                torch.cuda.synchronize(d)

    def free():
        # Recover the allocator state after an OOM so the following real
        # relaxation does not inherit a fragmented / near-full cache.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    best_rate, results, oom = 0.0, [], False
    for w in widths:
        if w > max_structs:
            break
        try:
            batch = Batch(engine, pool[:w])
            batch.rebuild()
            batch.compute(need_virial=need_virial)   # warm-up + worst-case memory
            sync()
            t0 = time.perf_counter()
            for _ in range(repeat):
                batch.compute(need_virial=need_virial)
            sync()
            rate = w * repeat / (time.perf_counter() - t0)
        except Exception as e:                      # OOM (or other) -> stop widening
            if "out of memory" in str(e).lower():
                oom = True
                free()
            break
        results.append((w, rate))
        best_rate = max(best_rate, rate)
        # A collapse means we are past the useful width; no point going wider.
        if rate < 0.5 * best_rate:
            break

    if not results:
        # Even a single structure did not fit; the first relaxation will raise
        # the real OOM with actionable guidance. Give the smallest budget.
        engine.max_batch_atoms = per_struct
        if verbose:
            print(f"  autotune: OOM at 1 structure -> max_batch_atoms="
                  f"{per_struct} (will retry with the smallest batch)")
        return engine.max_batch_atoms

    good = [w for w, r in results if r >= (1.0 - tolerance) * best_rate]
    chosen = max(good) * per_struct
    engine.max_batch_atoms = min(ceiling, max(per_struct, chosen))
    if verbose:
        table = "  ".join(f"{w}:{r:.0f}/s" for w, r in results)
        n_struct = engine.max_batch_atoms // per_struct
        note = "  (memory-limited)" if oom else ""
        print(f"  autotune: {table}  ->  max_batch_atoms="
              f"{engine.max_batch_atoms} (~{n_struct} structures){note}")
    return engine.max_batch_atoms


def describe_engine(engine) -> str:
    if isinstance(engine, MaceEngine):
        dt = str(engine.dtype).replace("torch.", "")
        devs = ",".join(str(d) for d in engine.devices)
        el = (f" Z={engine.n_elements}/{engine.n_elements_full}"
              if engine.restricted_to else f" Z={engine.n_elements}")
        cueq = " cueq" if getattr(engine, "cueq_active", False) else ""
        return (f"MACE[{devs} {dt} r_max={engine.r_max:g}{el}"
                f" nl={_NL_BACKEND}{cueq}]")
    return "EMT[cpu]"

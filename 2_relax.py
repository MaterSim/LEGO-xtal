#!/usr/bin/env python3
"""Juliette TiO2 relaxation v3.4 for floating-octahedra v33 output.

Pipeline:
    v33 ranked CIF ingestion (raw CIF is authoritative)
    -> exact PyXtal-representation deduplication
    -> raw SO3 diagnostic
    -> optional full-representation single-rutile SO3 refinement (lattice + Wyckoff coordinates)
    -> raw/SO3 displacement and chemistry comparison
    -> final-stage StructureMatcher deduplication on raw/SO3 structures
    -> strict training-database overlap comparison

No tabular wp/x/y/z decoder or GULP/ReaxFF relaxation is used. The finalized
v33 files ``pre_joint_tio2/sample_XXXXXX.cif`` are consumed directly and
matched to ``floating_builder_selected.csv`` by zero-based sample index /
one-based ``final_rank``. SO3 similarity is diagnostic/refinement only and is
never used to rank distinct candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
from multiprocessing import Pool
import os
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from time import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment, minimize
from ase import Atoms
from ase.io import read as ase_read, write as ase_write
from ase.db import connect
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from lego.builder import builder
from lego.util import calculate_S


CHEMISTRY_CUTOFF = 5.0
TIO_CUTOFF = 3.0
ANGLE_BINS = np.linspace(0.0, 180.0, 10)
OO_BINS = np.linspace(0.0, 6.0, 13)
SHIFTS = np.asarray(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=float,
)
ZERO_SHIFT = int(np.flatnonzero(np.all(SHIFTS == 0, axis=1))[0])


# -----------------------------------------------------------------------------
# Resources
# -----------------------------------------------------------------------------

def _cpu_affinity_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def resolve_ncpu(requested: int | None, reserve_for_scheduler: int = 1) -> int:
    affinity = _cpu_affinity_count()
    slurm_raw = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        allocated = int(slurm_raw) if slurm_raw is not None else affinity
    except ValueError:
        allocated = affinity
    allocated = max(1, min(allocated, affinity))
    requested = 0 if requested is None else int(requested)
    if requested < 0:
        raise ValueError("--ncpu cannot be negative")
    if requested == 0:
        return max(1, allocated - max(0, int(reserve_for_scheduler)))
    return max(1, min(requested, allocated))



def set_worker_thread_limits() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    torch.set_num_threads(1)



# -----------------------------------------------------------------------------
# v33 output ingestion and exact pre-SO3 deduplication
# -----------------------------------------------------------------------------

def _metadata_value(meta: dict, key: str):
    value = meta.get(key)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def load_v33_candidates(
    generation_dir: Path,
    cif_dir: Path | None,
    selected_metrics: Path | None,
):
    generation_dir = generation_dir.resolve()
    cif_dir = (cif_dir or generation_dir / "pre_joint_tio2").resolve()
    selected_metrics = (
        selected_metrics or generation_dir / "floating_builder_selected.csv"
    ).resolve()

    if not cif_dir.is_dir():
        raise FileNotFoundError(cif_dir)
    if not selected_metrics.is_file():
        raise FileNotFoundError(selected_metrics)

    meta_df = pd.read_csv(selected_metrics)
    if "final_rank" not in meta_df.columns:
        raise ValueError(
            f"{selected_metrics} has no final_rank column and cannot be mapped "
            "to v33 sample_XXXXXX.cif files"
        )
    if meta_df["final_rank"].duplicated().any():
        duplicates = sorted(
            meta_df.loc[meta_df["final_rank"].duplicated(False), "final_rank"]
            .astype(int)
            .unique()
            .tolist()
        )
        raise ValueError(f"Duplicate final_rank values in selected metadata: {duplicates}")

    rows_by_rank = {
        int(row["final_rank"]): row.to_dict()
        for _, row in meta_df.iterrows()
    }

    cif_paths = sorted(cif_dir.glob("sample_*.cif"))
    if not cif_paths:
        raise FileNotFoundError(f"No sample_*.cif files found in {cif_dir}")

    candidates = []
    failures = []
    for cif_path in cif_paths:
        stem = cif_path.stem
        try:
            sample_index = int(stem.rsplit("_", 1)[1])
        except Exception:
            failures.append(
                {
                    "cif_path": str(cif_path),
                    "failure_stage": "filename",
                    "error": "Cannot parse zero-based sample index from CIF filename",
                }
            )
            continue

        final_rank = sample_index + 1
        meta = rows_by_rank.get(final_rank)
        if meta is None:
            failures.append(
                {
                    "cif_path": str(cif_path),
                    "source_row": sample_index,
                    "final_rank": final_rank,
                    "failure_stage": "metadata",
                    "error": "No matching final_rank in floating_builder_selected.csv",
                }
            )
            continue

        try:
            atoms = ase_read(cif_path)
            xtal = pyxtal()
            xtal.from_seed(atoms)
            if xtal is None or not xtal.valid or not xtal.atom_sites:
                raise ValueError("PyXtal symmetry reconstruction returned an invalid structure")
            xtal.tag = {
                "source_row": int(sample_index),
                "final_rank": int(final_rank),
                "candidate_id": _metadata_value(meta, "candidate_id"),
                "raw_cif_path": str(cif_path),
            }
            detected_spg = int(xtal.group.number)
            generated_spg = int(meta["spg"]) if _metadata_value(meta, "spg") is not None else None
            candidates.append(
                {
                    "source_row": int(sample_index),
                    "final_rank": int(final_rank),
                    "candidate_id": _metadata_value(meta, "candidate_id"),
                    "cif_path": cif_path,
                    "atoms": atoms,
                    "xtal": xtal,
                    "meta": meta,
                    "generated_spg": generated_spg,
                    "detected_spg": detected_spg,
                    "spg_agrees": (
                        None if generated_spg is None else bool(generated_spg == detected_spg)
                    ),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "cif_path": str(cif_path),
                    "source_row": sample_index,
                    "final_rank": final_rank,
                    "failure_stage": "cif_ingestion",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    candidates.sort(key=lambda item: item["source_row"])
    return candidates, failures, selected_metrics, cif_dir


def representation_key(xtal: pyxtal, decimals: int) -> str:
    x = np.round(np.asarray(xtal.get_1d_rep_x(), dtype=float), decimals).tolist()
    site_signature = [
        (str(site.specie), site.wp.get_label()) for site in xtal.atom_sites
    ]
    payload = {
        "spg": int(xtal.group.number),
        "sites": site_signature,
        "x": x,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def deduplicate_raw_candidates(candidates: list[dict], decimals: int):
    groups = defaultdict(list)
    for item in candidates:
        groups[representation_key(item["xtal"], decimals)].append(item)

    unique = []
    duplicates = []
    for key, members in groups.items():
        members.sort(key=lambda item: item["source_row"])
        representative = members[0]
        source_rows = [int(item["source_row"]) for item in members]
        representative = dict(representative)
        representative["source_rows"] = source_rows
        representative["generation_count"] = len(source_rows)
        representative["dedup_key"] = key
        representative["xtal"].tag = {
            **deepcopy(getattr(representative["xtal"], "tag", {}) or {}),
            "representative_source_row": int(representative["source_row"]),
            "source_rows": source_rows,
            "generation_count": len(source_rows),
            "dedup_key": key,
        }
        unique.append(representative)
        for duplicate in members[1:]:
            duplicates.append(
                {
                    "source_row": int(duplicate["source_row"]),
                    "representative_source_row": int(representative["source_row"]),
                    "status": "duplicate_before_so3",
                    "dedup_key": key,
                }
            )

    unique.sort(key=lambda item: item["source_row"])
    return unique, duplicates


def same_order_movement_metrics(reference_atoms: Atoms, stage_atoms: Atoms) -> dict:
    """Measure internal motion and cell change.

    For SO3, atom order is representation-preserving and the exact same-order
    comparison is used.  A species-block Hungarian fallback is available for
    relaxed structures whose writer reordered atoms.  Internal displacement is
    measured after mapping both fractional coordinate sets through the raw cell,
    so cell strain is reported separately rather than folded into atomic motion.
    """
    if len(reference_atoms) != len(stage_atoms):
        raise ValueError(
            f"atom count changed: {len(reference_atoms)} -> {len(stage_atoms)}"
        )

    ref_symbols = np.asarray(reference_atoms.get_chemical_symbols(), dtype=object)
    stage_symbols = np.asarray(stage_atoms.get_chemical_symbols(), dtype=object)
    if Counter(ref_symbols.tolist()) != Counter(stage_symbols.tolist()):
        raise ValueError("chemical composition changed")

    ref_frac = np.asarray(reference_atoms.get_scaled_positions(wrap=True), dtype=float)
    stage_frac = np.asarray(stage_atoms.get_scaled_positions(wrap=True), dtype=float)
    ref_cell = np.asarray(reference_atoms.cell.array, dtype=float)
    stage_cell = np.asarray(stage_atoms.cell.array, dtype=float)

    if np.array_equal(ref_symbols, stage_symbols):
        stage_ordered = stage_frac
        assignment_mode = "same_order"
    else:
        stage_ordered = np.zeros_like(ref_frac)
        assignment_mode = "species_hungarian"
        for symbol in sorted(set(ref_symbols.tolist())):
            ref_ids = np.flatnonzero(ref_symbols == symbol)
            stage_ids = np.flatnonzero(stage_symbols == symbol)
            delta = (
                stage_frac[stage_ids][None, :, None, :]
                - ref_frac[ref_ids][:, None, None, :]
                + SHIFTS[None, None, :, :]
            )
            cart = np.einsum("...i,ij->...j", delta, ref_cell)
            cost = np.linalg.norm(cart, axis=-1).min(-1)
            rows, cols = linear_sum_assignment(cost)
            stage_ordered[ref_ids[rows]] = stage_frac[stage_ids[cols]]

    delta = stage_ordered - ref_frac
    delta -= np.round(delta)
    displacement = delta @ ref_cell
    norms = np.linalg.norm(displacement, axis=1)

    deformation = stage_cell @ np.linalg.inv(ref_cell)
    strain_like = deformation - np.eye(3)
    ref_volume = abs(float(np.linalg.det(ref_cell)))
    stage_volume = abs(float(np.linalg.det(stage_cell)))

    return {
        "assignment_mode": assignment_mode,
        "rms_displacement_A": float(np.sqrt(np.mean(norms**2))),
        "mean_displacement_A": float(np.mean(norms)),
        "max_displacement_A": float(np.max(norms)),
        "cell_deformation_frobenius": float(np.linalg.norm(strain_like)),
        "relative_volume_change": float(
            (stage_volume - ref_volume) / max(ref_volume, 1e-12)
        ),
    }


# -----------------------------------------------------------------------------
# Chemistry descriptors
# -----------------------------------------------------------------------------

def _stable_logistic_switch(distances, cutoff, width):
    scale = max(float(width), 0.03)
    argument = np.clip(
        (np.asarray(distances, dtype=float) - float(cutoff)) / scale,
        -60.0,
        60.0,
    )
    return 1.0 / (1.0 + np.exp(argument))


def periodic_neighbor_vectors(frac, cell):
    frac = np.asarray(frac, dtype=float)
    delta = frac[:, None, None, :] - frac[None, :, None, :] + SHIFTS[None, None, :, :]
    cart = np.einsum("...i,ij->...j", delta, cell)
    dist = np.linalg.norm(cart, axis=-1)
    ids = np.arange(len(frac))
    dist[ids, ids, ZERO_SHIFT] = np.inf
    vectors, distances = [], []
    for i in range(len(frac)):
        d = dist[i].reshape(-1)
        v = cart[i].reshape(-1, 3)
        mask = np.isfinite(d) & (d > 1e-6)
        order = np.argsort(d[mask])
        distances.append(d[mask][order])
        vectors.append(v[mask][order])
    return distances, vectors


def periodic_cross_vectors(center_frac, neighbor_frac, cell):
    center_frac = np.asarray(center_frac, dtype=float)
    neighbor_frac = np.asarray(neighbor_frac, dtype=float)
    delta = (
        neighbor_frac[None, :, None, :]
        - center_frac[:, None, None, :]
        + SHIFTS[None, None, :, :]
    )
    cart = np.einsum("...i,ij->...j", delta, cell)
    dist = np.linalg.norm(cart, axis=-1)
    image = np.argmin(dist, axis=-1)
    rows = np.arange(len(center_frac))[:, None]
    cols = np.arange(len(neighbor_frac))[None, :]
    return dist[rows, cols, image], cart[rows, cols, image]


def _soft_histogram(values, weights, bins):
    centers = 0.5 * (bins[:-1] + bins[1:])
    sigma = max(float(np.diff(bins).mean()) * 0.45, 0.08)
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0 or weights.sum() <= 1e-12:
        return np.zeros(len(centers), dtype=float)
    hist = (
        weights[:, None]
        * np.exp(-0.5 * ((values[:, None] - centers[None, :]) / sigma) ** 2)
    ).sum(0)
    return hist / max(hist.sum(), 1e-12)


def soft_angle_hist(distances, vectors, cutoff, width, bins=ANGLE_BINS):
    centers = 0.5 * (bins[:-1] + bins[1:])
    sigma = max(float(np.diff(bins).mean()) * 0.45, 2.0)
    histogram = np.zeros(len(centers), dtype=float)
    for d, v in zip(distances, vectors):
        weights = _stable_logistic_switch(d, cutoff, width)
        keep = np.flatnonzero(weights > 0.05)
        for ii in range(len(keep)):
            i = keep[ii]
            ni = np.linalg.norm(v[i])
            for jj in range(ii + 1, len(keep)):
                j = keep[jj]
                nj = np.linalg.norm(v[j])
                if ni <= 1e-10 or nj <= 1e-10:
                    continue
                angle = np.degrees(
                    np.arccos(np.clip(np.dot(v[i], v[j]) / (ni * nj), -1, 1))
                )
                histogram += (
                    weights[i]
                    * weights[j]
                    * np.exp(-0.5 * ((centers - angle) / sigma) ** 2)
                )
    return histogram / max(histogram.sum(), 1e-12)


def framework_descriptor(frac, cell, chemistry_cutoff=CHEMISTRY_CUTOFF):
    distances, vectors = periodic_neighbor_vectors(frac, cell)
    cutoff = float(chemistry_cutoff)
    shells = [d[d <= cutoff] for d in distances]
    shell_values = np.concatenate(shells) if shells else np.asarray([], dtype=float)
    if shell_values.size == 0:
        raise ValueError("No Ti neighbours inside the chemistry cutoff")
    cn_values = np.asarray([np.count_nonzero(d <= cutoff) for d in distances], dtype=float)
    nn_mean = float(np.mean(shell_values))
    nn_width = max(float(np.std(shell_values)), 0.03)
    angle = soft_angle_hist(distances, vectors, cutoff, max(nn_width, 0.08))
    nearest = np.asarray([d[0] for d in distances], dtype=float)
    volume = abs(float(np.linalg.det(cell)))
    return {
        "ti_ti_cn": float(np.mean(cn_values)),
        "ti_ti_mean": nn_mean,
        "ti_ti_width": nn_width,
        "minimum_ti_ti_distance": float(np.min(nearest)),
        "volume_per_ti": volume / len(frac),
        "ti_ti_angle_profile": angle,
    }


def tio2_environment_descriptor(ti_frac, o_frac, cell, cutoff=TIO_CUTOFF, smooth_width=None):
    if len(ti_frac) < 1 or len(o_frac) < 1:
        raise ValueError("Ti/O framework is empty")
    dist, vec = periodic_cross_vectors(ti_frac, o_frac, cell)
    hard = dist <= float(cutoff)
    ti_cn = hard.sum(1).astype(float)
    o_cn = hard.sum(0).astype(float)
    bond = dist[hard]
    hard_bond_shell_valid = bool(bond.size > 0)
    if hard_bond_shell_valid:
        hard_ti_o_mean = float(np.mean(bond))
        hard_ti_o_width = max(float(np.std(bond)), 0.03)
        inferred_smooth_width = max(hard_ti_o_width, 0.08)
    else:
        hard_ti_o_mean = float("nan")
        hard_ti_o_width = float("nan")
        inferred_smooth_width = max(float(np.min(dist)) * 0.10, 0.08)
    effective_smooth_width = (
        inferred_smooth_width if smooth_width is None else max(float(smooth_width), 0.03)
    )
    smooth = _stable_logistic_switch(dist, cutoff, effective_smooth_width)
    smooth_denom = max(float(smooth.sum()), 1e-12)
    proj_ti_cn = smooth.sum(1)
    proj_o_cn = smooth.sum(0)
    proj_mean = float((smooth * dist).sum() / smooth_denom)
    proj_var = float(
        (smooth * (dist - proj_mean) ** 2).sum() / smooth_denom
    )
    angle_values, angle_weights = [], []
    oo_values, oo_weights = [], []
    for i in range(len(ti_frac)):
        ids = np.flatnonzero(smooth[i] > 0.05)
        for a in range(len(ids)):
            j = ids[a]
            nj = np.linalg.norm(vec[i, j])
            for b in range(a + 1, len(ids)):
                k = ids[b]
                nk = np.linalg.norm(vec[i, k])
                if min(nj, nk) <= 1e-10:
                    continue
                w = smooth[i, j] * smooth[i, k]
                angle = np.degrees(
                    np.arccos(np.clip(np.dot(vec[i, j], vec[i, k]) / (nj * nk), -1, 1))
                )
                angle_values.append(angle)
                angle_weights.append(w)
                oo_values.append(np.linalg.norm(vec[i, j] - vec[i, k]))
                oo_weights.append(w)
    angle_profile = _soft_histogram(angle_values, angle_weights, ANGLE_BINS)
    oo_profile = _soft_histogram(oo_values, oo_weights, OO_BINS)
    oo_distances, _ = periodic_neighbor_vectors(o_frac, cell)
    min_oo = min(float(d[0]) for d in oo_distances)
    ti_distances, _ = periodic_neighbor_vectors(ti_frac, cell)
    min_titi = min(float(d[0]) for d in ti_distances)
    return {
        "ti_o_cn": float(np.mean(ti_cn)),
        "ti_o_cn_std": float(np.std(ti_cn)),
        "o_ti_cn": float(np.mean(o_cn)),
        "o_ti_cn_std": float(np.std(o_cn)),
        "ti_o_mean": hard_ti_o_mean,
        "ti_o_width": hard_ti_o_width,
        "hard_ti_o_shell_valid": hard_bond_shell_valid,
        "proj_ti_o_cn": float(np.mean(proj_ti_cn)),
        "proj_ti_o_cn_std": float(np.std(proj_ti_cn)),
        "proj_o_ti_cn": float(np.mean(proj_o_cn)),
        "proj_o_ti_cn_std": float(np.std(proj_o_cn)),
        "proj_ti_o_mean": proj_mean,
        "proj_ti_o_width": float(math.sqrt(max(proj_var, 1e-12))),
        "projection_smooth_width": float(effective_smooth_width),
        "minimum_ti_ti_distance": min_titi,
        "minimum_ti_o_distance": float(np.min(dist)),
        "minimum_o_o_distance": min_oo,
        "angle_profile": angle_profile,
        "shell_o_o_profile": oo_profile,
    }


def split_structure(atoms: Atoms):
    frac = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    ti_frac = frac[symbols == "Ti"]
    o_frac = frac[symbols == "O"]
    cell = np.asarray(atoms.cell.array, dtype=float)
    if len(ti_frac) < 1 or len(o_frac) != 2 * len(ti_frac):
        raise ValueError(f"structure is not TiO2: Ti={len(ti_frac)}, O={len(o_frac)}")
    return ti_frac, o_frac, cell


def descriptor_from_atoms(atoms: Atoms, cutoff: float, smooth_width: float | None = None):
    ti_frac, o_frac, cell = split_structure(atoms)
    framework = framework_descriptor(ti_frac, cell)
    chemistry = tio2_environment_descriptor(
        ti_frac, o_frac, cell, cutoff=cutoff, smooth_width=smooth_width
    )
    return {**framework, **chemistry}


def _jensen_shannon(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    ps, qs = float(p.sum()), float(q.sum())
    if ps <= 1e-15 and qs <= 1e-15:
        return 0.0
    if ps <= 1e-15 or qs <= 1e-15:
        return math.log(2.0)
    p, q = p / ps, q / qs
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# -----------------------------------------------------------------------------
# Single-reference SO3 diagnostic and native LEGO optimization
# -----------------------------------------------------------------------------

_REFERENCE_SO3_EVALUATOR = None


def make_reference_so3_evaluator(reference_tio2: Path, rcut: float):
    evaluator = builder.__new__(builder)
    evaluator.elements = ["Ti", "O"]
    evaluator.dim = 3
    evaluator.rank = 0
    evaluator.verbose = False
    evaluator.criteria = {}
    evaluator.calculator = None
    evaluator.ref_environments = None
    evaluator.reference_environment_bank = None
    evaluator.use_target_coordination = False
    evaluator.last_optimization_results = []
    evaluator.set_descriptor_calculator(mykwargs={"rcut": float(rcut)})
    evaluator.set_reference_enviroments(str(reference_tio2))
    return evaluator


def _init_reference_so3_worker(reference_tio2: str, rcut: float):
    global _REFERENCE_SO3_EVALUATOR
    set_worker_thread_limits()
    _REFERENCE_SO3_EVALUATOR = make_reference_so3_evaluator(
        Path(reference_tio2), float(rcut)
    )


def _reference_so3_diagnostic_worker(payload):
    source_row, stage, xtal = payload
    try:
        value = float(_REFERENCE_SO3_EVALUATOR.get_similarity(xtal))
        if not math.isfinite(value):
            raise ValueError("non-finite single-reference SO3 energy")
        return {
            "source_row": int(source_row),
            "stage": str(stage),
            "so3_reference_energy": value,
            "worker_pid": int(os.getpid()),
            "error": None,
        }
    except Exception as exc:
        return {
            "source_row": int(source_row),
            "stage": str(stage),
            "so3_reference_energy": math.nan,
            "worker_pid": int(os.getpid()),
            "error": f"{type(exc).__name__}: {exc}",
        }


def evaluate_reference_so3_stage(candidates, reference_tio2: Path, rcut: float, stage: str, ncpu: int):
    tasks = [(int(source_row), str(stage), xtal) for source_row, xtal in candidates]
    values = {}
    rows = []
    worker_tasks = Counter()
    first_error_printed = False
    progress_every = max(1, len(tasks) // 10) if tasks else 1
    initargs = (str(reference_tio2), float(rcut))

    if max(1, int(ncpu)) == 1:
        _init_reference_so3_worker(*initargs)
        iterator = map(_reference_so3_diagnostic_worker, tasks)
        pool = None
    else:
        pool = Pool(
            processes=max(1, int(ncpu)),
            initializer=_init_reference_so3_worker,
            initargs=initargs,
        )
        iterator = pool.imap_unordered(
            _reference_so3_diagnostic_worker,
            tasks,
            chunksize=1,
        )

    try:
        valid = 0
        for done, row in enumerate(iterator, start=1):
            rows.append(row)
            worker_tasks[int(row["worker_pid"])] += 1
            if row["error"] is None:
                valid += 1
                values[int(row["source_row"])] = float(row["so3_reference_energy"])
            elif not first_error_printed:
                print(
                    f"SO3 diagnostic [{stage}] first failure: "
                    f"source_row={row['source_row']}; {row['error']}",
                    flush=True,
                )
                first_error_printed = True
            if done % progress_every == 0 or done == len(tasks):
                print(
                    f"SO3 diagnostic [{stage}]: {done}/{len(tasks)}; "
                    f"valid={valid}; failed={done-valid}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(
        f"SO3 diagnostic [{stage}] worker tasks={dict(sorted(worker_tasks.items()))}",
        flush=True,
    )
    rows.sort(key=lambda row: int(row["source_row"]))
    return values, rows


_SITE_SO3_EVALUATOR = None
_SITE_SO3_NM_STEPS = 50
_SITE_SO3_LBFGS_STEPS = 150


def _site_reference_matrix(evaluator, xtal: pyxtal) -> np.ndarray:
    refs = np.asarray(evaluator.ref_environments, dtype=float)
    if refs.ndim == 1:
        refs = refs.reshape(1, -1)
    if refs.shape[0] != len(evaluator.elements):
        raise ValueError(
            "Element-reference count does not match evaluator elements: "
            f"{refs.shape[0]} versus {len(evaluator.elements)}"
        )
    by_element = {
        str(element): refs[index]
        for index, element in enumerate(evaluator.elements)
    }
    rows = []
    for index, site in enumerate(xtal.atom_sites):
        symbol = str(site.specie)
        if symbol not in by_element:
            raise ValueError(
                f"No SO3 reference for independent site {index} species {symbol!r}"
            )
        rows.append(by_element[symbol])
    if not rows:
        raise ValueError("No independent sites available for SO3 projection")
    return np.vstack(rows)


def _init_site_so3_worker(reference_tio2: str, rcut: float, nm_steps: int, lbfgs_steps: int):
    global _SITE_SO3_EVALUATOR, _SITE_SO3_NM_STEPS, _SITE_SO3_LBFGS_STEPS
    set_worker_thread_limits()
    _SITE_SO3_EVALUATOR = make_reference_so3_evaluator(
        Path(reference_tio2), float(rcut)
    )
    _SITE_SO3_NM_STEPS = int(nm_steps)
    _SITE_SO3_LBFGS_STEPS = int(lbfgs_steps)


def _full_so3_worker(payload):
    source_row, raw_xtal = payload
    source_row = int(source_row)
    try:
        xtal = raw_xtal.copy()
        xtal.tag = deepcopy(getattr(raw_xtal, "tag", {}) or {})
        x0 = np.asarray(xtal.get_1d_rep_x(), dtype=float)
        n_abc, n_ang = Lattice.get_dofs(xtal.lattice.ltype)
        lattice_dof = int(n_abc + n_ang)
        if lattice_dof > len(x0):
            raise ValueError(
                f"Lattice DOF {lattice_dof} exceeds 1D representation length {len(x0)}"
            )
        ref_matrix = _site_reference_matrix(_SITE_SO3_EVALUATOR, xtal)

        # Match PyXtal LEGO's native local-optimization variable space:
        # lattice lengths, lattice angles, then Wyckoff free coordinates.
        bounds = (
            [(1.5, 50.0)] * n_abc
            + [(30.0, 150.0)] * n_ang
            + [(0.0, 1.0)] * (len(x0) - lattice_dof)
        )
        if len(bounds) != len(x0):
            raise ValueError(
                f"SO3 bounds length {len(bounds)} != representation length {len(x0)}"
            )

        lower = np.asarray([b[0] for b in bounds], dtype=float)
        upper = np.asarray([b[1] for b in bounds], dtype=float)
        x_start = np.clip(x0, lower, upper)

        def objective(x):
            value = float(calculate_S(
                np.asarray(x, dtype=float),
                xtal,
                ref_matrix,
                _SITE_SO3_EVALUATOR.calculator,
            ))
            if not math.isfinite(value):
                return 1.0e300
            return value

        sim0 = float(objective(x_start))
        best_x = x_start.copy()
        best_sim = sim0
        evaluations = 1

        x = x_start.copy()
        for method, steps in (
            ("Nelder-Mead", _SITE_SO3_NM_STEPS),
            ("L-BFGS-B", _SITE_SO3_LBFGS_STEPS),
        ):
            if steps <= 0:
                continue
            res = minimize(
                objective,
                x,
                method=method,
                bounds=bounds,
                options={"maxiter": int(steps)},
            )
            evaluations += int(getattr(res, "nfev", 0) or 0)
            x = np.clip(np.asarray(res.x, dtype=float), lower, upper)
            stage_sim = float(objective(x))
            evaluations += 1
            if math.isfinite(stage_sim) and stage_sim < best_sim:
                best_sim = stage_sim
                best_x = x.copy()

        improved = bool(best_sim < sim0 - 1.0e-12)
        if improved:
            final_xtal = raw_xtal.copy()
            final_xtal.update_from_1d_rep(best_x)
            final_xtal.tag = deepcopy(getattr(raw_xtal, "tag", {}) or {})
        else:
            final_xtal = raw_xtal
            best_sim = sim0

        return {
            "source_row": source_row,
            "success": True,
            "initial_so3_reference_energy": sim0,
            "final_so3_reference_energy": float(best_sim),
            "improved": improved,
            "raw_fallback_used": bool(not improved),
            "lattice_dof": lattice_dof,
            "site_coordinate_dof": int(len(x0) - lattice_dof),
            "objective_evaluations": int(evaluations),
            "worker_pid": int(os.getpid()),
            "xtal": final_xtal,
            "error": None,
        }
    except Exception as exc:
        return {
            "source_row": source_row,
            "success": False,
            "initial_so3_reference_energy": math.nan,
            "final_so3_reference_energy": math.nan,
            "improved": False,
            "raw_fallback_used": True,
            "lattice_dof": math.nan,
            "site_coordinate_dof": math.nan,
            "objective_evaluations": 0,
            "worker_pid": int(os.getpid()),
            "xtal": raw_xtal,
            "error": f"{type(exc).__name__}: {exc}",
        }


def optimize_single_reference_so3(raw_candidates, reference_tio2: Path, rcut: float, ncpu: int, nm_steps: int, lbfgs_steps: int):
    """Optimize lattice and Wyckoff free coordinates against rutile SO3.

    The full native PyXtal 1D representation is optimized with the same local
    variable bounds used by PyXtal LEGO: lattice lengths 1.5--50 A, lattice
    angles 30--150 degrees, and Wyckoff free coordinates 0--1. Reference rows
    are assigned by independent-site species: Ti->rutile Ti and O->rutile O.
    """
    tasks = [(int(source_row), xtal) for source_row, xtal in raw_candidates]
    initargs = (str(reference_tio2), float(rcut), int(nm_steps), int(lbfgs_steps))
    if max(1, int(ncpu)) == 1:
        _init_site_so3_worker(*initargs)
        iterator = map(_full_so3_worker, tasks)
        pool = None
    else:
        pool = Pool(
            processes=max(1, int(ncpu)),
            initializer=_init_site_so3_worker,
            initargs=initargs,
        )
        iterator = pool.imap_unordered(_full_so3_worker, tasks, chunksize=1)

    rows = []
    projected_by_source = {}
    worker_tasks = Counter()
    first_error_printed = False
    success_count = 0
    progress_every = max(1, len(tasks) // 10) if tasks else 1
    try:
        for done, result in enumerate(iterator, start=1):
            worker_tasks[int(result["worker_pid"])] += 1
            source_row = int(result["source_row"])
            projected_by_source[source_row] = result.pop("xtal")
            row = {k: v for k, v in result.items() if k != "worker_pid"}
            rows.append(row)
            if bool(row["success"]):
                success_count += 1
            elif not first_error_printed:
                print(
                    f"Full-representation SO3 first failure: source_row={source_row}; {row['error']}",
                    flush=True,
                )
                first_error_printed = True
            if done % progress_every == 0 or done == len(tasks):
                print(
                    f"Full-representation SO3 refinement: {done}/{len(tasks)}; "
                    f"success={success_count}; failed={done-success_count}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    projected = [
        (int(source_row), projected_by_source[int(source_row)])
        for source_row, _ in raw_candidates
    ]
    rows.sort(key=lambda row: int(row["source_row"]))
    print(
        f"Full-representation SO3 worker tasks={dict(sorted(worker_tasks.items()))}",
        flush=True,
    )
    return projected, rows



def stage_metric_record(
    source_row: int,
    candidate_meta: dict,
    stage: str,
    atoms: Atoms,
    cutoff: float,
):
    record = {
        "source_row": int(source_row),
        "candidate_id": candidate_meta.get("candidate_id"),
        "final_rank": candidate_meta.get("final_rank"),
        "ranking_score": candidate_meta.get("ranking_score"),
        "stage": stage,
        "spg_detected": None,
        "volume_A3": float(abs(np.linalg.det(atoms.cell.array))),
        "chemistry_descriptor_evaluable": False,
        "chemistry_descriptor_reason": None,
    }
    try:
        desc = descriptor_from_atoms(atoms, cutoff)
        record["chemistry_descriptor_evaluable"] = bool(
            desc.get("hard_ti_o_shell_valid", True)
        )
        if not record["chemistry_descriptor_evaluable"]:
            record["chemistry_descriptor_reason"] = (
                f"No Ti-O neighbours within {float(cutoff):.3f} A"
            )
        record.update(
            {
                k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                for k, v in desc.items()
                if k not in {"angle_profile", "shell_o_o_profile", "ti_ti_angle_profile"}
            }
        )
    except Exception as exc:
        desc = None
        record["chemistry_descriptor_reason"] = f"{type(exc).__name__}: {exc}"
    try:
        tmp = pyxtal()
        tmp.from_seed(atoms)
        record["spg_detected"] = int(tmp.group.number)
    except Exception:
        pass
    if desc is not None and candidate_meta:
        angle_cols = [f"target_angle_bin_{i}" for i in range(len(ANGLE_BINS) - 1)]
        oo_cols = [f"target_shell_o_o_bin_{i}" for i in range(len(OO_BINS) - 1)]
        if all(c in candidate_meta and pd.notna(candidate_meta[c]) for c in angle_cols):
            target = np.asarray([candidate_meta[c] for c in angle_cols], dtype=float)
            record["angle_jsd_to_generation_target"] = _jensen_shannon(target, desc["angle_profile"])
        if all(c in candidate_meta and pd.notna(candidate_meta[c]) for c in oo_cols):
            target = np.asarray([candidate_meta[c] for c in oo_cols], dtype=float)
            record["shell_o_o_jsd_to_generation_target"] = _jensen_shannon(target, desc["shell_o_o_profile"])
    return record


def _strip_reason_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[c for c in frame.columns if "reason" in str(c).lower()],
        errors="ignore",
    )
_MATCH_STRUCTURES = None
_MATCHER = None


def _init_structure_match_worker(structures, matcher_kwargs):
    global _MATCH_STRUCTURES, _MATCHER
    _MATCH_STRUCTURES = structures
    kwargs = dict(matcher_kwargs)
    primitive_cell = bool(kwargs.pop("primitive_cell", True))
    scale = bool(kwargs.pop("scale", True))
    attempt_supercell = bool(kwargs.pop("attempt_supercell", True))
    allow_subset = bool(kwargs.pop("allow_subset", False))
    _MATCHER = StructureMatcher(
        primitive_cell=primitive_cell,
        scale=scale,
        attempt_supercell=attempt_supercell,
        allow_subset=allow_subset,
        comparator=ElementComparator(),
        **kwargs,
    )


def _match_structure_pair(pair):
    i, j = pair
    try:
        matched = bool(_MATCHER.fit(_MATCH_STRUCTURES[i], _MATCH_STRUCTURES[j]))
        rms = max_dist = None
        if matched:
            distances = _MATCHER.get_rms_dist(_MATCH_STRUCTURES[i], _MATCH_STRUCTURES[j])
            if distances is not None:
                rms, max_dist = map(float, distances)
        return i, j, matched, rms, max_dist, None
    except Exception as exc:
        return i, j, False, None, None, f"{type(exc).__name__}: {exc}"


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def deduplicate_final_stage(
    final_candidates,
    candidate_summary: dict,
    output_dir: Path,
    ncpu: int,
    ltol: float,
    stol: float,
    angle_tol: float,
    volume_tol: float,
    chunksize: int,
):
    """Deduplicate the final raw/SO3 structures without an energy oracle.

    SO3 similarity to rutile is deliberately not used for representative
    selection or ranking. Within each StructureMatcher component, retain the
    candidate with the earliest v33 final_rank, then source_row. The final table
    preserves that v33 ordering after duplicate components are collapsed.
    """
    prepared = []
    failures = []
    adaptor = AseAtomsAdaptor()
    for source_row, xtal in final_candidates:
        source_row = int(source_row)
        try:
            atoms = xtal.to_ase(resort=False)
            structure = adaptor.get_structure(atoms)
            meta = candidate_summary[source_row]
            prepared.append(
                (
                    {
                        "source_row": source_row,
                        "v33_final_rank": int(meta["final_rank"]),
                        "candidate_id": meta.get("candidate_id"),
                        "final_stage": (
                            "so3"
                            if bool(meta.get("so3_refinement_applied", False))
                            else "raw"
                        ),
                        "so3_raw_energy": meta.get("so3_raw_energy"),
                        "so3_final_energy": meta.get("so3_projected_energy"),
                        "delta_so3_final_minus_raw": meta.get(
                            "delta_so3_projected_minus_raw"
                        ),
                    },
                    atoms,
                    structure,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "source_row": source_row,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    pd.DataFrame(failures).to_csv(
        output_dir / "final_stage_structure_failures.csv", index=False
    )
    if not prepared:
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "ranked_candidates.csv", index=False)
        return empty, empty, empty

    structures = [item[2] for item in prepared]
    pairs = []
    for i, (_, _, s_i) in enumerate(prepared):
        vpa_i = s_i.volume / max(s_i.num_sites, 1)
        formula_i = s_i.composition.reduced_formula
        for j in range(i + 1, len(prepared)):
            s_j = structures[j]
            if formula_i != s_j.composition.reduced_formula:
                continue
            if math.isfinite(volume_tol):
                vpa_j = s_j.volume / max(s_j.num_sites, 1)
                rel = abs(vpa_i - vpa_j) / max(vpa_i, vpa_j, 1e-12)
                if rel > volume_tol:
                    continue
            pairs.append((i, j))

    print(
        f"Final-stage StructureMatcher: {len(prepared)} structures; "
        f"{len(pairs)} candidate pairs; "
        f"{min(ncpu, max(len(pairs), 1))} worker(s)",
        flush=True,
    )
    uf = _UnionFind(len(prepared))
    match_rows = []
    matcher_kwargs = {"ltol": ltol, "stol": stol, "angle_tol": angle_tol}
    if pairs:
        worker_count = min(ncpu, len(pairs))
        progress_every = max(1, len(pairs) // 10)
        if worker_count == 1:
            _init_structure_match_worker(structures, matcher_kwargs)
            iterator = map(_match_structure_pair, pairs)
            pool = None
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(
                processes=worker_count,
                initializer=_init_structure_match_worker,
                initargs=(structures, matcher_kwargs),
            )
            iterator = pool.imap_unordered(
                _match_structure_pair, pairs, chunksize=max(1, chunksize)
            )
        try:
            for done, (i, j, matched, rms, max_dist, error) in enumerate(
                iterator, start=1
            ):
                if matched:
                    uf.union(i, j)
                match_rows.append(
                    {
                        "source_row_a": prepared[i][0]["source_row"],
                        "source_row_b": prepared[j][0]["source_row"],
                        "matched": matched,
                        "normalized_rms_distance": rms,
                        "normalized_max_distance": max_dist,
                        "error": error,
                    }
                )
                if done % progress_every == 0 or done == len(pairs):
                    print(
                        f"Final-stage structure matching: {done}/{len(pairs)}",
                        flush=True,
                    )
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    components = defaultdict(list)
    for i in range(len(prepared)):
        components[uf.find(i)].append(i)

    candidates_dir = output_dir / "candidates"
    if candidates_dir.exists():
        shutil.rmtree(candidates_dir)
    candidates_dir.mkdir(parents=True)

    retained = []
    duplicates = []
    for component_id, members in enumerate(components.values(), start=1):
        members.sort(
            key=lambda idx: (
                prepared[idx][0]["v33_final_rank"],
                prepared[idx][0]["source_row"],
            )
        )
        rep = members[0]
        record, atoms, structure = prepared[rep]
        retained.append(
            (
                {
                    **record,
                    "structure_match_component": component_id,
                    "post_refinement_multiplicity": len(members),
                },
                structure,
            )
        )
        for idx in members[1:]:
            duplicate = prepared[idx][0]
            duplicates.append(
                {
                    "structure_match_component": component_id,
                    "source_row": duplicate["source_row"],
                    "v33_final_rank": duplicate["v33_final_rank"],
                    "retained_source_row": record["source_row"],
                    "retained_v33_final_rank": record["v33_final_rank"],
                }
            )

    retained.sort(
        key=lambda item: (
            item[0]["v33_final_rank"],
            item[0]["source_row"],
        )
    )
    ranked = []
    for rank, (record, structure) in enumerate(retained, start=1):
        cif_path = candidates_dir / (
            f"rank_{rank:04d}_row_{record['source_row']}.cif"
        )
        structure.to(filename=str(cif_path), fmt="cif")
        ranked.append(
            {
                "rank": rank,
                **record,
                "cif_path": str(cif_path),
            }
        )

    match_df = pd.DataFrame(match_rows)
    duplicate_df = pd.DataFrame(duplicates)
    ranked_df = pd.DataFrame(ranked)
    match_df.to_csv(output_dir / "structure_match_pairs.csv", index=False)
    duplicate_df.to_csv(
        output_dir / "post_refinement_duplicates.csv", index=False
    )
    ranked_df.to_csv(output_dir / "ranked_candidates.csv", index=False)
    return ranked_df, duplicate_df, match_df



def export_unique_standard_vasp(
    ranked_df: pd.DataFrame,
    final_candidates,
    output_dir: Path,
    symprec: float = 0.1,
    angle_tolerance: float = 5.0,
):
    """Export all final-stage StructureMatcher-unique candidates as
    symmetrized standard conventional VASP structures.

    This export deliberately ignores training-set overlap. Structures are taken
    from the in-memory final raw/SO3 PyXtal objects, not reparsed from CIF.
    """
    export_dir = output_dir / "unique_standard_vasp"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    xtal_by_row = {int(source_row): xtal for source_row, xtal in final_candidates}
    adaptor = AseAtomsAdaptor()
    report_rows = []

    for _, row in ranked_df.iterrows():
        rank = int(row["rank"])
        source_row = int(row["source_row"])
        final_stage = str(row.get("final_stage", "unknown"))
        output_path = export_dir / (
            f"rank_{rank:04d}_row_{source_row:04d}_{final_stage}_std.vasp"
        )
        try:
            xtal = xtal_by_row[source_row]
            structure = adaptor.get_structure(xtal.to_ase(resort=False))
            input_analyzer = SpacegroupAnalyzer(
                structure, symprec=symprec, angle_tolerance=angle_tolerance
            )
            input_spg_number = int(input_analyzer.get_space_group_number())
            input_spg_symbol = str(input_analyzer.get_space_group_symbol())

            refined = input_analyzer.get_refined_structure()
            refined_analyzer = SpacegroupAnalyzer(
                refined, symprec=symprec, angle_tolerance=angle_tolerance
            )
            standard = refined_analyzer.get_conventional_standard_structure()
            standard = standard.get_sorted_structure(
                key=lambda site: (0 if site.specie.symbol == "Ti" else 1, site.specie.symbol)
            )
            standard_analyzer = SpacegroupAnalyzer(
                standard, symprec=symprec, angle_tolerance=angle_tolerance
            )
            standard_spg_number = int(standard_analyzer.get_space_group_number())
            standard_spg_symbol = str(standard_analyzer.get_space_group_symbol())

            comment = (
                f"TiO2 SG{standard_spg_number} {standard_spg_symbol} "
                f"rank={rank} source_row={source_row} stage={final_stage}"
            )
            Poscar(standard, comment=comment).write_file(str(output_path))
            report_rows.append(
                {
                    "rank": rank,
                    "source_row": source_row,
                    "final_stage": final_stage,
                    "status": "ok",
                    "output_vasp": str(output_path),
                    "input_spg_number": input_spg_number,
                    "input_spg_symbol": input_spg_symbol,
                    "standard_spg_number": standard_spg_number,
                    "standard_spg_symbol": standard_spg_symbol,
                    "input_natoms": int(structure.num_sites),
                    "standard_natoms": int(standard.num_sites),
                    "a_A": float(standard.lattice.a),
                    "b_A": float(standard.lattice.b),
                    "c_A": float(standard.lattice.c),
                    "alpha_deg": float(standard.lattice.alpha),
                    "beta_deg": float(standard.lattice.beta),
                    "gamma_deg": float(standard.lattice.gamma),
                    "volume_A3": float(standard.volume),
                    "error": None,
                }
            )
        except Exception as exc:
            report_rows.append(
                {
                    "rank": rank,
                    "source_row": source_row,
                    "final_stage": final_stage,
                    "status": "failed",
                    "output_vasp": str(output_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(output_dir / "unique_standard_vasp_report.csv", index=False)
    ok = int((report_df.get("status") == "ok").sum()) if not report_df.empty else 0
    failed = int(len(report_df) - ok)
    print(
        f"Unique standard VASP export: {ok}/{len(report_df)} written; failed={failed}; "
        f"directory={export_dir}",
        flush=True,
    )
    return report_df, export_dir

def _structure_space_group_number(structure: Structure) -> int | None:
    try:
        xtal = pyxtal()
        xtal.from_seed(AseAtomsAdaptor.get_atoms(structure))
        if xtal is not None and xtal.valid:
            return int(xtal.group.number)
    except Exception:
        pass
    return None


def compare_ranked_to_training_set_strict(
    ranked_df: pd.DataFrame,
    training_db_path: Path,
    output_dir: Path,
    ncpu: int,
    ltol: float,
    stol: float,
    angle_tol: float,
    volume_tol: float,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conservatively identify final candidates that reproduce training structures.

    This matcher is intentionally stricter than final-stage candidate deduplication:
    no uniform lattice scaling and no supercell search are allowed. Candidate and
    training structures must also have the same detected space group, primitive
    site count, reduced formula, and similar volume per atom before StructureMatcher
    is invoked.
    """
    output_columns = [
        "candidate_rank",
        "candidate_source_row",
        "training_db_row",
        "training_label",
        "candidate_space_group",
        "training_space_group",
        "candidate_primitive_sites",
        "training_primitive_sites",
        "candidate_volume_per_atom",
        "training_volume_per_atom",
        "relative_volume_per_atom_difference",
        "normalized_rms_distance",
        "normalized_max_distance",
        "error",
    ]
    if ranked_df.empty:
        match_df = pd.DataFrame(columns=output_columns)
        match_df.to_csv(output_dir / "training_set_matches.csv", index=False)
        return ranked_df, match_df

    candidate_structures = []
    candidate_meta = []
    for _, row in ranked_df.iterrows():
        try:
            structure = Structure.from_file(str(row["cif_path"]))
            primitive = structure.get_primitive_structure()
            candidate_structures.append(structure)
            candidate_meta.append(
                {
                    "rank": int(row["rank"]),
                    "source_row": int(row["source_row"]),
                    "space_group": _structure_space_group_number(structure),
                    "primitive_sites": int(primitive.num_sites),
                    "volume_per_atom": float(structure.volume / max(structure.num_sites, 1)),
                }
            )
        except Exception as exc:
            print(
                f"Cannot parse ranked candidate {row.get('rank')} for training overlap: "
                f"{type(exc).__name__}: {exc}"
            )

    training_structures = []
    training_meta = []
    adaptor = AseAtomsAdaptor()
    with connect(training_db_path) as db:
        for row in db.select():
            try:
                structure = adaptor.get_structure(row.toatoms())
                primitive = structure.get_primitive_structure()
            except Exception as exc:
                print(
                    f"Cannot load training DB row {row.id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            label = None
            for key in ("name", "label", "prototype", "pearson_symbol"):
                if hasattr(row, key):
                    value = getattr(row, key)
                    if value is not None:
                        label = str(value)
                        break
            spg = (
                int(row.space_group_number)
                if hasattr(row, "space_group_number")
                else _structure_space_group_number(structure)
            )
            training_structures.append(structure)
            training_meta.append(
                {
                    "training_db_row": int(row.id),
                    "training_label": label or f"row_{row.id}",
                    "space_group": spg,
                    "primitive_sites": int(primitive.num_sites),
                    "volume_per_atom": float(structure.volume / max(structure.num_sites, 1)),
                    "formula": structure.composition.reduced_formula,
                }
            )

    if not candidate_structures or not training_structures:
        annotated = ranked_df.copy()
        annotated["in_training_set"] = False
        annotated["training_match_count"] = 0
        annotated["training_db_rows_json"] = "[]"
        match_df = pd.DataFrame(columns=output_columns)
        annotated.to_csv(output_dir / "ranked_candidates.csv", index=False)
        match_df.to_csv(output_dir / "training_set_matches.csv", index=False)
        return annotated, match_df

    combined = candidate_structures + training_structures
    offset = len(candidate_structures)
    pairs = []
    pair_prefilter_meta = {}
    for i, structure in enumerate(candidate_structures):
        meta = candidate_meta[i]
        formula = structure.composition.reduced_formula
        for j, train_meta in enumerate(training_meta):
            if formula != train_meta["formula"]:
                continue
            if meta["space_group"] is None or train_meta["space_group"] is None:
                continue
            if meta["space_group"] != train_meta["space_group"]:
                continue
            if meta["primitive_sites"] != train_meta["primitive_sites"]:
                continue
            rel_volume = abs(
                meta["volume_per_atom"] - train_meta["volume_per_atom"]
            ) / max(meta["volume_per_atom"], train_meta["volume_per_atom"], 1e-12)
            if rel_volume > volume_tol:
                continue
            pair = (i, offset + j)
            pairs.append(pair)
            pair_prefilter_meta[pair] = rel_volume

    matcher_kwargs = {
        "ltol": float(ltol),
        "stol": float(stol),
        "angle_tol": float(angle_tol),
        "primitive_cell": True,
        "scale": False,
        "attempt_supercell": False,
        "allow_subset": False,
    }
    print(
        f"Strict training-set overlap: {len(candidate_structures)} candidates x "
        f"{len(training_structures)} training structures; {len(pairs)} strict-prefilter "
        f"pairs; scale=False, attempt_supercell=False, ltol={ltol}, stol={stol}, "
        f"angle_tol={angle_tol}, volume_tol={volume_tol}; "
        f"{min(ncpu, max(len(pairs), 1))} worker(s)"
    )

    raw_results = []
    if pairs:
        worker_count = min(ncpu, len(pairs))
        if worker_count == 1:
            _init_structure_match_worker(combined, matcher_kwargs)
            iterator = map(_match_structure_pair, pairs)
            for result in tqdm(iterator, total=len(pairs), desc="Strict training overlap"):
                raw_results.append(result)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=worker_count,
                initializer=_init_structure_match_worker,
                initargs=(combined, matcher_kwargs),
            ) as pool:
                iterator = pool.imap_unordered(
                    _match_structure_pair,
                    pairs,
                    chunksize=max(1, int(chunksize)),
                )
                for result in tqdm(
                    iterator,
                    total=len(pairs),
                    desc="Strict training overlap",
                ):
                    raw_results.append(result)

    matches_by_candidate = defaultdict(list)
    match_rows = []
    for i, combined_j, matched, rms, max_dist, error in raw_results:
        if not matched:
            continue
        j = combined_j - offset
        cmeta = candidate_meta[i]
        tmeta = training_meta[j]
        rel_volume = pair_prefilter_meta[(i, combined_j)]
        record = {
            "candidate_rank": cmeta["rank"],
            "candidate_source_row": cmeta["source_row"],
            "training_db_row": tmeta["training_db_row"],
            "training_label": tmeta["training_label"],
            "candidate_space_group": cmeta["space_group"],
            "training_space_group": tmeta["space_group"],
            "candidate_primitive_sites": cmeta["primitive_sites"],
            "training_primitive_sites": tmeta["primitive_sites"],
            "candidate_volume_per_atom": cmeta["volume_per_atom"],
            "training_volume_per_atom": tmeta["volume_per_atom"],
            "relative_volume_per_atom_difference": rel_volume,
            "normalized_rms_distance": rms,
            "normalized_max_distance": max_dist,
            "error": error,
        }
        match_rows.append(record)
        matches_by_candidate[cmeta["source_row"]].append(record)

    annotated = ranked_df.copy()
    annotations = []
    for _, row in annotated.iterrows():
        source_row = int(row["source_row"])
        records = matches_by_candidate.get(source_row, [])
        records.sort(
            key=lambda rec: (
                float("inf")
                if rec["normalized_rms_distance"] is None
                else rec["normalized_rms_distance"],
                rec["relative_volume_per_atom_difference"],
                rec["training_db_row"],
            )
        )
        best = records[0] if records else None
        annotations.append(
            {
                "in_training_set": bool(records),
                "training_match_count": len(records),
                "training_db_rows_json": json.dumps(
                    [rec["training_db_row"] for rec in records],
                    separators=(",", ":"),
                ),
                "best_training_db_row": None if best is None else best["training_db_row"],
                "best_training_label": None if best is None else best["training_label"],
                "best_training_rms": None if best is None else best["normalized_rms_distance"],
                "best_training_max_dist": None if best is None else best["normalized_max_distance"],
                "best_training_relative_volume_difference": (
                    None
                    if best is None
                    else best["relative_volume_per_atom_difference"]
                ),
            }
        )

    annotation_df = pd.DataFrame(annotations)
    annotated = pd.concat(
        [annotated.reset_index(drop=True), annotation_df],
        axis=1,
    )
    match_df = pd.DataFrame(match_rows, columns=output_columns)
    annotated.to_csv(output_dir / "ranked_candidates.csv", index=False)
    match_df.to_csv(output_dir / "training_set_matches.csv", index=False)
    return annotated, match_df




# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Juliette TiO2 raw/SO3 audit v3.3 for floating-octahedra v33 output"
        )
    )
    parser.add_argument(
        "--generation-dir",
        required=True,
        help=(
            "Completed v33 generation directory containing pre_joint_tio2 "
            "and floating_builder_selected.csv"
        ),
    )
    parser.add_argument(
        "--cif-dir",
        default=None,
        help=(
            "Override v33 ranked CIF directory; default: "
            "<generation-dir>/pre_joint_tio2"
        ),
    )
    parser.add_argument(
        "--selected-metrics",
        default=None,
        help=(
            "Override v33 selected metadata CSV; default: "
            "<generation-dir>/floating_builder_selected.csv"
        ),
    )
    parser.add_argument(
        "--reference-tio2",
        required=True,
        help="Single TiO2 SO3 reference, normally rutile.cif",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--ncpu", type=int, default=0)
    parser.add_argument("--ti-o-cutoff", type=float, default=TIO_CUTOFF)
    parser.add_argument("--rcut", type=float, default=2.4)
    parser.add_argument("--nm-steps", type=int, default=50)
    parser.add_argument("--lbfgs-steps", type=int, default=150)
    parser.add_argument("--skip-so3-refinement", action="store_true")
    parser.add_argument("--dedup-decimals", type=int, default=8)
    parser.add_argument("--dedup-ltol", type=float, default=0.20)
    parser.add_argument("--dedup-stol", type=float, default=0.30)
    parser.add_argument("--dedup-angle-tol", type=float, default=5.0)
    parser.add_argument("--dedup-volume-tol", type=float, default=0.20)
    parser.add_argument(
        "--standard-vasp-symprec",
        type=float,
        default=0.1,
        help="Symmetry tolerance for unique standard-conventional VASP export",
    )
    parser.add_argument(
        "--standard-vasp-angle-tolerance",
        type=float,
        default=5.0,
        help="Angle tolerance in degrees for unique standard-conventional VASP export",
    )
    parser.add_argument("--match-chunksize", type=int, default=1)
    parser.add_argument("--training-db", default=None)
    parser.add_argument("--training-match-ltol", type=float, default=0.05)
    parser.add_argument("--training-match-stol", type=float, default=0.08)
    parser.add_argument("--training-match-angle-tol", type=float, default=1.0)
    parser.add_argument("--training-match-volume-tol", type=float, default=0.08)
    return parser.parse_args()


def main():
    started = time()
    args = parse_args()
    if args.rcut <= 0:
        raise ValueError("--rcut must be positive")
    if args.nm_steps < 0 or args.lbfgs_steps < 0:
        raise ValueError("optimizer step counts cannot be negative")
    if not args.skip_so3_refinement and args.nm_steps + args.lbfgs_steps <= 0:
        raise ValueError(
            "at least one SO3 optimizer stage must have positive steps unless "
            "--skip-so3-refinement is used"
        )
    if args.begin < 0:
        raise ValueError("--begin cannot be negative")
    if args.end != -1 and args.end <= args.begin:
        raise ValueError("--end must be -1 or greater than --begin")

    ncpu = resolve_ncpu(args.ncpu)
    set_worker_thread_limits()

    generation_dir = Path(args.generation_dir)
    cif_dir = Path(args.cif_dir) if args.cif_dir is not None else None
    selected_metrics = (
        Path(args.selected_metrics) if args.selected_metrics is not None else None
    )
    reference_tio2 = Path(args.reference_tio2)
    if not generation_dir.is_dir():
        raise FileNotFoundError(generation_dir)
    if not reference_tio2.is_file():
        raise FileNotFoundError(reference_tio2)

    training_db_path = (
        Path(args.training_db) if args.training_db is not None else None
    )
    if training_db_path is not None and not training_db_path.is_file():
        raise FileNotFoundError(training_db_path)

    output_dir = Path(
        args.output_dir or f"{generation_dir.name}-relax-v3.4"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- Juliette TiO2 raw/SO3 audit v3.4 ---")
    print(f"v33 generation directory: {generation_dir}")
    print(f"Single SO3 reference: {reference_tio2}")
    print(f"Output directory: {output_dir}")
    print(
        f"Resolved resources: ncpu={ncpu}; "
        f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'unset')}; "
        f"CPU_affinity={_cpu_affinity_count()}"
    )
    print(
        "Input mode=v33 ranked CIFs; tabular decoder=REMOVED; GULP=REMOVED; "
        f"SO3 refinement={'OFF' if args.skip_so3_refinement else 'ON'}"
    )

    ingestion_start = time()
    candidates, ingestion_failures, selected_metrics_path, resolved_cif_dir = (
        load_v33_candidates(generation_dir, cif_dir, selected_metrics)
    )
    pd.DataFrame(ingestion_failures).to_csv(
        output_dir / "ingestion_failures.csv", index=False
    )
    if not candidates:
        raise RuntimeError("No v33 ranked CIFs survived direct ingestion")

    ingestion_rows = [
        {
            "source_row": item["source_row"],
            "final_rank": item["final_rank"],
            "candidate_id": item["candidate_id"],
            "cif_path": str(item["cif_path"]),
            "generated_spg": item["generated_spg"],
            "detected_spg": item["detected_spg"],
            "spg_agrees": item["spg_agrees"],
        }
        for item in candidates
    ]
    pd.DataFrame(ingestion_rows).to_csv(
        output_dir / "v33_cif_ingestion.csv", index=False
    )

    unique, duplicates = deduplicate_raw_candidates(
        candidates, args.dedup_decimals
    )
    pd.DataFrame(duplicates).to_csv(
        output_dir / "pre_so3_duplicates.csv", index=False
    )
    selected = unique[args.begin:] if args.end == -1 else unique[args.begin:args.end]
    if not selected:
        raise ValueError("selected unique candidate range is empty")
    ingestion_seconds = time() - ingestion_start

    print(f"Resolved v33 CIF directory: {resolved_cif_dir}")
    print(f"Resolved v33 metadata: {selected_metrics_path}")
    print(
        f"Ingested {len(candidates)} CIFs; exact unique={len(unique)}; "
        f"selected={len(selected)}; ingestion_failures={len(ingestion_failures)}"
    )

    raw_candidates = []
    raw_atoms_by_source = {}
    meta_by_source = {}
    candidate_summary = {}
    stage_rows = []
    movement_rows = []

    raw_cif_dir = output_dir / "raw_cifs"
    so3_cif_dir = output_dir / "so3_cifs"
    for directory in (raw_cif_dir, so3_cif_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    for item in selected:
        source_row = int(item["source_row"])
        xtal = item["xtal"]
        raw_atoms = item["atoms"]
        meta = item["meta"]
        raw_candidates.append((source_row, xtal))
        raw_atoms_by_source[source_row] = raw_atoms
        meta_by_source[source_row] = meta
        raw_cif_path = raw_cif_dir / (
            f"rank_{int(item['final_rank']):04d}_row_{source_row:04d}_raw.cif"
        )
        ase_write(str(raw_cif_path), raw_atoms, format="cif")
        candidate_summary[source_row] = {
            "source_row": source_row,
            "candidate_id": item["candidate_id"],
            "final_rank": item["final_rank"],
            "chemistry_score": _metadata_value(meta, "chemistry_score"),
            "total_loss": _metadata_value(meta, "total_loss"),
            "ti_fingerprint_q90": _metadata_value(meta, "ti_fingerprint_q90"),
            "generated_spg": item["generated_spg"],
            "raw_detected_spg": item["detected_spg"],
            "raw_spg_agrees_with_generation": item["spg_agrees"],
            "so3_success": None,
            "so3_refinement_applied": False,
            "failure_stage": None,
            "generation_count": item["generation_count"],
            "source_rows_json": json.dumps(
                item["source_rows"], separators=(",", ":")
            ),
            "dedup_key": item["dedup_key"],
            "v33_input_cif_path": str(item["cif_path"]),
            "raw_cif_path": str(raw_cif_path),
        }
        stage_rows.append(
            stage_metric_record(
                source_row, meta, "raw", raw_atoms, args.ti_o_cutoff
            )
        )

    diagnostic_rows = []
    diagnostic_start = time()
    raw_values, raw_diag = evaluate_reference_so3_stage(
        raw_candidates, reference_tio2, args.rcut, "raw", ncpu
    )
    diagnostic_rows.extend(raw_diag)
    for source_row, value in raw_values.items():
        candidate_summary[source_row]["so3_raw_energy"] = value

    projection_start = time()
    if args.skip_so3_refinement:
        projected_xtals = [
            (int(source_row), xtal) for source_row, xtal in raw_candidates
        ]
        projection_rows = []
        for source_row, _ in projected_xtals:
            raw_value = candidate_summary[source_row].get(
                "so3_raw_energy", math.nan
            )
            projection_rows.append(
                {
                    "source_row": source_row,
                    "success": True,
                    "initial_so3_reference_energy": raw_value,
                    "final_so3_reference_energy": raw_value,
                    "improved": False,
                    "raw_fallback_used": True,
                    "lattice_dof": math.nan,
                    "site_coordinate_dof": math.nan,
                    "objective_evaluations": 0,
                    "refinement_skipped": True,
                    "error": None,
                }
            )
    else:
        projected_xtals, projection_rows = optimize_single_reference_so3(
            raw_candidates,
            reference_tio2,
            args.rcut,
            ncpu,
            args.nm_steps,
            args.lbfgs_steps,
        )
        for row in projection_rows:
            row["refinement_skipped"] = False
    projection_seconds = time() - projection_start
    pd.DataFrame(projection_rows).to_csv(
        output_dir / "so3_projection_results.csv", index=False
    )

    projection_by_source = {
        int(row["source_row"]): row for row in projection_rows
    }
    for source_row, xtal in projected_xtals:
        source_row = int(source_row)
        row = projection_by_source[source_row]
        summary = candidate_summary[source_row]
        summary["so3_success"] = bool(row["success"])
        summary["so3_initial_energy"] = row["initial_so3_reference_energy"]
        summary["so3_final_energy"] = row["final_so3_reference_energy"]
        summary["so3_improved"] = bool(row["improved"])
        summary["so3_raw_fallback_used"] = bool(row["raw_fallback_used"])
        summary["so3_refinement_applied"] = bool(
            not args.skip_so3_refinement and row["success"] and row["improved"]
        )
        if not row["success"]:
            summary["failure_stage"] = "so3"
            summary["so3_error"] = row["error"]

        final_atoms = xtal.to_ase(resort=False)
        so3_cif_path = so3_cif_dir / (
            f"rank_{int(summary['final_rank']):04d}_row_{source_row:04d}_"
            f"{'so3' if summary['so3_refinement_applied'] else 'raw_fallback'}.cif"
        )
        ase_write(str(so3_cif_path), final_atoms, format="cif")
        summary["so3_cif_path"] = str(so3_cif_path)
        stage_rows.append(
            stage_metric_record(
                source_row,
                meta_by_source[source_row],
                "so3" if not args.skip_so3_refinement else "raw_final",
                final_atoms,
                args.ti_o_cutoff,
            )
        )
        try:
            movement = same_order_movement_metrics(
                raw_atoms_by_source[source_row], final_atoms
            )
            movement_rows.append(
                {
                    "source_row": source_row,
                    "from_stage": "raw",
                    "to_stage": (
                        "so3" if not args.skip_so3_refinement else "raw_final"
                    ),
                    **movement,
                    "error": None,
                }
            )
            for key, value in movement.items():
                summary[f"raw_to_so3_{key}"] = value
        except Exception as exc:
            movement_rows.append(
                {
                    "source_row": source_row,
                    "from_stage": "raw",
                    "to_stage": (
                        "so3" if not args.skip_so3_refinement else "raw_final"
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    projected_values, projected_diag = evaluate_reference_so3_stage(
        projected_xtals,
        reference_tio2,
        args.rcut,
        "so3" if not args.skip_so3_refinement else "raw_final",
        ncpu,
    )
    diagnostic_rows.extend(projected_diag)
    for source_row, value in projected_values.items():
        summary = candidate_summary[source_row]
        summary["so3_projected_energy"] = value
        raw_value = summary.get("so3_raw_energy", math.nan)
        summary["delta_so3_projected_minus_raw"] = (
            value - raw_value
            if math.isfinite(value) and math.isfinite(raw_value)
            else math.nan
        )
    diagnostic_seconds = time() - diagnostic_start

    final_candidates = [
        (int(source_row), xtal) for source_row, xtal in projected_xtals
    ]
    dedup_start = time()
    ranked_df, _, _ = deduplicate_final_stage(
        final_candidates,
        candidate_summary,
        output_dir,
        ncpu,
        args.dedup_ltol,
        args.dedup_stol,
        args.dedup_angle_tol,
        args.dedup_volume_tol,
        args.match_chunksize,
    )
    dedup_seconds = time() - dedup_start

    standard_vasp_report, unique_standard_vasp_dir = export_unique_standard_vasp(
        ranked_df,
        final_candidates,
        output_dir,
        symprec=args.standard_vasp_symprec,
        angle_tolerance=args.standard_vasp_angle_tolerance,
    )

    training_seconds = 0.0
    if training_db_path is not None:
        training_start = time()
        ranked_df, _ = compare_ranked_to_training_set_strict(
            ranked_df,
            training_db_path,
            output_dir,
            ncpu,
            args.training_match_ltol,
            args.training_match_stol,
            args.training_match_angle_tol,
            args.training_match_volume_tol,
            args.match_chunksize,
        )
        training_seconds = time() - training_start

    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "so3_stage_diagnostics.csv", index=False
    )
    pd.DataFrame(stage_rows).to_csv(
        output_dir / "relaxation_stage_metrics.csv", index=False
    )
    pd.DataFrame(movement_rows).to_csv(
        output_dir / "stage_movement_metrics.csv", index=False
    )
    pd.DataFrame(
        [candidate_summary[key] for key in sorted(candidate_summary)]
    ).to_csv(output_dir / "relaxation_candidate_summary.csv", index=False)

    pipeline_summary = {
        "version": "3.4",
        "input": {
            "mode": "v33_ranked_cif_direct",
            "generation_dir": str(generation_dir),
            "cif_dir": str(resolved_cif_dir),
            "selected_metadata": str(selected_metrics_path),
            "tabular_decoder": False,
            "cif_authoritative": True,
        },
        "pipeline": [
            "v33_ranked_cif_ingestion",
            "exact_pyxtal_representation_deduplication",
            "raw_single_reference_so3_diagnostic",
            (
                "site_only_single_reference_so3_refinement"
                if not args.skip_so3_refinement
                else "so3_refinement_skipped"
            ),
            "raw_so3_movement_and_chemistry_comparison",
            "raw_and_so3_cif_export",
            "final_stage_structurematcher_deduplication",
            "unique_symmetrized_standard_conventional_vasp_export",
            (
                "strict_training_overlap"
                if training_db_path is not None
                else "training_overlap_skipped"
            ),
        ],
        "removed_paths": [
            "tabular_wp_xyz_decoder",
            "GULP_relaxation",
            "ReaxFF_Ti_O_potential",
            "SO3_similarity_ranking",
        ],
        "so3_objective": {
            "descriptor": "SO3",
            "target": "single rutile TiO2 reference",
            "optimizer": (
                "full-representation SciPy minimization of lego.util.calculate_S"
                if not args.skip_so3_refinement
                else "diagnostic only"
            ),
            "raw_fallback": True,
            "rcut": float(args.rcut),
            "lattice_mode": "optimized_with_native_lego_bounds",
            "reference_routing": (
                "site-labeled by species: Ti->rutile Ti, O->rutile O"
            ),
            "used_for_final_ranking": False,
            "nm_steps": int(args.nm_steps),
            "lbfgs_steps": int(args.lbfgs_steps),
        },
        "final_deduplication": {
            "representative_policy": (
                "earliest v33 final_rank, then source_row; no energy/SO3 ranking"
            ),
            "ltol": float(args.dedup_ltol),
            "stol": float(args.dedup_stol),
            "angle_tol": float(args.dedup_angle_tol),
            "volume_tol": float(args.dedup_volume_tol),
        },
        "resources": {"ncpu": int(ncpu)},
        "counts": {
            "v33_cifs_found": int(len(candidates) + len(ingestion_failures)),
            "ingested": int(len(candidates)),
            "ingestion_failures": int(len(ingestion_failures)),
            "exact_unique": int(len(unique)),
            "selected": int(len(selected)),
            "projection_success": int(
                sum(bool(r["success"]) for r in projection_rows)
            ),
            "ranked_unique": int(len(ranked_df)),
            "unique_standard_vasp_written": int((standard_vasp_report.get("status") == "ok").sum()) if not standard_vasp_report.empty else 0,
        },
        "timing_seconds": {
            "ingestion_and_pre_so3_dedup": float(ingestion_seconds),
            "so3_projection": float(projection_seconds),
            "so3_diagnostics": float(diagnostic_seconds),
            "final_stage_dedup": float(dedup_seconds),
            "training_overlap": float(training_seconds),
            "total": float(time() - started),
        },
    }
    with open(
        output_dir / "pipeline_summary.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(pipeline_summary, handle, indent=2)

    print(f"Candidate summary: {output_dir / 'relaxation_candidate_summary.csv'}")
    print(f"Stage metrics: {output_dir / 'relaxation_stage_metrics.csv'}")
    print(f"Movement metrics: {output_dir / 'stage_movement_metrics.csv'}")
    print(f"Raw CIFs: {raw_cif_dir}")
    print(f"SO3/fallback CIFs: {so3_cif_dir}")
    print(f"SO3 diagnostics: {output_dir / 'so3_stage_diagnostics.csv'}")
    print(f"Ranked candidates: {output_dir / 'ranked_candidates.csv'}")
    print(f"Unique standard VASP: {unique_standard_vasp_dir}")
    print(f"Unique standard VASP report: {output_dir / 'unique_standard_vasp_report.csv'}")
    if training_db_path is not None:
        print(f"Training matches: {output_dir / 'training_set_matches.csv'}")
    print(f"Total wall time: {(time() - started) / 60.0:.2f} min")


if __name__ == "__main__":
    main()

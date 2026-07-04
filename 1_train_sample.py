#!/usr/bin/env python3
"""Two-stage TiO2 LEGO-Xtal generator v18 with a differentiable Torch L/R builder.

The VAE learns the current factorized representation but generation uses only
space group and Ti Wyckoff skeleton.  Compact Ti-framework chemistry targets
are extracted from the training CSV and drive a symmetry-exact differentiable
builder for lattice and Ti free Wyckoff parameters.

Phase A generates a symmetry-exact Ti framework from Ti-Ti chemistry. Phase B
then generates an exact-stoichiometry oxygen sublattice against Ti-O coordination,
O-Ti-O angular, local shell O-O, O-to-Ti sharing, and collision objectives. Both
stages use persistent asynchronous one-process-per-GPU workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.io import write
from pyxtal.symmetry import Group

from lego.VAE_factorized import FactorizedVAE


BASE_COLUMNS = ["spg", "a", "b", "c", "alpha", "beta", "gamma"]
TI_ROLE = 6
O_ROLE = 3
ANGLE_BINS = np.linspace(0.0, 180.0, 10)
CHEMISTRY_CUTOFF = 5.0
MAX_TI_ATOMS = 32
MAX_TI_NEIGHBORS = 16
SHIFTS = np.asarray(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=float,
)
ZERO_SHIFT = int(np.flatnonzero(np.all(SHIFTS == 0, axis=1))[0])


def _stable_logistic_switch(distances, cutoff, width):
    """Numerically stable smooth neighbour-shell weights."""
    scale = max(float(width), 0.03)
    argument = np.clip(
        (np.asarray(distances, dtype=float) - float(cutoff)) / scale,
        -60.0,
        60.0,
    )
    return 1.0 / (1.0 + np.exp(argument))


def find_indexed_columns(columns, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    return sorted(int(m.group(1)) for c in columns if (m := pattern.match(str(c))))


def validate_layout(df):
    missing = set(BASE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing base columns: {sorted(missing)}")
    wp = find_indexed_columns(df.columns, "wp")
    target = find_indexed_columns(df.columns, "target_coord")
    if not wp or wp != list(range(len(wp))) or target != wp:
        raise ValueError(f"Invalid contiguous slot layout: wp={wp}, target={target}")
    for i in wp:
        required = {f"wp{i}", f"x{i}", f"y{i}", f"z{i}", f"target_coord{i}"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing slot-{i} columns: {sorted(missing)}")
    return len(wp)


def canonicalize_species_order(df, num_wps):
    output = df.copy()
    rows, n_ti_max, n_o_max = [], 0, 0
    for row_index, row in df.iterrows():
        ti, oxygen = [], []
        for i in range(num_wps):
            wp = int(row[f"wp{i}"])
            role = int(row[f"target_coord{i}"])
            xyz = tuple(float(row[f"{axis}{i}"]) for axis in "xyz")
            if wp < 0:
                if role != 0:
                    raise ValueError(f"Row {row_index}, slot {i}: padding role is not zero.")
                continue
            if role == TI_ROLE:
                ti.append((wp, xyz))
            elif role == O_ROLE:
                oxygen.append((wp, xyz))
            else:
                raise ValueError(f"Row {row_index}, slot {i}: unsupported role {role}.")
        if not ti or not oxygen:
            raise ValueError(f"Row {row_index} does not contain both Ti and O blocks.")
        n_ti_max, n_o_max = max(n_ti_max, len(ti)), max(n_o_max, len(oxygen))
        rows.append((ti, oxygen))
    for pos, (ti, oxygen) in enumerate(rows):
        ordered = [(wp, xyz, TI_ROLE) for wp, xyz in ti]
        ordered += [(wp, xyz, O_ROLE) for wp, xyz in oxygen]
        ordered += [(-1, (-1.0, -1.0, -1.0), 0)] * (num_wps - len(ordered))
        idx = output.index[pos]
        for i, (wp, xyz, role) in enumerate(ordered):
            output.at[idx, f"wp{i}"] = wp
            output.at[idx, f"target_coord{i}"] = role
            for axis, value in zip("xyz", xyz):
                output.at[idx, f"{axis}{i}"] = value
    return output, n_ti_max, n_o_max


def encode_wp_token(values):
    return "|".join(str(int(v)) for v in values)


def decode_wp_token(token, expected_slots=None):
    values = [int(x) for x in str(token).strip().split("|")]
    if expected_slots is not None and len(values) != int(expected_slots):
        raise ValueError(f"Token {token!r} has {len(values)} slots, expected {expected_slots}.")
    return values


def _wyckoff_free_parameters(spg, wp_index, xyz, row_label):
    group = Group(int(spg))
    wp = group[int(wp_index)]
    xyz = np.asarray(xyz, dtype=float)
    generator = wp.search_generator(xyz, tol=1e-2, symmetrize=True)
    if generator is None:
        generator = wp.search_generator(wp.project(xyz), tol=1e-6, symmetrize=True)
    if generator is None:
        raise ValueError(f"{row_label}: cannot map coordinate to {wp.get_label()}.")
    free = np.asarray(wp.get_free_xyzs(generator), dtype=float) % 1.0
    padded = np.zeros(3, dtype=float)
    padded[: int(wp.get_dof())] = free
    return padded


def _wyckoff_position_from_parameters(spg, wp_index, parameters, group=None):
    wp = (group if group is not None else Group(int(spg)))[int(wp_index)]
    dof = int(wp.get_dof())
    return np.asarray(
        wp.get_position_from_free_xyzs(np.asarray(parameters, dtype=float)[:dof] % 1.0),
        dtype=float,
    ) % 1.0


def build_factorized_blocks(df, num_wps, n_ti_max, n_o_max):
    global_df = df[BASE_COLUMNS].copy()
    ti_records, o_records = [], []
    for row_index, row in df.iterrows():
        spg = int(row["spg"])
        ti, oxygen = [], []
        for i in range(num_wps):
            wp = int(row[f"wp{i}"])
            if wp < 0:
                continue
            role = int(row[f"target_coord{i}"])
            xyz = [float(row[f"{axis}{i}"]) for axis in "xyz"]
            free = _wyckoff_free_parameters(spg, wp, xyz, f"row {row_index}, slot {i}")
            site = (wp, free)
            (ti if role == TI_ROLE else oxygen).append(site)
        ti += [(-1, np.full(3, -1.0))] * (n_ti_max - len(ti))
        oxygen += [(-1, np.full(3, -1.0))] * (n_o_max - len(oxygen))
        tr = {"si_skeleton_token": encode_wp_token(wp for wp, _ in ti)}
        od = {"o_skeleton_token": encode_wp_token(wp for wp, _ in oxygen)}
        for i, (_, free) in enumerate(ti):
            for j in range(3):
                tr[f"si_u{j}_{i}"] = float(free[j])
        for i, (_, free) in enumerate(oxygen):
            for j in range(3):
                od[f"o_u{j}_{i}"] = float(free[j])
        ti_records.append(tr); o_records.append(od)
    return global_df, pd.DataFrame(ti_records), pd.DataFrame(o_records)


def _cell_matrix(parameters):
    a, b, c, alpha, beta, gamma = map(float, parameters)
    ca, cb, cg, sg = np.cos(alpha), np.cos(beta), np.cos(gamma), np.sin(gamma)
    if min(a, b, c) <= 0 or abs(sg) < 1e-8:
        raise ValueError("Invalid cell parameters.")
    y3 = c * (ca - cb * cg) / sg
    z2 = c * c - (c * cb) ** 2 - y3 ** 2
    if z2 <= 1e-10:
        raise ValueError("Non-positive cell metric.")
    return np.asarray([[a, 0, 0], [b * cg, b * sg, 0], [c * cb, y3, np.sqrt(z2)]])


def _deduplicate_fractional(frac, tol=1e-6):
    unique = []
    for point in np.asarray(frac, dtype=float).reshape(-1, 3) % 1.0:
        if not any(np.linalg.norm((point - other) - np.round(point - other)) <= tol
                   for other in unique):
            unique.append(point)
    return np.asarray(unique, dtype=float).reshape(-1, 3)




def periodic_neighbor_vectors(frac, cell):
    frac = np.asarray(frac, dtype=float)
    delta = frac[:, None, None, :] - frac[None, :, None, :] + SHIFTS[None, None, :, :]
    cart = np.einsum("...i,ij->...j", delta, cell)
    dist = np.linalg.norm(cart, axis=-1)
    ids = np.arange(len(frac)); dist[ids, ids, ZERO_SHIFT] = np.inf
    vectors, distances = [], []
    for i in range(len(frac)):
        d = dist[i].reshape(-1)
        v = cart[i].reshape(-1, 3)
        mask = np.isfinite(d) & (d > 1e-6)
        order = np.argsort(d[mask])
        distances.append(d[mask][order]); vectors.append(v[mask][order])
    return distances, vectors




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
                angle = np.degrees(np.arccos(np.clip(np.dot(v[i], v[j]) / (ni * nj), -1, 1)))
                histogram += weights[i] * weights[j] * np.exp(-0.5 * ((centers - angle) / sigma) ** 2)
    return histogram / max(histogram.sum(), 1e-12)


def framework_descriptor(frac, cell, chemistry_cutoff=CHEMISTRY_CUTOFF):
    """Global Ti-community chemistry under the hard radial cutoff."""
    distances, vectors = periodic_neighbor_vectors(frac, cell)
    cutoff = float(chemistry_cutoff)
    shell_values = np.concatenate([d[d <= cutoff] for d in distances])
    if shell_values.size == 0:
        raise ValueError("No Ti neighbours inside the chemistry cutoff.")
    cn_values = np.asarray([np.count_nonzero(d <= cutoff) for d in distances], dtype=float)
    nn_mean = float(np.mean(shell_values))
    nn_width = max(float(np.std(shell_values)), 0.03)
    angle = soft_angle_hist(distances, vectors, cutoff, max(nn_width, 0.08))
    all_nearest = np.asarray([d[0] for d in distances], dtype=float)
    lengths = np.linalg.norm(cell, axis=1)
    volume = abs(float(np.linalg.det(cell)))
    return {
        "target_ti_cn": float(np.mean(cn_values)),
        "target_ti_nn_mean": nn_mean,
        "target_ti_nn_width": nn_width,
        "target_ti_shell_cutoff": cutoff,
        "target_ti_sphere_radius": 0.45 * float(np.percentile(all_nearest, 5)),
        "minimum_ti_ti_distance": float(np.min(all_nearest)),
        "volume_per_ti": volume / len(frac),
        "aspect_ratio": float(lengths.max() / max(lengths.min(), 1e-12)),
        "angle_profile": angle,
        "n_ti": int(len(frac)),
    }


def ti_skeleton_from_row(row, num_wps):
    return [int(row[f"wp{i}"]) for i in range(num_wps)
            if int(row[f"wp{i}"]) >= 0 and int(row[f"target_coord{i}"]) == TI_ROLE]


def ti_framework_from_row(row, num_wps):
    spg = int(row["spg"])
    group = Group(spg)
    frac = []
    for i in range(num_wps):
        if int(row[f"target_coord{i}"]) != TI_ROLE:
            continue
        wp_index = int(row[f"wp{i}"])
        generator = np.asarray([row[f"x{i}"], row[f"y{i}"], row[f"z{i}"]], dtype=float)
        orbit = _deduplicate_fractional([op.operate(generator) for op in group[wp_index].ops])
        frac.extend(orbit.tolist())
    return _deduplicate_fractional(frac), _cell_matrix(row[BASE_COLUMNS[1:]].to_numpy(float))


def framework_fingerprint(frac, cell):
    distances, _ = periodic_neighbor_vectors(frac, cell)
    values = np.sort(np.concatenate([d[: min(12, len(d))] for d in distances]))
    rounded = np.round(values, 3).tobytes()
    return hashlib.sha1(rounded).hexdigest()[:16]



CHEMISTRY_CACHE_VERSION = 2


def _extract_training_record(payload):
    row_index, row_dict, num_wps, chemistry_cutoff, has_label = payload
    row = pd.Series(row_dict)
    try:
        frac, cell = ti_framework_from_row(row, num_wps)
        descriptor = framework_descriptor(frac, cell, chemistry_cutoff=chemistry_cutoff)
        wps = ti_skeleton_from_row(row, num_wps)
        group = Group(int(row["spg"]))
        source = str(row["label"]) if has_label else framework_fingerprint(frac, cell)
        record = {
            "training_row": int(row_index),
            "source_group": source,
            "spg": int(row["spg"]),
            "lattice_type": str(group.lattice_type),
            "ti_skeleton_token_unpadded": encode_wp_token(wps),
            "ti_multiplicities": encode_wp_token(group[wp].multiplicity for wp in wps),
            **{k: v for k, v in descriptor.items() if k != "angle_profile"},
        }
        for i, value in enumerate(descriptor["angle_profile"]):
            record[f"angle_bin_{i}"] = float(value)
        return record, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _cpu_affinity_count():
    """Return CPUs available to this process after Slurm/cgroup binding."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def resolve_ncpu(requested, reserve_for_scheduler=1):
    """Resolve CPU worker count from an explicit value or the active allocation.

    Automatic mode (requested <= 0) prefers SLURM_CPUS_PER_TASK, caps it by
    the process CPU affinity, and reserves one CPU for the scheduler/main
    process.  An explicit positive value is treated as the requested worker
    count but is still capped by the CPUs actually available to the process.
    """
    affinity = _cpu_affinity_count()
    slurm_raw = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        allocated = int(slurm_raw) if slurm_raw is not None else affinity
    except ValueError:
        allocated = affinity
    allocated = max(1, min(allocated, affinity))

    requested = 0 if requested is None else int(requested)
    if requested < 0:
        raise ValueError("--ncpu cannot be negative.")
    if requested == 0:
        return max(1, allocated - max(0, int(reserve_for_scheduler)))
    return max(1, min(requested, allocated))


def set_worker_thread_limits():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")


class TiTrainingDistribution:
    def __init__(self, canonical_df, num_wps, chemistry_cutoff=CHEMISTRY_CUTOFF,
                 ncpu=1, cache_csv=None, cache_meta=None, cache_key=None):
        self.rng = np.random.default_rng(0)
        if cache_csv and cache_meta and os.path.exists(cache_csv) and os.path.exists(cache_meta):
            try:
                with open(cache_meta, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
                if meta.get("cache_key") == cache_key and meta.get("version") == CHEMISTRY_CACHE_VERSION:
                    self.frame = pd.read_csv(cache_csv)
                    self.failures = int(meta.get("failures", 0))
                    self.source_groups = self.frame.groupby("source_group", sort=False).indices
                    print(f"Reused Ti chemistry cache: {cache_csv}", flush=True)
                    return
            except Exception as exc:
                print(f"Ignoring invalid chemistry cache: {type(exc).__name__}: {exc}", flush=True)

        records, failures = [], 0
        payloads = [
            (int(idx), row.to_dict(), int(num_wps), float(chemistry_cutoff), "label" in canonical_df.columns)
            for idx, row in canonical_df.iterrows()
        ]
        total = len(payloads)
        print(f"Extracting Ti chemistry from {total} training rows with {ncpu} CPU worker(s)...", flush=True)
        if int(ncpu) == 1:
            iterator = map(_extract_training_record, payloads)
            for done, (record, error) in enumerate(iterator, 1):
                if record is None: failures += 1
                else: records.append(record)
                if done % 500 == 0 or done == total:
                    print(f"Ti chemistry extraction: {done}/{total}; valid={len(records)}; failures={failures}", flush=True)
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=int(ncpu), mp_context=ctx,
                                     initializer=set_worker_thread_limits) as pool:
                iterator = pool.map(_extract_training_record, payloads, chunksize=8)
                for done, (record, error) in enumerate(iterator, 1):
                    if record is None: failures += 1
                    else: records.append(record)
                    if done % 500 == 0 or done == total:
                        print(f"Ti chemistry extraction: {done}/{total}; valid={len(records)}; failures={failures}", flush=True)
        if not records:
            raise RuntimeError("No valid Ti-framework chemistry records could be extracted.")
        self.frame = pd.DataFrame(records).sort_values("training_row").reset_index(drop=True)
        self.failures = failures
        self.source_groups = self.frame.groupby("source_group", sort=False).indices
        if cache_csv and cache_meta:
            self.frame.to_csv(cache_csv, index=False)
            with open(cache_meta, "w", encoding="utf-8") as handle:
                json.dump({"version": CHEMISTRY_CACHE_VERSION, "cache_key": cache_key,
                           "failures": failures}, handle, indent=2)

    def save(self, path):
        self.frame.to_csv(path, index=False)

    def _row_to_target(self, row):
        angle = np.asarray([row[f"angle_bin_{i}"] for i in range(len(ANGLE_BINS) - 1)], dtype=float)
        return TiChemistryTarget(
            source_group=str(row.source_group), target_ti_cn=float(row.target_ti_cn),
            target_ti_nn_mean=float(row.target_ti_nn_mean),
            target_ti_nn_width=float(row.target_ti_nn_width),
            target_ti_shell_cutoff=float(row.target_ti_shell_cutoff),
            target_ti_sphere_radius=float(row.target_ti_sphere_radius),
            target_volume_per_ti=float(row.volume_per_ti), angle_profile=angle,
        )

    def sample_any(self, rng):
        source = rng.choice(self.frame.source_group.unique())
        subset = self.frame[self.frame.source_group == source]
        return self._row_to_target(subset.iloc[int(rng.integers(0, len(subset)))])

    def sample(self, spg, padded_token, rng):
        wps = [w for w in decode_wp_token(padded_token) if w >= 0]
        token = encode_wp_token(wps)
        group = Group(int(spg))
        multiplicities = encode_wp_token(group[w].multiplicity for w in wps)
        lattice_type = str(group.lattice_type)
        levels = [
            (self.frame.spg == int(spg)) & (self.frame.ti_skeleton_token_unpadded == token),
            (self.frame.lattice_type == lattice_type) & (self.frame.ti_multiplicities == multiplicities),
            self.frame.ti_multiplicities == multiplicities,
            np.ones(len(self.frame), dtype=bool),
        ]
        for mask in levels:
            candidates = self.frame.loc[mask]
            if len(candidates): break
        source = rng.choice(candidates.source_group.unique())
        subset = candidates[candidates.source_group == source]
        return self._row_to_target(subset.iloc[int(rng.integers(0, len(subset)))])


@dataclass(frozen=True)
class TiChemistryTarget:
    source_group: str
    target_ti_cn: float
    target_ti_nn_mean: float
    target_ti_nn_width: float
    target_ti_shell_cutoff: float
    target_ti_sphere_radius: float
    target_volume_per_ti: float
    angle_profile: np.ndarray


class TorchSymmetryConstrainedTiBuilder:
    """Optimize symmetry-allowed lattice and Wyckoff variables with autograd."""

    def __init__(
        self,
        initializations=16,
        screen_steps=30,
        refine_starts=4,
        refine_steps=60,
        cn_tolerance=0.75,
        minimum_distance=2.0,
        maximum_loss=5.0,
        chemistry_cutoff=CHEMISTRY_CUTOFF,
        max_ti_atoms=MAX_TI_ATOMS,
        max_neighbors=MAX_TI_NEIGHBORS,
        lr=0.06,
        device=None,
        seed=42,
    ):
        self.initializations = int(initializations)
        self.screen_steps = int(screen_steps)
        self.refine_starts = int(refine_starts)
        self.refine_steps = int(refine_steps)
        self.cn_tolerance = float(cn_tolerance)
        self.minimum_distance = float(minimum_distance)
        self.maximum_loss = float(maximum_loss)
        self.chemistry_cutoff = float(chemistry_cutoff)
        self.max_ti_atoms = int(max_ti_atoms)
        self.max_neighbors = int(max_neighbors)
        self.lr = float(lr)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)
        self._template_cache = {}
        self._shifts = torch.as_tensor(SHIFTS, dtype=torch.float32, device=self.device)

    @staticmethod
    def _lattice_spec(lattice_type):
        lt = str(lattice_type).lower()
        if lt == "cubic": return ("a",)
        if lt in {"tetragonal", "hexagonal", "trigonal"}: return ("a", "c")
        if lt == "orthorhombic": return ("a", "b", "c")
        if lt == "monoclinic": return ("a", "b", "c", "beta")
        return ("a", "b", "c", "alpha", "beta", "gamma")

    @staticmethod
    def _affine_map(function, dof):
        zero = np.asarray(function(np.zeros(dof)), dtype=float)
        matrix = np.zeros((3, dof), dtype=float)
        for k in range(dof):
            x = np.zeros(dof); x[k] = 0.137
            y = np.asarray(function(x), dtype=float)
            delta = (y - zero + 0.5) % 1.0 - 0.5
            matrix[:, k] = delta / 0.137
        return matrix, zero

    def _template(self, spg, padded_token, target):
        key = (int(spg), str(padded_token))
        if key in self._template_cache:
            return self._template_cache[key]
        group = Group(int(spg))
        wps = tuple(w for w in decode_wp_token(padded_token) if w >= 0)
        if not wps:
            return None
        n_ti = sum(int(group[w].multiplicity) for w in wps)
        if n_ti < 2 or n_ti > self.max_ti_atoms:
            return None
        spec = self._lattice_spec(group.lattice_type)
        site_dofs=[]; orbit_rot=[]; orbit_trans=[]; gen_A=[]; gen_b=[]
        for w in wps:
            wp=group[w]; dof=int(wp.get_dof()); site_dofs.append(dof)
            A,b=self._affine_map(lambda u, wp=wp: wp.get_position_from_free_xyzs(u), dof)
            gen_A.append(A); gen_b.append(b)
            rots=[]; trans=[]
            for op in wp.ops:
                o=np.asarray(op.operate([0.,0.,0.]),float)
                cols=[]
                for axis in range(3):
                    e=np.zeros(3); e[axis]=0.173
                    q=np.asarray(op.operate(e),float)
                    cols.append(((q-o+0.5)%1.0-0.5)/0.173)
                rots.append(np.stack(cols,axis=1)); trans.append(o)
            orbit_rot.append(np.asarray(rots,float)); orbit_trans.append(np.asarray(trans,float))
        template={
            'spg':int(spg),'group':group,'wps':wps,'spec':spec,
            'lattice_type':str(group.lattice_type).lower(),'site_dofs':tuple(site_dofs),
            'gen_A':[torch.tensor(x,dtype=torch.float32,device=self.device) for x in gen_A],
            'gen_b':[torch.tensor(x,dtype=torch.float32,device=self.device) for x in gen_b],
            'orbit_rot':[torch.tensor(x,dtype=torch.float32,device=self.device) for x in orbit_rot],
            'orbit_trans':[torch.tensor(x,dtype=torch.float32,device=self.device) for x in orbit_trans],
            'n_ti':n_ti,
        }
        self._template_cache[key]=template
        return template

    def _initial_raw(self, template, target, nstart):
        base=float(target.target_ti_nn_mean)*max(template['n_ti'],1)**(1/3)
        nlat=len(template['spec']); ncoord=sum(template['site_dofs'])
        raw=torch.randn((nstart,nlat+ncoord),device=self.device,dtype=torch.float32)
        # lattice logits centered around a chemistry-derived scale
        raw[:,:nlat] *= 0.35
        raw[:,:nlat] += math.log(math.expm1(max(base - 1.2, 0.5)))
        return raw

    def _lattice_and_frac(self, template, raw):
        B=raw.shape[0]; nlat=len(template['spec'])
        vals=raw[:,:nlat]
        lengths=torch.nn.functional.softplus(vals)+1.2
        lt=template['lattice_type']
        pi=torch.tensor(math.pi,device=self.device)
        if lt=='cubic':
            a=lengths[:,0]; abc=torch.stack([a,a,a],1); ang=torch.full((B,3),math.pi/2,device=self.device)
        elif lt=='tetragonal':
            a,c=lengths[:,0],lengths[:,1]; abc=torch.stack([a,a,c],1); ang=torch.full((B,3),math.pi/2,device=self.device)
        elif lt in {'hexagonal','trigonal'}:
            a,c=lengths[:,0],lengths[:,1]; abc=torch.stack([a,a,c],1)
            ang=torch.tensor([math.pi/2,math.pi/2,2*math.pi/3],device=self.device).repeat(B,1)
        elif lt=='orthorhombic':
            abc=lengths[:,:3]; ang=torch.full((B,3),math.pi/2,device=self.device)
        elif lt=='monoclinic':
            abc=lengths[:,:3]; beta=math.pi/3 + torch.sigmoid(vals[:,3])*math.pi/3
            ang=torch.stack([torch.full_like(beta,math.pi/2),beta,torch.full_like(beta,math.pi/2)],1)
        else:
            abc=lengths[:,:3]; ang=math.pi/4 + torch.sigmoid(vals[:,3:6])*math.pi/2
        a,b,c=abc[:,0],abc[:,1],abc[:,2]; alpha,beta,gamma=ang[:,0],ang[:,1],ang[:,2]
        ca,cb,cg,sg=torch.cos(alpha),torch.cos(beta),torch.cos(gamma),torch.sin(gamma).clamp_min(1e-4)
        y3=c*(ca-cb*cg)/sg; z2=(c*c-(c*cb)**2-y3*y3).clamp_min(1e-6)
        zero=torch.zeros_like(a)
        row1=torch.stack([a,zero,zero],1); row2=torch.stack([b*cg,b*sg,zero],1); row3=torch.stack([c*cb,y3,torch.sqrt(z2)],1)
        cell=torch.stack([row1,row2,row3],1)
        frac=[]; cursor=nlat; free=[]
        for dof,A,b0,R,t in zip(template['site_dofs'],template['gen_A'],template['gen_b'],template['orbit_rot'],template['orbit_trans']):
            u=torch.sigmoid(raw[:,cursor:cursor+dof]); cursor+=dof; free.append(u)
            gen=(u@A.T+b0) % 1.0
            orbit=(torch.einsum('oij,bj->boi',R,gen)+t[None,:,:]) % 1.0
            frac.append(orbit)
        frac=torch.cat(frac,dim=1)
        return abc,ang,cell,frac,free

    def _geometry(self, template, target, raw, include_angle):
        abc,ang,cell,frac,free=self._lattice_and_frac(template,raw)
        B,N=frac.shape[:2]; shifts=self._shifts
        delta=frac[:,:,None,None,:]-frac[:,None,:,None,:]+shifts[None,None,None,:,:]
        vec=torch.einsum('bijnk,bkl->bijnl',delta,cell)
        dist=torch.linalg.norm(vec,dim=-1).clamp_min(1e-6)
        eye=torch.eye(N,device=self.device,dtype=torch.bool)[None,:,:,None]
        zero=(torch.arange(27,device=self.device)==ZERO_SHIFT)[None,None,None,:]
        dist=dist.masked_fill(eye & zero, 1e6)
        flat_d=dist.reshape(B,N,-1); flat_v=vec.reshape(B,N,-1,3)
        width=max(float(target.target_ti_nn_width),0.10)
        w=torch.sigmoid((self.chemistry_cutoff-flat_d)/width)
        cn=w.sum(-1)
        denom=w.sum((1,2)).clamp_min(1e-6)
        mean=(w*flat_d).sum((1,2))/denom
        var=(w*(flat_d-mean[:,None,None])**2).sum((1,2))/denom
        cn_mean=cn.mean(1)
        cn_loss=((cn_mean-float(target.target_ti_cn))/2.0)**2
        mean_loss=((mean-float(target.target_ti_nn_mean))/max(width,0.15))**2
        width_loss=((torch.sqrt(var+1e-8)-float(target.target_ti_nn_width))/max(width,0.15))**2
        min_d=flat_d.min(-1).values.min(-1).values
        exclusion=1.8*float(target.target_ti_sphere_radius)
        overlap=torch.relu(exclusion-min_d).pow(2)/max(exclusion**2,0.1)
        volume=torch.abs(torch.linalg.det(cell))/N
        vol_loss=torch.log((volume/float(target.target_volume_per_ti)).clamp_min(1e-4)).pow(2)
        aspect=abc.max(-1).values/abc.min(-1).values.clamp_min(1e-4)
        shape_loss=torch.relu(aspect-5.0).pow(2)/25.0
        angle_loss=torch.zeros((B,),device=self.device)
        if include_angle:
            K=min(self.max_neighbors,flat_d.shape[-1])
            topd,idx=torch.topk(flat_d,K,dim=-1,largest=False)
            topv=torch.gather(flat_v,2,idx[...,None].expand(-1,-1,-1,3))
            norm=torch.linalg.norm(topv,dim=-1).clamp_min(1e-6)
            cos=torch.einsum('bnik,bnjk->bnij',topv,topv)/(norm[:,:,:,None]*norm[:,:,None,:])
            pairw=torch.sigmoid((self.chemistry_cutoff-topd)/width)
            pw=pairw[:,:,:,None]*pairw[:,:,None,:]
            tri=torch.triu(torch.ones((K,K),device=self.device,dtype=torch.bool),diagonal=1)
            cosv=cos[:,:,tri]; pvw=pw[:,:,tri]
            centers=torch.cos(torch.tensor(0.5*(ANGLE_BINS[:-1]+ANGLE_BINS[1:])*math.pi/180,device=self.device))
            sigma=0.16
            hist=(pvw[...,None]*torch.exp(-0.5*((cosv[...,None]-centers)/sigma)**2)).sum((1,2))
            hist=hist/hist.sum(-1,keepdim=True).clamp_min(1e-8)
            target_hist=torch.tensor(target.angle_profile,dtype=torch.float32,device=self.device)
            angle_loss=((hist-target_hist[None,:])**2).mean(-1)*len(target.angle_profile)
        total=cn_loss+mean_loss+0.5*width_loss+4.0*overlap+0.15*vol_loss+0.1*shape_loss+(0.7*angle_loss if include_angle else 0.0)
        detail={'cn_loss':cn_loss,'distance_loss':mean_loss+0.5*width_loss,'angle_loss':angle_loss,'overlap_loss':overlap,'cell_loss':0.15*vol_loss+0.1*shape_loss,
                'minimum_ti_ti_distance':min_d,'volume_per_ti':volume,'aspect_ratio':aspect,'achieved_cn':cn_mean,'achieved_nn_mean':mean,'achieved_nn_width':torch.sqrt(var+1e-8)}
        return total,detail,(abc,ang,cell,frac,free)

    def _optimize(self, template, target, raw, steps, include_angle):
        raw=raw.detach().clone().requires_grad_(True)
        opt=torch.optim.Adam([raw],lr=self.lr)
        best_raw=raw.detach().clone(); best=torch.full((raw.shape[0],),float('inf'),device=self.device)
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            total,detail,_=self._geometry(template,target,raw,include_angle)
            # total is batch-aggregated; keep all starts moving while selecting by per-start diagnostics later
            total.mean().backward(); torch.nn.utils.clip_grad_norm_([raw],10.0); opt.step()
        return raw.detach()

    def build(self, spg, padded_token, target, sample_id):
        template=self._template(spg,padded_token,target)
        if template is None: return None,[]
        raw=self._initial_raw(template,target,self.initializations)
        raw=self._optimize(template,target,raw,self.screen_steps,False)
        # rank starts independently by evaluating one at a time; only a small retained set reaches angles
        scores=[]
        with torch.no_grad():
            for i in range(raw.shape[0]):
                val,_,_=self._geometry(template,target,raw[i:i+1],False); scores.append(float(val[0]))
        order=np.argsort(scores)[:min(self.refine_starts,len(scores))]
        refined=self._optimize(template,target,raw[order],self.refine_steps,True)
        attempts=[]
        with torch.no_grad():
            for rank in range(refined.shape[0]):
                total,detail,geom=self._geometry(template,target,refined[rank:rank+1],True)
                abc,ang,cell,frac,free=geom
                lattice=torch.cat([abc[0],ang[0]]).cpu().numpy()
                free_np=np.zeros((len(template['wps']),3),float)
                for j,u in enumerate(free): free_np[j,:u.shape[1]]=u[0].cpu().numpy()
                item={'sample_id':sample_id,'initialization_id':int(order[rank]),'screen_rank':rank,'screen_loss':float(scores[order[rank]]),'success':bool(torch.isfinite(total[0])),'total_loss':float(total[0])}
                for k,v in detail.items(): item[k]=float(v.mean())
                item.update({'lattice':lattice,'free':free_np,'frac':frac[0].cpu().numpy(),'generators':None,'cell':cell[0].cpu().numpy()})
                attempts.append(item)
        attempts.sort(key=lambda x:x['total_loss'])
        if not attempts: return None,[]
        for rank,item in enumerate(attempts):
            item['cn_error']=abs(item['achieved_cn']-float(target.target_ti_cn))
            item['numerical_success']=bool(item['success'])
            item['chemistry_success']=bool(
                item['success'] and item['cn_error'] <= self.cn_tolerance and
                item['minimum_ti_ti_distance'] >= self.minimum_distance and
                item['total_loss'] <= self.maximum_loss
            )
            item['final_rank']=rank
        valid=[item for item in attempts if item['chemistry_success']]
        chosen=(valid[0] if valid else attempts[0])
        chosen['retained_rank']=chosen['final_rank']; chosen['retained_count']=len(valid)
        chosen['screened_initializations']=self.initializations; chosen['refined_initializations']=len(order)
        return chosen,attempts




def sample_rule_based_skeletons(draw, rng, max_independent_sites, max_ti_atoms):
    spgs=[]; tokens=[]; attempts=0
    while len(spgs)<draw and attempts<draw*100:
        attempts+=1; spg=int(rng.integers(1,231))
        try: group=Group(spg)
        except Exception: continue
        allowed=[i for i in range(len(group)) if 1<=int(group[i].multiplicity)<=max_ti_atoms]
        if not allowed: continue
        nsite=int(rng.integers(1,min(max_independent_sites,3,len(allowed))+1))
        picked=[]; total=0
        for w in rng.permutation(allowed):
            m=int(group[int(w)].multiplicity)
            if total+m<=max_ti_atoms:
                picked.append(int(w)); total+=m
            if len(picked)>=nsite: break
        if not picked or total<2: continue
        padded=picked+[-1]*(max_independent_sites-len(picked))
        spgs.append(spg); tokens.append(encode_wp_token(padded))
    return spgs,tokens



def draw_proposals(model, mode, draw, rng, temperature, composition, num_wps, max_ti_atoms):
    """Draw a complete S,W proposal population with explicit source semantics."""
    proposals = []
    stats = {"invalid_space_group": 0, "no_compatible_si_skeleton": 0}
    if mode not in {"interpolation", "exploration", "mixed"}:
        raise ValueError(f"Unsupported --sw-mode: {mode}")

    if mode == "interpolation":
        n_interpolation, n_exploration = int(draw), 0
    elif mode == "exploration":
        n_interpolation, n_exploration = 0, int(draw)
    else:
        # Give the extra slot to exploration for odd populations.
        n_interpolation = int(draw) // 2
        n_exploration = int(draw) - n_interpolation

    if n_interpolation:
        state = model.sample_ti_skeletons(
            n_interpolation, temperature=temperature, hard=True,
            composition_ratio=composition, max_independent_sites=num_wps,
        )
        proposals.extend(
            (int(state["spg"][p]), str(state["si_skeleton_token"][p]), "interpolation")
            for p in np.flatnonzero(state["valid_mask"])
        )
        for key in stats:
            stats[key] += int(state["stats"].get(key, 0))

    if n_exploration:
        ss, ww = sample_rule_based_skeletons(
            n_exploration, rng, num_wps, max_ti_atoms
        )
        proposals.extend(zip(ss, ww, ["exploration"] * len(ss)))
    if mode == "mixed" and len(proposals) > 1:
        rng.shuffle(proposals)
    return proposals, stats


def resolve_ngpu(requested):
    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if requested < 0:
        raise ValueError("--ngpu cannot be negative.")
    if requested == 0:
        return visible
    if requested > visible:
        raise ValueError(
            f"Requested --ngpu={requested}, but only {visible} CUDA device(s) are visible."
        )
    return int(requested)


def deterministic_seed(global_seed, target_id, cycle, proposal_slot, initialization=0):
    payload = f"{global_seed}:{target_id}:{cycle}:{proposal_slot}:{initialization}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**31 - 1)


def proposal_hash(spg, token):
    return f"{int(spg)}:{str(token)}"


def candidate_score(item, target):
    invalid = 0 if item.get("minimum_ti_ti_distance", 0.0) >= 2.0 else 1
    return (
        invalid,
        abs(float(item.get("achieved_cn", 1e9)) - float(target.target_ti_cn)),
        float(item.get("distance_loss", 1e9)),
        float(item.get("angle_loss", 1e9)),
        float(item.get("total_loss", 1e9)),
    )


def build_output_row(result, spg, padded_token, num_wps, n_ti_max):
    lattice = result["lattice"]; free = result["free"]
    record = dict(zip(BASE_COLUMNS, [int(spg), *map(float, lattice)]))
    wps = [w for w in decode_wp_token(padded_token) if w >= 0]
    sites = []
    for i, wp in enumerate(wps):
        xyz = _wyckoff_position_from_parameters(spg, wp, free[i])
        sites.append((wp, xyz, TI_ROLE))
    sites += [(-1, np.full(3, -1.0), 0)] * (num_wps - len(sites))
    for i, (wp, xyz, role) in enumerate(sites):
        record[f"wp{i}"] = int(wp)
        record[f"x{i}"], record[f"y{i}"], record[f"z{i}"] = map(float, xyz)
        record[f"target_coord{i}"] = int(role)
    return record


def save_ti_cif(result, path):
    atoms = Atoms("Ti" * len(result["frac"]), scaled_positions=result["frac"],
                  cell=result["cell"], pbc=True)
    write(path, atoms, format="cif")


def parse_composition(value):
    parts = tuple(int(x.strip()) for x in value.split(","))
    if len(parts) != 2 or min(parts) <= 0:
        raise ValueError("--composition must contain two positive integers.")
    return parts



TIO_CUTOFF = 3.0
OO_BINS = np.linspace(0.0, 6.0, 13)


def oxygen_framework_from_row(row, num_wps):
    spg = int(row["spg"])
    group = Group(spg)
    frac = []
    for i in range(num_wps):
        if int(row[f"target_coord{i}"]) != O_ROLE:
            continue
        wp_index = int(row[f"wp{i}"])
        generator = np.asarray([row[f"x{i}"], row[f"y{i}"], row[f"z{i}"]], dtype=float)
        orbit = _deduplicate_fractional([op.operate(generator) for op in group[wp_index].ops])
        frac.extend(orbit.tolist())
    return _deduplicate_fractional(frac)


def periodic_cross_vectors(center_frac, neighbor_frac, cell):
    center_frac = np.asarray(center_frac, dtype=float)
    neighbor_frac = np.asarray(neighbor_frac, dtype=float)
    delta = neighbor_frac[None, :, None, :] - center_frac[:, None, None, :] + SHIFTS[None, None, :, :]
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
    hist = (weights[:, None] * np.exp(-0.5 * ((values[:, None] - centers[None, :]) / sigma) ** 2)).sum(0)
    return hist / max(hist.sum(), 1e-12)


def tio2_environment_descriptor(ti_frac, o_frac, cell, cutoff=TIO_CUTOFF):
    if len(ti_frac) < 1 or len(o_frac) < 1:
        raise ValueError("Ti/O framework is empty.")
    dist, vec = periodic_cross_vectors(ti_frac, o_frac, cell)
    hard = dist <= float(cutoff)
    if not np.any(hard):
        raise ValueError("No Ti-O neighbours inside cutoff.")
    ti_cn = hard.sum(1).astype(float)
    o_cn = hard.sum(0).astype(float)
    bond = dist[hard]
    width = max(float(np.std(bond)), 0.03)
    smooth = _stable_logistic_switch(dist, cutoff, max(width, 0.08))
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
                angle = np.degrees(np.arccos(np.clip(np.dot(vec[i, j], vec[i, k]) / (nj * nk), -1, 1)))
                angle_values.append(angle); angle_weights.append(w)
                oo_values.append(np.linalg.norm(vec[i, j] - vec[i, k])); oo_weights.append(w)
    angle_profile = _soft_histogram(angle_values, angle_weights, ANGLE_BINS)
    oo_profile = _soft_histogram(oo_values, oo_weights, OO_BINS)
    oo_distances, _ = periodic_neighbor_vectors(o_frac, cell)
    min_oo = min(float(d[0]) for d in oo_distances)
    return {
        "target_ti_o_cn": float(np.mean(ti_cn)),
        "target_ti_o_cn_std": float(np.std(ti_cn)),
        "target_o_ti_cn": float(np.mean(o_cn)),
        "target_o_ti_cn_std": float(np.std(o_cn)),
        "target_ti_o_mean": float(np.mean(bond)),
        "target_ti_o_width": width,
        "target_ti_o_cutoff": float(cutoff),
        "minimum_ti_o_distance": float(np.min(dist)),
        "minimum_o_o_distance": min_oo,
        "oti_o_angle_profile": angle_profile,
        "shell_o_o_profile": oo_profile,
    }


O_CHEMISTRY_CACHE_VERSION = 1


def _extract_oxygen_training_record(payload):
    row_index, row_dict, num_wps, cutoff, has_label = payload
    row = pd.Series(row_dict)
    try:
        ti_frac, cell = ti_framework_from_row(row, num_wps)
        o_frac = oxygen_framework_from_row(row, num_wps)
        ti_desc = framework_descriptor(ti_frac, cell, chemistry_cutoff=CHEMISTRY_CUTOFF)
        o_desc = tio2_environment_descriptor(ti_frac, o_frac, cell, cutoff=cutoff)
        source = str(row["label"]) if has_label else framework_fingerprint(ti_frac, cell)
        group = Group(int(row["spg"]))
        o_wps = [int(row[f"wp{i}"]) for i in range(num_wps)
                 if int(row[f"wp{i}"]) >= 0 and int(row[f"target_coord{i}"]) == O_ROLE]
        record = {
            "training_row": int(row_index), "source_group": source,
            "spg": int(row["spg"]), "lattice_type": str(group.lattice_type),
            "volume_per_ti": float(ti_desc["volume_per_ti"]),
            "ti_ti_cn": float(ti_desc["target_ti_cn"]),
            "ti_ti_mean": float(ti_desc["target_ti_nn_mean"]),
            "ti_ti_width": float(ti_desc["target_ti_nn_width"]),
            "o_skeleton_token_unpadded": encode_wp_token(o_wps),
            "n_ti": int(len(ti_frac)), "n_o": int(len(o_frac)),
            **{k: v for k, v in o_desc.items() if k not in {"oti_o_angle_profile", "shell_o_o_profile"}},
        }
        for i, value in enumerate(o_desc["oti_o_angle_profile"]):
            record[f"oti_o_angle_bin_{i}"] = float(value)
        for i, value in enumerate(o_desc["shell_o_o_profile"]):
            record[f"shell_o_o_bin_{i}"] = float(value)
        return record, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class TiOChemistryTarget:
    source_group: str
    target_ti_o_cn: float
    target_ti_o_cn_std: float
    target_o_ti_cn: float
    target_o_ti_cn_std: float
    target_ti_o_mean: float
    target_ti_o_width: float
    target_ti_o_cutoff: float
    minimum_ti_o_distance: float
    minimum_o_o_distance: float
    oti_o_angle_profile: np.ndarray
    shell_o_o_profile: np.ndarray


class OxygenTrainingDistribution:
    def __init__(self, canonical_df, num_wps, cutoff=TIO_CUTOFF, ncpu=1,
                 cache_csv=None, cache_meta=None, cache_key=None):
        if cache_csv and cache_meta and os.path.exists(cache_csv) and os.path.exists(cache_meta):
            try:
                with open(cache_meta, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
                if meta.get("cache_key") == cache_key and meta.get("version") == O_CHEMISTRY_CACHE_VERSION:
                    self.frame = pd.read_csv(cache_csv)
                    self.failures = int(meta.get("failures", 0))
                    print(f"Reused oxygen chemistry cache: {cache_csv}", flush=True)
                    return
            except Exception as exc:
                print(f"Ignoring invalid oxygen chemistry cache: {type(exc).__name__}: {exc}", flush=True)
        payloads = [(int(idx), row.to_dict(), int(num_wps), float(cutoff), "label" in canonical_df.columns)
                    for idx, row in canonical_df.iterrows()]
        records, failures = [], 0
        print(f"Extracting Ti-O/O-O chemistry from {len(payloads)} rows with {ncpu} CPU worker(s)...", flush=True)
        if int(ncpu) == 1:
            iterator = map(_extract_oxygen_training_record, payloads)
            for done, (record, error) in enumerate(iterator, 1):
                failures += int(record is None)
                if record is not None: records.append(record)
                if done % 500 == 0 or done == len(payloads):
                    print(f"O chemistry extraction: {done}/{len(payloads)}; valid={len(records)}; failures={failures}", flush=True)
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=int(ncpu), mp_context=ctx,
                                     initializer=set_worker_thread_limits) as pool:
                iterator = pool.map(_extract_oxygen_training_record, payloads, chunksize=8)
                for done, (record, error) in enumerate(iterator, 1):
                    failures += int(record is None)
                    if record is not None: records.append(record)
                    if done % 500 == 0 or done == len(payloads):
                        print(f"O chemistry extraction: {done}/{len(payloads)}; valid={len(records)}; failures={failures}", flush=True)
        if not records:
            raise RuntimeError("No valid Ti-O chemistry records could be extracted.")
        self.frame = pd.DataFrame(records).sort_values("training_row").reset_index(drop=True)
        self.failures = failures
        if cache_csv and cache_meta:
            self.frame.to_csv(cache_csv, index=False)
            with open(cache_meta, "w", encoding="utf-8") as handle:
                json.dump({"version": O_CHEMISTRY_CACHE_VERSION, "cache_key": cache_key,
                           "failures": failures}, handle, indent=2)

    def _row_to_target(self, row):
        angles = np.asarray([row[f"oti_o_angle_bin_{i}"] for i in range(len(ANGLE_BINS)-1)], float)
        oo = np.asarray([row[f"shell_o_o_bin_{i}"] for i in range(len(OO_BINS)-1)], float)
        return TiOChemistryTarget(
            source_group=str(row.source_group), target_ti_o_cn=float(row.target_ti_o_cn),
            target_ti_o_cn_std=float(row.target_ti_o_cn_std), target_o_ti_cn=float(row.target_o_ti_cn),
            target_o_ti_cn_std=float(row.target_o_ti_cn_std), target_ti_o_mean=float(row.target_ti_o_mean),
            target_ti_o_width=float(row.target_ti_o_width), target_ti_o_cutoff=float(row.target_ti_o_cutoff),
            minimum_ti_o_distance=float(row.minimum_ti_o_distance), minimum_o_o_distance=float(row.minimum_o_o_distance),
            oti_o_angle_profile=angles, shell_o_o_profile=oo,
        )

    def sample_for_framework(self, framework, rng, nearest=64):
        frame = self.frame
        same = frame[frame.lattice_type == framework["lattice_type"]]
        candidates = same if len(same) >= min(16, nearest) else frame
        scales = {
            "ti_ti_cn": max(float(frame.ti_ti_cn.std()), 0.5),
            "ti_ti_mean": max(float(frame.ti_ti_mean.std()), 0.1),
            "volume_per_ti": max(float(np.log(frame.volume_per_ti.clip(lower=1e-6)).std()), 0.1),
        }
        score = ((candidates.ti_ti_cn - framework["ti_ti_cn"]) / scales["ti_ti_cn"]) ** 2
        score += ((candidates.ti_ti_mean - framework["ti_ti_mean"]) / scales["ti_ti_mean"]) ** 2
        score += ((np.log(candidates.volume_per_ti.clip(lower=1e-6)) - math.log(max(framework["volume_per_ti"],1e-6))) / scales["volume_per_ti"]) ** 2
        subset = candidates.loc[score.nsmallest(min(int(nearest), len(candidates))).index]
        groups = subset.source_group.unique()
        source = rng.choice(groups)
        source_rows = subset[subset.source_group == source]
        return self._row_to_target(source_rows.iloc[int(rng.integers(0, len(source_rows)))])

    def learned_tokens(self, spg, n_o, max_sites):
        subset = self.frame[(self.frame.spg == int(spg)) & (self.frame.n_o == int(n_o))]
        tokens = []
        for token in subset.o_skeleton_token_unpadded.astype(str):
            values = decode_wp_token(token)
            if len(values) <= int(max_sites): tokens.append(token)
        return tokens


def _canonical_o_token(values, max_sites):
    values = sorted(int(v) for v in values)
    return encode_wp_token(values + [-1] * (int(max_sites) - len(values)))


def sample_legal_o_skeletons(spg, required_o, draw, max_sites, rng, learned_tokens=(), mode="exploration"):
    group = Group(int(spg))
    mult = [int(group[i].multiplicity) for i in range(len(group))]
    allowed = [i for i, m in enumerate(mult) if 1 <= m <= required_o]
    learned = []
    for token in learned_tokens:
        vals = [w for w in decode_wp_token(token) if w >= 0]
        if len(vals) <= max_sites and sum(mult[w] for w in vals) == required_o:
            learned.append(_canonical_o_token(vals, max_sites))
    learned = list(dict.fromkeys(learned))
    if mode == "interpolation": n_l, n_e = draw, 0
    elif mode == "exploration": n_l, n_e = 0, draw
    elif mode == "mixed": n_l, n_e = draw // 2, draw - draw // 2
    else: raise ValueError(f"Unsupported oxygen proposal mode: {mode}")
    proposals = []
    if n_l and learned:
        order = rng.permutation(len(learned))
        for i in order[:min(n_l, len(learned))]: proposals.append((learned[int(i)], "interpolation"))
    elif n_l:
        n_e += n_l

    # Enumerate exact multiplicity multisets with replacement. Repeated Wyckoff
    # classes are legal because they represent separate independent orbits.
    legal = []
    cap = max(5000, 200 * max(1, draw))
    def visit(start, remaining, picked):
        if len(legal) >= cap: return
        if remaining == 0:
            if picked: legal.append(_canonical_o_token(picked, max_sites))
            return
        if len(picked) >= max_sites: return
        slots_left = max_sites - len(picked)
        for pos in range(start, len(allowed)):
            w = allowed[pos]; m = mult[w]
            if m > remaining: continue
            rem = remaining - m
            if rem and slots_left <= 1: continue
            visit(pos, rem, picked + [w])
            if len(legal) >= cap: return
    visit(0, int(required_o), [])
    legal = list(dict.fromkeys(legal))
    existing = {x[0] for x in proposals}
    legal = [token for token in legal if token not in existing]
    if legal and n_e:
        order = rng.permutation(len(legal))
        proposals.extend((legal[int(i)], "exploration") for i in order[:min(n_e, len(legal))])
    if mode == "mixed" and len(proposals) > 1: rng.shuffle(proposals)
    return proposals[:draw]


class TorchSymmetryConstrainedOBuilder:
    def __init__(self, initializations=8, screen_steps=25, refine_starts=2, refine_steps=60,
                 ti_cn_tolerance=0.75, o_cn_tolerance=0.75, distance_tolerance=0.35,
                 minimum_ti_o=1.45, minimum_o_o=1.6, maximum_loss=8.0,
                 max_neighbors=12, lr=0.06, device=None, seed=42):
        self.initializations=int(initializations); self.screen_steps=int(screen_steps)
        self.refine_starts=int(refine_starts); self.refine_steps=int(refine_steps)
        self.ti_cn_tolerance=float(ti_cn_tolerance); self.o_cn_tolerance=float(o_cn_tolerance)
        self.distance_tolerance=float(distance_tolerance); self.minimum_ti_o=float(minimum_ti_o)
        self.minimum_o_o=float(minimum_o_o); self.maximum_loss=float(maximum_loss)
        self.max_neighbors=int(max_neighbors); self.lr=float(lr)
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng=np.random.default_rng(seed); self._template_cache={}
        self._shifts=torch.as_tensor(SHIFTS,dtype=torch.float32,device=self.device)

    @staticmethod
    def _affine_map(function, dof):
        zero=np.asarray(function(np.zeros(dof)),float); matrix=np.zeros((3,dof),float)
        for k in range(dof):
            x=np.zeros(dof); x[k]=0.137; y=np.asarray(function(x),float)
            matrix[:,k]=((y-zero+0.5)%1.0-0.5)/0.137
        return matrix,zero

    def _template(self, spg, token):
        key=(int(spg),str(token))
        if key in self._template_cache: return self._template_cache[key]
        group=Group(int(spg)); wps=tuple(w for w in decode_wp_token(token) if w>=0)
        if not wps: return None
        site_dofs=[]; orbit_rot=[]; orbit_trans=[]; gen_A=[]; gen_b=[]
        for w in wps:
            wp=group[w]; dof=int(wp.get_dof()); site_dofs.append(dof)
            A,b=self._affine_map(lambda u,wp=wp:wp.get_position_from_free_xyzs(u),dof)
            gen_A.append(torch.tensor(A,dtype=torch.float32,device=self.device)); gen_b.append(torch.tensor(b,dtype=torch.float32,device=self.device))
            rots=[]; trans=[]
            for op in wp.ops:
                o=np.asarray(op.operate([0.,0.,0.]),float); cols=[]
                for axis in range(3):
                    e=np.zeros(3); e[axis]=0.173; q=np.asarray(op.operate(e),float)
                    cols.append(((q-o+0.5)%1.0-0.5)/0.173)
                rots.append(np.stack(cols,axis=1)); trans.append(o)
            orbit_rot.append(torch.tensor(np.asarray(rots),dtype=torch.float32,device=self.device))
            orbit_trans.append(torch.tensor(np.asarray(trans),dtype=torch.float32,device=self.device))
        out={'wps':wps,'site_dofs':tuple(site_dofs),'gen_A':gen_A,'gen_b':gen_b,
             'orbit_rot':orbit_rot,'orbit_trans':orbit_trans,
             'n_o':sum(int(group[w].multiplicity) for w in wps)}
        self._template_cache[key]=out; return out

    def _expand(self, template, raw):
        B=raw.shape[0]; cursor=0; frac=[]; free=[]
        for dof,A,b,R,t in zip(template['site_dofs'],template['gen_A'],template['gen_b'],template['orbit_rot'],template['orbit_trans']):
            u=torch.sigmoid(raw[:,cursor:cursor+dof]); cursor+=dof; free.append(u)
            gen=(u@A.T+b)%1.0
            orbit=(torch.einsum('oij,bj->boi',R,gen)+t[None,:,:])%1.0
            frac.append(orbit)
        return torch.cat(frac,1),free

    def _geometry(self, template, framework, target, raw, include_topology):
        o_frac,free=self._expand(template,raw); B,No=o_frac.shape[:2]
        ti=torch.as_tensor(framework['ti_frac'],dtype=torch.float32,device=self.device)
        cell=torch.as_tensor(framework['cell'],dtype=torch.float32,device=self.device)
        Nt=ti.shape[0]
        delta=o_frac[:,None,:,None,:]-ti[None,:,None,None,:]+self._shifts[None,None,None,:,:]
        vec_all=torch.einsum('btosk,kl->btosl',delta,cell)
        dist_all=torch.linalg.norm(vec_all,dim=-1).clamp_min(1e-6)
        dist,image=dist_all.min(-1)
        vec=torch.gather(vec_all,3,image[...,None,None].expand(-1,-1,-1,1,3)).squeeze(3)
        width=max(float(target.target_ti_o_width),0.08); cutoff=float(target.target_ti_o_cutoff)
        w=torch.sigmoid((cutoff-dist)/width)
        ti_cn=w.sum(-1); o_cn=w.sum(1)
        ti_cn_loss=((ti_cn-float(target.target_ti_o_cn))**2).mean(1)
        o_cn_loss=((o_cn-float(target.target_o_ti_cn))**2).mean(1)
        denom=w.sum((1,2)).clamp_min(1e-6)
        mean=(w*dist).sum((1,2))/denom
        var=(w*(dist-mean[:,None,None])**2).sum((1,2))/denom
        distance_loss=((mean-float(target.target_ti_o_mean))/max(width,0.12))**2
        distance_loss+=0.5*((torch.sqrt(var+1e-8)-float(target.target_ti_o_width))/max(width,0.12))**2
        min_tio=dist.amin((1,2)); tio_overlap=torch.relu(self.minimum_ti_o-min_tio).pow(2)/max(self.minimum_ti_o**2,0.1)
        d_oo=o_frac[:,:,None,None,:]-o_frac[:,None,:,None,:]+self._shifts[None,None,None,:,:]
        v_oo=torch.einsum('bijnk,kl->bijnl',d_oo,cell); r_oo=torch.linalg.norm(v_oo,dim=-1).clamp_min(1e-6)
        eye=torch.eye(No,device=self.device,dtype=torch.bool)[None,:,:,None]
        zero=(torch.arange(27,device=self.device)==ZERO_SHIFT)[None,None,None,:]
        r_oo=r_oo.masked_fill(eye & zero,1e6)
        min_oo=r_oo.amin((1,2,3)); oo_overlap=torch.relu(self.minimum_o_o-min_oo).pow(2)/max(self.minimum_o_o**2,0.1)
        angle_loss=torch.zeros(B,device=self.device); shell_oo_loss=torch.zeros(B,device=self.device)
        if include_topology:
            K=min(self.max_neighbors,No); topd,idx=torch.topk(dist,K,dim=-1,largest=False)
            topv=torch.gather(vec,2,idx[...,None].expand(-1,-1,-1,3))
            norm=torch.linalg.norm(topv,dim=-1).clamp_min(1e-6)
            cos=torch.einsum('btik,btjk->btij',topv,topv)/(norm[:,:,:,None]*norm[:,:,None,:])
            pw=torch.sigmoid((cutoff-topd)/width); pairw=pw[:,:,:,None]*pw[:,:,None,:]
            tri=torch.triu(torch.ones((K,K),device=self.device,dtype=torch.bool),diagonal=1)
            cosv=cos[:,:,tri]; pvw=pairw[:,:,tri]
            centers=torch.cos(torch.tensor(0.5*(ANGLE_BINS[:-1]+ANGLE_BINS[1:])*math.pi/180,device=self.device))
            ah=(pvw[...,None]*torch.exp(-0.5*((cosv[...,None]-centers)/0.16)**2)).sum((1,2))
            ah=ah/ah.sum(-1,keepdim=True).clamp_min(1e-8)
            at=torch.tensor(target.oti_o_angle_profile,dtype=torch.float32,device=self.device)
            angle_loss=((ah-at[None,:])**2).mean(-1)*len(at)
            pairdist=torch.linalg.norm(topv[:,:,:,None,:]-topv[:,:,None,:,:],dim=-1)[:,:,tri]
            oc=torch.tensor(0.5*(OO_BINS[:-1]+OO_BINS[1:]),dtype=torch.float32,device=self.device)
            osig=max(float(np.diff(OO_BINS).mean())*0.45,0.08)
            oh=(pvw[...,None]*torch.exp(-0.5*((pairdist[...,None]-oc)/osig)**2)).sum((1,2))
            oh=oh/oh.sum(-1,keepdim=True).clamp_min(1e-8)
            ot=torch.tensor(target.shell_o_o_profile,dtype=torch.float32,device=self.device)
            shell_oo_loss=((oh-ot[None,:])**2).mean(-1)*len(ot)
        total=ti_cn_loss+0.6*o_cn_loss+distance_loss+0.7*angle_loss+0.4*shell_oo_loss+6.0*tio_overlap+6.0*oo_overlap
        detail={'ti_cn_loss':ti_cn_loss,'o_cn_loss':o_cn_loss,'distance_loss':distance_loss,
                'angle_loss':angle_loss,'shell_o_o_loss':shell_oo_loss,'ti_o_overlap_loss':tio_overlap,
                'o_o_overlap_loss':oo_overlap,'achieved_ti_o_cn':ti_cn.mean(1),'ti_o_cn_std':ti_cn.std(1,unbiased=False),
                'achieved_o_ti_cn':o_cn.mean(1),'o_ti_cn_std':o_cn.std(1,unbiased=False),
                'achieved_ti_o_mean':mean,'achieved_ti_o_width':torch.sqrt(var+1e-8),
                'minimum_ti_o_distance':min_tio,'minimum_o_o_distance':min_oo,
                'ti_cn_q90_error':torch.quantile(torch.abs(ti_cn-float(target.target_ti_o_cn)),0.9,dim=1),
                'o_cn_q90_error':torch.quantile(torch.abs(o_cn-float(target.target_o_ti_cn)),0.9,dim=1)}
        return total,detail,(o_frac,free)

    def _optimize(self,template,framework,target,raw,steps,include_topology):
        if raw.shape[1] == 0:
            return raw.detach()
        raw=raw.detach().clone().requires_grad_(True); opt=torch.optim.Adam([raw],lr=self.lr)
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True); total,_,_=self._geometry(template,framework,target,raw,include_topology)
            total.mean().backward(); torch.nn.utils.clip_grad_norm_([raw],10.0); opt.step()
        return raw.detach()

    def build(self,framework,token,target,sample_id):
        template=self._template(framework['spg'],token)
        if template is None or template['n_o'] != 2*len(framework['ti_frac']): return None,[]
        nvar=sum(template['site_dofs']); raw=torch.randn((self.initializations,nvar),device=self.device)
        raw=self._optimize(template,framework,target,raw,self.screen_steps,False)
        scores=[]
        with torch.no_grad():
            for i in range(len(raw)):
                val,_,_=self._geometry(template,framework,target,raw[i:i+1],False); scores.append(float(val[0]))
        order=np.argsort(scores)[:min(self.refine_starts,len(scores))]
        refined=self._optimize(template,framework,target,raw[order],self.refine_steps,True)
        attempts=[]
        with torch.no_grad():
            for rank in range(len(refined)):
                total,detail,geom=self._geometry(template,framework,target,refined[rank:rank+1],True)
                o_frac,free=geom; free_np=np.zeros((len(template['wps']),3),float)
                for j,u in enumerate(free): free_np[j,:u.shape[1]]=u[0].cpu().numpy()
                item={'sample_id':sample_id,'initialization_id':int(order[rank]),'screen_rank':rank,
                      'screen_loss':float(scores[order[rank]]),'success':bool(torch.isfinite(total[0])),
                      'total_loss':float(total[0]),'o_free':free_np,'o_frac':o_frac[0].cpu().numpy()}
                for k,v in detail.items(): item[k]=float(v[0])
                item['ti_cn_error']=abs(item['achieved_ti_o_cn']-target.target_ti_o_cn)
                item['o_cn_error']=abs(item['achieved_o_ti_cn']-target.target_o_ti_cn)
                item['distance_error']=abs(item['achieved_ti_o_mean']-target.target_ti_o_mean)
                item['chemistry_success']=bool(item['success'] and item['ti_cn_error']<=self.ti_cn_tolerance and
                    item['o_cn_error']<=self.o_cn_tolerance and item['distance_error']<=self.distance_tolerance and
                    item['minimum_ti_o_distance']>=self.minimum_ti_o and item['minimum_o_o_distance']>=self.minimum_o_o and
                    item['total_loss']<=self.maximum_loss)
                attempts.append(item)
        attempts.sort(key=lambda x:(not x['chemistry_success'],x['ti_cn_error'],x['distance_error'],x['total_loss']))
        return (next((x for x in attempts if x['chemistry_success']),attempts[0] if attempts else None),attempts)






def phase_a_row_to_framework(row, diag, num_wps):
    spg=int(row['spg']); group=Group(spg); ti_frac=[]; ti_wps=[]
    for i in range(num_wps):
        if int(row.get(f'target_coord{i}',0)) != TI_ROLE: continue
        wp=int(row[f'wp{i}']); ti_wps.append(wp)
        xyz=np.asarray([row[f'x{i}'],row[f'y{i}'],row[f'z{i}']],float)
        ti_frac.extend(_deduplicate_fractional([op.operate(xyz) for op in group[wp].ops]).tolist())
    ti_frac=_deduplicate_fractional(ti_frac); cell=_cell_matrix(np.asarray([row[c] for c in BASE_COLUMNS[1:]],float))
    desc=framework_descriptor(ti_frac,cell)
    return {'spg':spg,'lattice_type':str(group.lattice_type),'cell':cell,'ti_frac':ti_frac,
            'ti_wps':ti_wps,'ti_ti_cn':float(desc['target_ti_cn']),'ti_ti_mean':float(desc['target_ti_nn_mean']),
            'volume_per_ti':float(desc['volume_per_ti']),'row':dict(row),'phase_a_diag':diag}


def build_complete_output_row(framework, oxygen_result, token, num_wps):
    record=dict(framework['row']); sites=[]
    for i,wp in enumerate(framework['ti_wps']):
        xyz=np.asarray([record[f'x{i}'],record[f'y{i}'],record[f'z{i}']],float); sites.append((wp,xyz,TI_ROLE))
    o_wps=[w for w in decode_wp_token(token) if w>=0]
    for i,wp in enumerate(o_wps):
        xyz=_wyckoff_position_from_parameters(framework['spg'],wp,oxygen_result['o_free'][i]); sites.append((wp,xyz,O_ROLE))
    if len(sites)>num_wps: raise ValueError(f"Complete structure needs {len(sites)} independent sites, capacity is {num_wps}.")
    sites += [(-1,np.full(3,-1.0),0)]*(num_wps-len(sites))
    for i,(wp,xyz,role) in enumerate(sites):
        record[f'wp{i}']=int(wp); record[f'target_coord{i}']=int(role)
        record[f'x{i}'],record[f'y{i}'],record[f'z{i}']=map(float,xyz)
    return record


def save_tio2_cif(framework,oxygen_result,path):
    frac=np.vstack([framework['ti_frac'],oxygen_result['o_frac']]); symbols=['Ti']*len(framework['ti_frac'])+['O']*len(oxygen_result['o_frac'])
    write(path,Atoms(symbols,scaled_positions=frac,cell=framework['cell'],pbc=True),format='cif')

def oxygen_candidate_score(item,target):
    return (not item.get('chemistry_success',False),
            abs(float(item.get('achieved_ti_o_cn',1e9))-float(target.target_ti_o_cn)),
            abs(float(item.get('achieved_ti_o_mean',1e9))-float(target.target_ti_o_mean)),
            float(item.get('angle_loss',1e9)),float(item.get('shell_o_o_loss',1e9)),
            float(item.get('total_loss',1e9)))






def _safe_r2(target, achieved):
    target = np.asarray(target, dtype=float)
    achieved = np.asarray(achieved, dtype=float)
    mask = np.isfinite(target) & np.isfinite(achieved)
    target, achieved = target[mask], achieved[mask]
    if target.size < 2:
        return np.nan
    denom = float(np.sum((target - np.mean(target)) ** 2))
    if denom <= 1e-14:
        return np.nan
    return float(1.0 - np.sum((achieved - target) ** 2) / denom)


def _error_summary(target, achieved):
    target = np.asarray(target, dtype=float)
    achieved = np.asarray(achieved, dtype=float)
    mask = np.isfinite(target) & np.isfinite(achieved)
    target, achieved = target[mask], achieved[mask]
    if target.size == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan,
                "q50_abs_error": np.nan, "q90_abs_error": np.nan,
                "q95_abs_error": np.nan, "r2": np.nan}
    err = achieved - target
    ae = np.abs(err)
    return {
        "n": int(target.size),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "q50_abs_error": float(np.quantile(ae, 0.50)),
        "q90_abs_error": float(np.quantile(ae, 0.90)),
        "q95_abs_error": float(np.quantile(ae, 0.95)),
        "r2": _safe_r2(target, achieved),
    }


def _jensen_shannon(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, 0.0, None); q = np.clip(q, 0.0, None)
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


def _empirical_wasserstein_1(a, b):
    a = np.sort(np.asarray(a, dtype=float)); b = np.sort(np.asarray(b, dtype=float))
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    grid = np.unique(np.concatenate([a, b]))
    if grid.size < 2:
        return 0.0
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.sum(np.abs(cdf_a[:-1] - cdf_b[:-1]) * np.diff(grid)))


def _ks_statistic(a, b):
    a = np.sort(np.asarray(a, dtype=float)); b = np.sort(np.asarray(b, dtype=float))
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    grid = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def evaluate_pre_relaxation(final_rows, final_diag, selected_indices, num_wps,
                            oxygen_training, output_folder, cutoff):
    """Evaluate raw symmetry-constrained TiO2 candidates before decoding/relaxation."""
    metric_rows = []
    selected_index_set = set(int(i) for i in selected_indices)
    angle_target_cols = [f"target_angle_bin_{i}" for i in range(len(ANGLE_BINS) - 1)]
    angle_ach_cols = [f"achieved_angle_bin_{i}" for i in range(len(ANGLE_BINS) - 1)]
    oo_target_cols = [f"target_shell_o_o_bin_{i}" for i in range(len(OO_BINS) - 1)]
    oo_ach_cols = [f"achieved_shell_o_o_bin_{i}" for i in range(len(OO_BINS) - 1)]
    for pool_index, (row, diag) in enumerate(zip(final_rows, final_diag)):
        series = pd.Series(row)
        ti_frac, cell = ti_framework_from_row(series, num_wps)
        o_frac = oxygen_framework_from_row(series, num_wps)
        achieved = tio2_environment_descriptor(ti_frac, o_frac, cell, cutoff=float(cutoff))
        record = {
            "candidate_id": int(diag["candidate_id"]),
            "pool_index": int(pool_index),
            "selected": bool(pool_index in selected_index_set),
            "final_rank": np.nan,
            "spg": int(diag["spg"]),
            "ranking_score": float(diag["ranking_score"]),
            "total_loss": float(diag["total_loss"]),
            "target_ti_o_cn": float(diag["target_ti_o_cn"]),
            "achieved_ti_o_cn": float(achieved["target_ti_o_cn"]),
            "target_ti_o_cn_std": float(diag["target_ti_o_cn_std"]),
            "achieved_ti_o_cn_std": float(achieved["target_ti_o_cn_std"]),
            "target_o_ti_cn": float(diag["target_o_ti_cn"]),
            "achieved_o_ti_cn": float(achieved["target_o_ti_cn"]),
            "target_o_ti_cn_std": float(diag["target_o_ti_cn_std"]),
            "achieved_o_ti_cn_std": float(achieved["target_o_ti_cn_std"]),
            "target_ti_o_mean": float(diag["target_ti_o_mean"]),
            "achieved_ti_o_mean": float(achieved["target_ti_o_mean"]),
            "target_ti_o_width": float(diag["target_ti_o_width"]),
            "achieved_ti_o_width": float(achieved["target_ti_o_width"]),
            "minimum_ti_o_distance": float(achieved["minimum_ti_o_distance"]),
            "minimum_o_o_distance": float(achieved["minimum_o_o_distance"]),
        }
        target_angle = np.asarray([diag[c] for c in angle_target_cols], dtype=float)
        target_oo = np.asarray([diag[c] for c in oo_target_cols], dtype=float)
        achieved_angle = np.asarray(achieved["oti_o_angle_profile"], dtype=float)
        achieved_oo = np.asarray(achieved["shell_o_o_profile"], dtype=float)
        record["angle_jsd"] = _jensen_shannon(target_angle, achieved_angle)
        record["shell_o_o_jsd"] = _jensen_shannon(target_oo, achieved_oo)
        record.update({c: float(v) for c, v in zip(angle_target_cols, target_angle)})
        record.update({c: float(v) for c, v in zip(angle_ach_cols, achieved_angle)})
        record.update({c: float(v) for c, v in zip(oo_target_cols, target_oo)})
        record.update({c: float(v) for c, v in zip(oo_ach_cols, achieved_oo)})
        metric_rows.append(record)
    metrics = pd.DataFrame(metric_rows)
    rank_lookup = {int(pool_idx): rank + 1 for rank, pool_idx in enumerate(selected_indices)}
    metrics["final_rank"] = [rank_lookup.get(int(i), np.nan) for i in metrics["pool_index"]]
    metrics.to_csv(os.path.join(output_folder, "pre_relaxation_candidate_metrics.csv"), index=False)

    scalar_pairs = [
        ("ti_o_cn", "target_ti_o_cn", "achieved_ti_o_cn"),
        ("ti_o_cn_std", "target_ti_o_cn_std", "achieved_ti_o_cn_std"),
        ("o_ti_cn", "target_o_ti_cn", "achieved_o_ti_cn"),
        ("o_ti_cn_std", "target_o_ti_cn_std", "achieved_o_ti_cn_std"),
        ("ti_o_mean", "target_ti_o_mean", "achieved_ti_o_mean"),
        ("ti_o_width", "target_ti_o_width", "achieved_ti_o_width"),
    ]
    summary_rows = []
    for population, frame in (("accepted_pool", metrics), ("selected", metrics[metrics.selected])):
        for metric, target_col, achieved_col in scalar_pairs:
            rec = {"population": population, "metric": metric, "metric_type": "target_realization"}
            rec.update(_error_summary(frame[target_col], frame[achieved_col]))
            rec.update(wasserstein_1=np.nan, ks_statistic=np.nan, jsd=np.nan)
            summary_rows.append(rec)
        for metric in ("angle_jsd", "shell_o_o_jsd", "ranking_score", "total_loss",
                       "minimum_ti_o_distance", "minimum_o_o_distance"):
            values = frame[metric].to_numpy(float)
            summary_rows.append({
                "population": population, "metric": metric, "metric_type": "distribution_summary",
                "n": int(np.isfinite(values).sum()), "mae": np.nan, "rmse": np.nan, "bias": np.nan,
                "q50_abs_error": float(np.nanquantile(values, 0.50)),
                "q90_abs_error": float(np.nanquantile(values, 0.90)),
                "q95_abs_error": float(np.nanquantile(values, 0.95)), "r2": np.nan,
                "wasserstein_1": np.nan, "ks_statistic": np.nan, "jsd": np.nan,
            })

        training_frame = oxygen_training.frame
        distribution_pairs = [
            ("ti_o_cn", "achieved_ti_o_cn", "target_ti_o_cn"),
            ("ti_o_cn_std", "achieved_ti_o_cn_std", "target_ti_o_cn_std"),
            ("o_ti_cn", "achieved_o_ti_cn", "target_o_ti_cn"),
            ("o_ti_cn_std", "achieved_o_ti_cn_std", "target_o_ti_cn_std"),
            ("ti_o_mean", "achieved_ti_o_mean", "target_ti_o_mean"),
            ("ti_o_width", "achieved_ti_o_width", "target_ti_o_width"),
        ]
        for metric, generated_col, training_col in distribution_pairs:
            summary_rows.append({
                "population": population, "metric": metric, "metric_type": "training_distribution",
                "n": int(len(frame)), "mae": np.nan, "rmse": np.nan, "bias": np.nan,
                "q50_abs_error": np.nan, "q90_abs_error": np.nan, "q95_abs_error": np.nan,
                "r2": np.nan,
                "wasserstein_1": _empirical_wasserstein_1(frame[generated_col], training_frame[training_col]),
                "ks_statistic": _ks_statistic(frame[generated_col], training_frame[training_col]),
                "jsd": np.nan,
            })
        train_angle = training_frame[[f"oti_o_angle_bin_{i}" for i in range(len(ANGLE_BINS)-1)]].to_numpy(float).mean(0)
        train_oo = training_frame[[f"shell_o_o_bin_{i}" for i in range(len(OO_BINS)-1)]].to_numpy(float).mean(0)
        gen_angle = frame[angle_ach_cols].to_numpy(float).mean(0)
        gen_oo = frame[oo_ach_cols].to_numpy(float).mean(0)
        summary_rows.extend([
            {"population": population, "metric": "angle_profile", "metric_type": "training_distribution",
             "n": int(len(frame)), "mae": np.nan, "rmse": np.nan, "bias": np.nan,
             "q50_abs_error": np.nan, "q90_abs_error": np.nan, "q95_abs_error": np.nan, "r2": np.nan,
             "wasserstein_1": np.nan, "ks_statistic": np.nan, "jsd": _jensen_shannon(gen_angle, train_angle)},
            {"population": population, "metric": "shell_o_o_profile", "metric_type": "training_distribution",
             "n": int(len(frame)), "mae": np.nan, "rmse": np.nan, "bias": np.nan,
             "q50_abs_error": np.nan, "q90_abs_error": np.nan, "q95_abs_error": np.nan, "r2": np.nan,
             "wasserstein_1": np.nan, "ks_statistic": np.nan, "jsd": _jensen_shannon(gen_oo, train_oo)},
        ])

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(output_folder, "pre_relaxation_metrics_summary.csv"), index=False)
    with open(os.path.join(output_folder, "pre_relaxation_metrics_summary.json"), "w", encoding="utf-8") as handle:
        json_ready = summary.astype(object).where(pd.notna(summary), None).to_dict(orient="records")
        json.dump(json_ready, handle, indent=2)

    print("Pre-relaxation target-realization metrics (accepted pool -> selected):", flush=True)
    for metric, _, _ in scalar_pairs:
        pool_rec = summary[(summary.population == "accepted_pool") & (summary.metric == metric) &
                           (summary.metric_type == "target_realization")].iloc[0]
        sel_rec = summary[(summary.population == "selected") & (summary.metric == metric) &
                          (summary.metric_type == "target_realization")].iloc[0]
        print(f"  {metric:14s} MAE {pool_rec.mae:.6g} -> {sel_rec.mae:.6g}; "
              f"q90 {pool_rec.q90_abs_error:.6g} -> {sel_rec.q90_abs_error:.6g}", flush=True)
    for metric in ("angle_jsd", "shell_o_o_jsd", "ranking_score"):
        pool_rec = summary[(summary.population == "accepted_pool") & (summary.metric == metric) &
                           (summary.metric_type == "distribution_summary")].iloc[0]
        sel_rec = summary[(summary.population == "selected") & (summary.metric == metric) &
                          (summary.metric_type == "distribution_summary")].iloc[0]
        print(f"  {metric:14s} median {pool_rec.q50_abs_error:.6g} -> {sel_rec.q50_abs_error:.6g}; "
              f"q90 {pool_rec.q90_abs_error:.6g} -> {sel_rec.q90_abs_error:.6g}", flush=True)
    return {
        "candidate_metrics_csv": os.path.join(output_folder, "pre_relaxation_candidate_metrics.csv"),
        "summary_csv": os.path.join(output_folder, "pre_relaxation_metrics_summary.csv"),
        "summary_json": os.path.join(output_folder, "pre_relaxation_metrics_summary.json"),
    }


@dataclass
class StreamingTiState:
    target_id: int
    target: TiChemistryTarget
    cycle: int = 0
    attempted: set | None = None
    elites: list | None = None
    cycle_pool: list | None = None
    proposals: list | None = None
    submitted: int = 0
    returned: int = 0

    def __post_init__(self):
        self.attempted = set() if self.attempted is None else self.attempted
        self.elites = [] if self.elites is None else self.elites
        self.cycle_pool = [] if self.cycle_pool is None else self.cycle_pool
        self.proposals = [] if self.proposals is None else self.proposals


@dataclass
class StreamingOState:
    target_id: int
    phase_a_sample_id: int
    framework: dict
    target: TiOChemistryTarget
    cycle: int = 0
    attempted: set | None = None
    cycle_pool: list | None = None
    proposals: list | None = None
    submitted: int = 0
    returned: int = 0
    no_legal_proposals: bool = False

    def __post_init__(self):
        self.attempted = set() if self.attempted is None else self.attempted
        self.cycle_pool = [] if self.cycle_pool is None else self.cycle_pool
        self.proposals = [] if self.proposals is None else self.proposals


def _streaming_builder_worker(worker_id, device_id, task_queue, result_queue, ti_builder_config, o_builder_config):
    """Persistent worker owning both Phase-A and Phase-B builders on one device."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    if device_id is None:
        device = "cpu"
    else:
        torch.cuda.set_device(int(device_id))
        device = f"cuda:{int(device_id)}"
    ti_builder = TorchSymmetryConstrainedTiBuilder(device=device, **ti_builder_config)
    o_builder = TorchSymmetryConstrainedOBuilder(device=device, **o_builder_config)
    while True:
        task = task_queue.get()
        if task is None:
            break
        stage = str(task.get("stage"))
        task_id = int(task["task_id"])
        seed = int(task["seed"])
        try:
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32 - 1))
            if device_id is not None:
                torch.cuda.manual_seed_all(seed)
            if stage == "ti":
                selected, attempts = ti_builder.build(
                    int(task["spg"]), str(task["token"]), task["target"], task_id
                )
                result_queue.put({
                    "stage": "ti", "worker_id": worker_id, "task_id": task_id,
                    "target_id": int(task["target_id"]), "cycle": int(task["cycle"]),
                    "proposal_slot": int(task["proposal_slot"]),
                    "spg": int(task["spg"]), "token": str(task["token"]),
                    "proposal_source": str(task["proposal_source"]),
                    "selected": selected, "attempts": attempts, "error": None,
                })
            elif stage == "oxygen":
                selected, attempts = o_builder.build(
                    task["framework"], str(task["token"]), task["target"], task_id
                )
                result_queue.put({
                    "stage": "oxygen", "worker_id": worker_id, "task_id": task_id,
                    "target_id": int(task["target_id"]), "cycle": int(task["cycle"]),
                    "proposal_slot": int(task["proposal_slot"]),
                    "token": str(task["token"]),
                    "proposal_source": str(task["proposal_source"]),
                    "selected": selected, "attempts": attempts, "error": None,
                })
            else:
                raise ValueError(f"Unknown streaming task stage: {stage!r}")
        except Exception as exc:
            base = {
                "stage": stage, "worker_id": worker_id, "task_id": task_id,
                "target_id": int(task.get("target_id", -1)),
                "cycle": int(task.get("cycle", -1)),
                "proposal_slot": int(task.get("proposal_slot", -1)),
                "proposal_source": str(task.get("proposal_source", "unknown")),
                "selected": None, "attempts": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            if stage == "ti":
                base.update({"spg": int(task.get("spg", -1)), "token": str(task.get("token", ""))})
            elif stage == "oxygen":
                base.update({"token": str(task.get("token", ""))})
            result_queue.put(base)


class StreamingBuilderPool:
    def __init__(self, ngpu, ti_builder_config, o_builder_config, max_in_flight):
        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue(maxsize=max(4, int(max_in_flight) + max(1, int(ngpu))))
        self.result_queue = self.ctx.Queue()
        self.processes = []
        devices = list(range(ngpu)) if ngpu > 0 else [None]
        for worker_id, device_id in enumerate(devices):
            process = self.ctx.Process(
                target=_streaming_builder_worker,
                args=(worker_id, device_id, self.task_queue, self.result_queue,
                      ti_builder_config, o_builder_config),
                daemon=True,
            )
            process.start()
            self.processes.append(process)

    @property
    def workers(self):
        return len(self.processes)

    def submit(self, task):
        self.task_queue.put(task)

    def get(self):
        return self.result_queue.get()

    def close(self):
        for _ in self.processes:
            self.task_queue.put(None)
        for process in self.processes:
            process.join()
        self.task_queue.close()
        self.result_queue.close()


def run_streaming_phase_ab(args, canonical, num_wps, n_ti_max, ncpu, ngpu, output_folder,
                           data_name, cache_key, model, training, timings, global_seed):
    """Bounded streaming Phase-A -> Phase-B scheduler.

    The only production target is the final number of complete TiO2 structures.
    Ti frameworks are generated only to keep the oxygen parent pool populated.
    GPU submissions are bounded by gpu_queue_depth * number_of_workers.
    """
    rng = np.random.default_rng(global_seed + 31)
    composition = parse_composition(args.composition)
    ti_folder = os.path.join(output_folder, "pre_o_ti")
    oxygen_folder = os.path.join(output_folder, "pre_joint_tio2")
    accepted_oxygen_folder = os.path.join(output_folder, "accepted_tio2_pool")
    os.makedirs(ti_folder, exist_ok=True)
    os.makedirs(oxygen_folder, exist_ok=True)
    os.makedirs(accepted_oxygen_folder, exist_ok=True)

    oxygen_cache_csv = os.path.join(output_folder, "oxygen_training_environment_statistics.csv")
    oxygen_cache_meta = os.path.join(output_folder, "oxygen_training_environment_statistics.meta.json")
    to = time.perf_counter()
    oxygen_training = OxygenTrainingDistribution(
        canonical, num_wps, args.ti_o_cutoff, ncpu, oxygen_cache_csv, oxygen_cache_meta,
        hashlib.sha1(f"{cache_key}:oxygen:{args.ti_o_cutoff}".encode()).hexdigest(),
    )
    timings["oxygen_chemistry_extraction_s"] = time.perf_counter() - to

    ti_builder_config = {
        "initializations": args.starts_per_template,
        "screen_steps": args.builder_screen_steps,
        "refine_starts": args.builder_refine_starts,
        "refine_steps": args.builder_refine_steps,
        "cn_tolerance": args.cn_tolerance,
        "minimum_distance": args.minimum_ti_ti_distance,
        "maximum_loss": args.maximum_total_loss,
        "chemistry_cutoff": args.chemistry_cutoff,
        "max_ti_atoms": args.max_ti_atoms,
        "max_neighbors": args.max_ti_neighbors,
        "lr": args.builder_lr,
        "seed": global_seed,
    }
    o_builder_config = {
        "initializations": args.o_starts_per_template,
        "screen_steps": args.o_builder_screen_steps,
        "refine_starts": args.o_builder_refine_starts,
        "refine_steps": args.o_builder_refine_steps,
        "ti_cn_tolerance": args.o_ti_cn_tolerance,
        "o_cn_tolerance": args.o_o_cn_tolerance,
        "distance_tolerance": args.o_distance_tolerance,
        "minimum_ti_o": args.minimum_ti_o_distance,
        "minimum_o_o": args.minimum_o_o_distance,
        "maximum_loss": args.o_maximum_total_loss,
        "max_neighbors": args.o_max_neighbors,
        "lr": args.o_builder_lr,
        "seed": global_seed,
    }

    workers = max(1, int(ngpu) if int(ngpu) > 0 else 1)
    max_in_flight = max(1, int(args.gpu_queue_depth) * workers)
    desired_o_parents = int(args.active_o_parents) if args.active_o_parents > 0 else 2 * workers
    ti_in_flight_target = max(1, min(max_in_flight - 1, int(round(max_in_flight * args.ti_task_fraction)))) if max_in_flight > 1 else 1
    min_generation_target = max(args.sample, int(math.ceil(args.sample * (1.0 + args.min_sample_overhead))))
    max_generation_target = max(min_generation_target, int(math.ceil(args.sample * (1.0 + args.max_sample_overhead))))
    ranking_check_interval = max(1, int(math.ceil(args.sample * args.ranking_check_fraction)))
    print(
        "Streaming Phase A/B: "
        f"target_complete={args.sample}, gpu_workers={workers}, "
        f"max_in_flight={max_in_flight}, desired_active_o_parents={desired_o_parents}, "
        f"Ti/O_slot_target={ti_in_flight_target}/{max_in_flight - ti_in_flight_target}, "
        f"accepted_pool_range={min_generation_target}-{max_generation_target}",
        flush=True,
    )

    pool = StreamingBuilderPool(ngpu, ti_builder_config, o_builder_config, max_in_flight)
    tg = time.perf_counter()

    active_ti, active_o = {}, {}
    ti_rr, o_rr = deque(), deque()
    final_rows, final_diag = [], []
    ti_rows, ti_diag, ti_attempt_diag = [], [], []
    o_attempt_diag = []

    next_ti_target_id = 0
    next_o_target_id = 0
    next_task_id = 0
    ti_template_attempts = 0
    o_template_attempts = 0
    outstanding_total = 0
    outstanding_ti = 0
    outstanding_o = 0
    ti_targets_considered = 0
    ti_targets_exhausted = 0
    ti_frameworks_accepted = 0
    ti_frameworks_sent_to_o = 0
    o_targets_exhausted = 0
    ti_source_attempts = {"interpolation": 0, "exploration": 0}
    ti_source_selected = {"interpolation": 0, "exploration": 0}
    o_source_attempts = {"interpolation": 0, "exploration": 0}
    o_source_selected = {"interpolation": 0, "exploration": 0}
    ti_accepted_cycles = {}
    o_accepted_cycles = {}
    worker_tasks = {}
    termination_reason = None
    ranking_checks = []
    previous_top_ids = None
    previous_boundary_score = None
    stable_ranking_checks = 0
    ranking_converged = False
    next_ranking_check = int(args.sample)

    def progress(prefix="Progress"):
        print(
            f"{prefix}: accepted_pool={len(final_rows)}; final_target={args.sample}; "
            f"Ti_frameworks={ti_frameworks_accepted}; O_exhausted={o_targets_exhausted}; "
            f"active_Ti/O={len(active_ti)}/{len(active_o)}; "
            f"in_flight_Ti/O={outstanding_ti}/{outstanding_o}; "
            f"Ti/O_attempts={ti_template_attempts}/{o_template_attempts}",
            flush=True,
        )

    def prepare_ti_cycle(state):
        state.cycle_pool = list(state.elites)
        state.proposals = []
        state.submitted = 0
        state.returned = 0
        needed = max(0, args.search_population - len(state.elites))
        proposals, rounds = [], 0
        while len(proposals) < needed and ti_template_attempts + len(proposals) < args.max_global_attempts:
            rounds += 1
            batch, stats = draw_proposals(
                model, args.sw_mode, needed - len(proposals), rng,
                args.temperature, composition, num_wps, args.max_ti_atoms,
            )
            for spg, token, source in batch:
                h = proposal_hash(spg, token)
                if h in state.attempted:
                    continue
                state.attempted.add(h)
                proposals.append((spg, token, source))
                if len(proposals) >= needed:
                    break
            if not batch or rounds >= 20:
                break
        state.proposals = proposals

    def add_ti_target():
        nonlocal next_ti_target_id, ti_targets_considered
        state = StreamingTiState(next_ti_target_id, training.sample_any(rng))
        active_ti[state.target_id] = state
        ti_rr.append(state.target_id)
        next_ti_target_id += 1
        ti_targets_considered += 1
        prepare_ti_cycle(state)
        return state

    def ti_needs_parents():
        return (len(final_rows) < max_generation_target and not ranking_converged and
                (len(active_o) + len(active_ti)) < desired_o_parents and
                ti_template_attempts < args.max_global_attempts)

    def ensure_ti_parent_capacity():
        while ti_needs_parents():
            add_ti_target()

    def submit_next_ti_task():
        nonlocal next_task_id, ti_template_attempts, outstanding_total, outstanding_ti
        if not active_ti:
            return False
        for _ in range(len(ti_rr)):
            tid = ti_rr.popleft()
            state = active_ti.get(tid)
            if state is None:
                continue
            ti_rr.append(tid)
            if state.submitted >= len(state.proposals):
                continue
            if ti_template_attempts >= args.max_global_attempts:
                return False
            slot = state.submitted
            spg, token, source = state.proposals[slot]
            state.submitted += 1
            ti_template_attempts += 1
            next_task_id += 1
            outstanding_total += 1
            outstanding_ti += 1
            ti_source_attempts[source] = ti_source_attempts.get(source, 0) + 1
            pool.submit({
                "stage": "ti", "task_id": next_task_id, "target_id": state.target_id,
                "cycle": state.cycle, "proposal_slot": slot,
                "spg": spg, "token": token, "proposal_source": source,
                "target": state.target,
                "seed": deterministic_seed(global_seed, state.target_id, state.cycle, slot),
            })
            return True
        return False

    def prepare_o_cycle(state):
        state.cycle_pool = []
        state.proposals = []
        state.submitted = 0
        state.returned = 0
        state.no_legal_proposals = False
        max_sites = num_wps - len(state.framework["ti_wps"])
        if max_sites <= 0:
            state.no_legal_proposals = True
            return
        required = 2 * len(state.framework["ti_frac"])
        learned = oxygen_training.learned_tokens(state.framework["spg"], required, max_sites)
        proposals, rounds = [], 0
        while len(proposals) < args.o_search_population and o_template_attempts + len(proposals) < args.o_max_global_attempts:
            rounds += 1
            batch = sample_legal_o_skeletons(
                state.framework["spg"], required, args.o_search_population - len(proposals),
                max_sites, rng, learned, args.o_sw_mode,
            )
            for token, source in batch:
                if token in state.attempted:
                    continue
                state.attempted.add(token)
                proposals.append((token, source))
                if len(proposals) >= args.o_search_population:
                    break
            if not batch or rounds >= 20:
                break
        state.proposals = proposals
        if not proposals:
            state.no_legal_proposals = True

    def add_o_target(framework):
        nonlocal next_o_target_id, ti_frameworks_sent_to_o
        state = StreamingOState(
            target_id=next_o_target_id,
            phase_a_sample_id=int(framework.get("phase_a_sample_id", next_o_target_id)),
            framework=framework,
            target=oxygen_training.sample_for_framework(framework, rng, args.o_target_nearest),
        )
        active_o[state.target_id] = state
        o_rr.append(state.target_id)
        next_o_target_id += 1
        ti_frameworks_sent_to_o += 1
        prepare_o_cycle(state)
        return state

    def submit_next_o_task():
        nonlocal next_task_id, o_template_attempts, outstanding_total, outstanding_o
        if not active_o:
            return False
        for _ in range(len(o_rr)):
            oid = o_rr.popleft()
            state = active_o.get(oid)
            if state is None:
                continue
            o_rr.append(oid)
            if state.submitted >= len(state.proposals):
                continue
            if o_template_attempts >= args.o_max_global_attempts:
                return False
            slot = state.submitted
            token, source = state.proposals[slot]
            state.submitted += 1
            o_template_attempts += 1
            next_task_id += 1
            outstanding_total += 1
            outstanding_o += 1
            o_source_attempts[source] = o_source_attempts.get(source, 0) + 1
            pool.submit({
                "stage": "oxygen", "task_id": next_task_id, "target_id": state.target_id,
                "cycle": state.cycle, "proposal_slot": slot,
                "token": token, "proposal_source": source,
                "framework": state.framework, "target": state.target,
                "seed": deterministic_seed(global_seed + 100003, state.target_id, state.cycle, slot),
            })
            return True
        return False

    def accept_ti_state(state, solved):
        nonlocal ti_frameworks_accepted
        row = build_output_row(solved, solved["spg"], solved["token"], num_wps, n_ti_max)
        ti_rows.append(row)
        diag = {k: v for k, v in solved.items() if k not in {"lattice", "free", "frac", "generators", "cell"}}
        diag.update({
            "target_id": state.target_id,
            "phase_a_sample_id": ti_frameworks_accepted,
            "spg": solved["spg"], "ti_skeleton_token": solved["token"],
            "target_source_group": state.target.source_group,
            "target_ti_cn": state.target.target_ti_cn,
            "target_ti_nn_mean": state.target.target_ti_nn_mean,
            "target_ti_nn_width": state.target.target_ti_nn_width,
            "target_ti_shell_cutoff": state.target.target_ti_shell_cutoff,
            "target_ti_sphere_radius": state.target.target_ti_sphere_radius,
            "sw_mode": solved["proposal_source"], "proposal_source": solved["proposal_source"],
        })
        ti_diag.append(diag)
        save_ti_cif(solved, os.path.join(ti_folder, f"sample_{ti_frameworks_accepted:06d}.cif"))
        ti_source_selected[solved["proposal_source"]] = ti_source_selected.get(solved["proposal_source"], 0) + 1
        ti_accepted_cycles[state.cycle + 1] = ti_accepted_cycles.get(state.cycle + 1, 0) + 1
        framework = phase_a_row_to_framework(row, diag, num_wps)
        framework["phase_a_sample_id"] = ti_frameworks_accepted
        ti_frameworks_accepted += 1
        active_ti.pop(state.target_id, None)
        add_o_target(framework)

    def resolve_ti_state_if_ready(state):
        nonlocal ti_targets_exhausted
        if state.submitted < len(state.proposals) or state.returned < state.submitted:
            return False
        valid = [x for x in state.cycle_pool if x.get("chemistry_success", False)]
        if valid:
            valid.sort(key=lambda x: float(x.get("total_loss", 1e9)))
            accept_ti_state(state, valid[0])
            return True
        state.cycle_pool.sort(key=lambda x: candidate_score(x, state.target))
        state.elites = state.cycle_pool[:args.search_elites]
        if state.cycle + 1 >= args.search_cycles or ti_template_attempts >= args.max_global_attempts or not state.proposals:
            ti_targets_exhausted += 1
            active_ti.pop(state.target_id, None)
            try:
                ti_rr.remove(state.target_id)
            except ValueError:
                pass
            if ti_targets_exhausted == 1 or ti_targets_exhausted % args.progress_every == 0:
                progress(prefix=f"Phase A exhausted={ti_targets_exhausted}")
            return True
        state.cycle += 1
        prepare_ti_cycle(state)
        return True

    def accept_o_state(state, solved):
        row = build_complete_output_row(state.framework, solved, solved["token"], num_wps)
        final_rows.append(row)
        diag = {k: v for k, v in solved.items() if k not in {"o_free", "o_frac"}}
        diag.update({
            "target_id": state.target_id,
            "phase_a_sample_id": state.phase_a_sample_id,
            "spg": state.framework["spg"],
            "o_skeleton_token": solved["token"],
            "proposal_source": solved["proposal_source"],
            "target_source_group": state.target.source_group,
            "target_ti_o_cn": state.target.target_ti_o_cn,
            "target_ti_o_cn_std": state.target.target_ti_o_cn_std,
            "target_o_ti_cn": state.target.target_o_ti_cn,
            "target_o_ti_cn_std": state.target.target_o_ti_cn_std,
            "target_ti_o_mean": state.target.target_ti_o_mean,
            "target_ti_o_width": state.target.target_ti_o_width,
        })
        for i, value in enumerate(state.target.oti_o_angle_profile):
            diag[f"target_angle_bin_{i}"] = float(value)
        for i, value in enumerate(state.target.shell_o_o_profile):
            diag[f"target_shell_o_o_bin_{i}"] = float(value)
        candidate_id = len(final_rows) - 1
        ranking_score = float(solved.get("total_loss", 1e9))
        diag["candidate_id"] = int(candidate_id)
        diag["ranking_score"] = ranking_score
        final_diag.append(diag)
        save_tio2_cif(
            state.framework, solved,
            os.path.join(accepted_oxygen_folder, f"candidate_{candidate_id:06d}.cif"),
        )
        o_source_selected[solved["proposal_source"]] = o_source_selected.get(solved["proposal_source"], 0) + 1
        o_accepted_cycles[state.cycle + 1] = o_accepted_cycles.get(state.cycle + 1, 0) + 1
        active_o.pop(state.target_id, None)
        try:
            o_rr.remove(state.target_id)
        except ValueError:
            pass
        if len(final_rows) == 1 or len(final_rows) % args.progress_every == 0:
            progress(prefix="Streaming progress")
        update_ranking_convergence()

    def update_ranking_convergence():
        nonlocal previous_top_ids, previous_boundary_score
        nonlocal stable_ranking_checks, ranking_converged, next_ranking_check, termination_reason
        accepted = len(final_rows)
        if accepted < args.sample or accepted < next_ranking_check:
            return
        ranked_idx = sorted(range(accepted), key=lambda i: float(final_diag[i].get("ranking_score", 1e9)))
        top_idx = ranked_idx[:args.sample]
        lo = max(0, int(math.floor(0.95 * args.sample)))
        boundary_scores = [float(final_diag[i].get("ranking_score", 1e9)) for i in top_idx[lo:args.sample]]
        boundary_score = float(np.median(boundary_scores)) if boundary_scores else float("inf")
        top_ids = {int(final_diag[i]["candidate_id"]) for i in top_idx}
        boundary_change = None
        turnover = None
        stable_now = False
        if previous_boundary_score is not None and previous_top_ids is not None:
            boundary_change = abs(boundary_score - previous_boundary_score) / max(abs(previous_boundary_score), 1e-12)
            turnover = 1.0 - len(top_ids & previous_top_ids) / max(args.sample, 1)
            stable_now = (
                accepted >= min_generation_target
                and boundary_change <= args.ranking_boundary_tol
                and turnover <= args.ranking_turnover_tol
            )
            stable_ranking_checks = stable_ranking_checks + 1 if stable_now else 0
        record = {
            "accepted_pool": int(accepted),
            "boundary_score": boundary_score,
            "boundary_change": boundary_change,
            "top_n_turnover": turnover,
            "stable_now": bool(stable_now),
            "stable_count": int(stable_ranking_checks),
        }
        ranking_checks.append(record)
        if accepted >= min_generation_target:
            bc = "baseline" if boundary_change is None else f"{100.0 * boundary_change:.3f}%"
            tv = "baseline" if turnover is None else f"{100.0 * turnover:.3f}%"
            print(
                f"Ranking check @ {accepted}: boundary={boundary_score:.6f}; "
                f"change={bc}; top{args.sample}_turnover={tv}; "
                f"stable={stable_ranking_checks}/{args.ranking_stable_checks}",
                flush=True,
            )
        previous_boundary_score = boundary_score
        previous_top_ids = top_ids
        next_ranking_check = accepted + ranking_check_interval
        if stable_ranking_checks >= args.ranking_stable_checks:
            ranking_converged = True
            termination_reason = "ranking_converged"
            print(
                f"Ranking converged with {accepted} accepted complete TiO2 candidates; "
                f"final selection will retain best {args.sample}.",
                flush=True,
            )

    def resolve_o_state_if_ready(state):
        nonlocal o_targets_exhausted
        if state.submitted < len(state.proposals) or state.returned < state.submitted:
            return False
        valid = [x for x in state.cycle_pool if x.get("chemistry_success", False)]
        if valid:
            valid.sort(key=lambda x: oxygen_candidate_score(x, state.target))
            accept_o_state(state, valid[0])
            return True
        if (state.cycle + 1 >= args.o_search_cycles or o_template_attempts >= args.o_max_global_attempts or
                state.no_legal_proposals):
            o_targets_exhausted += 1
            active_o.pop(state.target_id, None)
            try:
                o_rr.remove(state.target_id)
            except ValueError:
                pass
            if o_targets_exhausted == 1 or o_targets_exhausted % args.progress_every == 0:
                progress(prefix=f"Phase B exhausted={o_targets_exhausted}")
            return True
        state.cycle += 1
        prepare_o_cycle(state)
        return True

    def resolve_ready_states():
        changed = True
        while changed:
            changed = False
            for state in list(active_o.values()):
                changed = resolve_o_state_if_ready(state) or changed
            for state in list(active_ti.values()):
                changed = resolve_ti_state_if_ready(state) or changed

    def submit_bounded_work():
        submitted_any = False
        ensure_ti_parent_capacity()
        while outstanding_total < max_in_flight and not ranking_converged and len(final_rows) < max_generation_target:
            ensure_ti_parent_capacity()
            if not active_o:
                if submit_next_ti_task():
                    submitted_any = True
                    continue
                if submit_next_o_task():
                    submitted_any = True
                    continue
                break
            if outstanding_ti < ti_in_flight_target and submit_next_ti_task():
                submitted_any = True
                continue
            if submit_next_o_task():
                submitted_any = True
                continue
            if submit_next_ti_task():
                submitted_any = True
                continue
            break
        return submitted_any

    try:
        while len(final_rows) < max_generation_target and not ranking_converged:
            resolve_ready_states()
            update_ranking_convergence()
            submit_bounded_work()
            if ranking_converged or len(final_rows) >= max_generation_target:
                if termination_reason is None:
                    termination_reason = "maximum_sample_overhead_reached"
                break
            if outstanding_total <= 0:
                resolve_ready_states()
                submit_bounded_work()
                if outstanding_total <= 0:
                    if ti_template_attempts >= args.max_global_attempts:
                        termination_reason = "ti_template_attempt_limit_reached"
                    elif o_template_attempts >= args.o_max_global_attempts:
                        termination_reason = "oxygen_template_attempt_limit_reached"
                    else:
                        termination_reason = "streaming_scheduler_no_submittable_work"
                    break
            result = pool.get()
            outstanding_total -= 1
            worker_id = int(result["worker_id"])
            worker_tasks[worker_id] = worker_tasks.get(worker_id, 0) + 1
            if result["stage"] == "ti":
                outstanding_ti -= 1
                state = active_ti.get(int(result["target_id"]))
                if state is None:
                    continue
                state.returned += 1
                for item in result["attempts"]:
                    row = {k: v for k, v in item.items() if k not in {"lattice", "free", "frac", "generators", "cell"}}
                    row.update({
                        "target_id": state.target_id, "cycle": state.cycle,
                        "spg": result.get("spg"), "ti_skeleton_token": result.get("token"),
                        "sw_mode": result.get("proposal_source"),
                        "proposal_source": result.get("proposal_source"),
                        "target_ti_cn": state.target.target_ti_cn,
                        "target_source_group": state.target.source_group,
                        "worker_id": worker_id, "worker_error": result.get("error"),
                    })
                    ti_attempt_diag.append(row)
                selected = result.get("selected")
                if selected is not None:
                    selected = selected.copy()
                    selected.update({
                        "spg": result["spg"], "token": result["token"],
                        "sw_mode": result["proposal_source"],
                        "proposal_source": result["proposal_source"],
                        "cycle": state.cycle, "worker_id": worker_id,
                    })
                    state.cycle_pool.append(selected)
                resolve_ti_state_if_ready(state)
            elif result["stage"] == "oxygen":
                outstanding_o -= 1
                state = active_o.get(int(result["target_id"]))
                if state is None:
                    continue
                state.returned += 1
                for item in result["attempts"]:
                    row = {k: v for k, v in item.items() if k not in {"o_free", "o_frac"}}
                    row.update({
                        "target_id": state.target_id, "phase_a_sample_id": state.phase_a_sample_id,
                        "cycle": state.cycle, "spg": state.framework["spg"],
                        "o_skeleton_token": result.get("token"),
                        "proposal_source": result.get("proposal_source"),
                        "worker_id": worker_id, "worker_error": result.get("error"),
                        "target_ti_o_cn": state.target.target_ti_o_cn,
                        "target_o_ti_cn": state.target.target_o_ti_cn,
                        "target_ti_o_mean": state.target.target_ti_o_mean,
                        "target_source_group": state.target.source_group,
                    })
                    o_attempt_diag.append(row)
                selected = result.get("selected")
                if selected is not None:
                    selected = selected.copy()
                    selected.update({
                        "token": result["token"], "proposal_source": result["proposal_source"],
                        "cycle": state.cycle, "worker_id": worker_id,
                    })
                    state.cycle_pool.append(selected)
                resolve_o_state_if_ready(state)
    finally:
        pool.close()

    timings["streaming_generation_s"] = time.perf_counter() - tg
    # Write diagnostics even if the final exact-count assertion fails.
    pd.DataFrame(ti_attempt_diag).to_csv(os.path.join(output_folder, "ti_builder_attempts.csv"), index=False)
    pd.DataFrame(ti_diag).to_csv(os.path.join(output_folder, "ti_builder_selected.csv"), index=False)
    pd.DataFrame(o_attempt_diag).to_csv(os.path.join(output_folder, "oxygen_builder_attempts.csv"), index=False)
    if ti_rows:
        ti_frame = pd.DataFrame(ti_rows)
        for column in canonical.columns:
            if column not in ti_frame.columns:
                ti_frame[column] = np.nan
        ti_frame = ti_frame[[c for c in canonical.columns if c in ti_frame.columns]]
        ti_frame.to_csv(os.path.join(output_folder, f"{data_name}-streamed-phaseA-ti-{len(ti_frame)}.csv"), index=False)

    if len(final_rows) < args.sample:
        under = {
            "requested_complete_tio2": int(args.sample),
            "accepted_complete_tio2_pool": int(len(final_rows)),
            "missing": int(args.sample - len(final_rows)),
            "ti_targets_considered": int(ti_targets_considered),
            "ti_targets_exhausted": int(ti_targets_exhausted),
            "ti_frameworks_accepted": int(ti_frameworks_accepted),
            "ti_frameworks_sent_to_oxygen": int(ti_frameworks_sent_to_o),
            "oxygen_targets_exhausted": int(o_targets_exhausted),
            "active_ti_targets": int(len(active_ti)),
            "active_o_targets": int(len(active_o)),
            "outstanding_ti_tasks": int(outstanding_ti),
            "outstanding_o_tasks": int(outstanding_o),
            "ti_template_attempts": int(ti_template_attempts),
            "max_ti_template_attempts": int(args.max_global_attempts),
            "oxygen_template_attempts": int(o_template_attempts),
            "max_oxygen_template_attempts": int(args.o_max_global_attempts),
            "max_in_flight_gpu_tasks": int(max_in_flight),
            "desired_active_o_parents": int(desired_o_parents),
            "reason": termination_reason or "streaming_generation_underfilled",
        }
        path = os.path.join(output_folder, "streaming_underfill.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(under, handle, indent=2)
        raise RuntimeError(
            f"Streaming Phase-A/B underfilled complete TiO2 pool: {len(final_rows)}/{args.sample}; "
            f"diagnostics={path}"
        )

    ranked_indices = sorted(
        range(len(final_rows)),
        key=lambda i: float(final_diag[i].get("ranking_score", 1e9)),
    )
    selected_indices = ranked_indices[:args.sample]
    selected_rows = [final_rows[i] for i in selected_indices]
    selected_diag = [dict(final_diag[i], final_rank=rank + 1) for rank, i in enumerate(selected_indices)]

    pd.DataFrame(final_diag).to_csv(os.path.join(output_folder, "oxygen_builder_accepted_pool.csv"), index=False)
    pd.DataFrame(selected_diag).to_csv(os.path.join(output_folder, "oxygen_builder_selected.csv"), index=False)
    pd.DataFrame(ranking_checks).to_csv(os.path.join(output_folder, "ranking_convergence.csv"), index=False)
    pre_relaxation_metrics = evaluate_pre_relaxation(
        final_rows, final_diag, selected_indices, num_wps, oxygen_training, output_folder, args.ti_o_cutoff
    )

    for old in Path(oxygen_folder).glob("sample_*.cif"):
        old.unlink()
    for rank, pool_index in enumerate(selected_indices):
        candidate_id = int(final_diag[pool_index]["candidate_id"])
        shutil.copy2(
            os.path.join(accepted_oxygen_folder, f"candidate_{candidate_id:06d}.cif"),
            os.path.join(oxygen_folder, f"sample_{rank:06d}.cif"),
        )

    final = pd.DataFrame(selected_rows)
    for column in canonical.columns:
        if column not in final.columns:
            final[column] = np.nan
    final = final[[c for c in canonical.columns if c in final.columns]]
    final_path = os.path.join(output_folder, f"{data_name}-phaseB-tio2-{len(final)}.csv")
    final.to_csv(final_path, index=False)
    summary = {
        "requested_complete_tio2": int(args.sample),
        "accepted_complete_tio2_pool": int(len(final_rows)),
        "selected_complete_tio2": int(len(selected_rows)),
        "discarded_by_final_ranking": int(len(final_rows) - len(selected_rows)),
        "pre_relaxation_metrics": pre_relaxation_metrics,
        "ti_targets_considered": int(ti_targets_considered),
        "ti_targets_exhausted": int(ti_targets_exhausted),
        "ti_frameworks_accepted": int(ti_frameworks_accepted),
        "ti_frameworks_sent_to_oxygen": int(ti_frameworks_sent_to_o),
        "oxygen_targets_exhausted": int(o_targets_exhausted),
        "ti_template_attempts": int(ti_template_attempts),
        "oxygen_template_attempts": int(o_template_attempts),
        "ti_framework_to_tio2_conversion": len(final_rows) / max(ti_frameworks_sent_to_o, 1),
        "ti_template_acceptance_fraction": ti_frameworks_accepted / max(ti_template_attempts, 1),
        "oxygen_template_acceptance_fraction": len(final_rows) / max(o_template_attempts, 1),
        "ngpu": int(ngpu),
        "gpu_workers": int(pool.workers),
        "ncpu": int(ncpu),
        "gpu_queue_depth": int(args.gpu_queue_depth),
        "max_in_flight_gpu_tasks": int(max_in_flight),
        "desired_active_o_parents": int(desired_o_parents),
        "ti_task_fraction": float(args.ti_task_fraction),
        "ti_in_flight_target": int(ti_in_flight_target),
        "min_sample_overhead": float(args.min_sample_overhead),
        "max_sample_overhead": float(args.max_sample_overhead),
        "minimum_generation_target": int(min_generation_target),
        "maximum_generation_target": int(max_generation_target),
        "ranking_check_fraction": float(args.ranking_check_fraction),
        "ranking_boundary_tolerance": float(args.ranking_boundary_tol),
        "ranking_turnover_tolerance": float(args.ranking_turnover_tol),
        "ranking_stable_checks_required": int(args.ranking_stable_checks),
        "ranking_stable_checks_reached": int(stable_ranking_checks),
        "generation_termination_reason": termination_reason or "maximum_sample_overhead_reached",
        "ranking_checks": ranking_checks,
        "worker_template_tasks": worker_tasks,
        "ti_proposal_source_attempts": ti_source_attempts,
        "ti_proposal_source_selected": ti_source_selected,
        "oxygen_proposal_source_attempts": o_source_attempts,
        "oxygen_proposal_source_selected": o_source_selected,
        "ti_accepted_by_cycle": ti_accepted_cycles,
        "oxygen_accepted_by_cycle": o_accepted_cycles,
        "parallelization": "bounded streaming Phase-A to Phase-B with one shared persistent worker per GPU",
        "scheduling": "bounded weighted-fair Ti/O submission with round-robin oxygen parents and adaptive ranking convergence",
        "timings_seconds": timings,
    }
    with open(os.path.join(output_folder, "streaming_phaseAB_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return final_path, oxygen_folder, summary

def main():
    parser = argparse.ArgumentParser(description="Streaming Ti/O generation-swap builder v18")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--nbatch", type=int, default=500)
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--composition", default="1,2")
    parser.add_argument("--context-end", type=float, default=0.8)
    parser.add_argument("--sw-mode", choices=("interpolation", "exploration", "mixed"), default="mixed")
    parser.add_argument(
        "--ncpu", type=int, default=0,
        help="CPU workers; 0 auto-detects the Slurm/cgroup allocation and reserves one CPU for scheduling.",
    )
    parser.add_argument(
        "--ngpu", type=int, default=0,
        help="GPU workers; 0 uses every CUDA device visible to this process.",
    )
    parser.add_argument(
        "--gpu-queue-depth", type=int, default=2,
        help="Maximum streaming GPU tasks in flight per GPU worker. Default keeps only O(NGPU) tasks submitted.",
    )
    parser.add_argument(
        "--active-o-parents", type=int, default=0,
        help="Target active Ti/O parent working set; 0 resolves to 2*NGPU.",
    )
    parser.add_argument(
        "--ti-task-fraction", type=float, default=0.375,
        help="Target fraction of bounded in-flight GPU tasks reserved for Ti while oxygen parents exist.",
    )
    parser.add_argument("--min-sample-overhead", type=float, default=0.10)
    parser.add_argument("--max-sample-overhead", type=float, default=1.00)
    parser.add_argument("--ranking-check-fraction", type=float, default=0.10)
    parser.add_argument("--ranking-boundary-tol", type=float, default=0.01)
    parser.add_argument("--ranking-turnover-tol", type=float, default=0.05)
    parser.add_argument("--ranking-stable-checks", type=int, default=2)
    parser.add_argument(
        "--progress-every", type=int, default=10,
        help="Print aggregate streaming progress every N complete TiO2 acceptances/exhaustions.",
    )
    parser.add_argument("--search-cycles", type=int, default=6)
    parser.add_argument("--search-population", type=int, default=8)
    parser.add_argument("--search-elites", type=int, default=2)
    parser.add_argument("--starts-per-template", type=int, default=2)
    parser.add_argument("--builder-screen-steps", type=int, default=15)
    parser.add_argument("--builder-refine-starts", type=int, default=2)
    parser.add_argument("--builder-refine-steps", type=int, default=50)
    parser.add_argument("--builder-lr", type=float, default=0.06)
    parser.add_argument("--cn-tolerance", type=float, default=0.75)
    parser.add_argument("--minimum-ti-ti-distance", type=float, default=2.0)
    parser.add_argument("--maximum-total-loss", type=float, default=5.0)
    parser.add_argument("--max-ti-atoms", type=int, default=MAX_TI_ATOMS)
    parser.add_argument("--chemistry-cutoff", type=float, default=CHEMISTRY_CUTOFF)
    parser.add_argument("--max-ti-neighbors", type=int, default=MAX_TI_NEIGHBORS)
    parser.add_argument("--max-global-attempts", type=int, default=10000)
    parser.add_argument("--o-sw-mode", choices=("interpolation", "exploration", "mixed"), default="exploration")
    parser.add_argument("--o-search-cycles", type=int, default=6)
    parser.add_argument("--o-search-population", type=int, default=8)
    parser.add_argument("--o-starts-per-template", type=int, default=4)
    parser.add_argument("--o-builder-screen-steps", type=int, default=25)
    parser.add_argument("--o-builder-refine-starts", type=int, default=2)
    parser.add_argument("--o-builder-refine-steps", type=int, default=60)
    parser.add_argument("--o-builder-lr", type=float, default=0.06)
    parser.add_argument("--ti-o-cutoff", type=float, default=TIO_CUTOFF)
    parser.add_argument("--o-ti-cn-tolerance", type=float, default=0.75)
    parser.add_argument("--o-o-cn-tolerance", type=float, default=0.75)
    parser.add_argument("--o-distance-tolerance", type=float, default=0.35)
    parser.add_argument("--minimum-ti-o-distance", type=float, default=1.45)
    parser.add_argument("--minimum-o-o-distance", type=float, default=1.60)
    parser.add_argument("--o-maximum-total-loss", type=float, default=8.0)
    parser.add_argument("--o-max-neighbors", type=int, default=12)
    parser.add_argument("--o-target-nearest", type=int, default=64)
    parser.add_argument("--o-max-global-attempts", type=int, default=20000)
    parser.add_argument("--output-dir", default="data/sample")
    args = parser.parse_args()

    positive = [args.epochs, args.nbatch, args.sample, args.search_cycles, args.search_population,
                args.search_elites, args.starts_per_template, args.builder_screen_steps,
                args.builder_refine_starts, args.builder_refine_steps, args.max_global_attempts,
                args.o_search_cycles, args.o_search_population, args.o_starts_per_template,
                args.o_builder_screen_steps, args.o_builder_refine_starts, args.o_builder_refine_steps,
                args.o_max_neighbors, args.o_target_nearest, args.o_max_global_attempts,
                args.gpu_queue_depth, args.ranking_stable_checks]
    if min(positive) <= 0:
        raise ValueError("Positive integer arguments must be greater than zero.")
    if args.search_elites >= args.search_population:
        raise ValueError("--search-elites must be smaller than --search-population.")
    if args.builder_refine_starts > args.starts_per_template:
        raise ValueError("--builder-refine-starts cannot exceed --starts-per-template.")
    if args.o_builder_refine_starts > args.o_starts_per_template:
        raise ValueError("--o-builder-refine-starts cannot exceed --o-starts-per-template.")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be greater than zero.")
    if args.active_o_parents < 0:
        raise ValueError("--active-o-parents cannot be negative.")
    if not (0.0 < args.ti_task_fraction < 1.0):
        raise ValueError("--ti-task-fraction must be between 0 and 1.")
    if args.min_sample_overhead < 0.0 or args.max_sample_overhead < args.min_sample_overhead:
        raise ValueError("Require 0 <= --min-sample-overhead <= --max-sample-overhead.")
    if args.ranking_check_fraction <= 0.0:
        raise ValueError("--ranking-check-fraction must be greater than zero.")
    if args.ranking_boundary_tol < 0.0 or args.ranking_turnover_tol < 0.0:
        raise ValueError("Ranking convergence tolerances cannot be negative.")

    ncpu = resolve_ncpu(args.ncpu)
    set_worker_thread_limits()
    composition = parse_composition(args.composition)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    timings = {}
    t0 = time.perf_counter()

    print(f"Loading training CSV: {args.data}", flush=True)
    df = pd.read_csv(args.data)
    df.columns = df.columns.astype(str).str.strip()
    if args.cutoff is not None:
        df = df.iloc[:args.cutoff].copy()
    num_wps = validate_layout(df)
    canonical, n_ti_max, n_o_max = canonicalize_species_order(df, num_wps)
    print(f"Canonicalized {len(canonical)} rows. Building factorized blocks...", flush=True)
    global_df, ti_df, o_df = build_factorized_blocks(canonical, num_wps, n_ti_max, n_o_max)
    timings["data_preparation_s"] = time.perf_counter() - t0
    discrete_cell = bool(np.max(np.abs(
        global_df[BASE_COLUMNS[1:]].to_numpy(float) -
        np.rint(global_df[BASE_COLUMNS[1:]].to_numpy(float))
    )) < 1e-6)
    global_discrete = ["spg"] + (BASE_COLUMNS[1:] if discrete_cell else [])
    data_name = Path(args.data).stem
    model_folder = os.path.join("models", data_name, "FactorizedVAE_tio2_phaseAB_v18")
    output_folder = os.path.join(args.output_dir, f"{data_name}-phaseAB-v18-seed{args.seed}")
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    stat = os.stat(args.data)
    cache_key = hashlib.sha1(json.dumps({
        "path": os.path.abspath(args.data), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "rows": len(canonical),
        "num_wps": num_wps, "cutoff": args.chemistry_cutoff,
    }, sort_keys=True).encode()).hexdigest()
    cache_csv = os.path.join(output_folder, "ti_training_framework_statistics.csv")
    cache_meta = os.path.join(output_folder, "ti_training_framework_statistics.meta.json")
    tc = time.perf_counter()
    training = TiTrainingDistribution(
        canonical, num_wps, args.chemistry_cutoff, ncpu, cache_csv, cache_meta, cache_key
    )
    timings["chemistry_extraction_s"] = time.perf_counter() - tc
    print(f"Ti chemistry records: {len(training.frame)}; extraction failures={training.failures}", flush=True)

    tv = time.perf_counter()
    model = FactorizedVAE(
        embedding_dim=128, compress_dims=(512, 512), decompress_dims=(512, 512),
        context_dim=128, l2scale=1e-5, batch_size=args.nbatch, epochs=args.epochs,
        loss_factor=2.0, kl_weight=1.0, kl_warmup_epochs=min(50, args.epochs),
        predicted_context_start=0.0, predicted_context_end=args.context_end,
        cuda=torch.cuda.is_available(), verbose=True, folder=model_folder,
    )
    model.fit(global_df, ti_df, o_df, global_discrete_columns=global_discrete,
              si_discrete_columns=["si_skeleton_token"], o_discrete_columns=["o_skeleton_token"])
    timings["vae_training_s"] = time.perf_counter() - tv

    ngpu = resolve_ngpu(args.ngpu)
    print(
        "Resolved resources: "
        f"ncpu={ncpu} worker(s), ngpu={ngpu}; "
        f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'unset')}, "
        f"CPU_affinity={_cpu_affinity_count()}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
        flush=True,
    )
    print("VAE training complete. Beginning bounded streaming Phase-A/B generation.", flush=True)

    final_path, final_cif_folder, summary = run_streaming_phase_ab(
        args=args, canonical=canonical, num_wps=num_wps, n_ti_max=n_ti_max,
        ncpu=ncpu, ngpu=ngpu, output_folder=output_folder, data_name=data_name,
        cache_key=cache_key, model=model, training=training, timings=timings,
        global_seed=args.seed,
    )
    model_path = os.path.join(model_folder, "models", "FactorizedVAE_final.pkl")
    model.save(model_path)
    timings["total_s"] = time.perf_counter() - t0
    summary["timings_seconds"] = timings
    summary["saved_model"] = model_path
    with open(os.path.join(output_folder, "streaming_phaseAB_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved complete TiO2 rows: {final_path}")
    print(f"Saved complete TiO2 CIFs: {final_cif_folder}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()


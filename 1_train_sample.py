#!/usr/bin/env python3
"""Juliette TiO2 constructive floating-octahedra generator v33.

Causal path:
physical source structures -> chemistry_model.json -> dominant Ti local regime
-> symmetry-valid Ti framework -> one floating TiO6 proposal per Ti
-> three-distinct-Ti vertex coincidence -> physical O clustering
-> post-construction chemistry evaluation/refinement/ranking.

No VAE, training CSV, fixed independent-site capacity, preselected O Wyckoff
skeleton, global pooled O-Ti-O construction loss, subgroup-augmented chemistry,
or SO3 topology construction is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.io import write
from pyxtal.symmetry import Group

BASE_COLUMNS = ["spg", "a", "b", "c", "alpha", "beta", "gamma"]
TI_ROLE = 6
O_ROLE = 3
MAX_TI_ATOMS = 32

SHIFTS = np.asarray(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=float,
)
ZERO_SHIFT = int(np.flatnonzero(np.all(SHIFTS == 0, axis=1))[0])


# -----------------------------------------------------------------------------
# Crystallographic helpers
# -----------------------------------------------------------------------------

def encode_wp_token(values):
    values = [int(v) for v in values]
    if not values:
        raise ValueError("Wyckoff token cannot be empty.")
    return "|".join(str(v) for v in values)


def decode_wp_token(token, expected_slots=None):
    values = [int(x) for x in str(token).strip().split("|") if str(x).strip()]
    if not values:
        raise ValueError(f"Empty Wyckoff token {token!r}.")
    return values


def _wyckoff_position_from_parameters(spg, wp_index, parameters, group=None):
    wp = (group if group is not None else Group(int(spg)))[int(wp_index)]
    dof = int(wp.get_dof())
    return np.asarray(
        wp.get_position_from_free_xyzs(
            np.asarray(parameters, dtype=float)[:dof] % 1.0
        ),
        dtype=float,
    ) % 1.0


def _deduplicate_fractional(frac, tol=1e-6):
    unique = []
    for point in np.asarray(frac, dtype=float).reshape(-1, 3) % 1.0:
        if not any(
            np.linalg.norm((point - other) - np.round(point - other)) <= tol
            for other in unique
        ):
            unique.append(point)
    return np.asarray(unique, dtype=float).reshape(-1, 3)


def _periodic_vectors_and_distances(frac_a, frac_b, cell):
    """Return image-resolved PBC vectors/distances for all 27 nearby images."""
    a = np.asarray(frac_a, float).reshape(-1, 3)
    b = np.asarray(frac_b, float).reshape(-1, 3)
    delta = b[None, :, None, :] - a[:, None, None, :] + SHIFTS[None, None, :, :]
    cart = np.einsum("...i,ij->...j", delta, np.asarray(cell, float))
    dist = np.linalg.norm(cart, axis=-1)
    return cart, dist


def _periodic_distance_matrix(frac_a, frac_b, cell):
    """Minimum distance per atom-index pair; not suitable for coordination counts."""
    _, dist = _periodic_vectors_and_distances(frac_a, frac_b, cell)
    return dist.min(-1)


def periodic_neighbor_vectors(frac, cell):
    frac = np.asarray(frac, dtype=float)
    delta = (
        frac[:, None, None, :]
        - frac[None, :, None, :]
        + SHIFTS[None, None, :, :]
    )
    cart = np.einsum("...i,ij->...j", delta, cell)
    dist = np.linalg.norm(cart, axis=-1)
    ids = np.arange(len(frac))
    dist[ids, ids, ZERO_SHIFT] = np.inf
    distances, vectors = [], []
    for i in range(len(frac)):
        d = dist[i].reshape(-1)
        v = cart[i].reshape(-1, 3)
        mask = np.isfinite(d) & (d > 1e-6)
        order = np.argsort(d[mask])
        distances.append(d[mask][order])
        vectors.append(v[mask][order])
    return distances, vectors


def _angles_deg(vectors):
    v = np.asarray(vectors, float).reshape(-1, 3)
    n = np.linalg.norm(v, axis=1)
    out = []
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            c = np.dot(v[i], v[j]) / max(n[i] * n[j], 1e-12)
            out.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return np.sort(np.asarray(out, float))


def save_tio2_cif(result, path):
    frac = np.vstack([result["ti_frac"], result["o_frac"]])
    symbols = ["Ti"] * len(result["ti_frac"]) + ["O"] * len(result["o_frac"])
    write(
        path,
        Atoms(symbols, scaled_positions=frac, cell=result["cell"], pbc=True),
        format="cif",
    )


def build_output_row(result, spg, ti_token):
    record = dict(zip(BASE_COLUMNS, [int(spg), *map(float, result["lattice"])]))
    ti_wps = decode_wp_token(ti_token)
    o_wps = list(result.get("o_wps", []))
    record["ti_skeleton_token"] = encode_wp_token(ti_wps)
    record["o_skeleton_token"] = encode_wp_token(o_wps) if o_wps else ""
    record["n_ti_independent_sites"] = int(len(ti_wps))
    record["n_o_independent_sites"] = int(len(o_wps))
    record["n_independent_sites"] = int(len(ti_wps) + len(o_wps))
    record["formula_units"] = int(len(result["ti_frac"]))

    slot = 0
    for local_id, wp in enumerate(ti_wps):
        xyz = _wyckoff_position_from_parameters(
            spg, wp, result["ti_free"][local_id]
        )
        record[f"wp{slot}"] = int(wp)
        record[f"x{slot}"], record[f"y{slot}"], record[f"z{slot}"] = map(float, xyz)
        record[f"target_coord{slot}"] = int(TI_ROLE)
        slot += 1

    for local_id, wp in enumerate(o_wps):
        xyz = np.asarray(result["o_generators"][local_id], float) % 1.0
        record[f"wp{slot}"] = int(wp)
        record[f"x{slot}"], record[f"y{slot}"], record[f"z{slot}"] = map(float, xyz)
        record[f"target_coord{slot}"] = int(O_ROLE)
        slot += 1
    return record


# -----------------------------------------------------------------------------
# chemistry_model.json
# -----------------------------------------------------------------------------
def _norm_name(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _flatten_dict(node, prefix=()):
    if isinstance(node, dict):
        yield prefix, node
        for key, value in node.items():
            yield from _flatten_dict(value, prefix + (_norm_name(key),))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _flatten_dict(value, prefix + (str(i),))


def _peak_list_from_node(node):
    if not isinstance(node, dict):
        return None
    for key in ("peaks", "retained_peaks", "modes", "components"):
        value = node.get(key)
        if isinstance(value, list) and value and all(
            isinstance(x, dict) for x in value
        ):
            peaks = [x for x in value if x.get("retained", True) is not False]
            if peaks and all(
                any(k in x for k in ("mu", "mean", "center")) for x in peaks
            ):
                return peaks
    return None


def _peak_value(peak, names, default=None):
    for name in names:
        if name in peak:
            return float(peak[name])
    if default is not None:
        return float(default)
    raise KeyError(f"Peak is missing all keys {names}: {peak}")


@dataclass(frozen=True)
class PeakMode:
    mu: float
    sigma: float
    weight: float
    sampling_min: float
    sampling_max: float


@dataclass(frozen=True)
class PeakChannel:
    name: str
    modes: tuple[PeakMode, ...]

    def probabilities(self):
        weights = np.asarray([max(0.0, m.weight) for m in self.modes], float)
        if weights.sum() <= 0:
            return np.full(len(weights), 1.0 / len(weights))
        return weights / weights.sum()

    def sample(self, rng, size=None):
        ids = rng.choice(
            len(self.modes),
            size=size,
            replace=True,
            p=self.probabilities(),
        )
        ids_arr = np.asarray(ids).reshape(-1)
        out = []
        for idx in ids_arr:
            mode = self.modes[int(idx)]
            value = rng.normal(mode.mu, max(mode.sigma, 1e-8))
            out.append(float(np.clip(value, mode.sampling_min, mode.sampling_max)))
        values = np.asarray(out, dtype=float)
        if size is None:
            return float(values[0])
        return values.reshape(np.asarray(ids).shape)

    def centers(self):
        return np.asarray([m.mu for m in self.modes], dtype=float)


class ChemistryModel:
    """Semantic reader for the peak-based chemistry model.

    Channel lookup is path-semantic rather than tied to a single JSON nesting
    layout.  This keeps the generator coupled to chemistry meaning, not to a
    serialization accident.
    """

    def __init__(
        self,
        path,
        building_center="Ti",
        attachment="O",
        center_attachment_cn=None,
        attachment_center_cn=None,
    ):
        self.path = str(path)
        with open(path, "r", encoding="utf-8") as handle:
            self.raw = json.load(handle)

        self.center = _norm_name(building_center)
        self.attachment = _norm_name(attachment)
        self.channels = {}
        for path_parts, node in _flatten_dict(self.raw):
            peaks = _peak_list_from_node(node)
            if peaks is None:
                continue
            modes = []
            for peak in peaks:
                mu = _peak_value(peak, ("mu", "mean", "center"))
                sigma = max(
                    _peak_value(peak, ("sigma", "std", "width"), 0.0), 1e-8
                )
                weight = _peak_value(
                    peak,
                    ("basin_weight", "weight", "integrated_weight", "area"),
                    1.0,
                )
                lo = _peak_value(
                    peak,
                    ("sampling_min", "sample_min", "lower"),
                    mu - 2.0 * sigma,
                )
                hi = _peak_value(
                    peak,
                    ("sampling_max", "sample_max", "upper"),
                    mu + 2.0 * sigma,
                )
                modes.append(PeakMode(mu, sigma, weight, lo, hi))
            name = "_".join(path_parts)
            self.channels[name] = PeakChannel(name, tuple(modes))

        if not self.channels:
            raise ValueError(
                f"No peak channels could be found in chemistry model {path}."
            )

        self.ti_o_radial = self._find_channel(
            include=(self.center, self.attachment, "rad"),
            exclude=("angle",),
        )
        self.o_ti_o_angle = self._find_channel(
            include=(self.attachment, self.center, self.attachment),
            any_of=("angle", "angular"),
        )
        self.ti_o_ti_angle = self._find_channel(
            include=(self.center, self.attachment, self.center),
            any_of=("angle", "angular"),
        )
        self.ti_ti_radial = self._find_channel(
            include=(self.center, self.center, "rad"),
            exclude=(self.attachment, "angle"),
        )
        self.ti_ti_ti_angles = self._find_shell_pair_angles()

        self.center_attachment_cn = (
            int(center_attachment_cn)
            if center_attachment_cn is not None
            else self._find_scalar(
                (
                    "center_attachment_cn",
                    "building_center_attachment_cn",
                    "ti_o_cn",
                    "expected_center_cn",
                ),
                default=6,
            )
        )
        self.attachment_center_cn = (
            int(attachment_center_cn)
            if attachment_center_cn is not None
            else self._find_scalar(
                (
                    "attachment_center_cn",
                    "attachment_building_center_cn",
                    "o_ti_cn",
                    "expected_attachment_cn",
                ),
                default=3,
            )
        )
        self.ti_o_cutoff = self._find_scalar(
            (
                "ti_o_shell_cutoff",
                "o_ti_shell_cutoff",
                "center_attachment_shell_cutoff",
                "shell_cutoff",
            ),
            default=max(m.sampling_max for m in self.ti_o_radial.modes) + 0.25,
            as_int=False,
        )

    def _find_scalar(self, candidate_names, default, as_int=True):
        candidates = {_norm_name(x) for x in candidate_names}
        hits = []
        for path_parts, node in _flatten_dict(self.raw):
            if not isinstance(node, dict):
                continue
            for key, value in node.items():
                if _norm_name(key) in candidates and isinstance(
                    value, (int, float)
                ):
                    hits.append(float(value))
        value = hits[0] if hits else float(default)
        return int(round(value)) if as_int else float(value)

    @staticmethod
    def _token_count(name, token):
        token = _norm_name(token)
        return _norm_name(name).split("_").count(token)

    def _find_channel(self, include=(), exclude=(), any_of=()):
        scored = []
        for name, channel in self.channels.items():
            norm = _norm_name(name)
            words = norm.split("_")
            if any(_norm_name(x) in words for x in exclude):
                continue

            ok = True
            required_counts = {}
            for token in include:
                t = _norm_name(token)
                required_counts[t] = required_counts.get(t, 0) + 1
            for token, count in required_counts.items():
                if words.count(token) < count and token not in norm:
                    ok = False
                    break
            if not ok:
                continue
            if any_of and not any(_norm_name(x) in words for x in any_of):
                continue
            score = sum(words.count(_norm_name(x)) for x in include)
            score += 3 * sum(_norm_name(x) in words for x in any_of)
            scored.append((score, -len(words), name, channel))
        if not scored:
            raise KeyError(
                f"Cannot resolve chemistry channel include={include}, "
                f"any_of={any_of}, exclude={exclude}. "
                f"Available channels={sorted(self.channels)}"
            )
        scored.sort(reverse=True)
        return scored[0][3]

    def _find_shell_pair_angles(self):
        result = {}
        pattern = re.compile(r"shell_?(\d+)_?(\d+)")
        for name, channel in self.channels.items():
            norm = _norm_name(name)
            words = norm.split("_")
            if "angle" not in words and "angular" not in words:
                continue
            if words.count(self.center) < 3 and (
                f"{self.center}_{self.center}_{self.center}" not in norm
            ):
                continue
            match = pattern.search(norm)
            if match:
                i, j = sorted((int(match.group(1)), int(match.group(2))))
                result[(i, j)] = channel
        if not result:
            raise KeyError(
                "No shell-pair-conditioned center-center-center angular "
                "channels were found."
            )
        return dict(sorted(result.items()))

    def describe(self):
        return {
            "path": self.path,
            "center_attachment_cn": self.center_attachment_cn,
            "attachment_center_cn": self.attachment_center_cn,
            "ti_o_cutoff": self.ti_o_cutoff,
            "ti_o_radial": self.ti_o_radial.name,
            "o_ti_o_angle": self.o_ti_o_angle.name,
            "ti_o_ti_angle": self.ti_o_ti_angle.name,
            "ti_ti_radial": self.ti_ti_radial.name,
            "ti_ti_ti_shell_pairs": {
                f"{i}_{j}": channel.name
                for (i, j), channel in self.ti_ti_ti_angles.items()
            },
        }


@dataclass(frozen=True)
class TiChemistryTarget:
    shell_distances: np.ndarray
    shell_sigmas: np.ndarray
    shell_pair_angles: dict
    minimum_ti_ti_distance: float


@dataclass(frozen=True)
class OxygenChemistryTarget:
    ti_o_cn: int
    o_ti_cn: int
    ti_o_cutoff: float
    ti_o_distance_targets: np.ndarray
    o_ti_o_angle_targets: np.ndarray
    ti_o_ti_angle_targets: np.ndarray


class ChemistrySampler:
    def __init__(self, chemistry, seed=42):
        self.chemistry = chemistry
        self.rng = np.random.default_rng(seed)

    def sample_ti_target(self, dominant=False):
        # Retained-peak order is shell identity.  For the constructive Ti entrance,
        # dominant=True uses the peak center of each retained shell and the
        # highest-retained-weight angular mode.  This is the narrow dominant
        # regime available from chemistry_model.json without inventing a joint
        # per-site fingerprint dataset.
        values = []
        sigmas = []
        for mode in self.chemistry.ti_ti_radial.modes:
            if dominant:
                value = mode.mu
            else:
                value = self.rng.normal(mode.mu, max(mode.sigma, 1e-8))
            values.append(
                float(np.clip(value, mode.sampling_min, mode.sampling_max))
            )
            sigmas.append(max(float(mode.sigma), 0.05))
        # IMPORTANT: retain chemistry_model.json peak order exactly.
        # Ti-Ti-Ti shell-pair channel labels (shell_1_2, shell_2_3, ...)
        # were assigned against this retained-peak order during chemistry
        # extraction. Sorting by sampled distance would corrupt shell identity.
        values = np.asarray(values, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)

        pair_angles = {}
        for pair, channel in self.chemistry.ti_ti_ti_angles.items():
            if dominant:
                mode = channel.modes[int(np.argmax([m.weight for m in channel.modes]))]
                pair_angles[pair] = float(mode.mu)
            else:
                pair_angles[pair] = float(channel.sample(self.rng))

        minimum_distance = max(1.0, 0.65 * float(values[0]))
        return TiChemistryTarget(
            shell_distances=values,
            shell_sigmas=sigmas,
            shell_pair_angles=pair_angles,
            minimum_ti_ti_distance=minimum_distance,
        )

    def sample_o_target(self, n_ti):
        c = self.chemistry
        n_tio = max(1, int(n_ti) * int(c.center_attachment_cn))
        n_oti_o = max(1, int(n_ti) * math.comb(c.center_attachment_cn, 2))
        n_o = max(1, int(round(n_ti * c.center_attachment_cn / c.attachment_center_cn)))
        n_ti_o_ti = max(1, n_o * math.comb(c.attachment_center_cn, 2))
        return OxygenChemistryTarget(
            ti_o_cn=int(c.center_attachment_cn),
            o_ti_cn=int(c.attachment_center_cn),
            ti_o_cutoff=float(c.ti_o_cutoff),
            ti_o_distance_targets=np.sort(
                c.ti_o_radial.sample(self.rng, n_tio).reshape(int(n_ti), int(c.center_attachment_cn)),
                axis=1,
            ),
            o_ti_o_angle_targets=np.sort(
                c.o_ti_o_angle.sample(self.rng, n_oti_o).reshape(
                    int(n_ti), math.comb(int(c.center_attachment_cn), 2)
                ),
                axis=1,
            ),
            ti_o_ti_angle_targets=np.sort(
                c.ti_o_ti_angle.sample(self.rng, n_ti_o_ti).reshape(
                    int(n_o), math.comb(int(c.attachment_center_cn), 2)
                ),
                axis=1,
            ),
        )




# -----------------------------------------------------------------------------
# Constructive floating-octahedra builder
# -----------------------------------------------------------------------------

class TorchFloatingOctahedraBuilder:
    """Construct TiO2 through symmetry-propagated floating TiO6 proposals."""

    BASE_OCT = np.asarray(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float32,
    )

    def __init__(
        self,
        ti_initializations=12,
        ti_screen_steps=60,
        ti_refine_starts=4,
        ti_refine_steps=120,
        octahedral_branches=12,
        float_steps=180,
        coincidence_sigma=0.24,
        cluster_tolerance=0.38,
        ti_fingerprint_q90_max=3.0,
        minimum_ti_ti=1.8,
        minimum_ti_o=1.35,
        minimum_o_o=1.45,
        max_formula_units=MAX_TI_ATOMS,
        lr=0.04,
        device=None,
    ):
        self.ti_initializations = int(ti_initializations)
        self.ti_screen_steps = int(ti_screen_steps)
        self.ti_refine_starts = int(ti_refine_starts)
        self.ti_refine_steps = int(ti_refine_steps)
        self.octahedral_branches = int(octahedral_branches)
        self.float_steps = int(float_steps)
        self.coincidence_sigma = float(coincidence_sigma)
        self.cluster_tolerance = float(cluster_tolerance)
        self.ti_fingerprint_q90_max = float(ti_fingerprint_q90_max)
        self.minimum_ti_ti = float(minimum_ti_ti)
        self.minimum_ti_o = float(minimum_ti_o)
        self.minimum_o_o = float(minimum_o_o)
        self.max_formula_units = int(max_formula_units)
        self.lr = float(lr)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._template_cache = {}
        self._shifts = torch.as_tensor(SHIFTS, dtype=torch.float32, device=self.device)
        self._base_oct = torch.as_tensor(self.BASE_OCT, device=self.device)

    @staticmethod
    def _lattice_spec(lattice_type):
        lt = str(lattice_type).lower()
        if lt == "cubic":
            return ("a",)
        if lt in {"tetragonal", "hexagonal", "trigonal"}:
            return ("a", "c")
        if lt == "orthorhombic":
            return ("a", "b", "c")
        if lt == "monoclinic":
            return ("a", "b", "c", "beta")
        return ("a", "b", "c", "alpha", "beta", "gamma")

    @staticmethod
    def _affine_map(function, dof):
        zero = np.asarray(function(np.zeros(dof)), dtype=float)
        matrix = np.zeros((3, dof), dtype=float)
        for k in range(dof):
            x = np.zeros(dof)
            x[k] = 0.137
            y = np.asarray(function(x), dtype=float)
            delta = (y - zero + 0.5) % 1.0 - 0.5
            matrix[:, k] = delta / 0.137
        return matrix, zero

    def _orbit_template(self, group, wps):
        site_dofs, orbit_rot, orbit_trans, gen_A, gen_b, orbit_offsets = [], [], [], [], [], []
        offset = 0
        for w in wps:
            wp = group[int(w)]
            dof = int(wp.get_dof())
            site_dofs.append(dof)
            A, b = self._affine_map(
                lambda u, wp=wp: wp.get_position_from_free_xyzs(u), dof
            )
            gen_A.append(torch.tensor(A, dtype=torch.float32, device=self.device))
            gen_b.append(torch.tensor(b, dtype=torch.float32, device=self.device))
            rots, trans = [], []
            for op in wp.ops:
                rots.append(np.asarray(op.rotation_matrix, float))
                trans.append(np.asarray(op.translation_vector, float))
            orbit_rot.append(torch.tensor(np.asarray(rots), dtype=torch.float32, device=self.device))
            orbit_trans.append(torch.tensor(np.asarray(trans), dtype=torch.float32, device=self.device))
            orbit_offsets.append(offset)
            offset += len(rots)
        return {
            "wps": tuple(int(w) for w in wps),
            "site_dofs": tuple(site_dofs),
            "gen_A": gen_A,
            "gen_b": gen_b,
            "orbit_rot": orbit_rot,
            "orbit_trans": orbit_trans,
            "orbit_offsets": tuple(orbit_offsets),
            "n_atoms": int(offset),
        }

    def _template(self, spg, ti_token):
        key = (int(spg), str(ti_token))
        if key in self._template_cache:
            return self._template_cache[key]
        group = Group(int(spg))
        ti_wps = decode_wp_token(ti_token)
        ti = self._orbit_template(group, ti_wps)
        if ti["n_atoms"] < 1 or ti["n_atoms"] > self.max_formula_units:
            return None
        out = {
            "spg": int(spg),
            "group": group,
            "lattice_type": str(group.lattice_type).lower(),
            "spec": self._lattice_spec(group.lattice_type),
            "ti": ti,
            "n_ti": int(ti["n_atoms"]),
        }
        self._template_cache[key] = out
        return out

    def _lattice(self, template, vals):
        B = vals.shape[0]
        lengths = torch.nn.functional.softplus(vals) + 1.2
        lt = template["lattice_type"]
        if lt == "cubic":
            a = lengths[:, 0]
            abc = torch.stack([a, a, a], 1)
            ang = torch.full((B, 3), math.pi / 2, device=self.device)
        elif lt == "tetragonal":
            a, c = lengths[:, 0], lengths[:, 1]
            abc = torch.stack([a, a, c], 1)
            ang = torch.full((B, 3), math.pi / 2, device=self.device)
        elif lt in {"hexagonal", "trigonal"}:
            a, c = lengths[:, 0], lengths[:, 1]
            abc = torch.stack([a, a, c], 1)
            ang = torch.tensor(
                [math.pi / 2, math.pi / 2, 2 * math.pi / 3], device=self.device
            ).repeat(B, 1)
        elif lt == "orthorhombic":
            abc = lengths[:, :3]
            ang = torch.full((B, 3), math.pi / 2, device=self.device)
        elif lt == "monoclinic":
            abc = lengths[:, :3]
            beta = math.pi / 3 + torch.sigmoid(vals[:, 3]) * math.pi / 3
            ang = torch.stack(
                [torch.full_like(beta, math.pi / 2), beta, torch.full_like(beta, math.pi / 2)], 1
            )
        else:
            abc = lengths[:, :3]
            ang = math.pi / 4 + torch.sigmoid(vals[:, 3:6]) * math.pi / 2

        a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
        alpha, beta, gamma = ang[:, 0], ang[:, 1], ang[:, 2]
        ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
        sg = torch.sin(gamma).clamp_min(1e-4)
        y3 = c * (ca - cb * cg) / sg
        z2_raw = c * c - (c * cb) ** 2 - y3 * y3
        z2 = z2_raw.clamp_min(1e-6)
        zero = torch.zeros_like(a)
        cell = torch.stack(
            [
                torch.stack([a, zero, zero], 1),
                torch.stack([b * cg, b * sg, zero], 1),
                torch.stack([c * cb, y3, torch.sqrt(z2)], 1),
            ],
            1,
        )
        return abc, ang, cell, z2_raw

    def _expand_ti(self, template, coord_raw):
        frac, free = [], []
        cursor = 0
        for dof, A, b, R, t in zip(
            template["ti"]["site_dofs"],
            template["ti"]["gen_A"],
            template["ti"]["gen_b"],
            template["ti"]["orbit_rot"],
            template["ti"]["orbit_trans"],
        ):
            u = torch.sigmoid(coord_raw[:, cursor:cursor + dof])
            cursor += dof
            free.append(u)
            gen = (u @ A.T + b) % 1.0
            orbit = (torch.einsum("oij,bj->boi", R, gen) + t[None, :, :]) % 1.0
            frac.append(orbit)
        return torch.cat(frac, dim=1), free

    def _same_species(self, frac, cell):
        B, N = frac.shape[:2]
        delta = (
            frac[:, :, None, None, :]
            - frac[:, None, :, None, :]
            + self._shifts[None, None, None, :, :]
        )
        vec = torch.einsum("bijnk,bkl->bijnl", delta, cell)
        dist = torch.linalg.norm(vec, dim=-1).clamp_min(1e-6)
        eye = torch.eye(N, device=self.device, dtype=torch.bool)[None, :, :, None]
        zero = (torch.arange(27, device=self.device) == ZERO_SHIFT)[None, None, None, :]
        dist = dist.masked_fill(eye & zero, 1e6)
        return vec, dist

    def _ti_fingerprint(self, ti_frac, cell, target, abc, z2_raw):
        B, Nt = ti_frac.shape[:2]
        vec_all, dist_all = self._same_species(ti_frac, cell)
        d = dist_all.reshape(B, Nt, -1)
        v = vec_all.reshape(B, Nt, -1, 3)
        shell = torch.as_tensor(target.shell_distances, dtype=torch.float32, device=self.device)
        sigma = torch.as_tensor(target.shell_sigmas, dtype=torch.float32, device=self.device).clamp_min(0.05)
        shell_err = torch.abs(d[..., None] - shell) / sigma
        local_shell = shell_err.min(dim=2).values
        fingerprint_terms = [local_shell]

        for (si, sj), target_angle in target.shell_pair_angles.items():
            i, j = int(si) - 1, int(sj) - 1
            if i >= len(shell) or j >= len(shell):
                continue
            idi = torch.argmin(torch.abs(d - shell[i]), dim=-1)
            ej = torch.abs(d - shell[j])
            if i == j:
                ej = ej.scatter(2, idi[..., None], 1e6)
            idj = torch.argmin(ej, dim=-1)
            vi = torch.gather(v, 2, idi[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
            vj = torch.gather(v, 2, idj[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
            cos = (vi * vj).sum(-1) / (
                torch.linalg.norm(vi, dim=-1) * torch.linalg.norm(vj, dim=-1)
            ).clamp_min(1e-8)
            theta = torch.rad2deg(torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7)))
            fingerprint_terms.append(((theta - float(target_angle)) / 15.0)[..., None])

        fp = torch.cat(fingerprint_terms, dim=-1)
        site_error = torch.sqrt(torch.mean(fp.pow(2), dim=-1) + 1e-8)
        min_titi = d.amin((1, 2))
        exclusion = max(self.minimum_ti_ti, float(target.minimum_ti_ti_distance))
        overlap = torch.relu(exclusion - min_titi).pow(2) / max(exclusion**2, 0.1)
        aspect = abc.max(-1).values / abc.min(-1).values.clamp_min(1e-4)
        shape = torch.relu(aspect - 6.0).pow(2) / 36.0
        c = abc[:, 2]
        margin = z2_raw / c.square().clamp_min(1e-8)
        metric = torch.relu(1e-4 - margin).pow(2) * 1e6
        loss = site_error.pow(2).mean(1) + 8.0 * overlap + 0.1 * shape + metric
        detail = {
            "ti_fingerprint_mean": site_error.mean(1),
            "ti_fingerprint_q90": torch.quantile(site_error, 0.9, dim=1),
            "minimum_ti_ti_distance": min_titi,
            "ti_ti_overlap_loss": overlap,
            "aspect_ratio": aspect,
            "metric_valid": margin > 1e-4,
            "cell_metric_margin": margin,
        }
        return loss, detail

    def _initial_ti_raw(self, template, target, nstart):
        base = float(np.max(target.shell_distances)) * max(template["n_ti"], 1) ** (1 / 3)
        nlat = len(template["spec"])
        ncoord = sum(template["ti"]["site_dofs"])
        raw = torch.randn((int(nstart), nlat + ncoord), device=self.device)
        raw[:, :nlat] *= 0.45
        raw[:, :nlat] += math.log(math.expm1(max(base - 1.2, 0.5)))
        return raw

    def _ti_geometry(self, template, target, raw):
        nlat = len(template["spec"])
        abc, ang, cell, z2_raw = self._lattice(template, raw[:, :nlat])
        ti_frac, ti_free = self._expand_ti(template, raw[:, nlat:])
        loss, detail = self._ti_fingerprint(ti_frac, cell, target, abc, z2_raw)
        return loss, detail, (abc, ang, cell, ti_frac, ti_free)

    def _optimize_ti(self, template, target, raw, steps):
        raw = raw.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=self.lr)
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            loss, _, _ = self._ti_geometry(template, target, raw)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_([raw], 10.0)
            opt.step()
        return raw.detach()

    @staticmethod
    def _axis_angle_rotation(w):
        theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-8)
        axis = w / theta
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        K = torch.stack(
            [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
        ).reshape(*w.shape[:-1], 3, 3)
        eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(*w.shape[:-1], 3, 3)
        s = torch.sin(theta)[..., None]
        c = torch.cos(theta)[..., None]
        return eye + s * K + (1.0 - c) * torch.matmul(K, K)

    def _initial_branch_raw(self, template, chemistry, nbranch):
        nsite = len(template["ti"]["wps"])
        raw = torch.randn((int(nbranch), nsite, 27), device=self.device)
        raw[:, :, :3] *= 1.5
        raw[:, :, 3:21] *= 0.35
        raw[:, :, 21:27] *= 0.5
        return raw

    def _floating_vertices(self, template, cell, ti_frac, branch_raw, chemistry):
        B = branch_raw.shape[0]
        inv_cell = torch.linalg.inv(cell)
        dominant_mode = chemistry.ti_o_radial.modes[
            int(np.argmax([m.weight for m in chemistry.ti_o_radial.modes]))
        ]
        r0 = float(dominant_mode.mu)
        rs = max(float(dominant_mode.sigma), 0.05)

        vertex_frac = []
        vertex_unwrapped = []
        owners = []
        vertex_site = []
        ti_cursor = 0
        for site_id, (Rop, _top) in enumerate(
            zip(template["ti"]["orbit_rot"], template["ti"]["orbit_trans"])
        ):
            n_orbit = Rop.shape[0]
            local = branch_raw[:, site_id]
            Rlocal = self._axis_angle_rotation(local[:, :3])
            dirs = torch.einsum("vj,bij->bvi", self._base_oct, Rlocal)
            dirs = dirs + 0.16 * torch.tanh(local[:, 3:21].reshape(B, 6, 3))
            dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp_min(1e-8)
            radii = r0 + rs * torch.tanh(local[:, 21:27])
            dcart = dirs * radii[..., None]
            dfrac = torch.einsum("bvi,bij->bvj", dcart, inv_cell)
            transformed = torch.einsum("oij,bvj->bovi", Rop, dfrac)
            centers = ti_frac[:, ti_cursor:ti_cursor + n_orbit]
            vu = centers[:, :, None, :] + transformed
            vf = vu % 1.0
            vertex_unwrapped.append(vu.reshape(B, n_orbit * 6, 3))
            vertex_frac.append(vf.reshape(B, n_orbit * 6, 3))
            for o in range(n_orbit):
                owners.extend([ti_cursor + o] * 6)
                vertex_site.extend([site_id] * 6)
            ti_cursor += n_orbit

        return (
            torch.cat(vertex_frac, dim=1),
            torch.cat(vertex_unwrapped, dim=1),
            torch.as_tensor(owners, dtype=torch.long, device=self.device),
            torch.as_tensor(vertex_site, dtype=torch.long, device=self.device),
        )

    def _vertex_image_distances(self, vertex_unwrapped, cell):
        """Distances from central proposals to all periodic proposal images."""
        delta = (
            vertex_unwrapped[:, None, :, None, :]
            + self._shifts[None, None, None, :, :]
            - vertex_unwrapped[:, :, None, None, :]
        )
        vec = torch.einsum("bijnk,bkl->bijnl", delta, cell)
        return torch.linalg.norm(vec, dim=-1)

    def _floating_loss(self, template, ti_target, chemistry, ti_raw, branch_raw):
        nlat = len(template["spec"])
        abc, ang, cell, z2_raw = self._lattice(template, ti_raw[:, :nlat])
        ti_frac, ti_free = self._expand_ti(template, ti_raw[:, nlat:])
        ti_loss, ti_detail = self._ti_fingerprint(ti_frac, cell, ti_target, abc, z2_raw)
        vf, vu, owners, _ = self._floating_vertices(
            template, cell, ti_frac, branch_raw, chemistry
        )
        dist_img = self._vertex_image_distances(vu, cell)
        B, V, _, NS = dist_img.shape
        Nt = int(template["n_ti"])

        kernel = torch.exp(-0.5 * (dist_img / self.coincidence_sigma).pow(2))

        # Physical owner identity is (Ti index, lattice-image shift).
        candidate_owner_class = (
            owners[:, None] + Nt * torch.arange(NS, device=self.device)[None, :]
        ).reshape(-1)
        kernel_flat = kernel.reshape(B, V, V * NS)
        owner_mass = torch.zeros(
            (B, V, Nt * NS), dtype=kernel.dtype, device=self.device
        )
        owner_mass.scatter_add_(
            2,
            candidate_owner_class[None, None, :].expand(B, V, -1),
            kernel_flat,
        )

        own_class = owners + Nt * ZERO_SHIFT
        own_mask = torch.nn.functional.one_hot(
            own_class, num_classes=Nt * NS
        ).bool()[None].expand(B, -1, -1)
        owner_presence = (1.0 - torch.exp(-owner_mass)).masked_fill(own_mask, 0.0)
        rho = owner_presence.sum(-1)

        occupancy = (rho - 2.0).pow(2).mean(1)
        overcoord = torch.relu(rho - 2.15).pow(2).mean(1)

        owner_dist = torch.full(
            (B, V, Nt * NS), 1e6, dtype=dist_img.dtype, device=self.device
        )
        owner_dist.scatter_reduce_(
            2,
            candidate_owner_class[None, None, :].expand(B, V, -1),
            dist_img.reshape(B, V, V * NS),
            reduce="amin",
            include_self=True,
        )
        owner_dist = owner_dist.masked_fill(own_mask, 1e6)
        nearest = torch.topk(
            owner_dist, k=min(3, Nt * NS - 1), dim=-1, largest=False
        ).values
        compact = nearest[:, :, :2].pow(2).mean((1, 2)) / max(
            self.coincidence_sigma**2, 1e-4
        )
        if nearest.shape[-1] >= 3:
            overcollapse = torch.relu(
                1.5 * self.coincidence_sigma - nearest[:, :, 2]
            ).pow(2).mean(1)
            overcollapse = overcollapse / max(
                self.coincidence_sigma**2, 1e-4
            )
        else:
            overcollapse = torch.zeros(B, device=self.device)

        # Exclude collapse among the six vertices belonging to the same physical
        # Ti center. Images of that Ti in neighboring cells are distinct owners.
        same_owner_central = (
            owners[None, :, None] == owners[None, None, :]
        )
        zero_image_dist = dist_img[..., ZERO_SHIFT]
        eye = torch.eye(V, device=self.device, dtype=torch.bool)[None]
        same_dist = zero_image_dist.masked_fill(
            ~same_owner_central | eye, 1e6
        )
        min_same = same_dist.amin((1, 2))
        same_collapse = torch.relu(
            0.65 * chemistry.ti_o_cutoff - min_same
        ).pow(2)
        same_collapse = same_collapse / max(
            chemistry.ti_o_cutoff**2, 0.1
        )

        # Image-resolved Ti-O coordination is part of the construction loss.
        # Three coincident floating proposals represent one future physical O,
        # so proposal-shell occupancy carries one-third atomic weight.
        tio_delta = (
            vu[:, None, :, None, :]
            + self._shifts[None, None, None, :, :]
            - ti_frac[:, :, None, None, :]
        )
        tio_vec = torch.einsum("btvsk,bkl->btvsl", tio_delta, cell)
        tio_dist = torch.linalg.norm(tio_vec, dim=-1)
        width = 0.08
        cn_weight = torch.sigmoid(
            (float(chemistry.ti_o_cutoff) - tio_dist) / width
        )
        ti_cn_img = cn_weight.sum((2, 3)) / 3.0
        ti_cn_loss = (
            ti_cn_img - float(chemistry.center_attachment_cn)
        ).pow(2).mean(1)
        ti_cn_over = torch.relu(
            ti_cn_img - float(chemistry.center_attachment_cn)
        ).pow(2).mean(1)

        local_penalty = torch.tanh(
            branch_raw[:, :, 3:21]
        ).pow(2).mean((1, 2))
        radial_penalty = torch.tanh(
            branch_raw[:, :, 21:27]
        ).pow(2).mean((1, 2))

        total = (
            0.55 * ti_loss
            + 1.8 * occupancy
            + 1.2 * compact
            + 2.5 * overcoord
            + 2.0 * overcollapse
            + 6.0 * same_collapse
            + 1.2 * ti_cn_loss
            + 2.0 * ti_cn_over
            + 0.08 * local_penalty
            + 0.04 * radial_penalty
        )
        detail = dict(ti_detail)
        detail.update(
            {
                "coincidence_occupancy_loss": occupancy,
                "coincidence_compactness_loss": compact,
                "coincidence_overcoord_loss": overcoord,
                "coincidence_overcollapse_loss": overcollapse,
                "same_ti_vertex_collapse_loss": same_collapse,
                "image_resolved_ti_cn_loss": ti_cn_loss,
                "image_resolved_ti_cn_over_loss": ti_cn_over,
                "image_resolved_ti_cn_mean": ti_cn_img.mean(1),
                "image_resolved_ti_cn_q90_error": torch.quantile(
                    torch.abs(
                        ti_cn_img - float(chemistry.center_attachment_cn)
                    ),
                    0.9,
                    dim=1,
                ),
                "proposal_distortion_loss": local_penalty,
                "proposal_radius_loss": radial_penalty,
                "proposal_rho_mean": rho.mean(1),
                "proposal_rho_q10": torch.quantile(rho, 0.1, dim=1),
                "proposal_rho_q90": torch.quantile(rho, 0.9, dim=1),
                "minimum_same_ti_vertex_distance": min_same,
            }
        )
        return total, detail, (
            abc, ang, cell, ti_frac, ti_free, vf, vu, owners
        )

    def _optimize_branches(self, template, ti_target, chemistry, ti_raw, branch_raw, steps):
        branch_raw = branch_raw.detach().clone().requires_grad_(True)
        ti_raw = ti_raw.detach().clone().repeat(len(branch_raw), 1).requires_grad_(True)
        opt = torch.optim.Adam([ti_raw, branch_raw], lr=self.lr)
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            loss, _, _ = self._floating_loss(
                template, ti_target, chemistry, ti_raw, branch_raw
            )
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_([ti_raw, branch_raw], 10.0)
            opt.step()
        return ti_raw.detach(), branch_raw.detach()

    def _cluster_vertices(self, vertex_frac, vertex_unwrapped, owners, cell):
        frac = np.asarray(vertex_frac, float) % 1.0
        unwrapped = np.asarray(vertex_unwrapped, float)
        owners = np.asarray(owners, int)
        dist = _periodic_distance_matrix(frac, frac, cell)
        n = len(frac)
        parent = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Graph construction remains PBC-distance based. Physical owner identity
        # is checked after each component is unwrapped.
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] <= self.cluster_tolerance:
                    union(i, j)

        comps = {}
        for i in range(n):
            comps.setdefault(find(i), []).append(i)
        clusters = list(comps.values())

        cluster_owner_ids = []
        cluster_shifts = np.zeros((n, 3), dtype=int)
        exact_triplets = 0
        for c in clusters:
            anchor = c[0]
            owner_ids = []
            for idx in c:
                delta = (
                    unwrapped[idx][None, :]
                    + SHIFTS
                    - unwrapped[anchor][None, :]
                )
                cart = delta @ np.asarray(cell, float)
                sid = int(np.argmin(np.linalg.norm(cart, axis=1)))
                shift = SHIFTS[sid].astype(int)
                cluster_shifts[idx] = shift
                owner_ids.append((int(owners[idx]), *map(int, shift)))
            cluster_owner_ids.append(owner_ids)
            if len(c) == 3 and len(set(owner_ids)) == 3:
                exact_triplets += 1

        exact = (
            len(clusters) == 2 * len(np.unique(owners))
            and exact_triplets == len(clusters)
        )
        if not exact:
            return None, {
                "cluster_success": False,
                "n_clusters": int(len(clusters)),
                "target_clusters": int(2 * len(np.unique(owners))),
                "exact_triplet_clusters": int(exact_triplets),
                "cluster_size_histogram_json": json.dumps(
                    {
                        str(k): int(sum(len(c) == k for c in clusters))
                        for k in sorted(set(map(len, clusters)))
                    },
                    separators=(",", ":"),
                ),
            }, None

        centroids = []
        assignment = np.empty(n, dtype=int)
        for cid, c in enumerate(clusters):
            aligned = []
            for idx in c:
                aligned.append(unwrapped[idx] + cluster_shifts[idx])
                assignment[idx] = cid
            centroids.append(np.mean(aligned, axis=0) % 1.0)
        return np.asarray(centroids), {
            "cluster_success": True,
            "n_clusters": int(len(clusters)),
            "target_clusters": int(2 * len(np.unique(owners))),
            "exact_triplet_clusters": int(len(clusters)),
            "cluster_size_histogram_json": json.dumps(
                {"3": int(len(clusters))}
            ),
        }, (assignment, cluster_shifts, clusters, cluster_owner_ids)

    def _recover_o_wyckoff(self, spg, o_frac, cell, tol=0.10):
        group = Group(int(spg))
        remaining = list(range(len(o_frac)))
        recovered = []
        generators = []
        while remaining:
            best = None
            for idx in remaining:
                pos = np.asarray(o_frac[idx], float)
                for wp_index in range(len(group)):
                    wp = group[wp_index]
                    mult = int(wp.multiplicity)
                    if mult > len(remaining):
                        continue
                    try:
                        orbit = wp.get_all_positions(pos)
                    except Exception:
                        orbit = None
                    if orbit is None or len(orbit) != mult:
                        continue
                    rem_frac = np.asarray([o_frac[i] for i in remaining], float)
                    dm = _periodic_distance_matrix(np.asarray(orbit), rem_frac, cell)
                    used = set()
                    matched = []
                    total = 0.0
                    ok = True
                    for row in range(len(orbit)):
                        order = np.argsort(dm[row])
                        choice = next((int(j) for j in order if int(j) not in used), None)
                        if choice is None or dm[row, choice] > tol:
                            ok = False
                            break
                        used.add(choice)
                        matched.append(remaining[choice])
                        total += float(dm[row, choice])
                    if ok:
                        score = (total / mult, -mult, wp_index)
                        if best is None or score < best[0]:
                            gen = wp.search_generator(pos, tol=max(tol / max(np.linalg.norm(cell, axis=1)), 1e-3))
                            if gen is None:
                                gen = pos
                            best = (score, int(wp_index), np.asarray(gen, float) % 1.0, matched)
            if best is None:
                return None
            _, wp_index, gen, matched = best
            recovered.append(wp_index)
            generators.append(gen)
            matched_set = set(matched)
            remaining = [i for i in remaining if i not in matched_set]
        return recovered, generators

    def _chemistry_diagnostics(self, ti_frac, o_frac, cell, o_target):
        tio_vec, tio_dist = _periodic_vectors_and_distances(
            ti_frac, o_frac, cell
        )
        cutoff = float(o_target.ti_o_cutoff)
        ti_cn = np.sum(tio_dist <= cutoff, axis=(1, 2))
        o_cn = np.sum(tio_dist <= cutoff, axis=(0, 2))

        Nt, No, NS = tio_dist.shape
        flat_ti_dist = tio_dist.reshape(Nt, No * NS)
        flat_ti_vec = tio_vec.reshape(Nt, No * NS, 3)
        k_ti = int(o_target.ti_o_cn)
        ti_order = np.argsort(flat_ti_dist, axis=1)[:, :k_ti]
        ti_d = np.take_along_axis(flat_ti_dist, ti_order, axis=1)
        bond_mae = float(
            np.mean(np.abs(ti_d - o_target.ti_o_distance_targets))
        )

        ti_angles = []
        for i in range(Nt):
            vec = flat_ti_vec[i, ti_order[i]]
            ti_angles.append(_angles_deg(vec))
        ti_angles = np.asarray(ti_angles)
        oti_mae = float(
            np.mean(np.abs(ti_angles - o_target.o_ti_o_angle_targets))
        )

        # Reverse image-resolved Ti neighborhoods around each central O.
        oti_vec, oti_dist = _periodic_vectors_and_distances(
            o_frac, ti_frac, cell
        )
        flat_o_dist = oti_dist.reshape(No, Nt * NS)
        flat_o_vec = oti_vec.reshape(No, Nt * NS, 3)
        k_o = int(o_target.o_ti_cn)
        o_order = np.argsort(flat_o_dist, axis=1)[:, :k_o]
        o_angles = []
        for j in range(No):
            vec = flat_o_vec[j, o_order[j]]
            o_angles.append(_angles_deg(vec))
        o_angles = np.asarray(o_angles)
        tio_mae = float(
            np.mean(np.abs(o_angles - o_target.ti_o_ti_angle_targets))
        )

        oo_vec, oo_dist = _periodic_vectors_and_distances(
            o_frac, o_frac, cell
        )
        for i in range(No):
            oo_dist[i, i, ZERO_SHIFT] = np.inf
        min_oo = float(np.min(oo_dist))

        tt_vec, tt_dist = _periodic_vectors_and_distances(
            ti_frac, ti_frac, cell
        )
        for i in range(Nt):
            tt_dist[i, i, ZERO_SHIFT] = np.inf
        min_titi = float(np.min(tt_dist))
        min_tio = float(np.min(tio_dist))

        geometry_valid = bool(
            min_titi >= self.minimum_ti_ti
            and min_tio >= self.minimum_ti_o
            and min_oo >= self.minimum_o_o
        )
        chemistry_score = (
            4.0 * np.mean(
                np.abs(ti_cn - int(o_target.ti_o_cn))
            )
            + 4.0 * np.mean(
                np.abs(o_cn - int(o_target.o_ti_cn))
            )
            + bond_mae / 0.10
            + oti_mae / 10.0
            + tio_mae / 10.0
        )
        return {
            "chemistry_score": float(chemistry_score),
            "exact_ti_cn6_fraction": float(
                np.mean(ti_cn == int(o_target.ti_o_cn))
            ),
            "exact_o_cn3_fraction": float(
                np.mean(o_cn == int(o_target.o_ti_cn))
            ),
            "achieved_ti_o_cn": float(np.mean(ti_cn)),
            "achieved_o_ti_cn": float(np.mean(o_cn)),
            "ti_cn_q90_error": float(
                np.quantile(
                    np.abs(ti_cn - int(o_target.ti_o_cn)), 0.9
                )
            ),
            "o_cn_q90_error": float(
                np.quantile(
                    np.abs(o_cn - int(o_target.o_ti_cn)), 0.9
                )
            ),
            "ti_o_bond_mae": bond_mae,
            "o_ti_o_angle_mae": oti_mae,
            "ti_o_ti_angle_mae": tio_mae,
            "minimum_ti_ti_distance": min_titi,
            "minimum_ti_o_distance": min_tio,
            "minimum_o_o_distance": min_oo,
            "geometry_valid": geometry_valid,
            "chemistry_hard_valid": bool(
                geometry_valid
                and np.all(ti_cn == int(o_target.ti_o_cn))
                and np.all(o_cn == int(o_target.o_ti_cn))
            ),
        }

    def build(self, spg, ti_token, ti_target, o_target, chemistry, sample_id):
        template = self._template(spg, ti_token)
        if template is None or template["n_ti"] < 3:
            return None, []

        raw = self._initial_ti_raw(template, ti_target, self.ti_initializations)
        raw = self._optimize_ti(template, ti_target, raw, self.ti_screen_steps)
        with torch.no_grad():
            score = self._ti_geometry(template, ti_target, raw)[0].cpu().numpy()
        order = np.argsort(score)[:min(self.ti_refine_starts, len(score))]
        refined = self._optimize_ti(
            template, ti_target, raw[order], self.ti_refine_steps
        )

        attempts = []
        accepted_frameworks = []
        with torch.no_grad():
            loss, detail, _ = self._ti_geometry(template, ti_target, refined)
            for i in range(len(refined)):
                row = {
                    "sample_id": int(sample_id),
                    "stage": "ti_framework",
                    "framework_rank": int(i),
                    "total_loss": float(loss[i]),
                }
                for key, value in detail.items():
                    row[key] = bool(value[i]) if value.dtype == torch.bool else float(value[i])
                row["ti_framework_valid"] = bool(
                    row["metric_valid"]
                    and row["minimum_ti_ti_distance"] >= self.minimum_ti_ti
                    and row["ti_fingerprint_q90"] <= self.ti_fingerprint_q90_max
                )
                attempts.append(row)
                if row["ti_framework_valid"]:
                    accepted_frameworks.append((i, refined[i:i + 1]))

        if not accepted_frameworks:
            return None, attempts

        branch_results = []
        for framework_rank, ti_raw_single in accepted_frameworks:
            branches = self._initial_branch_raw(template, chemistry, self.octahedral_branches)
            ti_raw_b, branch_raw = self._optimize_branches(
                template, ti_target, chemistry, ti_raw_single, branches, self.float_steps
            )
            with torch.no_grad():
                total, detail, geom = self._floating_loss(
                    template, ti_target, chemistry, ti_raw_b, branch_raw
                )
                abc, ang, cell, ti_frac, ti_free, vf, vu, owners = geom
                for b in range(len(branch_raw)):
                    cluster_o, cluster_diag, cluster_state = self._cluster_vertices(
                        vf[b].cpu().numpy(),
                        vu[b].cpu().numpy(),
                        owners.cpu().numpy(),
                        cell[b].cpu().numpy(),
                    )
                    audit = {
                        "sample_id": int(sample_id),
                        "stage": "floating_branch",
                        "framework_rank": int(framework_rank),
                        "branch_id": int(b),
                        "total_loss": float(total[b]),
                    }
                    for key, value in detail.items():
                        audit[key] = bool(value[b]) if value.dtype == torch.bool else float(value[b])
                    audit.update(cluster_diag)
                    attempts.append(audit)
                    if cluster_o is None:
                        continue

                    ti_np = ti_frac[b].cpu().numpy()
                    cell_np = cell[b].cpu().numpy()
                    chem_diag = self._chemistry_diagnostics(ti_np, cluster_o, cell_np, o_target)
                    audit.update(chem_diag)
                    if not chem_diag["geometry_valid"]:
                        continue
                    wyckoff = self._recover_o_wyckoff(spg, cluster_o, cell_np)
                    symmetry_recovered = wyckoff is not None
                    audit["o_symmetry_recovered"] = bool(symmetry_recovered)
                    if not symmetry_recovered:
                        continue
                    o_wps, o_generators = wyckoff

                    ti_free_np = np.zeros((len(template["ti"]["wps"]), 3), float)
                    for j, u in enumerate(ti_free):
                        ti_free_np[j, :u.shape[1]] = u[b].cpu().numpy()
                    lattice = torch.cat([abc[b], ang[b]]).cpu().numpy()

                    item = dict(audit)
                    item.update(
                        {
                            "success": True,
                            "approach_valid": True,
                            "topology_valid": True,
                            "lattice": lattice,
                            "cell": cell_np,
                            "ti_frac": ti_np,
                            "o_frac": cluster_o,
                            "ti_free": ti_free_np,
                            "o_wps": list(map(int, o_wps)),
                            "o_generators": np.asarray(o_generators, float),
                        }
                    )
                    branch_results.append(item)

        branch_results.sort(
            key=lambda x: (
                not x.get("chemistry_hard_valid", False),
                x["chemistry_score"],
                x["total_loss"],
                x["ti_fingerprint_q90"],
            )
        )
        return (branch_results[0] if branch_results else None), attempts



# -----------------------------------------------------------------------------
# Symmetry/Wyckoff proposal engine
# -----------------------------------------------------------------------------

def canonical_ti_token(values):
    values = tuple(int(v) for v in values)
    if not values:
        raise ValueError("A Ti skeleton must contain at least one Wyckoff orbit.")
    return encode_wp_token(values)


def _enumerate_ti_skeletons_for_group(
    spg,
    max_formula_units,
    max_combinations,
):
    """Enumerate legal Ti orbit multisets for one space group.

    Repeated Wyckoff classes are allowed because they represent distinct
    independent orbits of the same Wyckoff type.
    """
    group = Group(int(spg))
    multiplicities = [int(group[i].multiplicity) for i in range(len(group))]
    allowed = [
        i
        for i, multiplicity in enumerate(multiplicities)
        if 1 <= multiplicity <= int(max_formula_units)
    ]
    legal = []
    cap = max(1, int(max_combinations))

    def visit(start, picked, total_atoms):
        if len(legal) >= cap:
            return
        if picked and 1 <= total_atoms <= int(max_formula_units):
            legal.append(canonical_ti_token(picked))

        for position in range(start, len(allowed)):
            wp = int(allowed[position])
            new_total = total_atoms + multiplicities[wp]
            if new_total > int(max_formula_units):
                continue
            visit(position, picked + [wp], new_total)
            if len(legal) >= cap:
                return

    visit(0, [], 0)
    return list(dict.fromkeys(legal))


class SymmetryProposalEngine:
    """Sample legal Ti entrances with replacement under flat SPG exploration.

    Space-group success statistics are deliberately not used here. Repeated
    (space group, Ti Wyckoff skeleton) entrances are allowed so the stochastic
    Ti/octahedral optimization can revisit the same crystallographic entrance.
    """

    def __init__(
        self,
        max_formula_units,
        max_group_skeletons=5000,
        seed=42,
    ):
        self.max_formula_units = int(max_formula_units)
        self.max_group_skeletons = int(max_group_skeletons)
        self.rng = np.random.default_rng(seed)
        self._cache = {}

    def _group_tokens(self, spg):
        spg = int(spg)
        if spg not in self._cache:
            try:
                tokens = _enumerate_ti_skeletons_for_group(
                    spg=spg,
                    max_formula_units=self.max_formula_units,
                    max_combinations=self.max_group_skeletons,
                )
            except Exception:
                tokens = []
            self._cache[spg] = tokens
        return self._cache[spg]

    def draw(self, count):
        proposals = []
        requested = int(count)
        attempts = 0
        max_attempts = max(1000, 200 * requested)

        while len(proposals) < requested and attempts < max_attempts:
            attempts += 1
            # Uniform over space-group numbers. Empty groups are skipped;
            # diagnostics never feed back into this proposal probability.
            spg = int(self.rng.integers(1, 231))
            tokens = self._group_tokens(spg)
            if not tokens:
                continue
            token = str(tokens[int(self.rng.integers(0, len(tokens)))])
            proposals.append((spg, token, "flat_spg_exploration_with_replacement"))

        return proposals




# -----------------------------------------------------------------------------
# Persistent builder workers
# -----------------------------------------------------------------------------

def _builder_worker(worker_id, device_id, task_queue, result_queue, builder_config, chemistry_path, chemistry_kwargs):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    if device_id is None:
        device = "cpu"
    else:
        torch.cuda.set_device(int(device_id))
        device = f"cuda:{int(device_id)}"

    chemistry = ChemistryModel(chemistry_path, **chemistry_kwargs)
    builder = TorchFloatingOctahedraBuilder(device=device, **builder_config)

    while True:
        task = task_queue.get()
        if task is None:
            break
        task_id = int(task["task_id"])
        seed = int(task["seed"])
        try:
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32 - 1))
            if device_id is not None:
                torch.cuda.manual_seed_all(seed)
            selected, attempts = builder.build(
                task["spg"],
                task["ti_token"],
                task["ti_target"],
                task["o_target"],
                chemistry,
                task_id,
            )
            result_queue.put(
                {
                    "worker_id": worker_id,
                    "task_id": task_id,
                    "metadata": task["metadata"],
                    "selected": selected,
                    "attempts": attempts,
                    "error": None,
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "worker_id": worker_id,
                    "task_id": task_id,
                    "metadata": task.get("metadata", {}),
                    "selected": None,
                    "attempts": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


class BuilderPool:
    def __init__(self, ngpu, builder_config, queue_depth, chemistry_path, chemistry_kwargs):
        self.ctx = mp.get_context("spawn")
        devices = list(range(ngpu)) if ngpu > 0 else [None]
        max_queue = max(4, int(queue_depth) * len(devices))
        self.task_queue = self.ctx.Queue(maxsize=max_queue)
        self.result_queue = self.ctx.Queue()
        self.processes = []
        for worker_id, device_id in enumerate(devices):
            process = self.ctx.Process(
                target=_builder_worker,
                args=(
                    worker_id,
                    device_id,
                    self.task_queue,
                    self.result_queue,
                    builder_config,
                    chemistry_path,
                    chemistry_kwargs,
                ),
                daemon=True,
            )
            process.start()
            self.processes.append(process)

    @property
    def workers(self):
        return len(self.processes)

    def submit(self, task):
        self.task_queue.put(task)

    def get_result(self):
        return self.result_queue.get()

    def close(self):
        for _ in self.processes:
            self.task_queue.put(None)
        for process in self.processes:
            process.join()


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

def deterministic_seed(global_seed, *parts):
    payload = ":".join(str(x) for x in (global_seed, *parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**31 - 1)


def resolve_ngpu(requested):
    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if requested < 0:
        raise ValueError("--ngpu cannot be negative.")
    if requested == 0:
        return visible
    if requested > visible:
        raise ValueError(
            f"Requested --ngpu={requested}, but only {visible} CUDA devices are visible."
        )
    return int(requested)


def _cpu_affinity_count():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def set_worker_thread_limits():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def run_generation(args, chemistry, ngpu, output_folder):
    sampler = ChemistrySampler(chemistry, args.seed + 101)
    proposal_engine = SymmetryProposalEngine(
        max_formula_units=args.max_formula_units,
        max_group_skeletons=args.max_group_skeletons,
        seed=args.seed + 211,
    )

    pool_folder = os.path.join(output_folder, "floating_candidate_pool")
    selected_folder = os.path.join(output_folder, "pre_joint_tio2")
    os.makedirs(pool_folder, exist_ok=True)
    os.makedirs(selected_folder, exist_ok=True)

    builder_config = {
        "ti_initializations": args.ti_starts,
        "ti_screen_steps": args.ti_screen_steps,
        "ti_refine_starts": args.ti_refine_starts,
        "ti_refine_steps": args.ti_refine_steps,
        "octahedral_branches": args.octahedral_branches,
        "float_steps": args.float_steps,
        "coincidence_sigma": args.coincidence_sigma,
        "cluster_tolerance": args.cluster_tolerance,
        "ti_fingerprint_q90_max": args.ti_fingerprint_q90_max,
        "minimum_ti_ti": args.minimum_ti_ti_distance,
        "minimum_ti_o": args.minimum_ti_o_distance,
        "minimum_o_o": args.minimum_o_o_distance,
        "max_formula_units": args.max_formula_units,
        "lr": args.builder_lr,
    }
    chemistry_kwargs = {
        "building_center": args.building_center,
        "attachment": args.attachment,
        "center_attachment_cn": args.center_attachment_cn,
        "attachment_center_cn": args.attachment_center_cn,
    }
    pool = BuilderPool(
        ngpu, builder_config, args.gpu_queue_depth, args.chemistry_model, chemistry_kwargs
    )

    task_id = tasks_submitted = tasks_completed = 0
    attempts_rows, candidates, candidate_rows = [], [], []
    framework_outcomes = []
    spg_stats = {}
    in_flight = set()
    consecutive_worker_errors = 0
    generation_target = max(args.sample, int(math.ceil(args.sample * (1.0 + args.sample_overhead))))

    def make_task():
        nonlocal task_id, tasks_submitted
        while tasks_submitted < args.max_framework_tasks:
            proposals = proposal_engine.draw(1)
            if not proposals:
                return None
            spg, ti_token, source = proposals[0]
            group = Group(int(spg))
            n_ti = sum(int(group[w].multiplicity) for w in decode_wp_token(ti_token))
            if n_ti < 3:
                continue
            ti_target = sampler.sample_ti_target(dominant=True)
            o_target = sampler.sample_o_target(n_ti)
            task_id += 1
            tasks_submitted += 1
            return {
                "task_id": task_id,
                "seed": deterministic_seed(args.seed, "floating", tasks_submitted, spg, ti_token),
                "spg": int(spg),
                "ti_token": str(ti_token),
                "ti_target": ti_target,
                "o_target": o_target,
                "metadata": {
                    "stream_index": int(tasks_submitted),
                    "spg": int(spg),
                    "ti_skeleton_token": str(ti_token),
                    "formula_units": int(n_ti),
                    "proposal_source": str(source),
                    "dominant_ti_shell_distances_json": json.dumps(
                        ti_target.shell_distances.tolist(), separators=(",", ":")
                    ),
                    "dominant_ti_shell_pair_angles_json": json.dumps(
                        {f"{i}_{j}": v for (i, j), v in ti_target.shell_pair_angles.items()},
                        separators=(",", ":"),
                    ),
                },
            }
        return None

    def submit_one():
        task = make_task()
        if task is None:
            return False
        pool.submit(task)
        in_flight.add(int(task["task_id"]))
        return True

    try:
        while len(in_flight) < pool.workers and submit_one():
            pass

        while in_flight and len(candidates) < generation_target:
            result = pool.get_result()
            in_flight.discard(int(result["task_id"]))
            tasks_completed += 1
            error = result.get("error")
            if error and not result.get("attempts"):
                consecutive_worker_errors += 1
                if consecutive_worker_errors >= pool.workers:
                    raise RuntimeError(
                        "All active floating-octahedra builder tasks failed before producing attempts. "
                        f"Latest worker error: {error}"
                    )
            else:
                consecutive_worker_errors = 0

            meta = result["metadata"]
            spg = int(meta["spg"])
            stats = spg_stats.setdefault(
                spg,
                {
                    "spg": spg,
                    "framework_attempts": 0,
                    "worker_errors": 0,
                    "ti_framework_successes": 0,
                    "exact_triplet_successes": 0,
                    "geometry_valid_successes": 0,
                    "hard_chemistry_successes": 0,
                    "candidate_successes": 0,
                },
            )
            stats["framework_attempts"] += 1
            if error:
                stats["worker_errors"] += 1

            task_attempts = result["attempts"]
            ti_success = any(
                bool(r.get("ti_framework_valid", False))
                for r in task_attempts
                if r.get("stage") == "ti_framework"
            )
            exact_success = any(
                bool(r.get("cluster_success", False))
                for r in task_attempts
                if r.get("stage") == "floating_branch"
            )
            geometry_success = any(
                bool(r.get("geometry_valid", False))
                for r in task_attempts
                if r.get("stage") == "floating_branch"
            )
            hard_success = any(
                bool(r.get("chemistry_hard_valid", False))
                for r in task_attempts
                if r.get("stage") == "floating_branch"
            )
            stats["ti_framework_successes"] += int(ti_success)
            stats["exact_triplet_successes"] += int(exact_success)
            stats["geometry_valid_successes"] += int(geometry_success)
            stats["hard_chemistry_successes"] += int(hard_success)

            framework_outcomes.append(
                {
                    "task_id": int(result["task_id"]),
                    "spg": spg,
                    "ti_framework_success": bool(ti_success),
                    "exact_triplet_success": bool(exact_success),
                    "geometry_valid_success": bool(geometry_success),
                    "hard_chemistry_success": bool(hard_success),
                    "candidate_success": False,
                }
            )

            for item in task_attempts:
                audit = dict(item)
                audit.update(meta)
                audit["worker_id"] = result["worker_id"]
                audit["worker_error"] = error
                attempts_rows.append(audit)

            selected = result["selected"]
            if selected is not None and selected.get("approach_valid", False):
                candidate_id = len(candidates)
                diag = {
                    k: v for k, v in selected.items()
                    if k not in {"lattice", "cell", "ti_frac", "o_frac", "ti_free", "o_wps", "o_generators"}
                }
                diag.update(meta)
                diag["candidate_id"] = int(candidate_id)
                candidates.append((selected, diag))
                stats["candidate_successes"] += 1
                framework_outcomes[-1]["candidate_success"] = True
                candidate_rows.append(build_output_row(selected, meta["spg"], meta["ti_skeleton_token"]))
                save_tio2_cif(
                    selected, os.path.join(pool_folder, f"candidate_{candidate_id:06d}.cif")
                )

            submit_one()

            if (
                tasks_completed == 1
                or tasks_completed % args.progress_every == 0
                or len(candidates) >= generation_target
            ):
                recent = framework_outcomes[-100:]
                recent_n = len(recent)
                recent_ti = sum(r["ti_framework_success"] for r in recent)
                recent_exact = sum(r["exact_triplet_success"] for r in recent)
                recent_geometry = sum(r["geometry_valid_success"] for r in recent)
                recent_hard = sum(r["hard_chemistry_success"] for r in recent)
                recent_candidate = sum(r["candidate_success"] for r in recent)
                print(
                    "Generation progress: "
                    f"frameworks={tasks_completed}/{tasks_submitted} completed/submitted; "
                    f"candidate_pool={len(candidates)}/{generation_target}; "
                    f"active={len(in_flight)}; "
                    f"recent100_ti={recent_ti}/{recent_n}; "
                    f"recent100_triplet={recent_exact}/{recent_n}; "
                    f"recent100_geometry={recent_geometry}/{recent_n}; "
                    f"recent100_hard={recent_hard}/{recent_n}; "
                    f"recent100_candidate={recent_candidate}/{recent_n}",
                    flush=True,
                )

        if not candidates:
            raise RuntimeError(
                "Floating-octahedra generation produced no exact three-Ti clustered TiO2 candidates."
            )
    finally:
        pool.close()

    pd.DataFrame(attempts_rows).to_csv(
        os.path.join(output_folder, "floating_builder_attempts.csv"), index=False
    )
    pd.DataFrame(framework_outcomes).to_csv(
        os.path.join(output_folder, "framework_outcomes.csv"), index=False
    )

    spg_rows = []
    for spg in sorted(spg_stats):
        row = dict(spg_stats[spg])
        attempts = max(1, int(row["framework_attempts"]))
        ti = max(1, int(row["ti_framework_successes"]))
        triplet = max(1, int(row["exact_triplet_successes"]))
        geometry = max(1, int(row["geometry_valid_successes"]))
        row["ti_success_rate"] = row["ti_framework_successes"] / attempts
        row["triplet_success_rate"] = row["exact_triplet_successes"] / attempts
        row["geometry_success_rate"] = row["geometry_valid_successes"] / attempts
        row["hard_chemistry_success_rate"] = row["hard_chemistry_successes"] / attempts
        row["candidate_success_rate"] = row["candidate_successes"] / attempts
        row["triplet_given_ti_rate"] = row["exact_triplet_successes"] / ti
        row["geometry_given_triplet_rate"] = row["geometry_valid_successes"] / triplet
        row["hard_given_geometry_rate"] = row["hard_chemistry_successes"] / geometry
        spg_rows.append(row)
    pd.DataFrame(spg_rows).to_csv(
        os.path.join(output_folder, "space_group_generation_statistics.csv"),
        index=False,
    )

    ranked_indices = sorted(
        range(len(candidates)),
        key=lambda i: (
            not bool(candidates[i][1].get("chemistry_hard_valid", False)),
            float(candidates[i][1]["chemistry_score"]),
            float(candidates[i][1]["total_loss"]),
            float(candidates[i][1]["ti_fingerprint_q90"]),
        ),
    )
    selected_indices = ranked_indices[:min(args.sample, len(ranked_indices))]
    selected_rows = [candidate_rows[i] for i in selected_indices]
    selected_diag = []
    for rank, pool_index in enumerate(selected_indices, start=1):
        diag = dict(candidates[pool_index][1])
        diag["final_rank"] = int(rank)
        selected_diag.append(diag)

    pd.DataFrame(selected_diag).to_csv(
        os.path.join(output_folder, "floating_builder_selected.csv"), index=False
    )
    for old in Path(selected_folder).glob("sample_*.cif"):
        old.unlink()
    for rank, pool_index in enumerate(selected_indices):
        cid = int(candidates[pool_index][1]["candidate_id"])
        shutil.copy2(
            os.path.join(pool_folder, f"candidate_{cid:06d}.cif"),
            os.path.join(selected_folder, f"sample_{rank:06d}.cif"),
        )

    final = pd.DataFrame(selected_rows)
    final_path = os.path.join(output_folder, f"generated_tio2_{len(final)}.csv")
    final.to_csv(final_path, index=False)

    summary = {
        "architecture": "v33_flat_spg_repeated_entrance_pbc_floating_octahedra",
        "requested_tio2": int(args.sample),
        "candidate_pool": int(len(candidates)),
        "selected_tio2": int(len(selected_rows)),
        "framework_tasks_submitted": int(tasks_submitted),
        "framework_tasks_completed": int(tasks_completed),
        "max_formula_units": int(args.max_formula_units),
        "chemistry_model": chemistry.describe(),
        "ngpu": int(ngpu),
        "gpu_workers": int(pool.workers),
        "parallelization": (
            "persistent one-process-per-visible-GPU workers; each free worker receives one "
            "complete Ti-framework task and internally evaluates multiple octahedral branches"
        ),
        "constructive_topology": (
            "one TiO6 proposal per Ti; six floating vertices per Ti; soft coincidence of "
            "two other distinct physical Ti owners (Ti index + lattice image) per vertex; "
            "exact PBC clustering requires 2N_Ti triplets with three distinct periodic Ti contributors"
        ),
        "dominant_ti_regime": (
            "site-wise Ti fingerprint against retained Ti-Ti shell centers and the highest-"
            "retained-weight mode of each shell-pair Ti-Ti-Ti angular channel"
        ),
        "oxygen_symmetry": (
            "no O Wyckoff skeleton is preselected; O positions are clustered first and "
            "PyXtal Wyckoff orbits are recovered afterwards"
        ),
        "entrance_sampling": (
            "space-group numbers are sampled uniformly; legal Ti Wyckoff skeletons are "
            "sampled with replacement; repeated entrances are allowed; cached success/failure "
            "statistics are diagnostic only and never bias proposal probabilities"
        ),
        "space_group_statistics_file": "space_group_generation_statistics.csv",
        "framework_outcomes_file": "framework_outcomes.csv",
        "removed_paths": [
            "random_full_Ti_O_coordinate_entrance",
            "preselected_O_Wyckoff_skeleton",
            "global_pooled_O_Ti_O_construction_loss",
            "VAE_crystallographic_entrance",
            "independent_site_capacity",
            "SO3_topology_construction",
        ],
    }
    with open(os.path.join(output_folder, "generation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if len(selected_rows) < args.sample:
        print(
            f"Generation underfilled: selected {len(selected_rows)}/{args.sample} exact-cluster candidates.",
            flush=True,
        )
    return final_path, selected_folder, summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Juliette flat-SPG repeated-entrance floating-octahedra TiO2 generator v33"
    )
    parser.add_argument("--chemistry-model", required=True)
    parser.add_argument("--building-center", default="Ti")
    parser.add_argument("--attachment", default="O")
    parser.add_argument("--center-attachment-cn", type=int)
    parser.add_argument("--attachment-center-cn", type=int)

    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument("--gpu-queue-depth", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)

    parser.add_argument("--max-group-skeletons", type=int, default=5000)
    parser.add_argument("--max-framework-tasks", type=int, default=10000)
    parser.add_argument("--max-formula-units", type=int, default=MAX_TI_ATOMS)

    parser.add_argument("--ti-starts", type=int, default=12)
    parser.add_argument("--ti-screen-steps", type=int, default=60)
    parser.add_argument("--ti-refine-starts", type=int, default=4)
    parser.add_argument("--ti-refine-steps", type=int, default=120)
    parser.add_argument("--ti-fingerprint-q90-max", type=float, default=3.0)

    parser.add_argument("--octahedral-branches", type=int, default=12)
    parser.add_argument("--float-steps", type=int, default=180)
    parser.add_argument("--coincidence-sigma", type=float, default=0.24)
    parser.add_argument("--cluster-tolerance", type=float, default=0.38)
    parser.add_argument("--builder-lr", type=float, default=0.04)

    parser.add_argument("--minimum-ti-ti-distance", type=float, default=1.8)
    parser.add_argument("--minimum-ti-o-distance", type=float, default=1.35)
    parser.add_argument("--minimum-o-o-distance", type=float, default=1.45)
    parser.add_argument("--sample-overhead", type=float, default=0.25)
    parser.add_argument("--output-dir", default="data/sample")
    args = parser.parse_args()

    positive = [
        args.sample, args.gpu_queue_depth, args.progress_every,
        args.max_formula_units, args.max_group_skeletons, args.max_framework_tasks,
        args.ti_starts, args.ti_screen_steps, args.ti_refine_starts,
        args.ti_refine_steps, args.octahedral_branches, args.float_steps,
    ]
    if min(positive) <= 0:
        raise ValueError("Positive integer arguments must be greater than zero.")
    if args.ti_refine_starts > args.ti_starts:
        raise ValueError("--ti-refine-starts cannot exceed --ti-starts.")
    if args.sample_overhead < 0:
        raise ValueError("--sample-overhead cannot be negative.")
    if args.coincidence_sigma <= 0 or args.cluster_tolerance <= 0:
        raise ValueError("Coincidence and cluster length scales must be positive.")

    set_worker_thread_limits()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    t0 = time.perf_counter()
    chemistry = ChemistryModel(
        args.chemistry_model,
        building_center=args.building_center,
        attachment=args.attachment,
        center_attachment_cn=args.center_attachment_cn,
        attachment_center_cn=args.attachment_center_cn,
    )
    if chemistry.center_attachment_cn != 6 or chemistry.attachment_center_cn != 3:
        raise ValueError(
            "The v31 constructive topology is specifically TiO6/OTi3 and requires center CN=6, attachment CN=3."
        )

    print("Resolved chemistry channels:", flush=True)
    print(json.dumps(chemistry.describe(), indent=2), flush=True)

    chemistry_name = Path(args.chemistry_model).parent.name or "chemistry"
    output_folder = os.path.join(
        args.output_dir, f"{chemistry_name}-floating-octahedra-v31-seed{args.seed}"
    )
    os.makedirs(output_folder, exist_ok=True)

    ngpu = resolve_ngpu(args.ngpu)
    print(
        "Resolved resources: "
        f"ngpu={ngpu}; CPU_affinity={_cpu_affinity_count()}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
        flush=True,
    )
    print(
        "No VAE/O-Wyckoff entrance. Beginning dominant-Ti -> floating-TiO6 -> exact triplet clustering.",
        flush=True,
    )

    final_path, final_cif_folder, summary = run_generation(
        args=args, chemistry=chemistry, ngpu=ngpu, output_folder=output_folder
    )
    summary["total_seconds"] = time.perf_counter() - t0
    with open(os.path.join(output_folder, "generation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved ranked TiO2 rows: {final_path}", flush=True)
    print(f"Saved ranked TiO2 CIFs: {final_cif_folder}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()


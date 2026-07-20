#!/usr/bin/env python3
"""Generate symmetry-constrained crystals from unified Juliette targets and Xn templates.

The generator uses coincidence-driven topology construction.  Each physical
building block owns an oriented local Xn topology.  Compatible ports are grouped
into an exact discrete topology before each continuous optimization block:

* pair reconciliation connects two parent blocks (carbon-like networks);
* capacitated shared-site reconciliation assigns high-CN parent ports to
  low-CN child centres with exact child arity (TiO2-like networks).

The discrete topology is refreshed dynamically as geometry improves, while all
continuous variables remain Wyckoff/lattice/block-pose variables.  Physical
atoms and X templates are expanded under the selected space group at every
objective evaluation; symmetry is never recovered post hoc.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import queue
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx
from scipy.optimize import linear_sum_assignment

SHIFTS = np.asarray(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=float,
)
ZERO_SHIFT = int(np.flatnonzero(np.all(SHIFTS == 0, axis=1))[0])
EPS = 1.0e-12


def encode_token(values) -> str:
    return "|".join(str(x) for x in values)


def decode_int_token(token: str) -> list[int]:
    return [int(x) for x in str(token).split("|") if x != ""]


def decode_str_token(token: str) -> list[str]:
    return [str(x) for x in str(token).split("|") if x != ""]


def _parse_formula(formula: str) -> dict[str, int]:
    """Parse a simple neutral chemical formula such as TiO2 or C8.

    Parentheses, charges, fractional occupancies, and isotope decorations are
    intentionally rejected.  Named construction-species labels are resolved
    before formula parsing, so labels such as O2_linear remain first-class.
    """
    import re

    text = str(formula).strip()
    if not text:
        raise ValueError("Empty target label")
    tokens = list(re.finditer(r"([A-Z][a-z]?)([0-9]*)", text))
    if not tokens or "".join(m.group(0) for m in tokens) != text:
        raise ValueError(
            f"Unknown target {formula!r}; it is neither a construction-species label "
            "nor a supported simple chemical formula"
        )
    composition: dict[str, int] = {}
    for match in tokens:
        element = match.group(1)
        count = int(match.group(2) or "1")
        if count <= 0:
            raise ValueError(f"Invalid zero/negative stoichiometry in {formula!r}")
        composition[element] = composition.get(element, 0) + count
    return composition


def resolve_targets(items: list[str], model: "XNModel") -> tuple[dict[str, int], list[dict]]:
    """Resolve unified ``--target LABEL=COUNT`` requests.

    A target label may be either:
      * an exact construction-species label in the learned model; or
      * a simple chemical formula whose elements each map to exactly one
        construction species in the model.

    Optional explicit recipes stored under ``model.raw['recipes']`` take
    precedence over formula inference.  The command line never exposes the
    resulting low-level construction-species counts.
    """
    if not items:
        raise ValueError("At least one positive --target LABEL=COUNT is required")

    recipes = model.raw.get("recipes", {})
    counts = {label: 0 for label in model.labels}
    resolved: list[dict] = []

    labels_by_formula: dict[str, list[str]] = defaultdict(list)
    for label in model.labels:
        labels_by_formula[str(model.final_formula[label])].append(label)

    seen_targets: set[str] = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --target {item!r}; expected LABEL=COUNT")
        target, value = item.split("=", 1)
        target = target.strip()
        multiplier = int(value)
        if not target or multiplier <= 0 or target in seen_targets:
            raise ValueError(f"Invalid or duplicate target {item!r}")
        seen_targets.add(target)

        expansion: dict[str, int]
        source: str
        if target in model.items:
            expansion = {target: 1}
            source = "construction_species"
        elif target in recipes:
            recipe = recipes[target]
            expansion = {
                str(label): int(number)
                for label, number in recipe.get("formula_unit", recipe.get("species", {})).items()
            }
            unknown = sorted(set(expansion) - set(model.labels))
            if unknown or not expansion or any(v <= 0 for v in expansion.values()):
                raise ValueError(f"Invalid recipe {target!r}; unknown={unknown}, expansion={expansion}")
            source = "model_recipe"
        else:
            formula = _parse_formula(target)
            expansion = {}
            for element, number in formula.items():
                choices = labels_by_formula.get(element, [])
                if not choices:
                    raise ValueError(
                        f"Target {target!r} requires element {element}, but the model has no "
                        f"construction species with final_formula={element!r}"
                    )
                if len(choices) != 1:
                    raise ValueError(
                        f"Target {target!r} is ambiguous for element {element}: {choices}. "
                        "Use explicit construction-species targets or add a named recipe to the model."
                    )
                expansion[choices[0]] = expansion.get(choices[0], 0) + int(number)
            source = "formula_inference"

        scaled = {label: int(number) * multiplier for label, number in expansion.items()}
        for label, number in scaled.items():
            counts[label] += number
        resolved.append(
            {
                "target": target,
                "count": multiplier,
                "source": source,
                "construction_species": scaled,
            }
        )

    if not any(counts.values()):
        raise ValueError("Resolved target contains no construction species")
    return counts, resolved


def deterministic_seed(seed: int, *parts) -> int:
    payload = ":".join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**31 - 1)


def _quat_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _angles_np(vectors: np.ndarray) -> np.ndarray:
    out = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a, b = vectors[i], vectors[j]
            c = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))
            out.append(math.degrees(math.acos(float(np.clip(c, -1.0, 1.0)))))
    return np.sort(np.asarray(out, float))


@dataclass(frozen=True)
class PairChannel:
    species_i: str
    species_j: str
    mu: float
    sigma: float
    sampling_min: float
    sampling_max: float
    first_shell_cutoff: float


class XNModel:
    def __init__(self, path: str):
        self.path = str(path)
        self.raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if self.raw.get("schema") != "juliette_building_block_xn_v1":
            raise ValueError(f"Unsupported model schema {self.raw.get('schema')!r}")
        self.items = {str(x["label"]): x for x in self.raw["construction_species"]}
        self.labels = tuple(self.items)
        self.label_to_id = {x: i for i, x in enumerate(self.labels)}
        self.id_to_label = {i: x for x, i in self.label_to_id.items()}
        self.final_formula = {x: str(self.items[x]["final_formula"]) for x in self.labels}
        self.block_atoms = {}
        self.block_center = {}
        self.template_vectors = {}
        self.template_cn = {}
        self.template_angles = {}
        self.template_angle_sigma = {}
        self.allowed = {}
        self.framework_models = {str(k): dict(v) for k, v in self.raw.get("framework_models", {}).items()}
        for label, item in self.items.items():
            block = item["building_block"]
            atoms = block.get("atoms", [])
            if not atoms:
                raise ValueError(f"{label}: empty building block")
            self.block_atoms[label] = (
                tuple(str(x["element"]) for x in atoms),
                np.asarray([x["position_A"] for x in atoms], float),
            )
            self.block_center[label] = np.asarray(block.get("reference_center_A", [0, 0, 0]), float)
            template = item["external_template"]
            vectors = np.asarray(template.get("canonical_vectors_A", []), float).reshape(-1, 3)
            cn = int(template["coordination_number"])
            if len(vectors) != cn:
                raise ValueError(f"{label}: X vector count != coordination number")
            self.template_vectors[label] = vectors
            self.template_cn[label] = cn
            angle_mean = np.asarray(template.get("angular_mean_deg", []), float)
            angle_sigma = np.asarray(template.get("angular_sigma_deg", []), float)
            if len(angle_mean) != len(angle_sigma):
                raise ValueError(f"{label}: angular mean/sigma lengths differ")
            order = np.argsort(angle_mean)
            self.template_angles[label] = angle_mean[order]
            self.template_angle_sigma[label] = np.maximum(angle_sigma[order], 1.0e-3)
            self.allowed[label] = tuple(str(x) for x in template.get("allowed_partner_labels", []))
        self.channels = {}
        for row in self.raw.get("pair_channels", []):
            if str(row.get("relation")) != "external_neighbor":
                continue
            channel = PairChannel(
                species_i=str(row["species_i"]),
                species_j=str(row["species_j"]),
                mu=float(row["distance_mu_A"]),
                sigma=max(float(row["distance_sigma_A"]), 1.0e-4),
                sampling_min=float(row["sampling_min_A"]),
                sampling_max=float(row["sampling_max_A"]),
                first_shell_cutoff=float(
                    row.get("first_shell_cutoff_A", float(row["sampling_max_A"]) + 0.05)
                ),
            )
            self.channels[(channel.species_i, channel.species_j)] = channel
        for label in self.labels:
            for partner in self.allowed[label]:
                if (label, partner) not in self.channels:
                    raise KeyError(f"Missing pair channel {label}->{partner}")

    def construction_plan(self, counts: dict[str, int]) -> dict:
        active = tuple(label for label in self.labels if int(counts.get(label, 0)) > 0)
        formulas = {self.final_formula[x] for x in active}
        if len(formulas) <= 1:
            return {"mode": "pair", "construction_counts": {x: int(counts.get(x, 0)) for x in self.labels},
                    "parents": active, "children": ()}
        cns = {x: int(self.template_cn[x]) for x in active}
        hi, lo = max(cns.values()), min(cns.values())
        parents = tuple(x for x in active if cns[x] == hi)
        children = tuple(x for x in active if cns[x] == lo)
        if not parents or not children or hi == lo:
            raise ValueError(f"Cannot infer construction roles for {active}")
        parent_ports = sum(int(counts[x]) * cns[x] for x in parents)
        child_capacity = sum(int(counts[x]) * cns[x] for x in children)
        if parent_ports != child_capacity:
            raise ValueError(f"Shared-site incidence mismatch: {parent_ports} != {child_capacity}")
        return {"mode": "shared_site",
                "construction_counts": {x: (int(counts[x]) if x in parents else 0) for x in self.labels},
                "parents": parents, "children": children,
                "parent_ports": parent_ports, "child_capacity": child_capacity}

    def pair(self, a: str, b: str) -> PairChannel:
        return self.channels[(str(a), str(b))]

    def physical_atoms_per_block(self, label: str) -> int:
        return len(self.block_atoms[str(label)][0])

    def physical_count(self, counts: dict[str, int]) -> int:
        return int(sum(counts[x] * self.physical_atoms_per_block(x) for x in counts))


class SymmetryProposalEngine:
    def __init__(self, model: XNModel, counts: dict[str, int], max_entries_per_group: int,
                 seed: int):
        from pyxtal.symmetry import Group

        self.model = model
        self.counts = dict(counts)
        self.active_labels = tuple(label for label in model.labels if self.counts.get(label, 0) > 0)
        self.total_blocks = int(sum(counts.values()))
        self.max_entries_per_group = int(max_entries_per_group)
        self.rng = np.random.default_rng(seed)
        self.entries = {}
        for spg in range(1, 231):
            group = Group(spg)
            tokens = self._exact_skeletons(group)
            rows = []
            for wp_token in tokens:
                for species_token in self._species_assignments(group, wp_token):
                    rows.append((wp_token, species_token))
                    if len(rows) >= self.max_entries_per_group:
                        break
                if len(rows) >= self.max_entries_per_group:
                    break
            if rows:
                self.entries[spg] = rows
        self.space_groups = sorted(self.entries)
        if not self.space_groups:
            raise RuntimeError(f"No space group realizes exact counts {self.counts}")

    def _exact_skeletons(self, group) -> list[str]:
        mult = [int(group[i].multiplicity) for i in range(len(group))]
        allowed = [i for i, m in enumerate(mult) if 1 <= m <= self.total_blocks]
        out = []

        def visit(start: int, selected: list[int], total: int):
            if len(out) >= self.max_entries_per_group:
                return
            if total == self.total_blocks:
                out.append(encode_token(selected))
                return
            for position in range(start, len(allowed)):
                wp = allowed[position]
                new_total = total + mult[wp]
                if new_total <= self.total_blocks:
                    visit(position, selected + [wp], new_total)

        visit(0, [], 0)
        return list(dict.fromkeys(out))

    def _species_assignments(self, group, wp_token: str) -> list[str]:
        wps = decode_int_token(wp_token)
        multiplicities = [int(group[x].multiplicity) for x in wps]
        target = tuple(self.counts[x] for x in self.active_labels)
        out = []

        def visit(index: int, remaining: tuple[int, ...], labels: list[str]):
            if index == len(multiplicities):
                if all(x == 0 for x in remaining):
                    out.append(encode_token(labels))
                return
            m = multiplicities[index]
            for species_id, label in enumerate(self.active_labels):
                if remaining[species_id] < m:
                    continue
                updated = list(remaining)
                updated[species_id] -= m
                visit(index + 1, tuple(updated), labels + [label])

        visit(0, target, [])
        return list(dict.fromkeys(out))

    def draw(self) -> tuple[int, str, str]:
        spg = int(self.space_groups[int(self.rng.integers(0, len(self.space_groups)))])
        rows = self.entries[spg]
        wp_token, species_token = rows[int(self.rng.integers(0, len(rows)))]
        return spg, wp_token, species_token


class XNBuilder:
    def __init__(self, model: XNModel, device: str, starts: int, screen_steps: int,
                 refine_steps: int, polish_steps: int, lr: float, minimum_distance: float,
                 soft_temperature: float, port_width: float, angular_weight: float,
                 radial_weight: float, overlap_weight: float, uniqueness_weight: float,
                 nonbonded_weight: float, nonbonded_margin: float, nonbonded_width: float,
                 angular_site_z_max: float, angular_vector_z_max: float,
                 assignment_refresh: int, target_counts: dict[str, int],
                 construction_symmetry: str, coincidence_rms_max: float,
                 coincidence_max_max: float, coincidence_weight: float,
                 distortion_weight: float, distortion_max: float,
                 framework_weight: float, framework_restraint_weight: float,
                 framework_keep: int, framework_patience: int, oxygen_coincidence_steps: int,
                 oxygen_contact_steps: int, oxygen_assigned_fraction_min: float,
                 oxygen_screen_rms_max: float, oxygen_screen_max_max: float):
        from pyxtal.symmetry import Group  # noqa: F401

        self.model = model
        self.device = torch.device(device)
        self.starts = int(starts)
        self.screen_steps = int(screen_steps)
        self.refine_steps = int(refine_steps)
        self.polish_steps = int(polish_steps)
        self.lr = float(lr)
        self.minimum_distance = float(minimum_distance)
        self.soft_temperature = float(soft_temperature)
        self.port_width = float(port_width)
        self.angular_weight = float(angular_weight)
        self.radial_weight = float(radial_weight)
        self.overlap_weight = float(overlap_weight)
        self.uniqueness_weight = float(uniqueness_weight)
        self.nonbonded_weight = float(nonbonded_weight)
        self.nonbonded_margin = float(nonbonded_margin)
        self.nonbonded_width = max(float(nonbonded_width), 1.0e-4)
        self.angular_site_z_max = float(angular_site_z_max)
        self.angular_vector_z_max = float(angular_vector_z_max)
        self.assignment_refresh = max(1, int(assignment_refresh))
        self.target_counts = {str(k): int(v) for k, v in target_counts.items()}
        self.plan = self.model.construction_plan(self.target_counts)
        self.construction_symmetry = str(construction_symmetry)
        self.coincidence_rms_max = float(coincidence_rms_max)
        self.coincidence_max_max = float(coincidence_max_max)
        self.coincidence_weight = float(coincidence_weight)
        self.distortion_weight = float(distortion_weight)
        self.distortion_max = float(distortion_max)
        self.framework_weight = float(framework_weight)
        self.framework_restraint_weight = float(framework_restraint_weight)
        self.framework_keep = max(1, int(framework_keep))
        self.framework_patience = max(1, int(framework_patience))
        self.oxygen_coincidence_steps = max(1, int(oxygen_coincidence_steps))
        self.oxygen_contact_steps = max(1, int(oxygen_contact_steps))
        self.oxygen_assigned_fraction_min = float(oxygen_assigned_fraction_min)
        if not 0.0 <= self.oxygen_assigned_fraction_min <= 1.0:
            raise ValueError("oxygen_assigned_fraction_min must be between 0 and 1")
        self.oxygen_screen_rms_max = float(oxygen_screen_rms_max)
        self.oxygen_screen_max_max = float(oxygen_screen_max_max)
        self._framework_reference = None
        self.shift_t = torch.as_tensor(SHIFTS, dtype=torch.float32, device=self.device)
        self._cache = {}

    @staticmethod
    def _lattice_spec(lattice_type: str) -> tuple[str, ...]:
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
    def _affine_map(function, dof: int) -> tuple[np.ndarray, np.ndarray]:
        zero = np.asarray(function(np.zeros(dof)), float)
        matrix = np.zeros((3, dof), float)
        for k in range(dof):
            x = np.zeros(dof)
            x[k] = 0.137
            y = np.asarray(function(x), float)
            matrix[:, k] = ((y - zero + 0.5) % 1.0 - 0.5) / 0.137
        return matrix, zero

    def _template(self, spg: int, wp_token: str, species_token: str) -> dict:
        from pyxtal.symmetry import Group

        key = (int(spg), str(wp_token), str(species_token))
        if key in self._cache:
            return self._cache[key]
        group = Group(int(spg))
        wps = decode_int_token(wp_token)
        labels = decode_str_token(species_token)
        if len(wps) != len(labels):
            raise ValueError("Wyckoff/species token lengths differ")
        site_dofs, gen_a, gen_b, orbit_rot, orbit_trans = [], [], [], [], []
        expanded_labels, parent_site, orbit_local = [], [], []
        for site_id, (wp_id, label) in enumerate(zip(wps, labels)):
            wp = group[int(wp_id)]
            dof = int(wp.get_dof())
            a, b = self._affine_map(lambda u, wp=wp: wp.get_position_from_free_xyzs(u), dof)
            rotations = np.asarray([op.rotation_matrix for op in wp.ops], float)
            translations = np.asarray([op.translation_vector for op in wp.ops], float)
            site_dofs.append(dof)
            gen_a.append(torch.as_tensor(a, dtype=torch.float32, device=self.device))
            gen_b.append(torch.as_tensor(b, dtype=torch.float32, device=self.device))
            orbit_rot.append(torch.as_tensor(rotations, dtype=torch.float32, device=self.device))
            orbit_trans.append(torch.as_tensor(translations, dtype=torch.float32, device=self.device))
            for local_id in range(len(rotations)):
                expanded_labels.append(label)
                parent_site.append(site_id)
                orbit_local.append(local_id)
        out = {
            "spg": int(spg),
            "group": group,
            "lattice_type": str(group.lattice_type).lower(),
            "spec": self._lattice_spec(group.lattice_type),
            "wps": tuple(wps),
            "site_labels": tuple(labels),
            "site_dofs": tuple(site_dofs),
            "gen_a": gen_a,
            "gen_b": gen_b,
            "orbit_rot": orbit_rot,
            "orbit_trans": orbit_trans,
            "expanded_labels": tuple(expanded_labels),
            "expanded_label_ids": torch.as_tensor(
                [self.model.label_to_id[x] for x in expanded_labels], dtype=torch.long, device=self.device
            ),
            "parent_site": tuple(parent_site),
            "orbit_local": tuple(orbit_local),
            "nblocks": len(expanded_labels),
        }
        self._cache[key] = out
        return out

    def _lattice(self, template: dict, raw: torch.Tensor):
        b = len(raw)
        lengths = torch.nn.functional.softplus(raw) + 1.2
        lt = template["lattice_type"]
        if lt == "cubic":
            a = lengths[:, 0]
            abc = torch.stack([a, a, a], 1)
            angles = torch.full((b, 3), math.pi / 2, device=self.device)
        elif lt == "tetragonal":
            a, c = lengths[:, 0], lengths[:, 1]
            abc = torch.stack([a, a, c], 1)
            angles = torch.full((b, 3), math.pi / 2, device=self.device)
        elif lt in {"hexagonal", "trigonal"}:
            a, c = lengths[:, 0], lengths[:, 1]
            abc = torch.stack([a, a, c], 1)
            angles = torch.tensor([math.pi / 2, math.pi / 2, 2 * math.pi / 3], device=self.device).repeat(b, 1)
        elif lt == "orthorhombic":
            abc = lengths[:, :3]
            angles = torch.full((b, 3), math.pi / 2, device=self.device)
        elif lt == "monoclinic":
            abc = lengths[:, :3]
            beta = math.pi / 3 + torch.sigmoid(raw[:, 3]) * math.pi / 3
            angles = torch.stack([torch.full_like(beta, math.pi / 2), beta, torch.full_like(beta, math.pi / 2)], 1)
        else:
            abc = lengths[:, :3]
            angles = math.pi / 4 + torch.sigmoid(raw[:, 3:6]) * math.pi / 2
        a, bb, c = abc.unbind(1)
        alpha, beta, gamma = angles.unbind(1)
        ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
        sg = torch.sin(gamma).clamp_min(1.0e-4)
        y3 = c * (ca - cb * cg) / sg
        z2_raw = c * c - (c * cb) ** 2 - y3 * y3
        z = torch.sqrt(z2_raw.clamp_min(1.0e-6))
        zero = torch.zeros_like(a)
        cell = torch.stack(
            [
                torch.stack([a, zero, zero], 1),
                torch.stack([bb * cg, bb * sg, zero], 1),
                torch.stack([c * cb, y3, z], 1),
            ],
            1,
        )
        return abc, angles, cell, z2_raw

    def _expand(self, template: dict, coord_raw: torch.Tensor, orient_raw: torch.Tensor,
                scale_raw: torch.Tensor, cell: torch.Tensor) -> dict:
        b = len(coord_raw)
        inv_cell = torch.linalg.inv(cell)
        centers, vectors, masks = [], [], []
        physical_frac, physical_symbols, physical_owner = [], [], []
        free = []
        cursor = 0
        for site_id, (dof, a, offset, rotations, translations, label) in enumerate(
            zip(
                template["site_dofs"], template["gen_a"], template["gen_b"],
                template["orbit_rot"], template["orbit_trans"], template["site_labels"]
            )
        ):
            u = torch.sigmoid(coord_raw[:, cursor:cursor + dof])
            cursor += dof
            free.append(u)
            generator = (u @ a.T + offset) % 1.0
            orbit = (torch.einsum("oij,bj->boi", rotations, generator) + translations[None]) % 1.0
            centers.append(orbit)
            qmat = _quat_matrix(orient_raw[:, site_id])
            scale = 0.75 + 0.5 * torch.sigmoid(scale_raw[:, site_id])
            canonical = torch.as_tensor(self.model.template_vectors[label], dtype=torch.float32, device=self.device)
            local = torch.einsum("pi,bij->bpj", canonical, qmat.transpose(-1, -2)) * scale[:, None, None]
            # Convert each fractional symmetry rotation to its Cartesian action.
            cart_ops = torch.einsum(
                "bij,ojk,bkl->boil", inv_cell, rotations.transpose(-1, -2), cell
            )
            orbit_vectors = torch.einsum("bpj,bojk->bopk", local, cart_ops)
            padded = torch.zeros((b, rotations.shape[0], max_cn, 3), device=self.device)
            mask = torch.zeros((rotations.shape[0], max_cn), dtype=torch.bool, device=self.device)
            if len(canonical):
                padded[:, :, :len(canonical)] = orbit_vectors
                mask[:, :len(canonical)] = True
            vectors.append(padded)
            masks.append(mask)

            symbols, block_pos = self.model.block_atoms[label]
            block = torch.as_tensor(block_pos - self.model.block_center[label], dtype=torch.float32, device=self.device)
            block_local = torch.einsum("pi,bij->bpj", block, qmat.transpose(-1, -2))
            block_orbit_cart = torch.einsum("bpj,bojk->bopk", block_local, cart_ops)
            block_orbit_frac = torch.einsum("bopj,bjk->bopk", block_orbit_cart, inv_cell)
            atoms = (orbit[:, :, None, :] + block_orbit_frac) % 1.0
            physical_frac.append(atoms.reshape(b, -1, 3))
            for orbit_id in range(rotations.shape[0]):
                for symbol in symbols:
                    physical_symbols.append(symbol)
                    physical_owner.append(len(sum(centers[:-1], [])) if False else None)
        center_frac = torch.cat(centers, 1)
        template_vectors = torch.cat(vectors, 1)
        port_mask = torch.cat(masks, 0)
        return {
            "center_frac": center_frac,
            "template_vectors_cart": template_vectors,
            "port_mask": port_mask,
            "physical_frac": torch.cat(physical_frac, 1),
            "physical_symbols": tuple(physical_symbols),
            "free": free,
        }

    def _center_geometry(self, frac: torch.Tensor, cell: torch.Tensor):
        delta = frac[:, None, :, None, :] + self.shift_t[None, None, None, :, :] - frac[:, :, None, None, :]
        vec = torch.einsum("bijnk,bkl->bijnl", delta, cell)
        dist = torch.linalg.norm(vec, dim=-1).clamp_min(1.0e-7)
        n = frac.shape[1]
        eye = torch.eye(n, dtype=torch.bool, device=self.device)[None, :, :, None]
        zero = (torch.arange(27, device=self.device) == ZERO_SHIFT)[None, None, None, :]
        return vec, dist.masked_fill(eye & zero, 1.0e6)

    def _raw_geometry(self, template: dict, raw: torch.Tensor):
        nlat = len(template["spec"])
        ncoord = sum(template["site_dofs"])
        nsite = len(template["site_labels"])
        abc, angles, cell, z2_raw = self._lattice(template, raw[:, :nlat])
        coord_raw = raw[:, nlat:nlat + ncoord]
        orient_raw = raw[:, nlat + ncoord:nlat + ncoord + 4 * nsite].reshape(len(raw), nsite, 4)
        scale_raw = raw[:, nlat + ncoord + 4 * nsite:nlat + ncoord + 5 * nsite]
        expanded = self._expand(template, coord_raw, orient_raw, scale_raw, cell)
        return abc, angles, cell, z2_raw, expanded

    def _assignment_cost(self, template: dict, raw: torch.Tensor):
        """Build endpoint-to-centre costs for a temporary discrete reconciliation.

        Every source X port is assigned to one distinct periodic candidate centre
        within that source site.  The assignment is refreshed during optimization,
        so neighbour identities can swap without becoming a permanent graph.
        """
        abc, angles, cell, z2_raw, expanded = self._raw_geometry(template, raw)
        centers = expanded["center_frac"]
        vectors = expanded["template_vectors_cart"]
        port_mask = expanded["port_mask"]
        batch, ncentre, nport = vectors.shape[:3]
        center_vec, center_dist = self._center_geometry(centers, cell)
        disp = center_vec.reshape(batch, ncentre, ncentre * 27, 3)
        distances = center_dist.reshape(batch, ncentre, ncentre * 27)
        candidate_atom = torch.arange(ncentre, device=self.device).repeat_interleave(27)
        candidate_shift = torch.arange(27, device=self.device).repeat(ncentre)
        candidate_vectors = vectors[:, candidate_atom]
        candidate_port_mask = port_mask[candidate_atom]
        forward = torch.linalg.norm(disp[:, :, None] - vectors[:, :, :, None], dim=-1)
        reciprocal_all = torch.linalg.norm(
            candidate_vectors[:, None] + disp[:, :, :, None], dim=-1
        ).masked_fill(~candidate_port_mask[None, None], 1.0e4)
        reciprocal = reciprocal_all.amin(-1)
        compatibility = torch.zeros((ncentre, ncentre * 27), dtype=torch.bool, device=self.device)
        mu = torch.zeros((ncentre, ncentre * 27), dtype=torch.float32, device=self.device)
        sigma = torch.ones_like(mu)
        shell_cutoff = torch.zeros_like(mu)
        for i, label_i in enumerate(template["expanded_labels"]):
            allowed = set(self.model.allowed[label_i])
            for m in range(ncentre * 27):
                j = int(candidate_atom[m])
                label_j = template["expanded_labels"][j]
                ok = label_j in allowed and not (i == j and int(candidate_shift[m]) == ZERO_SHIFT)
                compatibility[i, m] = ok
                if ok:
                    ch = self.model.pair(label_i, label_j)
                    mu[i, m] = ch.mu
                    sigma[i, m] = max(ch.sigma, 0.02)
                    shell_cutoff[i, m] = ch.first_shell_cutoff
        radial = torch.abs(distances - mu[None]) / sigma[None]
        cost = (forward / self.port_width).pow(2) + (reciprocal[:, :, None] / self.port_width).pow(2)
        cost = cost + self.radial_weight * radial[:, :, None].pow(2)
        valid = port_mask[None, :, :, None] & compatibility[None, :, None]
        cost = cost.masked_fill(~valid, 1.0e6)
        return cost, (abc, angles, cell, z2_raw, expanded, disp, distances,
                      candidate_atom, candidate_shift, compatibility, shell_cutoff)

    def _reconciliation_plan(self, template: dict) -> dict:
        """Infer a generic coincidence primitive from active learned templates.

        The inference uses only model semantics (labels, allowed partners,
        coordination numbers, and final formulas), never element-specific flags.
        A chemically homogeneous component is reconciled by pair matching.  A
        heterogeneous reciprocal component is reconciled as a shared-site system:
        the highest-CN species own the source ports and the lowest-CN species are
        exact-capacity child centres.
        """
        labels = tuple(template["expanded_labels"])
        active = tuple(dict.fromkeys(labels))
        formulas = {self.model.final_formula[x] for x in active}
        if len(formulas) == 1:
            total_ports = sum(self.model.template_cn[x] for x in labels)
            if total_ports % 2:
                raise ValueError(
                    f"Pair coincidence requires an even number of ports; got {total_ports}"
                )
            return {"mode": "pair", "labels": active}

        cns = {x: int(self.model.template_cn[x]) for x in active}
        max_cn, min_cn = max(cns.values()), min(cns.values())
        parents = tuple(x for x in active if cns[x] == max_cn)
        children = tuple(x for x in active if cns[x] == min_cn)
        if not parents or not children or set(parents) & set(children):
            raise ValueError(f"Cannot infer coincidence roles for labels {active}")
        for p in parents:
            if not any(c in self.model.allowed[p] and p in self.model.allowed[c] for c in children):
                raise ValueError(f"No reciprocal parent/child compatibility for {p}")
        parent_ports = sum(cns[x] for x in labels if x in parents)
        child_capacity = sum(cns[x] for x in labels if x in children)
        if parent_ports != child_capacity:
            raise ValueError(
                "Shared-site incidence mismatch: "
                f"parent_ports={parent_ports}, child_capacity={child_capacity}"
            )
        return {
            "mode": "shared_site",
            "parents": parents,
            "children": children,
            "parent_ports": parent_ports,
            "child_capacity": child_capacity,
        }

    def _make_assignment(self, template: dict, raw: torch.Tensor) -> torch.Tensor:
        """Construct one exact global coincidence topology per batch row.

        Returned entries retain the old directed endpoint representation
        ``candidate = neighbour_index * 27 + periodic_shift`` so the existing
        differentiable geometry loss and strict audit remain reusable.  Unlike
        the old local Hungarian assignments, every topology is globally
        reciprocal and obeys exact edge/site capacities.
        """
        with torch.no_grad():
            cost, state = self._assignment_cost(template, raw)
            expanded = state[4]
            port_mask = expanded["port_mask"]
            candidate_atom, candidate_shift = state[7], state[8]
            bsz, ncentre, nport, _ = cost.shape
            assignment = np.full((bsz, ncentre, nport), -1, dtype=np.int64)
            c = np.nan_to_num(
                cost.detach().cpu().numpy(), nan=1.0e6, posinf=1.0e6, neginf=1.0e6
            )
            pm = port_mask.detach().cpu().numpy()
            labels = tuple(template["expanded_labels"])
            plan = self._reconciliation_plan(template)
            reverse_shift = {
                s: int(np.flatnonzero(np.all(SHIFTS == -SHIFTS[s], axis=1))[0])
                for s in range(27)
            }

            if plan["mode"] == "pair":
                ports = [(i, p) for i in range(ncentre) for p in np.flatnonzero(pm[i])]
                for b in range(bsz):
                    graph = nx.Graph()
                    graph.add_nodes_from(range(len(ports)))
                    directed_best = {}
                    for a in range(len(ports)):
                        i, p = ports[a]
                        for d in range(a + 1, len(ports)):
                            j, q = ports[d]
                            if i == j or labels[j] not in self.model.allowed[labels[i]]                                     or labels[i] not in self.model.allowed[labels[j]]:
                                continue
                            cand_ij = slice(j * 27, (j + 1) * 27)
                            vals_ij = c[b, i, p, cand_ij]
                            s = int(np.argmin(vals_ij))
                            m_ij = j * 27 + s
                            rs = reverse_shift[s]
                            m_ji = i * 27 + rs
                            value = float(vals_ij[s] + c[b, j, q, m_ji])
                            if not np.isfinite(value) or value >= 2.0e5:
                                continue
                            graph.add_edge(a, d, weight=value)
                            directed_best[(a, d)] = (m_ij, m_ji)
                    matching = nx.algorithms.matching.min_weight_matching(
                        graph, weight="weight"
                    )
                    if len(matching) * 2 != len(ports):
                        continue
                    for a, d in matching:
                        if a > d:
                            a, d = d, a
                        i, p = ports[a]
                        j, q = ports[d]
                        m_ij, m_ji = directed_best[(a, d)]
                        assignment[b, i, p] = m_ij
                        assignment[b, j, q] = m_ji
            else:
                parent_labels = set(plan["parents"])
                child_labels = set(plan["children"])
                parent_ports = [
                    (i, p) for i in range(ncentre) if labels[i] in parent_labels
                    for p in np.flatnonzero(pm[i])
                ]
                children = [i for i in range(ncentre) if labels[i] in child_labels]
                slots = [j for j in children for _ in range(self.model.template_cn[labels[j]])]
                if len(parent_ports) != len(slots):
                    return torch.as_tensor(assignment, dtype=torch.long, device=self.device)
                for b in range(bsz):
                    matrix = np.full((len(parent_ports), len(slots)), 1.0e6, float)
                    best_shift = np.full_like(matrix, -1, dtype=int)
                    for r, (i, p) in enumerate(parent_ports):
                        for col, j in enumerate(slots):
                            if labels[j] not in self.model.allowed[labels[i]]:
                                continue
                            vals = c[b, i, p, j * 27:(j + 1) * 27]
                            s = int(np.argmin(vals))
                            matrix[r, col] = float(vals[s])
                            best_shift[r, col] = s
                    if np.any(~np.isfinite(matrix).any(axis=1)):
                        continue
                    rows, cols = linear_sum_assignment(matrix)
                    if len(rows) != len(parent_ports) or np.any(matrix[rows, cols] >= 1.0e5):
                        continue
                    child_members = defaultdict(list)
                    for r, col in zip(rows, cols):
                        i, p = parent_ports[r]
                        j = slots[col]
                        s = int(best_shift[r, col])
                        assignment[b, i, p] = j * 27 + s
                        child_members[j].append((i, reverse_shift[s]))

                    # Derive the reverse child-port topology from the same groups.
                    # This preserves exact reciprocity instead of allowing O->Ti
                    # assignments to be solved independently from Ti->O groups.
                    complete = True
                    for j in children:
                        members = child_members.get(j, [])
                        ports_j = np.flatnonzero(pm[j])
                        if len(members) != len(ports_j):
                            complete = False
                            break
                        rev = np.full((len(ports_j), len(members)), 1.0e6, float)
                        candidates = np.zeros((len(members),), dtype=int)
                        for m, (i, rs) in enumerate(members):
                            candidates[m] = i * 27 + rs
                            rev[:, m] = c[b, j, ports_j, candidates[m]]
                        rr, cc = linear_sum_assignment(rev)
                        if len(rr) != len(ports_j) or np.any(rev[rr, cc] >= 1.0e5):
                            complete = False
                            break
                        for rj, cm in zip(rr, cc):
                            assignment[b, j, ports_j[rj]] = candidates[cm]
                    if not complete:
                        assignment[b] = -1

            return torch.as_tensor(assignment, dtype=torch.long, device=self.device)

    def _loss(self, template: dict, raw: torch.Tensor, assignment: torch.Tensor):
        """Optimize geometry under a temporary exact coincidence topology."""
        cost_all, state = self._assignment_cost(template, raw)
        (abc, angles, cell, z2_raw, expanded, disp, distances, candidate_atom,
         candidate_shift, compatibility, shell_cutoff) = state
        port_mask = expanded["port_mask"]
        bsz, ncentre, nport = assignment.shape
        safe = assignment.clamp_min(0)
        chosen_cost = torch.gather(cost_all, 3, safe[..., None]).squeeze(-1)
        active = port_mask[None] & (assignment >= 0)
        expected_ports = port_mask.sum().clamp_min(1)
        topology_complete = active.sum((1, 2)) == expected_ports
        port_count = active.sum((1, 2)).clamp_min(1)
        endpoint_loss = (chosen_cost * active).sum((1, 2)) / port_count

        # Penalize multiple ports on the same source selecting the same physical
        # periodic candidate. Hungarian assignment should make this exactly zero;
        # keeping the term provides a diagnostic and guards malformed assignments.
        duplicate = torch.zeros(bsz, device=self.device)
        for p in range(nport):
            for q in range(p + 1, nport):
                both = active[:, :, p] & active[:, :, q]
                duplicate += ((assignment[:, :, p] == assignment[:, :, q]) & both).float().mean(1)

        # Contacts inside the learned first-shell boundary must be represented
        # by an assigned port. This is a temporary directed exemption; the strict
        # audit later requires exact reciprocal selection.
        selected_count = torch.zeros(
            (bsz, ncentre, distances.shape[-1]), dtype=torch.int16, device=self.device
        )
        selected_count.scatter_add_(
            2, safe.reshape(bsz, ncentre, -1),
            active.reshape(bsz, ncentre, -1).to(torch.int16),
        )
        selected = selected_count > 0
        nonbonded_mask = compatibility[None] & ~selected
        penetration = torch.relu(
            (shell_cutoff[None] + self.nonbonded_margin - distances) / self.nonbonded_width
        )
        nonbonded_terms = penetration.pow(2) * nonbonded_mask
        nonbonded_count = nonbonded_mask.sum((1, 2)).clamp_min(1)
        nonbonded_loss = nonbonded_terms.sum((1, 2)) / nonbonded_count
        minimum_unselected = distances.masked_fill(~nonbonded_mask, 1.0e6).amin((1, 2))

        physical = expanded["physical_frac"]
        _, physical_dist = self._center_geometry(physical, cell)
        minimum_physical = physical_dist.amin((1, 2, 3))
        overlap = torch.relu((self.minimum_distance - minimum_physical) / 0.05).pow(2)
        catastrophic_collapse = torch.relu((0.45 - minimum_physical) / 0.02).pow(2)

        # Model-normalized local angular loss.  This was previously absent even
        # though --angular-weight existed, so optimization could satisfy CN and
        # bond lengths while producing severely sheared Xn polyhedra.
        flat_disp = disp.reshape(bsz, ncentre, -1, 3)
        angular_loss = torch.zeros(bsz, device=self.device)
        angular_site_z = torch.zeros(bsz, device=self.device)
        angular_vector_z = torch.zeros(bsz, device=self.device)
        angular_centres = 0
        for i, label_i in enumerate(template["expanded_labels"]):
            cn = int(self.model.template_cn[label_i])
            if cn < 2:
                continue
            port_ids = torch.nonzero(port_mask[i], as_tuple=False).flatten()
            if len(port_ids) != cn:
                continue
            idx = assignment[:, i, port_ids]
            valid_i = (idx >= 0).all(1)
            safe_i = idx.clamp_min(0)
            vec = torch.gather(flat_disp[:, i], 1, safe_i[..., None].expand(-1, -1, 3))
            unit = vec / torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1.0e-6)
            gram = torch.matmul(unit, unit.transpose(1, 2)).clamp(-1.0, 1.0)
            tri = torch.triu_indices(cn, cn, offset=1, device=self.device)
            observed = torch.rad2deg(torch.acos(gram[:, tri[0], tri[1]]))
            observed = torch.sort(observed, dim=1).values
            target = torch.as_tensor(self.model.template_angles[label_i], dtype=observed.dtype, device=self.device)
            sigma_a = torch.as_tensor(self.model.template_angle_sigma[label_i], dtype=observed.dtype, device=self.device)
            z = torch.abs(observed - target[None]) / sigma_a[None].clamp_min(1.0e-3)
            z = torch.where(valid_i[:, None], z, torch.full_like(z, 10.0))
            angular_loss += z.pow(2).mean(1)
            angular_site_z = torch.maximum(angular_site_z, z.mean(1))
            angular_vector_z = torch.maximum(angular_vector_z, z.amax(1))
            angular_centres += 1
        if angular_centres:
            angular_loss = angular_loss / float(angular_centres)

        aspect = abc.amax(1) / abc.amin(1).clamp_min(1.0e-4)
        shape = torch.relu(aspect - 6.0).pow(2)
        metric = torch.relu(1.0e-5 - z2_raw / abc[:, 2].square().clamp_min(1.0e-8)).pow(2) * 1.0e6
        incomplete_penalty = (~topology_complete).float() * 1.0e4
        total = (endpoint_loss + incomplete_penalty + self.uniqueness_weight * duplicate
                 + self.nonbonded_weight * nonbonded_loss
                 + self.angular_weight * angular_loss
                 + self.overlap_weight * (overlap + catastrophic_collapse)
                 + 0.05 * shape + metric)
        total = torch.nan_to_num(total, nan=1.0e9, posinf=1.0e9, neginf=1.0e9)
        detail = {
            "topology_complete": topology_complete,
            "assigned_endpoint_loss": endpoint_loss,
            "assignment_duplicate_loss": duplicate,
            "minimum_physical_distance_A": minimum_physical,
            "minimum_unselected_distance_A": minimum_unselected,
            "unselected_first_shell_loss": nonbonded_loss,
            "overlap_loss": overlap,
            "catastrophic_collapse_loss": catastrophic_collapse,
            "model_angular_loss": angular_loss,
            "model_angular_site_z": angular_site_z,
            "model_angular_vector_z": angular_vector_z,
            "aspect_ratio": aspect,
            "metric_valid": z2_raw > 0,
        }
        return total, detail, (abc, angles, cell, expanded)

    def _initial_raw(self, template: dict) -> torch.Tensor:
        nlat = len(template["spec"])
        ncoord = sum(template["site_dofs"])
        nsite = len(template["site_labels"])
        pair_mus = [x.mu for x in self.model.channels.values()]
        mean_bond = float(np.mean(pair_mus))
        base = mean_bond * max(template["nblocks"], 1) ** (1 / 3) * 1.65
        raw = torch.randn((self.starts, nlat + ncoord + 5 * nsite), device=self.device)
        raw[:, :nlat] *= 0.4
        raw[:, :nlat] += math.log(math.expm1(max(base - 1.2, 0.5)))
        orient_start = nlat + ncoord
        raw[:, orient_start:orient_start + 4 * nsite] = torch.randn(
            (self.starts, 4 * nsite), device=self.device
        )
        return raw

    def _optimize(self, template: dict, raw: torch.Tensor, steps: int, phase: str,
                  lr: float, heartbeat=None):
        raw = raw.detach().clone()
        best = raw.clone()
        best_loss = torch.full((len(raw),), float("inf"), device=self.device)
        completed = 0
        last_heartbeat = time.perf_counter()
        while completed < int(steps):
            assignment = self._make_assignment(template, raw)
            block = min(self.assignment_refresh, int(steps) - completed)
            variable = raw.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([variable], lr=float(lr))
            for local_step in range(block):
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = self._loss(template, variable, assignment)
                finite_loss = torch.isfinite(loss)
                if not bool(finite_loss.any()):
                    break
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_([variable], 10.0)
                optimizer.step()
                with torch.no_grad():
                    bad_rows = ~torch.isfinite(variable).all(1)
                    if bool(bad_rows.any()):
                        variable[bad_rows] = best[bad_rows]
                    improved = torch.isfinite(loss) & (loss < best_loss)
                    best_loss = torch.where(improved, loss, best_loss)
                    best[improved] = variable.detach()[improved]
                completed += 1
                now = time.perf_counter()
                if heartbeat is not None and (completed == 1 or now - last_heartbeat >= 10.0 or completed == int(steps)):
                    heartbeat(phase, completed, int(steps), float(best_loss.min().detach().cpu()))
                    last_heartbeat = now
            raw = variable.detach()
        return best

    def _strict(self, template: dict, cell: np.ndarray, expanded: dict) -> dict:
        frac = expanded["center_frac"].detach().cpu().numpy().reshape(-1, 3)
        # Strict validation is performed from the realized final coordinates and
        # learned neighbour templates.  It must not require construction-port
        # vectors: derived shared-site atoms (for example O constructed from
        # three Ti-owned ports) do not own an independent port tensor here.
        labels = list(template["expanded_labels"])
        n = len(frac)
        delta = frac[None, :, None, :] - frac[:, None, None, :] + SHIFTS[None, None, :, :]
        cart = np.einsum("...i,ij->...j", delta, cell)
        dist = np.linalg.norm(cart, axis=-1)
        for i in range(n):
            dist[i, i, ZERO_SHIFT] = np.inf
        exact = []
        radial_errors = []
        angle_site_errors = []
        angle_vector_errors = []
        angle_site_z_errors = []
        angle_vector_z_errors = []
        reciprocal_hits = reciprocal_total = 0
        selected_neighbors = []
        for i, label_i in enumerate(labels):
            candidates = []
            for j, label_j in enumerate(labels):
                if label_j not in self.model.allowed[label_i]:
                    continue
                channel = self.model.pair(label_i, label_j)
                for shift in range(27):
                    if i == j and shift == ZERO_SHIFT:
                        continue
                    d = float(dist[i, j, shift])
                    if channel.sampling_min - 1.0e-8 <= d <= channel.sampling_max + 1.0e-8:
                        candidates.append((d, j, shift, channel))
            cn = self.model.template_cn[label_i]
            exact.append(len(candidates) == cn)
            candidates.sort(key=lambda x: x[0])
            chosen = candidates if len(candidates) == cn else sorted(
                [
                    (float(dist[i, j, s]), j, s, self.model.pair(label_i, labels[j]))
                    for j in range(n) if labels[j] in self.model.allowed[label_i]
                    for s in range(27) if not (i == j and s == ZERO_SHIFT)
                ], key=lambda x: x[0]
            )[:cn]
            selected_neighbors.append([(j, s) for _, j, s, _ in chosen])
            radial_errors.extend(abs(d - channel.mu) for d, _, _, channel in chosen)
            observed_vectors = np.asarray([cart[i, j, s] for _, j, s, _ in chosen], float)
            if len(observed_vectors) == cn:
                observed_angles = _angles_np(observed_vectors)
                target_angles = self.model.template_angles[label_i]
                if len(observed_angles) == len(target_angles):
                    errors = np.abs(observed_angles - target_angles)
                    sigma_a = self.model.template_angle_sigma[label_i]
                    z_errors = errors / np.maximum(sigma_a, 1.0e-3)
                    angle_site_errors.append(float(np.mean(errors)))
                    angle_vector_errors.extend(errors.tolist())
                    angle_site_z_errors.append(float(np.mean(z_errors)))
                    angle_vector_z_errors.extend(z_errors.tolist())
        reverse_shift = {
            i: int(np.flatnonzero(np.all(SHIFTS == -SHIFTS[i], axis=1))[0]) for i in range(27)
        }
        for i, entries in enumerate(selected_neighbors):
            for j, shift in entries:
                reciprocal_total += 1
                reciprocal_hits += int((i, reverse_shift[shift]) in selected_neighbors[j])
        physical = expanded["physical_frac"].detach().cpu().numpy().reshape(-1, 3)
        pdelta = physical[None, :, None, :] - physical[:, None, None, :] + SHIFTS[None, None, :, :]
        pdist = np.linalg.norm(np.einsum("...i,ij->...j", pdelta, cell), axis=-1)
        for i in range(len(physical)):
            pdist[i, i, ZERO_SHIFT] = np.inf
        min_physical = float(np.min(pdist)) if pdist.size else float("inf")

        selected_sets = [set(entries) for entries in selected_neighbors]
        minimum_unselected = float("inf")
        unselected_short_contacts = 0
        unselected_shell_contacts = 0
        for i, label_i in enumerate(labels):
            for j, label_j in enumerate(labels):
                if label_j not in self.model.allowed[label_i]:
                    continue
                channel = self.model.pair(label_i, label_j)
                for shift in range(27):
                    if i == j and shift == ZERO_SHIFT:
                        continue
                    if (j, shift) in selected_sets[i]:
                        continue
                    d = float(dist[i, j, shift])
                    minimum_unselected = min(minimum_unselected, d)
                    if d < channel.sampling_min - 1.0e-8:
                        unselected_short_contacts += 1
                    if d < channel.first_shell_cutoff - 1.0e-8:
                        unselected_shell_contacts += 1
        full_contact_shell_valid = unselected_shell_contacts == 0

        exact_fraction = float(np.mean(exact)) if exact else 1.0
        radial_max = float(np.max(radial_errors)) if radial_errors else 0.0
        angular_site_max = float(np.max(angle_site_errors)) if angle_site_errors else 0.0
        angular_vector_max = float(np.max(angle_vector_errors)) if angle_vector_errors else 0.0
        angular_site_z_max = float(np.max(angle_site_z_errors)) if angle_site_z_errors else 0.0
        angular_vector_z_max = float(np.max(angle_vector_z_errors)) if angle_vector_z_errors else 0.0
        reciprocal_fraction = float(reciprocal_hits / reciprocal_total) if reciprocal_total else 1.0
        valid = bool(
            exact_fraction >= 1.0 - 1.0e-12
            and reciprocal_fraction >= 1.0 - 1.0e-12
            and min_physical >= self.minimum_distance
            and full_contact_shell_valid
            and angular_site_z_max <= self.angular_site_z_max
            and angular_vector_z_max <= self.angular_vector_z_max
        )
        return {
            "exact_target_cn_fraction": exact_fraction,
            "reciprocal_neighbor_fraction": reciprocal_fraction,
            "local_radial_mae_A": float(np.mean(radial_errors)) if radial_errors else 0.0,
            "local_radial_vector_max_A": radial_max,
            "local_angular_site_max_deg": angular_site_max,
            "local_angular_vector_max_deg": angular_vector_max,
            "local_angular_site_z_max": angular_site_z_max,
            "local_angular_vector_z_max": angular_vector_z_max,
            "minimum_physical_distance_A": min_physical,
            "minimum_unselected_distance_A": minimum_unselected,
            "unselected_short_contact_count": int(unselected_short_contacts),
            "unselected_first_shell_contact_count": int(unselected_shell_contacts),
            "full_contact_shell_valid": bool(full_contact_shell_valid),
            "strict_valid": valid,
        }

    # ------------------------------------------------------------------
    # Construction-framework prebuilding
    # ------------------------------------------------------------------
    def _shared_prefix_size(self, template: dict) -> int:
        return len(template["spec"]) + sum(template["site_dofs"])

    def _framework_geometry(self, template: dict, prefix: torch.Tensor):
        bsz = len(prefix); nlat = len(template["spec"])
        abc, angles, cell, z2_raw = self._lattice(template, prefix[:, :nlat])
        coord_raw = prefix[:, nlat:]
        centres=[]; cursor=0
        for dof, a, offset, rotations, translations in zip(
            template["site_dofs"], template["gen_a"], template["gen_b"],
            template["orbit_rot"], template["orbit_trans"]):
            u=torch.sigmoid(coord_raw[:, cursor:cursor+dof]); cursor += dof
            generator=(u @ a.T + offset) % 1.0
            orbit=(torch.einsum("oij,bj->boi", rotations, generator)+translations[None]) % 1.0
            centres.append(orbit)
        return abc, angles, cell, z2_raw, torch.cat(centres, dim=1)

    def _framework_loss(self, template: dict, prefix: torch.Tensor):
        abc, angles, cell, z2_raw, centres = self._framework_geometry(template, prefix)
        labels=tuple(template["expanded_labels"]); bsz,ncentre,_=centres.shape
        # Keep every framework objective attached to the optimized prefix even
        # when all active penalties evaluate to exactly zero.  A plain
        # torch.zeros accumulator can otherwise produce a graphless scalar for
        # special/high-symmetry branches and make backward() fail.
        grad_anchor = prefix.sum(dim=1) * 0.0
        total = grad_anchor.clone()
        radial = grad_anchor.clone()
        angular = grad_anchor.clone()
        connectivity = grad_anchor.clone()
        site_score = grad_anchor[:, None].expand(-1, ncentre).clone()
        min_distance=torch.full((bsz,),1.0e6,device=self.device) + grad_anchor
        used=0
        # Shared-site construction normally has one parent label; support mixed
        # parent labels by evaluating each centre with its own learned model.
        delta=centres[:,None,:,None,:]-centres[:,:,None,None,:]+self.shift_t[None,None,None,:,:]
        cart=torch.einsum("bijnk,bkl->bijnl",delta,cell)
        dist=torch.linalg.norm(cart,dim=-1)
        eye=torch.eye(ncentre,dtype=torch.bool,device=self.device)[None,:,:,None]
        zero=torch.zeros(27,dtype=torch.bool,device=self.device); zero[ZERO_SHIFT]=True
        dist=dist.masked_fill(eye & zero[None,None,None,:],1.0e6)
        min_distance=dist.amin((1,2,3))
        for i,label in enumerate(labels):
            fm=self.model.framework_models.get(label)
            if not fm: continue
            k=min(int(fm["neighbor_count"]), max(1,ncentre*27-1))
            flat=dist[:,i].reshape(bsz,-1)
            vals,idx=torch.topk(flat,k,largest=False,dim=1)
            target=torch.as_tensor(fm["radial_mean_A"],dtype=vals.dtype,device=self.device)[:k]
            sigma=torch.as_tensor(fm["radial_sigma_A"],dtype=vals.dtype,device=self.device)[:k].clamp_min(0.03)
            rz=((vals-target[None])/sigma[None])
            radial += rz.pow(2).mean(1)
            local=rz.abs().mean(1)
            angular_groups=fm.get("angular_shell_pair_groups", [])
            if k>=2 and angular_groups:
                j=(idx//27); s=(idx%27)
                vec=cart[:,i][torch.arange(bsz,device=self.device)[:,None],j,s]
                unit=vec/torch.linalg.norm(vec,dim=-1,keepdim=True).clamp_min(1e-6)
                dot=torch.matmul(unit,unit.transpose(1,2)).clamp(-1,1)
                angular_sum=torch.zeros_like(total)
                angular_abs_sum=torch.zeros_like(total)
                angular_count=0
                for group in angular_groups:
                    rank_pairs=group.get("neighbor_rank_pairs", [])
                    if not rank_pairs:
                        continue
                    left=torch.as_tensor([p[0] for p in rank_pairs],dtype=torch.long,device=self.device)
                    right=torch.as_tensor([p[1] for p in rank_pairs],dtype=torch.long,device=self.device)
                    if int(left.max())>=k or int(right.max())>=k:
                        raise ValueError(f"Framework angular rank exceeds neighbor_count for {label}")
                    obs=torch.sort(torch.rad2deg(torch.acos(dot[:,left,right].clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))),dim=1).values
                    at=torch.as_tensor(group["angular_mean_deg"],dtype=obs.dtype,device=self.device)
                    asi=torch.as_tensor(group["angular_sigma_deg"],dtype=obs.dtype,device=self.device).clamp_min(2.0)
                    if len(at)!=obs.shape[1] or len(asi)!=obs.shape[1]:
                        raise ValueError(f"Malformed shell-pair angular model for {label}")
                    az=(obs-at[None])/asi[None]
                    angular_sum += az.pow(2).sum(1)
                    angular_abs_sum += az.abs().sum(1)
                    angular_count += obs.shape[1]
                if angular_count:
                    angular += angular_sum/float(angular_count)
                    local += angular_abs_sum/float(angular_count)
            site_score[:,i]=local; used += 1
            connectivity += torch.relu((vals[:,-1]-float(fm["connectivity_upper_A"]))/0.10).pow(2)
            lower=float(fm["radial_lower_bound_A"])
            total += 10.0*torch.relu((lower-min_distance)/0.05).pow(2)
        if used:
            radial/=used; angular/=used; connectivity/=used
        aspect=abc.amax(1)/abc.amin(1).clamp_min(1e-4)
        shape=torch.relu(aspect-6.0).pow(2)
        metric=torch.relu(1e-5-z2_raw/abc[:,2].square().clamp_min(1e-8)).pow(2)*1e6
        q90=torch.quantile(site_score,0.90,dim=1) if ncentre>1 else site_score[:,0]
        total += radial + angular + connectivity + 0.05*shape + metric
        total=torch.nan_to_num(total,nan=1e9,posinf=1e9,neginf=1e9)
        return total,{"framework_radial_loss":radial,"framework_angular_loss":angular,
            "framework_connectivity_loss":connectivity,"framework_score_q90":q90,
            "framework_min_distance_A":min_distance,"framework_aspect_ratio":aspect},(abc,angles,cell,centres)

    def _optimize_framework(self, template: dict, prefix: torch.Tensor, steps: int, heartbeat=None):
        variable=prefix.detach().clone().requires_grad_(True)
        best=variable.detach().clone()
        best_loss=torch.full((len(prefix),),float("inf"),device=self.device)
        stale=torch.zeros((len(prefix),),dtype=torch.long,device=self.device)
        active=torch.ones((len(prefix),),dtype=torch.bool,device=self.device)
        opt=torch.optim.Adam([variable],lr=self.lr)
        last=time.perf_counter()
        for step in range(int(steps)):
            if not bool(active.any()):
                break
            opt.zero_grad(set_to_none=True)
            loss,detail,_=self._framework_loss(template,variable)
            finite=torch.isfinite(loss)
            active &= finite
            if not bool(active.any()):
                break
            masked=torch.where(active,loss,torch.zeros_like(loss))
            objective = masked.sum() / active.sum().clamp_min(1)
            if not objective.requires_grad or objective.grad_fn is None:
                raise RuntimeError(
                    "Ti framework objective lost autograd connection: "
                    f"spg={template.get('spg')} wp={template.get('wps')} "
                    f"labels={template.get('expanded_labels')} "
                    f"prefix_requires_grad={variable.requires_grad} "
                    f"loss_requires_grad={loss.requires_grad}"
                )
            objective.backward()
            torch.nn.utils.clip_grad_norm_([variable],10.0)
            opt.step()
            with torch.no_grad():
                post,post_detail,_=self._framework_loss(template,variable)
                finite_post=torch.isfinite(post)&torch.isfinite(variable).all(1)
                improved=active&finite_post&(post<best_loss-1.0e-6)
                best_loss=torch.where(improved,post,best_loss)
                best[improved]=variable.detach()[improved]
                stale=torch.where(improved,torch.zeros_like(stale),stale+active.long())
                # A branch that already satisfies the learned q90 and lower-bound
                # gates may stop immediately; otherwise stop after no progress.
                qmax=max([float(self.model.framework_models.get(label,{}).get("score_q90_max",3.5))
                    for label in template["expanded_labels"]] or [3.5])
                lower=max([float(self.model.framework_models.get(label,{}).get("radial_lower_bound_A",0.0))
                    for label in template["expanded_labels"]] or [0.0])
                accepted_now=(post_detail["framework_score_q90"] <= qmax) & (
                    post_detail["framework_min_distance_A"] >= lower)
                active &= ~(accepted_now | (stale>=self.framework_patience) | ~finite_post)
                bad=~torch.isfinite(variable).all(1)
                if bool(bad.any()):
                    variable[bad]=best[bad]
            now=time.perf_counter()
            if heartbeat is not None and (step==0 or now-last>=10 or step+1==int(steps) or not bool(active.any())):
                heartbeat("ti_framework_prebuild",step+1,int(steps),float(best_loss.min().detach().cpu())); last=now
        return best

    # ------------------------------------------------------------------
    # Parent-only shared-site construction
    # ------------------------------------------------------------------
    def _shared_raw_geometry(self, template: dict, raw: torch.Tensor):
        """Expand parent construction centres and their attached Xn ports.

        ``full`` uses one pose/scale per independent Wyckoff orbit. ``centers``
        uses one pose/scale per expanded centre while preserving centre
        symmetry. ``off`` additionally gives every expanded centre a bounded
        independent fractional displacement.
        """
        bsz = len(raw)
        nlat = len(template["spec"])
        ncoord = sum(template["site_dofs"])
        nsite = len(template["site_labels"])
        nblock = int(template["nblocks"])
        npose = nsite if self.construction_symmetry == "full" else nblock
        cursor = 0
        abc, angles, cell, z2_raw = self._lattice(template, raw[:, cursor:cursor+nlat])
        cursor += nlat
        coord_raw = raw[:, cursor:cursor+ncoord]; cursor += ncoord
        orient = raw[:, cursor:cursor+4*npose].reshape(bsz, npose, 4); cursor += 4*npose
        scale_raw = raw[:, cursor:cursor+npose]; cursor += npose
        max_cn = max(self.model.template_cn.values(), default=0)
        distortion_raw = raw[:, cursor:cursor+3*npose*max_cn].reshape(bsz, npose, max_cn, 3); cursor += 3*npose*max_cn
        # Local port deformation is symmetry-coupled in ``full`` mode and
        # independent per expanded centre otherwise.  Remove the mean local
        # displacement so this term distorts the polyhedron rather than merely
        # translating it away from its construction centre.
        local_distortion = self.distortion_max * torch.tanh(distortion_raw)
        local_distortion = local_distortion - local_distortion.mean(dim=2, keepdim=True)
        displacement = None
        if self.construction_symmetry == "off":
            displacement = 0.18 * torch.tanh(raw[:, cursor:cursor+3*nblock].reshape(bsz, nblock, 3))

        inv_cell = torch.linalg.inv(cell)
        centers_list, vectors_list, labels, owner_site = [], [], [], []
        free = []
        coord_cursor = 0
        expanded_id = 0
        max_cn = max(self.model.template_cn.values(), default=0)
        for site_id, (dof, a, offset, rotations, translations, label) in enumerate(zip(
            template["site_dofs"], template["gen_a"], template["gen_b"],
            template["orbit_rot"], template["orbit_trans"], template["site_labels"]
        )):
            u = torch.sigmoid(coord_raw[:, coord_cursor:coord_cursor+dof]); coord_cursor += dof
            free.append(u)
            generator = (u @ a.T + offset) % 1.0
            orbit = (torch.einsum("oij,bj->boi", rotations, generator) + translations[None]) % 1.0
            norbit = rotations.shape[0]
            if displacement is not None:
                orbit = (orbit + displacement[:, expanded_id:expanded_id+norbit]) % 1.0
            centers_list.append(orbit)

            canonical = torch.as_tensor(self.model.template_vectors[label], dtype=torch.float32, device=self.device)
            cn = len(canonical)
            if self.construction_symmetry == "full":
                qmat = _quat_matrix(orient[:, site_id])
                scale = 0.75 + 0.5 * torch.sigmoid(scale_raw[:, site_id])
                local_shape = canonical[None, :, :] + local_distortion[:, site_id, :cn, :]
                local = torch.einsum("bpi,bij->bpj", local_shape, qmat.transpose(-1, -2)) * scale[:, None, None]
                cart_ops = torch.einsum("bij,ojk,bkl->boil", inv_cell, rotations.transpose(-1, -2), cell)
                orbit_vectors = torch.einsum("bpj,bojk->bopk", local, cart_ops)
            else:
                qmat = _quat_matrix(orient[:, expanded_id:expanded_id+norbit])
                scale = 0.75 + 0.5 * torch.sigmoid(scale_raw[:, expanded_id:expanded_id+norbit])
                local_shape = canonical[None, None, :, :] + local_distortion[:, expanded_id:expanded_id+norbit, :cn, :]
                orbit_vectors = torch.einsum("bopi,boij->bopj", local_shape, qmat.transpose(-1, -2))
                orbit_vectors = orbit_vectors * scale[:, :, None, None]
            padded = torch.zeros((bsz, norbit, max_cn, 3), device=self.device)
            padded[:, :, :len(canonical)] = orbit_vectors
            vectors_list.append(padded)
            labels.extend([label] * norbit)
            owner_site.extend([site_id] * norbit)
            expanded_id += norbit

        centers = torch.cat(centers_list, 1)
        vectors = torch.cat(vectors_list, 1)
        port_mask = torch.zeros((nblock, max_cn), dtype=torch.bool, device=self.device)
        for i, label in enumerate(labels):
            port_mask[i, :self.model.template_cn[label]] = True
        return abc, angles, cell, z2_raw, {
            "center_frac": centers, "template_vectors_cart": vectors,
            "port_mask": port_mask, "expanded_labels": tuple(labels),
            "free": free, "owner_site": tuple(owner_site),
            "local_distortion": local_distortion,
        }

    def _initial_shared_raw(self, template: dict) -> torch.Tensor:
        nlat = len(template["spec"]); ncoord = sum(template["site_dofs"])
        nsite = len(template["site_labels"]); nblock = int(template["nblocks"])
        npose = nsite if self.construction_symmetry == "full" else nblock
        extra = 3*nblock if self.construction_symmetry == "off" else 0
        max_cn = max(self.model.template_cn.values(), default=0)
        pair_mus = [x.mu for x in self.model.channels.values()]
        base = float(np.mean(pair_mus)) * max(nblock, 1) ** (1/3) * 1.8
        raw = torch.randn((self.starts, nlat+ncoord+5*npose+3*npose*max_cn+extra), device=self.device)
        # Begin close to the learned local template; topology/geometry optimization
        # introduces only the distortion needed to reach coincidence.
        d0 = nlat+ncoord+5*npose
        raw[:, d0:d0+3*npose*max_cn] *= 0.08
        raw[:, :nlat] *= 0.35
        raw[:, :nlat] += math.log(math.expm1(max(base-1.2, 0.5)))
        return raw

    @staticmethod
    def _periodic_unwrap_np(points: np.ndarray, cell: np.ndarray, anchor: np.ndarray):
        shifts_cart = SHIFTS @ cell
        out, shifts = [], []
        for point in points:
            choices = point[None, :] + shifts_cart
            sid = int(np.argmin(np.linalg.norm(choices-anchor[None, :], axis=1)))
            out.append(choices[sid]); shifts.append(SHIFTS[sid])
        return np.asarray(out), np.asarray(shifts)

    def _make_shared_groups(self, template: dict, raw: torch.Tensor):
        """Build a fast exact fixed-arity partition of all parent ports.

        For the common Ti4/O8 case (four parent centres, three-parent shared
        sites), an exact incidence pattern is known combinatorially: each child
        omits one of the four parents and each parent is omitted twice.  We only
        optimize which local port from each included parent enters each child.
        This avoids enumerating C(24, 3) triples and running a bounded exact-cover
        search for every batch member and topology refresh.
        """
        with torch.no_grad():
            _, _, cell_t, _, expanded = self._shared_raw_geometry(template, raw)
            centres_f = expanded["center_frac"].cpu().numpy()
            vectors = expanded["template_vectors_cart"].cpu().numpy()
            cells = cell_t.cpu().numpy()
        parent_labels = expanded["expanded_labels"]
        port_records = []
        ports_by_parent = defaultdict(list)
        for i, label in enumerate(parent_labels):
            for p in range(self.model.template_cn[label]):
                idx = len(port_records)
                port_records.append((i, p))
                ports_by_parent[i].append(idx)

        child_labels = tuple(self.plan["children"])
        if len(child_labels) != 1:
            raise NotImplementedError("shared-site construction currently requires one child label")
        child_label = child_labels[0]
        arity = int(self.model.template_cn[child_label])
        nchild = int(self.target_counts[child_label])
        if len(port_records) != nchild * arity:
            raise ValueError("Parent-port count does not match child-site capacity")

        parent_ids = sorted(ports_by_parent)
        counts = [len(ports_by_parent[i]) for i in parent_ids]
        all_groups = []
        rng = np.random.default_rng(1729)

        # Fast balanced construction used by Ti4O8 and analogous (m+1)-parent,
        # m-fold shared-site systems.
        balanced_case = (
            len(parent_ids) == arity + 1
            and len(set(counts)) == 1
            and nchild * (len(parent_ids) - arity) % len(parent_ids) == 0
        )

        for b in range(len(raw)):
            centre_cart = centres_f[b] @ cells[b]
            endpoints = np.asarray([centre_cart[i] + vectors[b, i, p] for i, p in port_records])
            best_solution = None
            best_score = float("inf")

            if balanced_case:
                omit_each = nchild * (len(parent_ids) - arity) // len(parent_ids)
                omissions = [pid for pid in parent_ids for _ in range(omit_each)]
                slots_for_parent = {
                    pid: [g for g, omitted in enumerate(omissions) if pid != omitted]
                    for pid in parent_ids
                }
                if any(len(slots_for_parent[pid]) != len(ports_by_parent[pid]) for pid in parent_ids):
                    omissions = []

                # Randomized exact assignments are extremely cheap here.  Keep
                # the best periodic coincidence score over several restarts.
                for trial in range(64 if omissions else 0):
                    groups = [[] for _ in range(nchild)]
                    for pid in parent_ids:
                        plist = np.asarray(ports_by_parent[pid], dtype=int).copy()
                        rng.shuffle(plist)
                        slot_order = np.asarray(slots_for_parent[pid], dtype=int).copy()
                        if trial:
                            rng.shuffle(slot_order)
                        for q, g in zip(plist, slot_order):
                            groups[int(g)].append(int(q))
                    if any(len(g) != arity for g in groups):
                        continue
                    solution = []
                    score = 0.0
                    valid = True
                    for combo in groups:
                        pts, shifts = self._periodic_unwrap_np(
                            endpoints[combo], cells[b], endpoints[combo[0]]
                        )
                        center = pts.mean(0)
                        spread = float(np.mean(np.sum((pts - center) ** 2, axis=1)))
                        parents = [port_records[q][0] for q in combo]
                        if len(set(parents)) != arity:
                            valid = False
                            break
                        score += spread
                        solution.append((tuple(combo), shifts.astype(float)))
                    if valid and score < best_score:
                        best_score = score
                        best_solution = solution

            # Generic bounded randomized greedy fallback.  It always consumes
            # ports exactly once and rejects only when no distinct-parent group
            # can be completed.
            if best_solution is None:
                for _ in range(128):
                    unused = set(range(len(port_records)))
                    solution = []
                    score = 0.0
                    while unused:
                        anchor = min(unused)
                        anchor_parent = port_records[anchor][0]
                        choices = [q for q in unused if port_records[q][0] != anchor_parent]
                        if len(choices) < arity - 1:
                            solution = None
                            break
                        rng.shuffle(choices)
                        best_local = None
                        for subset in itertools.combinations(choices[:min(len(choices), 16)], arity - 1):
                            combo = (anchor,) + tuple(subset)
                            parents = [port_records[q][0] for q in combo]
                            if len(set(parents)) != arity:
                                continue
                            pts, shifts = self._periodic_unwrap_np(
                                endpoints[list(combo)], cells[b], endpoints[anchor]
                            )
                            center = pts.mean(0)
                            spread = float(np.mean(np.sum((pts - center) ** 2, axis=1)))
                            if best_local is None or spread < best_local[0]:
                                best_local = (spread, combo, shifts.astype(float))
                        if best_local is None:
                            solution = None
                            break
                        spread, combo, shifts = best_local
                        unused.difference_update(combo)
                        solution.append((tuple(combo), shifts))
                        score += spread
                    if solution is not None and len(solution) == nchild and score < best_score:
                        best_score = score
                        best_solution = solution

            all_groups.append(best_solution if best_solution is not None and len(best_solution) == nchild else None)
        return port_records, child_label, all_groups

    @staticmethod
    def _slice_shared_topology(topology, selector):
        """Select batch members without rebuilding the discrete O topology."""
        port_records, child_label, all_groups = topology
        if torch.is_tensor(selector):
            selector = selector.detach().cpu().numpy()
        arr = np.asarray(selector)
        if arr.dtype == bool:
            indices = np.flatnonzero(arr).tolist()
        else:
            indices = arr.astype(int).reshape(-1).tolist()
        return port_records, child_label, [all_groups[i] for i in indices]

    def _shared_loss(self, template: dict, raw: torch.Tensor, topology, phase: str = "local_chemistry"):
        port_records, child_label, all_groups = topology
        abc, angles, cell, z2_raw, expanded = self._shared_raw_geometry(template, raw)
        centres_f = expanded["center_frac"]
        centres_c = torch.einsum("bni,bij->bnj", centres_f, cell)
        vectors = expanded["template_vectors_cart"]
        endpoints = torch.stack([centres_c[:, i] + vectors[:, i, p] for i, p in port_records], dim=1)
        bsz = len(raw); nchild = int(self.target_counts[child_label]); arity = self.model.template_cn[child_label]
        child_cart_rows=[]; coinc_rms=[]; coinc_max=[]; complete=[]
        member_parent_rows=[]
        for b in range(bsz):
            groups = all_groups[b]
            if groups is None:
                child_cart_rows.append(torch.zeros((nchild,3),device=self.device)); coinc_rms.append(torch.tensor(10.,device=self.device)); coinc_max.append(torch.tensor(10.,device=self.device)); complete.append(False); member_parent_rows.append([]); continue
            centers=[]; devs=[]; parent_images_for_groups=[]
            for ids, shifts_np in groups:
                ids_t=torch.as_tensor(ids,dtype=torch.long,device=self.device)
                shifts_t=torch.as_tensor(shifts_np,dtype=torch.float32,device=self.device)
                # The same periodic image shift used to unwrap a parent-owned port
                # must also be applied to its parent centre.  Otherwise a child
                # assembled across a cell boundary is compared with the wrong Ti
                # image, corrupting every radial and angular term.
                pts=endpoints[b,ids_t] + shifts_t @ cell[b]
                c=pts.mean(0); centers.append(c); devs.append(torch.linalg.norm(pts-c,dim=1))
                parent_images=[]
                for local_id, q in enumerate(ids):
                    parent=port_records[q][0]
                    parent_images.append((parent, shifts_t[local_id]))
                parent_images_for_groups.append(parent_images)
            dev=torch.cat(devs); child_cart_rows.append(torch.stack(centers)); coinc_rms.append(torch.sqrt(torch.mean(dev**2))); coinc_max.append(dev.max()); complete.append(True); member_parent_rows.append(parent_images_for_groups)
        child_cart=torch.stack(child_cart_rows); inv_cell=torch.linalg.inv(cell); child_frac=(torch.einsum("bni,bij->bnj",child_cart,inv_cell))%1.0
        coincidence_rms=torch.stack(coinc_rms); coincidence_max=torch.stack(coinc_max)
        coincidence_loss=(coincidence_rms/self.port_width).pow(2)

        radial_loss=torch.zeros(bsz,device=self.device); parent_angle_loss=torch.zeros_like(radial_loss); child_angle_loss=torch.zeros_like(radial_loss)
        parent_child_vectors=[[[] for _ in range(len(expanded["expanded_labels"]))] for _ in range(bsz)]
        for b in range(bsz):
            if not complete[b]: continue
            for g, parent_images in enumerate(member_parent_rows[b]):
                reverse_vectors=[]
                for parent, shift_t in parent_images:
                    label=expanded["expanded_labels"][parent]; ch=self.model.pair(label,child_label)
                    parent_image=centres_c[b,parent] + shift_t @ cell[b]
                    vec=child_cart[b,g]-parent_image
                    radial_loss[b]+=((torch.linalg.norm(vec)-ch.mu)/max(ch.sigma,0.02))**2
                    parent_child_vectors[b][parent].append(vec)
                    reverse_vectors.append(parent_image-child_cart[b,g])
                # reverse child Xn angular geometry, using the same unwrapped
                # periodic parent images that define this coincidence group.
                vecs=torch.stack(reverse_vectors)
                unit=vecs/torch.linalg.norm(vecs,dim=1,keepdim=True).clamp_min(1e-6)
                tri=torch.triu_indices(arity,arity,1,device=self.device)
                obs=torch.sort(torch.rad2deg(torch.acos((unit@unit.T).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)[tri[0],tri[1]]))).values
                target=torch.as_tensor(self.model.template_angles[child_label],device=self.device,dtype=obs.dtype)
                sig=torch.as_tensor(self.model.template_angle_sigma[child_label],device=self.device,dtype=obs.dtype)
                child_angle_loss[b]+=torch.mean(((obs-target)/sig.clamp_min(1e-3))**2)
            radial_loss[b]/=max(len(port_records),1); child_angle_loss[b]/=max(nchild,1)
            for i,label in enumerate(expanded["expanded_labels"]):
                vecs=parent_child_vectors[b][i]; cn=self.model.template_cn[label]
                if len(vecs)!=cn: parent_angle_loss[b]+=100.; continue
                vv=torch.stack(vecs); unit=vv/torch.linalg.norm(vv,dim=1,keepdim=True).clamp_min(1e-6)
                tri=torch.triu_indices(cn,cn,1,device=self.device)
                obs=torch.sort(torch.rad2deg(torch.acos((unit@unit.T).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)[tri[0],tri[1]]))).values
                target=torch.as_tensor(self.model.template_angles[label],device=self.device,dtype=obs.dtype)
                sig=torch.as_tensor(self.model.template_angle_sigma[label],device=self.device,dtype=obs.dtype)
                parent_angle_loss[b]+=torch.mean(((obs-target)/sig.clamp_min(1e-3))**2)
            parent_angle_loss[b]/=max(len(expanded["expanded_labels"]),1)

        final_frac=torch.cat([centres_f,child_frac],dim=1)

        # Contact-graph objective.  Assigned edges are image-resolved and use
        # the immutable periodic shifts stored by topology construction.  Only
        # genuinely unassigned Ti/O pairs use a minimum-image distance.
        nparent = len(expanded["expanded_labels"])
        assigned_pair_mask = torch.zeros((bsz, nparent, nchild), dtype=torch.bool, device=self.device)
        assigned_edge_dist_rows = []
        assigned_edge_low_rows = []
        assigned_edge_high_rows = []
        assigned_edge_mu_rows = []
        assigned_edge_sigma_rows = []
        for b in range(bsz):
            edge_dist = []
            edge_low = []
            edge_high = []
            edge_mu = []
            edge_sigma = []
            if complete[b]:
                for g, parent_images in enumerate(member_parent_rows[b]):
                    for parent, shift_t in parent_images:
                        label = expanded["expanded_labels"][parent]
                        ch = self.model.pair(label, child_label)
                        parent_image = centres_c[b, parent] + shift_t @ cell[b]
                        edge_dist.append(torch.linalg.norm(child_cart[b, g] - parent_image))
                        edge_low.append(float(ch.sampling_min))
                        edge_high.append(float(ch.sampling_max))
                        edge_mu.append(float(ch.mu))
                        edge_sigma.append(max(float(ch.sigma), 0.02))
                        assigned_pair_mask[b, parent, g] = True
            if edge_dist:
                assigned_edge_dist_rows.append(torch.stack(edge_dist))
                assigned_edge_low_rows.append(torch.as_tensor(edge_low, dtype=cell.dtype, device=self.device))
                assigned_edge_high_rows.append(torch.as_tensor(edge_high, dtype=cell.dtype, device=self.device))
                assigned_edge_mu_rows.append(torch.as_tensor(edge_mu, dtype=cell.dtype, device=self.device))
                assigned_edge_sigma_rows.append(torch.as_tensor(edge_sigma, dtype=cell.dtype, device=self.device))
            else:
                z = torch.zeros((1,), dtype=cell.dtype, device=self.device)
                assigned_edge_dist_rows.append(z)
                assigned_edge_low_rows.append(z)
                assigned_edge_high_rows.append(z)
                assigned_edge_mu_rows.append(z)
                assigned_edge_sigma_rows.append(torch.ones_like(z))

        assigned_window_loss = torch.zeros((bsz,), dtype=cell.dtype, device=self.device)
        assigned_radial_loss = torch.zeros_like(assigned_window_loss)
        assigned_fraction = torch.zeros_like(assigned_window_loss)
        for b in range(bsz):
            if not complete[b]:
                continue
            d = assigned_edge_dist_rows[b]
            lo = assigned_edge_low_rows[b]
            hi = assigned_edge_high_rows[b]
            mu = assigned_edge_mu_rows[b]
            sig = assigned_edge_sigma_rows[b]
            assigned_window_loss[b] = (
                torch.relu((lo-d)/sig).pow(2) + torch.relu((d-hi)/sig).pow(2)
            ).mean()
            assigned_radial_loss[b] = ((d-mu)/sig).pow(2).mean()
            assigned_fraction[b] = ((d >= lo) & (d <= hi)).float().mean()

        shifts_t = torch.as_tensor(SHIFTS, dtype=child_frac.dtype, device=child_frac.device)
        delta_pc = (
            child_frac[:, None, :, None, :]
            - centres_f[:, :, None, None, :]
            + shifts_t[None, None, None, :, :]
        )
        cart_pc = torch.einsum("bpksi,bij->bpksj", delta_pc, cell)
        dist_pc = torch.linalg.norm(cart_pc, dim=-1).amin(dim=3)
        shell_limit = torch.zeros((bsz, nparent, nchild), dtype=dist_pc.dtype, device=self.device)
        for i, label in enumerate(expanded["expanded_labels"]):
            ch = self.model.pair(label, child_label)
            shell_limit[:, i, :] = float(ch.first_shell_cutoff)
        complete_mask = torch.as_tensor(complete, dtype=torch.bool, device=self.device)[:, None, None]
        unassigned_mask = (~assigned_pair_mask) & complete_mask
        unselected_penetration = torch.relu(
            (shell_limit + self.nonbonded_margin - dist_pc) / self.nonbonded_width
        ).pow(2)
        unselected_count = unassigned_mask.sum((1,2)).clamp_min(1)
        unselected_contact_loss = (unselected_penetration * unassigned_mask).sum((1,2)) / unselected_count
        unselected_first_shell_count = (unassigned_mask & (dist_pc < shell_limit)).sum((1,2))

        _, pdist=self._center_geometry(final_frac,cell); min_dist=pdist.amin((1,2,3))
        overlap=torch.relu((self.minimum_distance-min_dist)/0.05).pow(2)
        aspect=abc.amax(1)/abc.amin(1).clamp_min(1e-4)
        shape=torch.relu(aspect-6.0).pow(2)
        metric=torch.relu(1e-5-z2_raw/abc[:,2].square().clamp_min(1e-8)).pow(2)*1e6
        complete_t=torch.as_tensor(complete,dtype=torch.bool,device=self.device)
        distortion_loss=(expanded["local_distortion"]/max(self.distortion_max,1.0e-6)).pow(2).mean(dim=(1,2,3))
        prefix=raw[:, :self._shared_prefix_size(template)]
        framework_loss, framework_detail, _ = self._framework_loss(template, prefix)
        framework_restraint=torch.zeros_like(framework_loss)
        if self._framework_reference is not None and len(self._framework_reference)==len(raw):
            framework_restraint=(prefix-self._framework_reference).pow(2).mean(1)
        if phase == "o_coincidence":
            coincidence_scale, angular_scale, contact_scale, exclusion_scale = 3.0, 0.05, 2.0, 1.0
            framework_restraint_scale = 0.35
        elif phase == "o_contact_graph":
            coincidence_scale, angular_scale, contact_scale, exclusion_scale = 1.5, 0.35, 8.0, 8.0
            framework_restraint_scale = 0.65
        else:
            coincidence_scale, angular_scale, contact_scale, exclusion_scale = 1.0, 1.0, 12.0, 12.0
            framework_restraint_scale = 1.0
        total=(coincidence_scale*self.coincidence_weight*coincidence_loss
            +self.radial_weight*(radial_loss + assigned_radial_loss)
            +angular_scale*self.angular_weight*(parent_angle_loss+child_angle_loss)
            +contact_scale*assigned_window_loss
            +exclusion_scale*self.nonbonded_weight*unselected_contact_loss
            +self.distortion_weight*distortion_loss+self.overlap_weight*overlap
            +self.framework_weight*framework_loss
            +framework_restraint_scale*self.framework_restraint_weight*framework_restraint
            +0.05*shape+metric+(~complete_t).float()*1e4)
        total=torch.nan_to_num(total,nan=1e9,posinf=1e9,neginf=1e9)
        details={"topology_complete":complete_t,"coincidence_rms_A":coincidence_rms,"coincidence_max_A":coincidence_max,"coincidence_loss":coincidence_loss,"shared_radial_loss":radial_loss,"parent_angular_loss":parent_angle_loss,"child_angular_loss":child_angle_loss,"template_distortion_loss":distortion_loss,"minimum_physical_distance_A":min_dist,"aspect_ratio":aspect,
            "framework_loss":framework_loss,"framework_restraint_loss":framework_restraint,
            "assigned_bond_window_loss":assigned_window_loss,
            "assigned_bond_fraction":assigned_fraction,
            "unselected_contact_loss":unselected_contact_loss,
            "unselected_first_shell_contact_count_soft":unselected_first_shell_count,
            **framework_detail}
        geom=(abc,angles,cell,expanded,child_frac,final_frac)
        return total,details,geom

    def _optimize_shared(self, template, raw, steps, phase, lr, heartbeat=None, topology=None):
        raw=raw.detach().clone()
        best=raw.clone()
        best_loss=torch.full((len(raw),),float("inf"),device=self.device)
        completed=0
        last=time.perf_counter()
        # A supplied topology is immutable across all subsequent O stages.
        # Rebuilding is permitted only for legacy callers that did not provide one.
        fixed_topology = topology is not None
        if topology is None:
            topology=self._make_shared_groups(template,raw)
        refresh=0 if fixed_topology else int(self.assignment_refresh)
        variable=raw.detach().clone().requires_grad_(True)
        opt=torch.optim.Adam([variable],lr=float(lr))
        while completed<int(steps):
            if refresh>0 and completed>0 and completed % refresh == 0:
                raw=variable.detach()
                topology=self._make_shared_groups(template,raw)
                variable=raw.clone().requires_grad_(True)
                opt=torch.optim.Adam([variable],lr=float(lr))
            opt.zero_grad(set_to_none=True)
            loss,_,_=self._shared_loss(template,variable,topology,phase=phase)
            if not bool(torch.isfinite(loss).any()):
                break
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_([variable],10.0)
            opt.step()
            with torch.no_grad():
                bad=~torch.isfinite(variable).all(1)
                if bool(bad.any()): variable[bad]=best[bad]
                post_loss,_,_=self._shared_loss(template,variable,topology,phase=phase)
                improved=torch.isfinite(post_loss)&(post_loss<best_loss)
                best_loss=torch.where(improved,post_loss,best_loss)
                best[improved]=variable.detach()[improved]
            completed+=1
            now=time.perf_counter()
            if heartbeat is not None and (completed==1 or now-last>=10 or completed==int(steps)):
                heartbeat(phase,completed,int(steps),float(best_loss.min().detach().cpu()))
                last=now
        return best, topology

    def _build_shared(self, template: dict, heartbeat=None):
        attempts=[]; candidates=[]
        raw=self._initial_shared_raw(template)
        psize=self._shared_prefix_size(template)

        # Stage 1: Ti framework only.
        framework=self._optimize_framework(
            template,raw[:,:psize],self.screen_steps+self.refine_steps,heartbeat)
        with torch.no_grad():
            floss,fdetail,_=self._framework_loss(template,framework)
            parent_models=[self.model.framework_models.get(x,{}) for x in template["expanded_labels"]]
            qmax=max([float(x.get("score_q90_max",3.5)) for x in parent_models] or [3.5])
            lower=max([float(x.get("radial_lower_bound_A",0.0)) for x in parent_models] or [0.0])
            ti_finite=torch.isfinite(floss)&torch.isfinite(framework).all(1)
            ti_accepted=ti_finite&(fdetail["framework_score_q90"]<=qmax)&(fdetail["framework_min_distance_A"]>=lower)
            for i in range(len(framework)):
                attempts.append({
                    "stage":"ti_framework_screen","branch":int(i),
                    "loss":float(floss[i]),"construction_symmetry":self.construction_symmetry,
                    "ti_framework_finite":bool(ti_finite[i]),
                    "ti_framework_accepted":bool(ti_accepted[i]),
                    "o_topology_complete":False,"o_topology_accepted":False,
                    "o_attachment_accepted":False,"strict_valid":False,
                    **{k:float(v[i]) for k,v in fdetail.items()},
                })
            eligible=torch.nonzero(ti_accepted, as_tuple=False).flatten()
            if len(eligible)==0:
                self._framework_reference=None
                return None,attempts
            order=eligible[torch.argsort(floss[eligible])]
            keep=order[:min(self.framework_keep,len(order))]
            framework=framework[keep]

        # Stage 2: instantiate parent templates and require an exact O topology
        # before spending any gradient steps on oxygen geometry.
        full=self._initial_shared_raw(template)[:len(framework)]
        full[:,:psize]=framework
        topology=self._make_shared_groups(template,full)
        topo_mask=torch.as_tensor([x is not None for x in topology[2]],dtype=torch.bool,device=self.device)
        for i in range(len(full)):
            attempts.append({
                "stage":"o_topology_screen","branch":int(i),
                "construction_symmetry":self.construction_symmetry,
                "ti_framework_finite":True,"ti_framework_accepted":True,
                "o_topology_complete":bool(topo_mask[i]),
                "o_topology_accepted":bool(topo_mask[i]),
                "o_attachment_accepted":False,"strict_valid":False,
            })
        if not bool(topo_mask.any()):
            self._framework_reference=None
            return None,attempts
        full=full[topo_mask]
        framework=framework[topo_mask]
        topology=self._slice_shared_topology(topology, topo_mask)
        self._framework_reference=framework.detach().clone()

        # Stage 3a: close the three parent-owned ports onto each O site while
        # allowing restrained Ti-framework motion.  Local angles are deliberately
        # weak here so coincidence and assigned Ti-O bonds converge first.
        coincident, topology=self._optimize_shared(
            template,full,self.oxygen_coincidence_steps,"o_coincidence",0.45*self.lr,heartbeat,topology=topology)

        # Stage 3b: reconcile the realized geometric contact graph.  Assigned
        # Ti-O edges are driven inside the learned bond interval and every
        # unassigned Ti-O pair is repelled beyond the learned first-shell cutoff.
        attached, topology=self._optimize_shared(
            template,coincident,self.oxygen_contact_steps,"o_contact_graph",0.25*self.lr,heartbeat,topology=topology)
        with torch.no_grad():
            aloss,adetail,_=self._shared_loss(template,attached,topology,phase="o_contact_graph")
            attach_finite=torch.isfinite(aloss)&torch.isfinite(attached).all(1)
            attach_topology=adetail["topology_complete"]
            attach_accepted=(attach_finite&attach_topology
                &(adetail["coincidence_rms_A"]<=self.oxygen_screen_rms_max)
                &(adetail["coincidence_max_A"]<=self.oxygen_screen_max_max)
                &(adetail["assigned_bond_fraction"]>=self.oxygen_assigned_fraction_min)
                &(adetail["unselected_first_shell_contact_count_soft"]==0))
            for i in range(len(attached)):
                row={
                    "stage":"o_attachment_screen","branch":int(i),
                    "loss":float(aloss[i]),"construction_symmetry":self.construction_symmetry,
                    "ti_framework_finite":True,"ti_framework_accepted":True,
                    "o_topology_complete":bool(attach_topology[i]),
                    "o_topology_accepted":bool(attach_topology[i]),
                    "o_attachment_accepted":bool(attach_accepted[i]),
                    "strict_valid":False,
                }
                for key,value in adetail.items():
                    row[key]=bool(value[i]) if value.dtype==torch.bool else float(value[i])
                attempts.append(row)
            if not bool(attach_accepted.any()):
                self._framework_reference=None
                return None,attempts
            topology=self._slice_shared_topology(topology, attach_accepted)
            attached=attached[attach_accepted]

        # Stage 4: final local-chemistry polish with the same image-resolved topology.
        self._framework_reference=attached[:,:psize].detach().clone()
        polished, topology=self._optimize_shared(
            template,attached,self.polish_steps,"local_chemistry_polish",0.12*self.lr,heartbeat,topology=topology)
        with torch.no_grad():
            loss,detail,geom=self._shared_loss(template,polished,topology,phase="local_chemistry")
            abc,angles,cell,expanded,child_frac,final_frac=geom
            child_label=self.plan["children"][0]
            labels=tuple(expanded["expanded_labels"])+tuple([child_label]*int(self.target_counts[child_label]))
            symbols=tuple(self.model.block_atoms[x][0][0] for x in labels)
            fake_template={"expanded_labels":labels}
            for index in range(len(polished)):
                ex={"center_frac":final_frac[index],"physical_frac":final_frac[index]}
                strict=self._strict(fake_template,cell[index].cpu().numpy(),ex)
                row={"stage":"strict_final_audit","branch":int(index),"loss":float(loss[index]),
                    "construction_symmetry":self.construction_symmetry,
                    "ti_framework_finite":True,"ti_framework_accepted":True,
                    "o_topology_complete":bool(detail["topology_complete"][index]),
                    "o_topology_accepted":bool(detail["topology_complete"][index]),
                    "o_attachment_accepted":True,**strict}
                for key,value in detail.items():
                    row[key]=bool(value[index]) if value.dtype==torch.bool else float(value[index])
                row["strict_valid"]=bool(row["strict_valid"]
                    and row["coincidence_rms_A"]<=self.coincidence_rms_max
                    and row["coincidence_max_A"]<=self.coincidence_max_max)
                attempts.append(row)
                if not row["strict_valid"]:
                    continue
                candidates.append({**row,"cell":cell[index].cpu().numpy(),
                    "lattice":np.concatenate([abc[index].cpu().numpy(),angles[index].cpu().numpy()]),
                    "frac":final_frac[index].cpu().numpy(),"symbols":symbols,
                    "center_frac":expanded["center_frac"][index].cpu().numpy(),
                    "free":np.zeros((len(template["wps"]),3),float)})
        self._framework_reference=None
        candidates.sort(key=lambda x:(x["coincidence_rms_A"],x["framework_score_q90"],
            x["local_angular_vector_max_deg"],x["loss"]))
        return (candidates[0] if candidates else None),attempts

    def build(self, spg: int, wp_token: str, species_token: str, heartbeat=None) -> tuple[dict | None, list[dict]]:
        template = self._template(spg, wp_token, species_token)
        if self.plan["mode"] == "shared_site":
            return self._build_shared(template, heartbeat=heartbeat)
        raw = self._initial_raw(template)
        screened = self._optimize(template, raw, self.screen_steps, "screen", self.lr, heartbeat)
        with torch.no_grad():
            screen_assignment = self._make_assignment(template, screened)
            loss, _, _ = self._loss(template, screened, screen_assignment)
            keep = torch.argsort(loss)[: min(max(4, self.starts // 4), len(screened))]
        refined = self._optimize(template, screened[keep], self.refine_steps, "refine", 0.45 * self.lr, heartbeat)
        polished = self._optimize(template, refined, self.polish_steps, "polish", 0.12 * self.lr, heartbeat)
        attempts = []
        candidates = []
        with torch.no_grad():
            final_assignment = self._make_assignment(template, polished)
            loss, detail, geom = self._loss(template, polished, final_assignment)
            abc, angles, cell, expanded = geom
            for index in range(len(polished)):
                ex = {
                    "center_frac": expanded["center_frac"][index],
                    "template_vectors_cart": expanded["template_vectors_cart"][index],
                    "physical_frac": expanded["physical_frac"][index],
                }
                strict = self._strict(template, cell[index].cpu().numpy(), ex)
                row = {
                    "stage": "coincidence_driven_topology_polish",
                    "branch": int(index),
                    "loss": float(loss[index]),
                    **strict,
                }
                for key, value in detail.items():
                    row[key] = bool(value[index]) if value.dtype == torch.bool else float(value[index])
                attempts.append(row)
                if not strict["strict_valid"]:
                    continue
                free_np = np.zeros((len(template["wps"]), 3), float)
                for site_id, u in enumerate(expanded["free"]):
                    free_np[site_id, :u.shape[1]] = u[index].cpu().numpy()
                candidates.append(
                    {
                        **row,
                        "cell": cell[index].cpu().numpy(),
                        "lattice": np.concatenate([abc[index].cpu().numpy(), angles[index].cpu().numpy()]),
                        "frac": expanded["physical_frac"][index].cpu().numpy(),
                        "symbols": expanded["physical_symbols"],
                        "center_frac": expanded["center_frac"][index].cpu().numpy(),
                        "free": free_np,
                    }
                )
        candidates.sort(
            key=lambda x: (
                x["local_angular_vector_max_deg"],
                x["local_radial_vector_max_A"],
                x["loss"],
            )
        )
        return (candidates[0] if candidates else None), attempts


def _write_cif(result: dict, path: Path):
    from ase import Atoms
    from ase.io import write

    atoms = Atoms(
        list(result["symbols"]),
        scaled_positions=np.asarray(result["frac"], float),
        cell=np.asarray(result["cell"], float),
        pbc=True,
    )
    write(path, atoms, format="cif")



# -----------------------------------------------------------------------------
# Conservative duplicate alerting and strict proper-rotation matching
# -----------------------------------------------------------------------------

def _periodic_pair_spectrum(cell: np.ndarray, frac: np.ndarray, symbols: tuple[str, ...],
                            cutoff: float = 6.0) -> dict[str, list[float]]:
    cell = np.asarray(cell, float)
    frac = np.asarray(frac, float) % 1.0
    symbols = tuple(map(str, symbols))
    delta = frac[None, :, None, :] - frac[:, None, None, :] + SHIFTS[None, None, :, :]
    cart = np.einsum("...i,ij->...j", delta, cell)
    dist = np.linalg.norm(cart, axis=-1)
    out: dict[str, list[float]] = defaultdict(list)
    n = len(frac)
    for i in range(n):
        for j in range(i, n):
            key = "|".join(sorted((symbols[i], symbols[j])))
            for sid, value in enumerate(dist[i, j]):
                if i == j and sid == ZERO_SHIFT:
                    continue
                # Avoid double counting central-cell i-j and j-i pairs while retaining self images.
                if i != j and j < i:
                    continue
                if 1.0e-7 < float(value) <= float(cutoff):
                    out[key].append(float(value))
    return {key: sorted(values) for key, values in sorted(out.items())}


def _dedup_fingerprint(result: dict, species_token: str) -> dict:
    cell = np.asarray(result["cell"], float)
    frac = np.asarray(result["frac"], float) % 1.0
    symbols = tuple(map(str, result["symbols"]))
    metric = cell @ cell.T
    eig = np.sort(np.linalg.eigvalsh(metric).clip(min=0.0))
    scale = max(float(np.trace(metric)), EPS)
    return {
        "atom_count": int(len(frac)),
        "composition": dict(sorted(Counter(symbols).items())),
        "species_token": str(species_token),
        "volume_per_atom_A3": float(abs(np.linalg.det(cell)) / max(len(frac), 1)),
        "metric_eigenvalues_normalized": (eig / scale).tolist(),
        "pair_spectrum_A": _periodic_pair_spectrum(cell, frac, symbols),
    }


def _fingerprint_alert(a: dict, b: dict) -> bool:
    if a["atom_count"] != b["atom_count"] or a["composition"] != b["composition"]:
        return False
    va, vb = float(a["volume_per_atom_A3"]), float(b["volume_per_atom_A3"])
    if abs(va - vb) / max(va, vb, EPS) > 0.04:
        return False
    ea = np.asarray(a["metric_eigenvalues_normalized"], float)
    eb = np.asarray(b["metric_eigenvalues_normalized"], float)
    if ea.shape != eb.shape or float(np.max(np.abs(ea - eb))) > 0.04:
        return False
    if set(a["pair_spectrum_A"]) != set(b["pair_spectrum_A"]):
        return False
    for key in a["pair_spectrum_A"]:
        da = np.asarray(a["pair_spectrum_A"][key], float)
        db = np.asarray(b["pair_spectrum_A"][key], float)
        if da.shape != db.shape:
            return False
        if len(da) and float(np.max(np.abs(da - db))) > 0.08:
            return False
    return True


def _prepare_reduced_structure(cell, frac, symbols):
    """Return an ASE Niggli-reduced representation without idealizing atoms."""
    from ase import Atoms
    atoms = Atoms(list(symbols), scaled_positions=np.asarray(frac, float) % 1.0,
                  cell=np.asarray(cell, float), pbc=True)
    try:
        from ase.build.tools import niggli_reduce
        atoms.wrap()
        niggli_reduce(atoms)
        atoms.wrap()
    except Exception:
        atoms.wrap()
    return np.asarray(atoms.cell.array, float), np.asarray(atoms.get_scaled_positions(), float) % 1.0, tuple(atoms.get_chemical_symbols())


def _signed_permutation_matrices() -> list[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        base = np.eye(3, dtype=int)[list(perm)]
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.diag(signs) @ base
            if abs(round(np.linalg.det(matrix))) == 1:
                mats.append(matrix.astype(float))
    return mats


_SIGNED_PERMUTATIONS = _signed_permutation_matrices()


def _strict_proper_exact_match(current: dict, previous: dict,
                               lattice_rel_tol: float = 0.005,
                               angle_tol_deg: float = 0.30,
                               rms_tol_A: float = 0.015,
                               max_tol_A: float = 0.030) -> dict | None:
    """Reject only an exceptionally strict, orientation-preserving exact match.

    Returning None means distinct *or uncertain*.  This deliberately prefers
    false uniques over any false duplicate rejection.
    """
    try:
        c1, f1, s1 = _prepare_reduced_structure(current["cell"], current["frac"], current["symbols"])
        c2, f2, s2 = _prepare_reduced_structure(previous["cell"], previous["frac"], previous["symbols"])
    except Exception:
        return None
    if len(f1) != len(f2) or Counter(s1) != Counter(s2):
        return None

    def cell_angles(cell):
        lengths = np.linalg.norm(cell, axis=1)
        vals = []
        for i, j in ((1, 2), (0, 2), (0, 1)):
            cosine = np.dot(cell[i], cell[j]) / max(lengths[i] * lengths[j], EPS)
            vals.append(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))
        return lengths, np.asarray(vals)

    l1, a1 = cell_angles(c1)
    inv_c1 = np.linalg.inv(c1)
    anchor_symbol = min(Counter(s1), key=lambda x: Counter(s1)[x])
    anchors1 = [i for i, x in enumerate(s1) if x == anchor_symbol]
    anchors2 = [i for i, x in enumerate(s2) if x == anchor_symbol]

    for U in _SIGNED_PERMUTATIONS:
        c2u = U @ c2
        f2u = (f2 @ np.linalg.inv(U)) % 1.0
        l2, a2 = cell_angles(c2u)
        if float(np.max(np.abs(l1 - l2) / np.maximum(l1, EPS))) > lattice_rel_tol:
            continue
        if float(np.max(np.abs(a1 - a2))) > angle_tol_deg:
            continue
        # Orthogonal Procrustes map between row-vector lattice bases.
        H = c2u.T @ c1
        u, _, vt = np.linalg.svd(H)
        R = u @ vt
        if np.linalg.det(R) <= 1.0 - 1.0e-8:
            continue
        if float(np.max(np.linalg.norm(c2u @ R - c1, axis=1))) > max_tol_A:
            continue
        cart2 = f2u @ c2u @ R
        frac2_in_1 = (cart2 @ inv_c1) % 1.0
        for i1 in anchors1:
            for i2 in anchors2:
                shift = (f1[i1] - frac2_in_1[i2]) % 1.0
                shifted = (frac2_in_1 + shift) % 1.0
                sq_sum = 0.0
                max_dist = 0.0
                total = 0
                valid = True
                for symbol in sorted(set(s1)):
                    ids1 = np.asarray([i for i, x in enumerate(s1) if x == symbol], int)
                    ids2 = np.asarray([i for i, x in enumerate(s2) if x == symbol], int)
                    delta = shifted[ids2][None, :, :] - f1[ids1][:, None, :]
                    delta -= np.round(delta)
                    dist = np.linalg.norm(np.einsum("...i,ij->...j", delta, c1), axis=-1)
                    rows, cols = linear_sum_assignment(dist)
                    matched = dist[rows, cols]
                    if len(matched) != len(ids1):
                        valid = False; break
                    sq_sum += float(np.sum(matched ** 2))
                    max_dist = max(max_dist, float(np.max(matched)) if len(matched) else 0.0)
                    total += len(matched)
                if not valid or total == 0:
                    continue
                rms = math.sqrt(sq_sum / total)
                if rms <= rms_tol_A and max_dist <= max_tol_A:
                    return {
                        "rms_A": float(rms),
                        "max_displacement_A": float(max_dist),
                        "rotation_determinant": float(np.linalg.det(R)),
                    }
    return None


class ConservativeDeduplicator:
    def __init__(self, index_path: Path, active_labels: tuple[str, ...], model: XNModel):
        self.index_path = Path(index_path)
        self.records: list[dict] = []
        # If multiple construction labels share a physical element, geometry-only
        # matching cannot safely prove label-preserving identity. Disable rejection.
        label_elements = []
        for label in active_labels:
            elements = tuple(model.block_atoms[label][0])
            label_elements.append(elements)
        self.label_safe = len(label_elements) == len(set(label_elements))
        if self.index_path.exists():
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    self.records.append(json.loads(line))
                except Exception:
                    continue

    def check(self, result: dict, species_token: str) -> dict:
        fingerprint = _dedup_fingerprint(result, species_token)
        alerts = 0
        exact_checks = 0
        if self.label_safe:
            for record in self.records:
                if not _fingerprint_alert(fingerprint, record["fingerprint"]):
                    continue
                alerts += 1
                exact_checks += 1
                match = _strict_proper_exact_match(result, record["structure"])
                if match is not None:
                    return {
                        "duplicate": True,
                        "duplicate_of_candidate_id": record.get("candidate_id"),
                        "fingerprint_alerts": alerts,
                        "exact_match_checks": exact_checks,
                        **match,
                        "fingerprint": fingerprint,
                    }
        return {
            "duplicate": False,
            "duplicate_of_candidate_id": None,
            "fingerprint_alerts": alerts,
            "exact_match_checks": exact_checks,
            "fingerprint": fingerprint,
        }

    def add(self, candidate_id: int, result: dict, species_token: str, fingerprint: dict) -> None:
        record = {
            "candidate_id": int(candidate_id),
            "species_token": str(species_token),
            "fingerprint": fingerprint,
            "structure": {
                "cell": np.asarray(result["cell"], float).tolist(),
                "frac": (np.asarray(result["frac"], float) % 1.0).tolist(),
                "symbols": list(map(str, result["symbols"])),
            },
        }
        self.records.append(record)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _set_thread_limits():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def _cpu_count() -> int:
    """Return CPUs allocated to this job/task, preferring scheduler metadata.

    Some Slurm ``srun`` configurations expose a one-core affinity mask to the
    parent Python process even when ``SLURM_CPUS_PER_TASK`` is larger.  The
    allocation metadata is therefore the authoritative source on Slurm.
    """
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):  # task-local first
        value = os.environ.get(key, "").strip()
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def _exact_entries_for_group(spg: int, labels: tuple[str, ...], counts: dict[str, int], cap: int):
    """Return bounded exact Wyckoff/species entrances without exhaustive enumeration.

    The old implementation first enumerated every multiplicity skeleton and then
    every species colouring of each skeleton.  For mixed construction species,
    that Cartesian expansion can become enormous before the output cap is ever
    consulted.  This routine performs one joint depth-first search instead and
    memoizes dead remainder states.  Runtime is therefore governed by ``cap``
    rather than by the total number of possible colourings.
    """
    from pyxtal.symmetry import Group

    group = Group(int(spg))
    active = tuple(x for x in labels if counts.get(x, 0) > 0)
    target = tuple(int(counts[x]) for x in active)
    total = int(sum(target))
    multiplicity = [int(group[i].multiplicity) for i in range(len(group))]
    allowed = tuple(i for i, m in enumerate(multiplicity) if 1 <= m <= total)
    if not active or not allowed:
        return []

    # Fast necessary conditions.  Each species count must be representable by
    # the available Wyckoff multiplicities because one orbit carries one label.
    unique_mult = sorted(set(multiplicity[i] for i in allowed))
    representable = [False] * (total + 1)
    representable[0] = True
    for value in range(1, total + 1):
        representable[value] = any(value >= m and representable[value - m] for m in unique_mult)
    if any(not representable[n] for n in target):
        return []

    rows: list[tuple[str, str]] = []
    dead: set[tuple[int, tuple[int, ...]]] = set()

    def visit(start_pos: int, remaining: tuple[int, ...], wps: list[int], species: list[str]) -> bool:
        if len(rows) >= cap:
            return True
        if all(x == 0 for x in remaining):
            rows.append((encode_token(wps), encode_token(species)))
            return len(rows) >= cap

        state = (start_pos, remaining)
        if state in dead:
            return False
        before = len(rows)

        # Nondecreasing Wyckoff indices avoid permutation duplicates while still
        # allowing repeated use of a Wyckoff position when PyXtal permits it.
        for pos in range(start_pos, len(allowed)):
            wp = allowed[pos]
            m = multiplicity[wp]
            for sid, label in enumerate(active):
                if remaining[sid] < m:
                    continue
                new_remaining = list(remaining)
                new_remaining[sid] -= m
                # Prune species remainders that cannot be composed from any
                # available multiplicities.
                if any(not representable[x] for x in new_remaining):
                    continue
                if visit(pos, tuple(new_remaining), wps + [wp], species + [label]):
                    return True

        if len(rows) == before:
            dead.add(state)
        return False

    visit(0, target, [], [])
    return rows


def _proposal_worker(worker_id, request_queue, result_queue, labels, counts, cap, seed):
    _set_thread_limits()
    rng = np.random.default_rng(int(seed) + 7919 * int(worker_id))
    cache = {}
    while True:
        request = request_queue.get()
        if request is None: break
        task_id = int(request)
        try:
            for _ in range(2000):
                spg = int(rng.integers(1, 231))
                if spg not in cache:
                    cache[spg] = _exact_entries_for_group(spg, labels, counts, cap)
                rows = cache[spg]
                if rows:
                    wp, species = rows[int(rng.integers(0, len(rows)))]
                    result_queue.put({"kind":"proposal", "task_id":task_id, "spg":spg,
                                      "wp_token":wp, "species_token":species, "error":None})
                    break
            else:
                raise RuntimeError("Could not find a compatible crystallographic entrance")
        except Exception as exc:
            result_queue.put({"kind":"proposal", "task_id":task_id,
                              "error":f"{type(exc).__name__}: {exc}"})


class ProposalPool:
    def __init__(self, nworkers, labels, counts, cap, seed):
        self.ctx = mp.get_context("spawn")
        self.requests = self.ctx.Queue(maxsize=max(4, 2 * nworkers))
        self.results = self.ctx.Queue()
        self.processes = []
        for wid in range(nworkers):
            p = self.ctx.Process(target=_proposal_worker,
                args=(wid, self.requests, self.results, labels, counts, cap, seed), daemon=True)
            p.start(); self.processes.append(p)
    def request(self, task_id): self.requests.put(int(task_id))
    def close(self):
        for _ in self.processes: self.requests.put(None)
        for p in self.processes: p.join()


def _builder_worker(worker_id, device_id, tasks, results, model_path, config):
    _set_thread_limits(); torch.set_num_threads(1)
    if device_id is None: device = "cpu"
    else:
        torch.cuda.set_device(int(device_id)); device = f"cuda:{int(device_id)}"
    model = XNModel(model_path)
    builder = XNBuilder(model=model, device=device, **config)
    while True:
        task = tasks.get()
        if task is None: break
        task_id = int(task["task_id"])
        seed = int(task["seed"])
        try:
            torch.manual_seed(seed); np.random.seed(seed % (2**32 - 1))
            if device_id is not None: torch.cuda.manual_seed_all(seed)
            def heartbeat(stage, step, total, best):
                results.put({"kind":"heartbeat", "worker_id":worker_id, "task_id":task_id,
                             "stage":stage, "step":step, "total":total, "best_loss":best})
            selected, attempts = builder.build(task["spg"], task["wp_token"],
                                                task["species_token"], heartbeat=heartbeat)
            results.put({"kind":"result", "worker_id":worker_id, "task":task,
                         "selected":selected, "attempts":attempts, "error":None})
        except Exception as exc:
            results.put({"kind":"result", "worker_id":worker_id, "task":task,
                         "selected":None, "attempts":[],
                         "error":f"{type(exc).__name__}: {exc}"})


class BuilderPool:
    def __init__(self, ngpu, queue_depth, model_path, config):
        self.ctx = mp.get_context("spawn")
        devices = list(range(ngpu)) if ngpu > 0 else [None]
        self.tasks = self.ctx.Queue(maxsize=max(2, queue_depth * len(devices)))
        self.results = self.ctx.Queue()
        self.processes = []
        for wid, did in enumerate(devices):
            p = self.ctx.Process(target=_builder_worker,
                args=(wid, did, self.tasks, self.results, model_path, config), daemon=True)
            p.start(); self.processes.append(p)
    @property
    def workers(self): return len(self.processes)
    def close(self):
        for _ in self.processes: self.tasks.put(None)
        for p in self.processes: p.join()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate crystals from Juliette Xn templates")
    p.add_argument("--chemistry-model", default="data/xn_templates/chemistry_model.json")
    p.add_argument("--target", action="append", default=[],
                   help="Requested construction label or named/formula recipe as LABEL=COUNT")
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--max-attempts", type=int, default=1000)
    p.add_argument("--max-runtime-minutes", type=float, default=115.0)
    p.add_argument("--max-entries-per-group", type=int, default=5000)
    p.add_argument("--proposal-workers", type=int, default=0)
    p.add_argument("--ngpu", type=int, default=0)
    p.add_argument("--gpu-queue-depth", type=int, default=2)
    p.add_argument("--starts", type=int, default=8)
    p.add_argument("--screen-steps", type=int, default=20)
    p.add_argument("--refine-steps", type=int, default=40)
    p.add_argument("--polish-steps", type=int, default=20)
    p.add_argument("--builder-lr", type=float, default=0.04)
    p.add_argument("--minimum-distance", type=float, default=1.0)
    p.add_argument("--soft-temperature", type=float, default=0.18)
    p.add_argument("--port-width", type=float, default=0.18)
    p.add_argument("--radial-weight", type=float, default=2.0)
    p.add_argument("--angular-weight", type=float, default=1.0)
    p.add_argument("--overlap-weight", type=float, default=10.0)
    p.add_argument("--uniqueness-weight", type=float, default=4.0)
    p.add_argument("--nonbonded-weight", type=float, default=8.0)
    p.add_argument("--nonbonded-margin", type=float, default=0.05)
    p.add_argument("--nonbonded-width", type=float, default=0.08)
    p.add_argument("--angular-site-z-max", type=float, default=3.0,
                   help="Maximum per-site mean angular error in learned sigma units")
    p.add_argument("--angular-vector-z-max", type=float, default=4.0,
                   help="Maximum individual angular error in learned sigma units")
    p.add_argument("--construction-symmetry", choices=("full", "centers", "off"), default="full")
    p.add_argument("--coincidence-rms-max", type=float, default=0.12)
    p.add_argument("--coincidence-max-max", type=float, default=0.20)
    p.add_argument("--coincidence-weight", type=float, default=6.0)
    p.add_argument("--distortion-weight", type=float, default=0.20)
    p.add_argument("--distortion-max", type=float, default=0.35)
    p.add_argument("--framework-weight", type=float, default=3.0)
    p.add_argument("--framework-restraint-weight", type=float, default=40.0)
    p.add_argument("--framework-keep", type=int, default=2)
    p.add_argument("--framework-patience", type=int, default=12,
                   help="Stop a Ti-framework branch after this many non-improving steps")
    p.add_argument("--oxygen-coincidence-steps", type=int, default=80)
    p.add_argument("--oxygen-contact-steps", type=int, default=80)
    p.add_argument("--oxygen-assigned-fraction-min", type=float, default=0.95)
    p.add_argument("--oxygen-screen-rms-max", type=float, default=0.60,
                   help="Relaxed O-coincidence RMS gate before final polish")
    p.add_argument("--oxygen-screen-max-max", type=float, default=1.00,
                   help="Relaxed maximum O-port deviation gate before final polish")
    p.add_argument("--assignment-refresh", type=int, default=0,
                   help="Steps between topology rebuilds inside a stage; 0 keeps one topology per stage (fast default)")
    p.add_argument("--progress-every", type=int, default=20,
                   help="Write one compact manager status line every N completed attempts")
    p.add_argument("--verbose-workers", action="store_true",
                   help="Print per-worker optimization-stage heartbeats")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-folder", default="generated_xn_v12")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_thread_limits()
    model = XNModel(args.chemistry_model)
    counts, resolved_targets = resolve_targets(args.target, model)
    construction_plan = model.construction_plan(counts)
    construction_counts = construction_plan["construction_counts"]
    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    ngpu = visible if args.ngpu == 0 else int(args.ngpu)
    if ngpu > visible: raise ValueError(f"Requested {ngpu} GPUs, only {visible} visible")
    ncpu = _cpu_count()
    # Each persistent GPU optimization process needs host-side Python/launch
    # capacity.  Proposal workers use only CPUs left after reserving one logical
    # CPU per GPU worker.  On GPU-dense allocations this intentionally becomes
    # one proposal worker; request more --cpus-per-task to increase it.
    auto_proposal_workers = max(1, ncpu - max(1, ngpu)) if ngpu > 0 else max(1, ncpu)
    proposal_workers = int(args.proposal_workers) if args.proposal_workers > 0 else auto_proposal_workers
    config = dict(starts=args.starts, screen_steps=args.screen_steps,
        refine_steps=args.refine_steps, polish_steps=args.polish_steps, lr=args.builder_lr,
        minimum_distance=args.minimum_distance, soft_temperature=args.soft_temperature,
        port_width=args.port_width, angular_weight=args.angular_weight,
        radial_weight=args.radial_weight, overlap_weight=args.overlap_weight,
        uniqueness_weight=args.uniqueness_weight, nonbonded_weight=args.nonbonded_weight,
        nonbonded_margin=args.nonbonded_margin, nonbonded_width=args.nonbonded_width,
        angular_site_z_max=args.angular_site_z_max,
        angular_vector_z_max=args.angular_vector_z_max, assignment_refresh=args.assignment_refresh,
        target_counts=counts, construction_symmetry=args.construction_symmetry,
        coincidence_rms_max=args.coincidence_rms_max, coincidence_max_max=args.coincidence_max_max,
        coincidence_weight=args.coincidence_weight, distortion_weight=args.distortion_weight,
        distortion_max=args.distortion_max, framework_weight=args.framework_weight,
        framework_restraint_weight=args.framework_restraint_weight, framework_keep=args.framework_keep,
        framework_patience=args.framework_patience,
        oxygen_coincidence_steps=args.oxygen_coincidence_steps,
        oxygen_contact_steps=args.oxygen_contact_steps,
        oxygen_assigned_fraction_min=args.oxygen_assigned_fraction_min,
        oxygen_screen_rms_max=args.oxygen_screen_rms_max,
        oxygen_screen_max_max=args.oxygen_screen_max_max)
    # Validate worker configuration in the manager before spawning persistent
    # processes.  This prevents a missing constructor option from killing every
    # GPU worker while the manager continues waiting on dead queues.
    import inspect
    ctor = inspect.signature(XNBuilder.__init__)
    required = {
        name for name, parameter in ctor.parameters.items()
        if name not in {"self", "model", "device"}
        and parameter.default is inspect.Parameter.empty
    }
    missing = sorted(required.difference(config))
    unexpected = sorted(set(config).difference(ctor.parameters))
    if missing or unexpected:
        raise RuntimeError(
            "Invalid XNBuilder worker configuration: "
            f"missing={missing}, unexpected={unexpected}"
        )

    proposal_pool = ProposalPool(proposal_workers, model.labels, construction_counts,
                                 args.max_entries_per_group, args.seed + 211)
    builder_pool = BuilderPool(ngpu, args.gpu_queue_depth, args.chemistry_model, config)
    output = Path(args.output_folder); pool_dir = output / "candidate_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    active_labels = tuple(label for label, count in counts.items() if count > 0)
    deduplicator = ConservativeDeduplicator(output / "dedup_index.jsonl", active_labels, model)
    attempts_rows=[]; accepted_rows=[]; duplicate_rows=[]
    stage_counts=Counter(ti_tested=0,ti_accepted=0,o_topology_tested=0,o_topology_accepted=0,
                         o_attachment_tested=0,o_attachment_accepted=0,strict_tested=0,strict_accepted=0)
    existing_candidate_ids = [int(r.get("candidate_id", -1)) for r in deduplicator.records]
    next_candidate_id = max(existing_candidate_ids, default=-1) + 1
    spg_stats=defaultdict(lambda: Counter(attempts=0, accepted=0, duplicates=0))
    start=time.perf_counter(); deadline=start + args.max_runtime_minutes*60.0
    next_id=0; requested=0; submitted=0; completed=0; proposal_pending=set(); gpu_inflight=set(); worker_errors=0
    max_gpu_inflight=max(1, builder_pool.workers * args.gpu_queue_depth)
    max_proposals=max(proposal_workers, max_gpu_inflight*2)
    stop_new=False
    last_manager_status = start
    print("--- Juliette multistage Ti/O generator v28 ---", flush=True)
    print(
        f"Targets: {args.target}; resolved={counts}; construction={construction_counts}; "
        f"mode={construction_plan['mode']}; symmetry={args.construction_symmetry}; "
        f"atoms={model.physical_count(counts)}; sample={args.sample}; "
        f"allocated_CPU={ncpu}; proposal_workers={proposal_workers}; GPU={builder_pool.workers}",
        flush=True,
    )
    try:
        while len(accepted_rows) < args.sample and completed < args.max_attempts:
            now=time.perf_counter()
            if now >= deadline: stop_new=True
            while (not stop_new and requested < args.max_attempts and
                   len(proposal_pending) < max_proposals):
                proposal_pool.request(next_id); proposal_pending.add(next_id)
                next_id += 1; requested += 1
            # Drain prepared proposals into GPU queue while capacity exists.
            while len(gpu_inflight) < max_gpu_inflight:
                try: msg=proposal_pool.results.get_nowait()
                except queue.Empty: break
                tid=int(msg["task_id"]); proposal_pending.discard(tid)
                if msg.get("error"):
                    attempts_rows.append({"attempt_id":tid,"stage":"proposal_error","error":msg["error"]})
                    completed += 1; continue
                task={**msg, "seed":deterministic_seed(args.seed, tid, msg["spg"], msg["wp_token"], msg["species_token"])}
                builder_pool.tasks.put(task); gpu_inflight.add(tid); submitted += 1
                spg_stats[int(msg["spg"])]["attempts"] += 1
            try:
                event=builder_pool.results.get(timeout=1.0)
            except queue.Empty:
                now = time.perf_counter()
                if now - last_manager_status >= 30.0:
                    print(
                        f"Manager: accepted={len(accepted_rows)}/{args.sample}; completed={completed}; "
                        f"Ti={stage_counts['ti_accepted']}/{stage_counts['ti_tested']}; "
                        f"Otop={stage_counts['o_topology_accepted']}/{stage_counts['o_topology_tested']}; "
                        f"Oattach={stage_counts['o_attachment_accepted']}/{stage_counts['o_attachment_tested']}; "
                        f"proposal_pending={len(proposal_pending)}; gpu_inflight={len(gpu_inflight)}; "
                        f"elapsed={(now-start)/60.0:.1f} min",
                        flush=True,
                    )
                    last_manager_status = now
                if stop_new and not gpu_inflight and not proposal_pending: break
                continue
            if event["kind"] == "heartbeat":
                if args.verbose_workers:
                    print(
                        f"Worker {event['worker_id']} task {event['task_id']}: "
                        f"{event['stage']} {event['step']}/{event['total']} "
                        f"best={event['best_loss']:.5g}",
                        flush=True,
                    )
                continue
            task=event["task"]; tid=int(task["task_id"]); gpu_inflight.discard(tid); completed += 1
            for row in event.get("attempts",[]):
                attempts_rows.append({"attempt_id":tid,"spg":task["spg"],"wp_token":task["wp_token"],
                    "species_token":task["species_token"],**row})
                stage=row.get("stage")
                if stage=="ti_framework_screen":
                    stage_counts["ti_tested"] += 1
                    stage_counts["ti_accepted"] += int(bool(row.get("ti_framework_accepted",False)))
                elif stage=="o_topology_screen":
                    stage_counts["o_topology_tested"] += 1
                    stage_counts["o_topology_accepted"] += int(bool(row.get("o_topology_accepted",False)))
                elif stage=="o_attachment_screen":
                    stage_counts["o_attachment_tested"] += 1
                    stage_counts["o_attachment_accepted"] += int(bool(row.get("o_attachment_accepted",False)))
                elif stage=="strict_final_audit":
                    stage_counts["strict_tested"] += 1
                    stage_counts["strict_accepted"] += int(bool(row.get("strict_valid",False)))
            if event.get("error"):
                worker_errors += 1
                err = str(event["error"])
                attempts_rows.append({"attempt_id":tid,"spg":task["spg"],"wp_token":task.get("wp_token"),
                    "species_token":task.get("species_token"),"stage":"worker_error","error":err})
                print(f"Worker error on attempt {tid} (spg={task['spg']}): {err}", flush=True)
                if worker_errors >= max(8, 2 * builder_pool.workers) and stage_counts["ti_tested"] == 0:
                    raise RuntimeError(
                        f"Aborting after {worker_errors} worker errors before any Ti framework was evaluated. "
                        f"First inspect {output/'attempts.csv'} for the exception."
                    )
            selected=event.get("selected")
            if selected is not None:
                dedup = deduplicator.check(selected, task["species_token"])
                if dedup["duplicate"]:
                    spg_stats[int(task["spg"])]["duplicates"] += 1
                    duplicate_rows.append({
                        "attempt_id": tid, "spg": task["spg"],
                        "wp_token": task["wp_token"], "species_token": task["species_token"],
                        "duplicate_of_candidate_id": dedup["duplicate_of_candidate_id"],
                        "matching_rms_A": dedup.get("rms_A"),
                        "matching_max_displacement_A": dedup.get("max_displacement_A"),
                        "rotation_determinant": dedup.get("rotation_determinant"),
                        "fingerprint_alerts": dedup["fingerprint_alerts"],
                        "exact_match_checks": dedup["exact_match_checks"],
                    })
                else:
                    cid=next_candidate_id + len(accepted_rows); cif=pool_dir/f"candidate_{cid:06d}_spg{task['spg']}.cif"
                    _write_cif(selected,cif)
                    deduplicator.add(cid, selected, task["species_token"], dedup["fingerprint"])
                    accepted_rows.append({"candidate_id":cid,"attempt_id":tid,"spg":task["spg"],
                        "wp_token":task["wp_token"],"species_token":task["species_token"],"cif":str(cif),
                        "loss":selected["loss"],"local_radial_mae_A":selected["local_radial_mae_A"],
                        "local_angular_site_max_deg":selected["local_angular_site_max_deg"],
                        "minimum_physical_distance_A":selected["minimum_physical_distance_A"],
                        "fingerprint_alerts":dedup["fingerprint_alerts"],
                        "exact_match_checks":dedup["exact_match_checks"]})
                    spg_stats[int(task["spg"])]["accepted"] += 1
                    print(
                        f"Accepted {len(accepted_rows)}/{args.sample}: "
                        f"spg={task['spg']} attempt={completed}",
                        flush=True,
                    )
            # Checkpoint diagnostics after every completed proposal.  These
            # files are small and must remain useful during long-running jobs.
            pd.DataFrame(attempts_rows).to_csv(output/"attempts.csv",index=False)
            pd.DataFrame(accepted_rows).to_csv(output/"accepted.csv",index=False)
            pd.DataFrame(duplicate_rows).to_csv(output/"duplicates.csv",index=False)
            if completed % args.progress_every == 0:
                elapsed_min = (time.perf_counter() - start) / 60.0
                print(
                    f"Status: accepted={len(accepted_rows)}/{args.sample}; "
                    f"attempts={completed}/{args.max_attempts}; "
                    f"Ti={stage_counts['ti_accepted']}/{stage_counts['ti_tested']}; "
                    f"O={stage_counts['o_attachment_accepted']}/{stage_counts['o_attachment_tested']}; "
                    f"active={len(gpu_inflight)}; elapsed={elapsed_min:.1f} min",
                    flush=True,
                )
        if time.perf_counter() >= deadline:
            print("Runtime deadline reached: no new proposals; all in-flight GPU jobs were drained.", flush=True)
    finally:
        proposal_pool.close(); builder_pool.close()
    pd.DataFrame(attempts_rows).to_csv(output/"attempts.csv",index=False)
    pd.DataFrame(accepted_rows).to_csv(output/"accepted.csv",index=False)
    pd.DataFrame(duplicate_rows).to_csv(output/"duplicates.csv",index=False)
    spg_rows=[]
    for spg in sorted(spg_stats):
        row=dict(spg_stats[spg]); row["spg"]=spg; row["acceptance_rate"]=row["accepted"]/max(row["attempts"],1); spg_rows.append(row)
    pd.DataFrame(spg_rows).to_csv(output/"space_group_statistics.csv",index=False)
    summary={"model":str(Path(args.chemistry_model).resolve()),"targets":resolved_targets,"resolved_counts":counts,"requested":args.sample,
        "accepted":len(accepted_rows),"duplicates_rejected":len(duplicate_rows),"attempts_completed":completed,"runtime_seconds":time.perf_counter()-start,
        "runtime_limit_reached":time.perf_counter()>=deadline,"cpu_proposal_workers":proposal_workers,
        "gpu_workers":builder_pool.workers,"visible_gpus":visible,
        "stage_counts":dict(stage_counts),
        "semantics":"unified_target_driven_symmetry_constrained_dynamic_Xn_reconciliation",
        "parallelization":"dynamic CPU proposal workers feeding persistent one-process-per-GPU optimization workers",
        "loss_kernel":"fully_vectorized_forward_and_reciprocal_X_port_matching",
        "deduplication":"fingerprint_alert_then_strict_proper_rotation_exact_match",
        "dedup_label_safe":bool(deduplicator.label_safe)}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(
        f"Finished: accepted={len(accepted_rows)}/{args.sample}; "
        f"attempts={completed}; Ti={stage_counts['ti_accepted']}/{stage_counts['ti_tested']}; "
        f"Otop={stage_counts['o_topology_accepted']}/{stage_counts['o_topology_tested']}; "
        f"Oattach={stage_counts['o_attachment_accepted']}/{stage_counts['o_attachment_tested']}; "
        f"runtime={summary['runtime_seconds'] / 60.0:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()

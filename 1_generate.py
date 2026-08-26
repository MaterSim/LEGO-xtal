#!/usr/bin/env python3
"""Juliette v78: efficient breadth-first Ti entrances + exploratory refinement.

v78 keeps the v74/v77 direct Ti3->O hypergraph constructor, topology-diverse
cover selection, strict chemistry audit, conservative exact deduplication, and safe
quotient-like canonical family accounting.  This revision keeps the useful breadth
logic from v77 while removing the two dominant inefficiencies seen in the 2000-token
benchmark.

The default crystallographic domain is SG17--230.  Globally, 95% of proposal draws
are forced toward a previously unseen exact Wyckoff entrance while unseen entrances
remain; only a 5% exploitation channel may revisit a known entrance.  A first visit
uses the cheap 16-start x 8-step scout.  Proposals that are simultaneously far from
any generic Ti3->O opportunity, incidence-deficient, and have essentially zero good
triplets may escape before any Ti token is spent; a small stochastic sentinel channel
measures false negatives.

Three expensive Ti branches are used per promoted entrance, all exploratory and
selected from max-min-diverse scout survivors.  The previously ineffective strong
branch is removed.  Learned Ti-framework radial/angular descriptors and the historical
Ti4 omission descriptor remain diagnostic-only.

There is still no preferred volume or artificial pressure.  Local shared-O chemistry
and learned nonbonded exclusions determine the cell scale.  Topology/family/symmetry
diagnostics only steer future sampling and never reject an otherwise strict output.
"""
from __future__ import annotations

import argparse
import csv
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
from scipy.optimize import linear_sum_assignment, least_squares

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
        self.nonbonded_exclusions = {}
        for row in self.raw.get("nonbonded_pair_exclusions", []):
            ei = str(row.get("element_i", row.get("species_i", "")))
            ej = str(row.get("element_j", row.get("species_j", "")))
            if not ei or not ej:
                continue
            key = tuple(sorted((ei, ej)))
            self.nonbonded_exclusions[key] = {
                "hard_min_A": float(row["hard_min_A"]),
                "soft_min_A": float(row.get("soft_min_A", row["hard_min_A"])),
                "reference_min_A": float(row.get("reference_min_A", row["hard_min_A"])),
                "source": str(row.get("selection", row.get("source", "learned_nonbonded"))),
            }
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

    def nonbonded_exclusion(self, a: str, b: str) -> dict:
        """Return the learned physical nonbonded wall for construction labels a,b."""
        ea = self.final_formula[str(a)]
        eb = self.final_formula[str(b)]
        key = tuple(sorted((ea, eb)))
        if key not in self.nonbonded_exclusions:
            raise KeyError(
                f"Missing learned physical nonbonded exclusion for {ea}-{eb}. "
                "Rerun the v65-compatible 0_learn.py before generation."
            )
        return self.nonbonded_exclusions[key]

    def nonbonded_hard_min(self, a: str, b: str) -> float:
        return float(self.nonbonded_exclusion(a, b)["hard_min_A"])

    def nonbonded_soft_min(self, a: str, b: str) -> float:
        return float(self.nonbonded_exclusion(a, b)["soft_min_A"])

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


def parse_space_groups(text: str) -> list[int]:
    value = str(text).strip().lower()
    if value in {"all", "1-230"}:
        return list(range(1, 231))
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            a, b = int(left), int(right)
            if a > b:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    if not out or out[0] < 1 or out[-1] > 230:
        raise ValueError(f"Invalid --space-groups {text!r}; values must be in 1..230")
    return out


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
                if int(group[wp].get_dof()) == 0 and wp in selected:
                    continue
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
    BASE_OCT = np.asarray([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.float32)
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
                 oxygen_screen_rms_max: float, oxygen_screen_max_max: float,
                 octahedral_branches: int, octahedron_prepare_steps: int,
                 octahedron_match_steps: int, octahedron_cluster_steps: int,
                 floating_coincidence_sigma: float, floating_cluster_tolerance: float,
                 ti_registry_path: str = "", oxygen_proposal_oversample: int = 4,
                 oxygen_proposal_descriptor_tol: float = 0.025,
                 oxygen_basin_prune_every: int = 25,
                 framework_memory_path: str = "", framework_basin_memory_path: str = "",
                 framework_intelligent_keep: int = 2, framework_memory_k: int = 8):
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
        if self.plan.get("mode") == "shared_site":
            if not self.model.nonbonded_exclusions:
                raise ValueError(
                    "v78 shared-site generation requires chemistry_model.json with "
                    "nonbonded_pair_exclusions. Rerun the matching v64+ 0_learn.py."
                )
            active_labels = [x for x, n in self.target_counts.items() if int(n) > 0]
            active_elements = sorted({self.model.final_formula[x] for x in active_labels})
            missing = []
            for i, ea in enumerate(active_elements):
                for eb in active_elements[i:]:
                    if tuple(sorted((ea, eb))) not in self.model.nonbonded_exclusions:
                        missing.append(f"{ea}-{eb}")
            if missing:
                raise ValueError(
                    "v78 chemistry model is missing learned physical nonbonded exclusions for "
                    + ", ".join(missing) + ". Rerun the matching v64+ 0_learn.py."
                )
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
        self.octahedral_branches = max(1, int(octahedral_branches))
        self.octahedron_prepare_steps = max(1, int(octahedron_prepare_steps))
        self.octahedron_match_steps = max(1, int(octahedron_match_steps))
        self.octahedron_cluster_steps = max(1, int(octahedron_cluster_steps))
        self.floating_coincidence_sigma = max(float(floating_coincidence_sigma), 1.0e-3)
        self.floating_cluster_tolerance = max(float(floating_cluster_tolerance), 1.0e-3)
        self.ti_registry_path = str(ti_registry_path)
        self.oxygen_proposal_oversample = max(1, int(oxygen_proposal_oversample))
        self.oxygen_proposal_descriptor_tol = max(float(oxygen_proposal_descriptor_tol), 1.0e-6)
        self.oxygen_basin_prune_every = max(0, int(oxygen_basin_prune_every))
        self.framework_memory_path = str(framework_memory_path)
        self.framework_basin_memory_path = str(framework_basin_memory_path)
        self.framework_intelligent_keep = max(1, int(framework_intelligent_keep))
        self.framework_memory_k = max(1, int(framework_memory_k))
        # v75 direct hypergraph O construction.  Keep these as stable scientific defaults
        # rather than expanding the public CLI during the first benchmark.
        self.analytic_o_max_assignments = 2
        self.analytic_o_radial_slack_A = 0.06
        self.analytic_o_nonbonded_slack_A = 0.03
        # v75 direct Ti3->O hypergraph construction.  These are deliberately
        # internal defaults: the public production interface remains compact.
        self.hypergraph_max_covers = 4
        # v74 enumerates more mathematically valid covers, then selects a small
        # max-min subset by the actual Ti3->O connectivity rather than accepting
        # the first few chemistry-ranked solutions.  This is a ranking/budget
        # mechanism only; topology fingerprints never reject a strict candidate.
        self.hypergraph_enumerate_covers = 48
        self.hypergraph_site_enumerate_covers = 64
        self.hypergraph_neighbor_cap = 28
        self.hypergraph_candidate_cap = 720
        self.hypergraph_per_parent_cap = 120
        self.hypergraph_search_node_cap = 30000
        self.hypergraph_symmetry_match_A = 0.12
        self.hypergraph_pair_slack_A = 0.12
        self.hypergraph_angle_z_soft_max = max(8.0, self.angular_vector_z_max + 2.0)
        # v75 near-hypergraph framework steering.  These are intentionally internal:
        # they define a chemistry surrogate, not user-tuned phase targets.
        self.framework_triplet_neighbor_k = 12
        self.framework_triplet_good_score = 4.0
        self.framework_triplet_good_width = 0.80
        # v75 retains branch-specific chemistry steering rather than four copies of
        # one attractor. Mode order: strong, medium, exploratory, alternate-triplet.
        self.framework_branch_mode_names = ("strong", "medium", "explore", "alternate")
        self.framework_branch_pair_weights = (0.45, 0.38, 0.22, 0.40)
        self.framework_branch_triplet_weights = (1.35, 0.85, 0.30, 1.05)
        self.framework_branch_incidence_weights = (2.50, 1.50, 0.45, 1.75)
        self.framework_branch_diversity_weight = 0.40
        self.framework_branch_diversity_sigma = 0.060
        # v78 keeps the learned Ti-framework descriptors diagnostic-only.  The broad-entrance
        # experiment deliberately applies no learned-framework force.  Radial/angular q90
        # values are still logged so breadth can be audited after the run without
        # pre-judging unfamiliar but locally valid Ti frameworks.
        self.framework_prior_envelope_weights = (0.0, 0.0, 0.0, 0.0)
        self.framework_prior_angular_factor = 0.50
        self.framework_prior_catastrophic_tail_factor = 0.05
        # Backward/default scalar values are retained for code paths that do not
        # explicitly carry a branch mode (cheap screening, diagnostics, O polish).
        self.framework_pair_tension_weight = self.framework_branch_pair_weights[0]
        self.framework_triplet_tension_weight = self.framework_branch_triplet_weights[0]
        self.framework_incidence_deficit_weight = self.framework_branch_incidence_weights[0]
        # Site-level covers are exploratory.  Once a full symmetry-orbit cover
        # exists it always wins; fallback branches are capped to avoid flooding the
        # pool with symmetry-broken variants of the same connectivity basin.
        self.hypergraph_site_fallback_max_covers = 4
        # Per-worker incremental caches. Other workers communicate by append-only JSONL.
        self._chem_memory_rows = []
        self._chem_memory_offset = 0
        self._basin_memory_offset = 0
        self._basin_stats = {}
        # Assigned by the persistent builder worker. The synchronized counter is
        # shared by every GPU process and makes the Ti-token budget global.
        self.ti_token_counter = None
        self.ti_token_budget = 20000
        self._base_oct = torch.as_tensor(self.BASE_OCT, dtype=torch.float32, device=self.device)
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

    def _framework_near_hypergraph_loss(self, template: dict, cart: torch.Tensor,
                                        dist: torch.Tensor, chemistry_cost: torch.Tensor,
                                        share_max: torch.Tensor, target_modes: torch.Tensor,
                                        sigma_modes: torch.Tensor, self_zero: torch.Tensor,
                                        branch_modes: torch.Tensor | None = None):
        """Differentiable near-miss Ti3->O coverage surrogate.

        The exact hypergraph builder is discrete and therefore cannot pull a nearly
        viable Ti framework toward coverability.  This surrogate uses the same
        learned O-centred chemistry but remains differentiable with respect to Ti
        Wyckoff and cell variables.  For each Ti, a small set of its best periodic
        co-owner images is selected by pair chemistry; all neighbour pairs among that
        set define candidate Ti3 triplets.  The loss asks for at least the learned Ti
        coordination number of low-cost triplets around every Ti.

        Selection indices are piecewise constant (top-k), while the selected scores
        and coordinates remain differentiable.  No experimental framework, volume,
        or Ti--Ti RDF target enters this loss.
        """
        bsz, ncentre, _nj, nshift = dist.shape
        if ncentre <= 0:
            z = dist.sum((1, 2, 3)) * 0.0
            return z, z, z, z
        owner_flat = torch.arange(ncentre, dtype=torch.long, device=self.device).repeat_interleave(nshift)
        triplet_losses = []
        deficit_losses = []
        good_counts = []
        labels = tuple(template["expanded_labels"])
        if branch_modes is None:
            branch_modes = torch.zeros((bsz,), dtype=torch.long, device=self.device)
        else:
            branch_modes = branch_modes.to(device=self.device, dtype=torch.long).reshape(-1)
            if len(branch_modes) != bsz:
                raise ValueError("branch_modes length must match framework batch")

        for i in range(ncentre):
            pair_score = chemistry_cost[:, i].reshape(bsz, -1)
            pair_dist = dist[:, i].reshape(bsz, -1)
            share_limit = share_max[i, :, None].expand(ncentre, nshift).reshape(1, -1)
            # A smooth beyond-sphere bound keeps far-away images from occupying the
            # neighbour shortlist merely because one broad angular mode fits them.
            beyond = torch.nn.functional.softplus((pair_dist - share_limit) / 0.12).pow(2)
            guided = pair_score + 2.0 * beyond
            k = min(int(self.framework_triplet_neighbor_k), int(guided.shape[-1]))
            if k < 2:
                z = guided[:, 0] * 0.0 + 1.0
                triplet_losses.append(z); deficit_losses.append(z); good_counts.append(z * 0.0)
                continue
            vals, idx = torch.topk(guided, k=k, dim=-1, largest=False)
            vec_flat = cart[:, i].reshape(bsz, -1, 3)
            vec = torch.gather(vec_flat, 1, idx[..., None].expand(-1, -1, 3))
            owners = torch.gather(owner_flat[None].expand(bsz, -1), 1, idx)
            aa, bb = torch.triu_indices(k, k, offset=1, device=self.device)
            va, vb = vec[:, aa], vec[:, bb]
            dab = torch.linalg.norm(va - vb, dim=-1).clamp_min(1.0e-7)
            oa, ob = owners[:, aa], owners[:, bb]
            tab = target_modes[oa, ob]
            sab = sigma_modes[oa, ob].clamp_min(1.0e-3)
            ab_cost = ((dab[..., None] - tab) / sab).pow(2).amin(-1)
            ab_share = share_max[oa, ob]
            ab_cost = ab_cost + 2.0 * torch.nn.functional.softplus((dab - ab_share) / 0.12).pow(2)
            # Three pair distances can each look plausible while still being
            # impossible to intersect at one O.  For the TiO2-like shared-site
            # chemistry used here, the triangle circumradius provides a strong,
            # differentiable near-miss test: a common O sphere cannot exist when
            # R_circ exceeds the available Ti--O radius.  For mixed parent labels
            # we use the mean learned radial support as a conservative surrogate.
            la = torch.linalg.norm(va, dim=-1).clamp_min(1.0e-7)
            lb = torch.linalg.norm(vb, dim=-1).clamp_min(1.0e-7)
            twice_area = torch.linalg.norm(torch.cross(va, vb, dim=-1), dim=-1).clamp_min(1.0e-5)
            circum = la * lb * dab / (2.0 * twice_area)
            child = tuple(self.plan.get("children", ()))[0]
            radial_max = torch.as_tensor(
                [float(self.model.pair(label, child).sampling_max) for label in labels],
                dtype=dist.dtype, device=self.device)
            available_r = (radial_max[i] + radial_max[oa] + radial_max[ob]) / 3.0
            sphere_miss = torch.nn.functional.softplus((circum - available_r) / 0.08).pow(2)
            raw = (vals[:, aa] + vals[:, bb] + ab_cost) / 3.0 + 3.0 * sphere_miss
            raw = torch.nan_to_num(raw, nan=1.0e4, posinf=1.0e4, neginf=1.0e4).clamp(0.0, 1.0e4)
            target_cn = max(1, int(self.model.template_cn[labels[i]]))
            nbest = min(target_cn, int(raw.shape[-1]))
            # Most branches optimize the best CN triplets.  The alternate branch
            # intentionally uses a shifted window within the low-cost pool, so a
            # second chemically plausible connectivity basin can be pulled down
            # instead of every row chasing the identical nearest hyperedges.
            pool_k = min(int(raw.shape[-1]), max(nbest, nbest + max(1, target_cn // 2)))
            ranked = torch.topk(raw, k=pool_k, dim=-1, largest=False).values
            base_raw = ranked[:, :nbest]
            alt_start = min(max(1, target_cn // 2), max(pool_k - nbest, 0))
            alt_raw = ranked[:, alt_start:alt_start + nbest]
            if alt_raw.shape[1] != nbest:
                alt_raw = base_raw
            alternate = (branch_modes == 3)[:, None]
            best_raw = torch.where(alternate, alt_raw, base_raw)
            triplet_losses.append(torch.log1p(best_raw).mean(-1))
            # Coverage is assessed on the branch-selected CN triplets.  This keeps
            # the exploratory/alternate directions chemically meaningful instead of
            # rewarding a cloud of fractional mediocre combinations.
            goodness = torch.sigmoid((float(self.framework_triplet_good_score) - best_raw)
                                     / max(float(self.framework_triplet_good_width), 1.0e-3))
            good = goodness.sum(-1)
            good_counts.append(good)
            deficit = torch.relu(float(target_cn) - good) / float(target_cn)
            deficit_losses.append(deficit.pow(2))

        triplet_tension = torch.stack(triplet_losses, dim=1).mean(1)
        incidence_deficit = torch.stack(deficit_losses, dim=1).mean(1)
        good_stack = torch.stack(good_counts, dim=1)
        min_good_count = good_stack.amin(1)
        mean_good_count = good_stack.mean(1)
        return triplet_tension, incidence_deficit, min_good_count, mean_good_count

    def _framework_learned_envelope(self, template: dict, cart: torch.Tensor,
                                      dist: torch.Tensor, self_zero: torch.Tensor):
        """Broad learned Ti-framework envelope; flat inside the learned support.

        The learner stores, for each parent construction species, the ranked nearest
        same-element periodic-neighbour distances plus sorted Ti--Ti--Ti angles
        resolved by radial-shell pair.  v75 evaluates those descriptors site by site.
        Nothing is attracted to a mean while the robust q90 z-score remains below the
        learned ``score_q90_max``.  Beyond that boundary a differentiable soft
        restoring force grows with normalized q90 excess.  A very weak >2x-threshold
        tail term prevents a single catastrophic local outlier from hiding below q90.

        This is deliberately an *admissible-envelope* prior, not a phase/RDF target.
        """
        bsz, ncentre, _nj, _nshift = dist.shape
        anchor = dist.sum((1, 2, 3)) * 0.0
        labels = tuple(template.get("expanded_labels", ()))
        radial_chunks = []
        angular_chunks = []
        thresholds = []
        rank_z_chunks: dict[int, list[torch.Tensor]] = defaultdict(list)
        rank_d_chunks: dict[int, list[torch.Tensor]] = defaultdict(list)
        shell_angle_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)

        for i in range(min(ncentre, len(labels))):
            label = str(labels[i])
            fm = self.model.framework_models.get(label, {})
            k = int(fm.get("neighbor_count", 0) or 0)
            rmean = np.asarray(fm.get("radial_mean_A", []), float)
            rsig = np.asarray(fm.get("radial_sigma_A", []), float)
            if k <= 0 or len(rmean) != k or len(rsig) != k:
                continue
            thresholds.append(float(fm.get("score_q90_max", 3.5)))
            flat_d = dist[:, i].reshape(bsz, -1)
            flat_v = cart[:, i].reshape(bsz, -1, 3)
            if flat_d.shape[1] < k:
                continue
            nearest_d, nearest_idx = torch.topk(flat_d, k=k, dim=-1, largest=False, sorted=True)
            nearest_v = torch.gather(flat_v, 1, nearest_idx[..., None].expand(-1, -1, 3))
            rmean_t = torch.as_tensor(rmean, dtype=dist.dtype, device=self.device)[None, :]
            rsig_t = torch.as_tensor(np.maximum(rsig, 1.0e-3), dtype=dist.dtype, device=self.device)[None, :]
            rz = torch.abs(nearest_d - rmean_t) / rsig_t
            radial_chunks.append(rz)
            for rank in range(k):
                rank_z_chunks[int(rank)].append(rz[:, rank])
                rank_d_chunks[int(rank)].append(nearest_d[:, rank])

            unit = nearest_v / torch.linalg.norm(nearest_v, dim=-1, keepdim=True).clamp_min(1.0e-7)
            for group in fm.get("angular_shell_pair_groups", []):
                pairs = [(int(a), int(b)) for a, b in group.get("neighbor_rank_pairs", [])]
                amean = np.asarray(group.get("angular_mean_deg", []), float)
                asig = np.asarray(group.get("angular_sigma_deg", []), float)
                if not pairs or len(amean) != len(pairs) or len(asig) != len(pairs):
                    continue
                vals = []
                valid_group = True
                for a, b in pairs:
                    if a >= k or b >= k:
                        valid_group = False
                        break
                    cosang = (unit[:, a] * unit[:, b]).sum(-1).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
                    vals.append(torch.rad2deg(torch.acos(cosang)))
                if not valid_group or not vals:
                    continue
                # The learner sorts the angles *within each shell-pair group* before
                # taking robust medians, so reproduce exactly that representation.
                obs = torch.sort(torch.stack(vals, dim=1), dim=1).values
                amean_t = torch.as_tensor(amean, dtype=dist.dtype, device=self.device)[None, :]
                asig_t = torch.as_tensor(np.maximum(asig, 1.0e-3), dtype=dist.dtype, device=self.device)[None, :]
                az = torch.abs(obs - amean_t) / asig_t
                angular_chunks.append(az)
                sp = group.get("shell_pair", [0, 0])
                key = f"{int(sp[0])}{int(sp[1])}"
                shell_angle_chunks[key].append(az)

        if not radial_chunks:
            return anchor, {
                "framework_prior_radial_q90": anchor,
                "framework_prior_radial_z_max": anchor,
                "framework_prior_angular_q90": anchor,
                "framework_prior_angular_z_max": anchor,
                "framework_prior_envelope_threshold": torch.full_like(anchor, 3.5),
                "framework_prior_inside_envelope": torch.ones_like(anchor),
            }

        radial_all = torch.cat(radial_chunks, dim=1)
        radial_q90 = torch.quantile(radial_all, 0.90, dim=1)
        radial_z_max = radial_all.amax(1)
        if angular_chunks:
            angular_all = torch.cat(angular_chunks, dim=1)
            angular_q90 = torch.quantile(angular_all, 0.90, dim=1)
            angular_z_max = angular_all.amax(1)
        else:
            angular_q90 = anchor
            angular_z_max = anchor

        threshold = float(np.median(thresholds)) if thresholds else 3.5
        threshold = max(threshold, 1.0e-3)
        t = torch.full_like(anchor, threshold)
        radial_excess = torch.relu((radial_q90 - t) / t).pow(2)
        angular_excess = torch.relu((angular_q90 - t) / t).pow(2)
        radial_tail = torch.relu((radial_z_max - 2.0 * t) / t).pow(2)
        angular_tail = torch.relu((angular_z_max - 2.0 * t) / t).pow(2)
        envelope = (radial_excess
                    + float(self.framework_prior_angular_factor) * angular_excess
                    + float(self.framework_prior_catastrophic_tail_factor) * (radial_tail + angular_tail))

        detail = {
            "framework_prior_radial_q90": radial_q90,
            "framework_prior_radial_z_max": radial_z_max,
            "framework_prior_angular_q90": angular_q90,
            "framework_prior_angular_z_max": angular_z_max,
            "framework_prior_envelope_threshold": t,
            "framework_prior_inside_envelope": ((radial_q90 <= t) & (angular_q90 <= t)).to(dist.dtype),
        }
        for rank in range(6):
            zrows = rank_z_chunks.get(rank, [])
            drows = rank_d_chunks.get(rank, [])
            detail[f"framework_prior_radial_rank{rank+1}_z_mean"] = (
                torch.stack(zrows, dim=1).mean(1) if zrows else anchor)
            detail[f"framework_prior_radial_rank{rank+1}_distance_A"] = (
                torch.stack(drows, dim=1).mean(1) if drows else anchor)
        for key in ("00", "01", "11"):
            rows = shell_angle_chunks.get(key, [])
            if rows:
                merged = torch.cat(rows, dim=1)
                detail[f"framework_prior_angular_shell{key}_q90"] = torch.quantile(merged, 0.90, dim=1)
            else:
                detail[f"framework_prior_angular_shell{key}_q90"] = anchor
        return envelope, detail

    def _framework_loss(self, template: dict, prefix: torch.Tensor,
                        branch_modes: torch.Tensor | None = None,
                        diversity_weight: float = 0.0):
        """v75 chemistry-driven Ti objective plus a flat learned-framework envelope.

        Pairwise shared-O compatibility and the near-hypergraph surrogate still
        drive coverability.  The learned Ti framework now contributes only when its
        ranked radial/angular q90 descriptors leave the broad learned support.
        Inside that support the framework prior is exactly zero, preserving global
        topological exploration.  No target volume or density is introduced.
        """
        abc, angles, cell, z2_raw, centres = self._framework_geometry(template, prefix)
        bsz, ncentre, _ = centres.shape
        grad_anchor = prefix.sum(dim=1) * 0.0
        delta = centres[:, None, :, None, :] - centres[:, :, None, None, :] + self.shift_t[None, None, None, :, :]
        cart = torch.einsum("bijnk,bkl->bijnl", delta, cell)
        dist = torch.linalg.norm(cart, dim=-1).clamp_min(1.0e-7)
        eye = torch.eye(ncentre, dtype=torch.bool, device=self.device)[None, :, :, None]
        zero = torch.zeros(len(SHIFTS), dtype=torch.bool, device=self.device); zero[ZERO_SHIFT] = True
        self_zero = eye & zero[None, None, None, :]
        dist = dist.masked_fill(self_zero, 1.0e6)

        labels = tuple(template["expanded_labels"])
        child_labels = tuple(self.plan.get("children", ()))
        if len(child_labels) != 1:
            raise ValueError("shared-site framework chemistry tension requires exactly one child construction label")
        child = child_labels[0]

        hard_floor = torch.zeros((ncentre, ncentre), dtype=cell.dtype, device=self.device)
        soft_floor = torch.zeros_like(hard_floor)
        share_max = torch.zeros_like(hard_floor)
        chemistry_cost = torch.full_like(dist, 1.0e6)
        child_angles = np.asarray(self.model.template_angles[child], float)
        child_angle_sigma = np.maximum(np.asarray(self.model.template_angle_sigma[child], float), 1.0e-3)
        nmodes = max(1, len(child_angles))
        target_modes = torch.zeros((ncentre, ncentre, nmodes), dtype=cell.dtype, device=self.device)
        sigma_modes = torch.ones_like(target_modes)

        # Convert the learned O-centred Ti--O--Ti angular/radial modes to Ti--Ti
        # separation modes.  These are local-chemistry consequences, not a learned
        # long-range Ti framework prior.
        for i, li in enumerate(labels):
            chi = self.model.pair(li, child)
            for j, lj in enumerate(labels):
                chj = self.model.pair(lj, child)
                hard_floor[i, j] = float(self.model.nonbonded_hard_min(li, lj))
                soft_floor[i, j] = float(self.model.nonbonded_soft_min(li, lj))
                share_max[i, j] = float(chi.sampling_max + chj.sampling_max)
                mode_costs = []
                for imode, (theta_deg, sigma_deg) in enumerate(zip(child_angles, child_angle_sigma)):
                    theta = math.radians(float(theta_deg))
                    ri, rj = float(chi.mu), float(chj.mu)
                    target = math.sqrt(max(ri*ri + rj*rj - 2.0*ri*rj*math.cos(theta), 1.0e-8))
                    dd_dtheta = abs(ri*rj*math.sin(theta) / max(target, 1.0e-6))
                    sigma_d = math.sqrt(float(chi.sigma)**2 + float(chj.sigma)**2
                                        + (dd_dtheta * math.radians(float(sigma_deg)))**2)
                    sigma_d = min(max(sigma_d, 0.12), 0.55)
                    target_modes[i, j, imode] = float(target)
                    sigma_modes[i, j, imode] = float(sigma_d)
                    mode_costs.append(((dist[:, i, j, :] - target) / sigma_d).pow(2))
                if mode_costs:
                    chemistry_cost[:, i, j, :] = torch.stack(mode_costs, dim=-1).amin(-1)

        hard_floor4 = hard_floor[None, :, :, None]
        soft_floor4 = soft_floor[None, :, :, None]
        share_max4 = share_max[None, :, :, None]
        pair_pen = torch.nn.functional.softplus((soft_floor4 - dist) / 0.08).pow(2)
        pair_pen = pair_pen.masked_fill(self_zero, 0.0)
        repulsive = pair_pen.sum((1, 2, 3)) / max(ncentre, 1)

        hard_margin = (dist - hard_floor4).masked_fill(self_zero, 1.0e6)
        min_pair_margin = hard_margin.amin((1, 2, 3))
        hard = torch.relu((-min_pair_margin) / 0.03).pow(4) * 40.0
        min_distance = dist.amin((1, 2, 3))

        shareable = (dist <= share_max4) & (~self_zero)
        shareable_count = shareable.sum((2, 3)).to(torch.float32)
        min_contact_count = shareable_count.amin(1)
        contact_cutoff = float(torch.max(share_max).detach().cpu())

        # Retain a weak pairwise guide so very poor starting cells still have a
        # smooth path toward the local-chemistry region.
        chem_flat = chemistry_cost.masked_fill(self_zero, 1.0e6).reshape(bsz, ncentre, -1)
        k = min(2, chem_flat.shape[-1])
        best_chem = torch.topk(chem_flat, k=k, dim=-1, largest=False).values
        chemistry_tension = best_chem.mean((1, 2))

        if branch_modes is None:
            branch_modes = torch.zeros((bsz,), dtype=torch.long, device=self.device)
        else:
            branch_modes = branch_modes.to(device=self.device, dtype=torch.long).reshape(-1)
            if len(branch_modes) != bsz:
                raise ValueError("branch_modes length must match framework batch")
        triplet_tension, incidence_deficit, min_good_triplets, mean_good_triplets = \
            self._framework_near_hypergraph_loss(
                template, cart, dist, chemistry_cost, share_max,
                target_modes, sigma_modes, self_zero, branch_modes=branch_modes)

        framework_prior_envelope, framework_prior_detail = self._framework_learned_envelope(
            template, cart, dist, self_zero)

        volume = torch.abs(torch.linalg.det(cell)) / max(ncentre, 1)  # diagnostic only
        aspect = abc.amax(1) / abc.amin(1).clamp_min(1.0e-4)
        shape_guard = torch.relu(aspect - 8.0).pow(2)
        qdet = torch.abs(torch.linalg.det(cell)) / torch.prod(abc, dim=1).clamp_min(1.0e-8)
        metric_guard = torch.relu(0.12 - qdet).pow(2) * 100.0
        metric_guard = metric_guard + torch.relu(1.0e-4 - z2_raw / abc[:, 2].square().clamp_min(1.0e-8)).pow(2) * 1.0e6

        pair_table = torch.as_tensor(self.framework_branch_pair_weights, dtype=cell.dtype, device=self.device)
        trip_table = torch.as_tensor(self.framework_branch_triplet_weights, dtype=cell.dtype, device=self.device)
        inc_table = torch.as_tensor(self.framework_branch_incidence_weights, dtype=cell.dtype, device=self.device)
        safe_modes = branch_modes.clamp(0, len(self.framework_branch_mode_names) - 1)
        pair_weight = pair_table[safe_modes]
        triplet_weight = trip_table[safe_modes]
        incidence_weight = inc_table[safe_modes]
        prior_table = torch.as_tensor(self.framework_prior_envelope_weights, dtype=cell.dtype, device=self.device)
        prior_weight = prior_table[safe_modes]

        # Differentiable within-batch Ti-framework repulsion.  It uses only a
        # short nearest-periodic-Ti spectrum plus scale-free cell metrics; it does
        # not encode any experimental framework target and never rejects a branch.
        batch_diversity_penalty = torch.zeros_like(repulsive)
        if float(diversity_weight) > 0.0 and bsz > 1:
            # Exclude all periodic images of the same crystallographic Ti owner;
            # diversity should separate inter-Ti frameworks, not merely cell-vector
            # self images.
            dflat = dist.masked_fill(eye, 1.0e6).reshape(bsz, -1)
            kd = min(16, int(dflat.shape[-1]))
            dspec = torch.topk(dflat, k=kd, dim=-1, largest=False).values / 4.0
            desc = torch.cat([
                dspec,
                torch.log(volume.clamp_min(1.0e-6))[:, None] / 4.0,
                torch.log(aspect.clamp_min(1.0))[:, None],
                qdet[:, None],
            ], dim=1)
            d2 = (desc[:, None, :] - desc[None, :, :]).pow(2).mean(-1)
            sigma2 = max(float(self.framework_branch_diversity_sigma), 1.0e-4) ** 2
            similarity = torch.exp(-0.5 * d2 / sigma2)
            eye_b = torch.eye(bsz, dtype=torch.bool, device=self.device)
            similarity = similarity.masked_fill(eye_b, 0.0)
            batch_diversity_penalty = similarity.sum(1) / max(bsz - 1, 1)

        total = (repulsive
                 + pair_weight * chemistry_tension
                 + triplet_weight * triplet_tension
                 + incidence_weight * incidence_deficit
                 + prior_weight * framework_prior_envelope
                 + float(diversity_weight) * batch_diversity_penalty
                 + hard + 0.01 * shape_guard + metric_guard + grad_anchor)
        total = torch.nan_to_num(total, nan=1e9, posinf=1e9, neginf=1e9)
        site_rep = pair_pen.sum((2, 3))
        q90 = torch.quantile(site_rep, 0.90, dim=1) if ncentre > 1 else site_rep[:, 0]
        return total, {
            "framework_radial_loss": repulsive,
            "framework_chemistry_tension_loss": chemistry_tension,
            "framework_near_triplet_loss": triplet_tension,
            "framework_incidence_deficit_loss": incidence_deficit,
            "framework_branch_mode": safe_modes.to(dtype=cell.dtype),
            "framework_branch_pair_weight": pair_weight,
            "framework_branch_triplet_weight": triplet_weight,
            "framework_branch_incidence_weight": incidence_weight,
            "framework_prior_envelope_weight": prior_weight,
            "framework_prior_envelope_loss": framework_prior_envelope,
            **framework_prior_detail,
            "framework_batch_diversity_penalty": batch_diversity_penalty,
            "framework_min_near_triplet_good_count": min_good_triplets,
            "framework_mean_near_triplet_good_count": mean_good_triplets,
            "framework_angular_loss": grad_anchor,
            "framework_connectivity_loss": triplet_tension + incidence_deficit,
            "framework_vacuum_loss": torch.zeros_like(total),
            "framework_min_ti_contact_count": min_contact_count,
            "framework_ti_contact_cutoff_A": torch.full_like(min_contact_count, contact_cutoff),
            "framework_score_q90": q90,
            "framework_min_distance_A": min_distance,
            "framework_min_nonbonded_margin_A": min_pair_margin,
            "framework_aspect_ratio": aspect,
            "framework_volume_per_parent_A3": volume,
            "framework_hard_wall_loss": hard,
            "framework_normalized_determinant": qdet,
            "framework_cell_guard_loss": 0.01 * shape_guard + metric_guard,
        }, (abc, angles, cell, centres)

    def _optimize_framework(self, template: dict, prefix: torch.Tensor, steps: int, heartbeat=None,
                            branch_modes: torch.Tensor | None = None,
                            diversity_weight: float = 0.0):
        variable = prefix.detach().clone().requires_grad_(True)
        best = variable.detach().clone()
        best_loss = torch.full((len(prefix),), float("inf"), device=self.device)
        stale = torch.zeros((len(prefix),), dtype=torch.long, device=self.device)
        active = torch.ones((len(prefix),), dtype=torch.bool, device=self.device)
        opt = torch.optim.Adam([variable], lr=self.lr)
        last = time.perf_counter()
        for step in range(int(steps)):
            if not bool(active.any()):
                break
            opt.zero_grad(set_to_none=True)
            loss, _, _ = self._framework_loss(template, variable, branch_modes=branch_modes,
                                                diversity_weight=diversity_weight)
            finite = torch.isfinite(loss) & torch.isfinite(variable).all(1)
            active &= finite
            if not bool(active.any()):
                break
            objective = torch.where(active, loss, torch.zeros_like(loss)).sum() / active.sum().clamp_min(1)
            objective.backward()
            torch.nn.utils.clip_grad_norm_([variable], 10.0)
            opt.step()
            with torch.no_grad():
                post, _, _ = self._framework_loss(template, variable, branch_modes=branch_modes,
                                                    diversity_weight=diversity_weight)
                finite_post = torch.isfinite(post) & torch.isfinite(variable).all(1)
                improved = active & finite_post & (post < best_loss - 1.0e-6)
                best_loss = torch.where(improved, post, best_loss)
                best[improved] = variable.detach()[improved]
                stale = torch.where(improved, torch.zeros_like(stale), stale + active.long())
                active &= ~((stale >= self.framework_patience) | ~finite_post)
                bad = ~torch.isfinite(variable).all(1)
                if bool(bad.any()):
                    variable[bad] = best[bad]
            now = time.perf_counter()
            if heartbeat is not None and (step == 0 or now-last >= 10 or step+1 == int(steps) or not bool(active.any())):
                heartbeat("ti_framework_prebuild", step+1, int(steps), float(best_loss.min().detach().cpu())); last = now
        return best

    def _prune_framework_prefix(self, template: dict, prefix: torch.Tensor, keep: int) -> torch.Tensor:
        """Breadth-first cheap prune before Ti tokens are spent.

        v78 deliberately avoids turning the cheap screen into a single-loss funnel.
        It first keeps the best member of coarse cell/packing bins, then selects
        representatives by max-min distance in a scale/shape descriptor.  Framework
        loss only breaks ties and fills any remaining slots.  The final strict chemistry
        and exact hypergraph stages remain authoritative.
        """
        if len(prefix) <= keep:
            return prefix
        with torch.no_grad():
            loss, detail, geom = self._framework_loss(template, prefix)
            _abc, _angles, _cell, _centres = geom
            volume = detail["framework_volume_per_parent_A3"]
            dmin = detail["framework_min_distance_A"]
            aspect = detail["framework_aspect_ratio"]
            qdet = detail["framework_normalized_determinant"]
            finite = torch.isfinite(loss) & torch.isfinite(prefix).all(1)
            indices = torch.nonzero(finite, as_tuple=False).flatten().tolist()
            if not indices:
                return prefix[:0]

            # Coarse bins prevent dozens of nearly identical cheap starts from
            # crowding out a different cell/packing basin.
            reps = {}
            for i in indices:
                key = (round(float(volume[i]) / 8.0),
                       round(float(dmin[i]) / 0.12),
                       round(float(aspect[i]) / 0.15),
                       round(float(qdet[i]) / 0.05))
                score = float(loss[i])
                if key not in reps or score < reps[key][0]:
                    reps[key] = (score, i)
            pool = [i for _score, i in reps.values()]
            if not pool:
                pool = indices[:]

            def desc(i):
                return np.asarray([
                    math.log(max(float(volume[i]), 1.0e-6)) / 4.0,
                    float(dmin[i]) / 4.0,
                    math.log(max(float(aspect[i]), 1.0)) / 2.0,
                    float(qdet[i]),
                ], dtype=float)

            # One chemically sensible anchor, then max-min geometric coverage.
            chosen = [min(pool, key=lambda j: float(loss[j]))]
            pool_set = set(pool)
            while len(chosen) < min(int(keep), len(pool)):
                best = None
                for i in pool:
                    if i in chosen:
                        continue
                    di = desc(i)
                    novelty = min(float(np.sqrt(np.mean((di - desc(j)) ** 2))) for j in chosen)
                    # Breadth dominates; framework loss is a small tie-break only.
                    score = novelty - 0.015 * math.log1p(max(float(loss[i]), 0.0))
                    if best is None or score > best[0]:
                        best = (score, i)
                if best is None:
                    break
                chosen.append(int(best[1]))

            if len(chosen) < keep:
                used = set(chosen)
                for i in sorted(indices, key=lambda j: float(loss[j])):
                    if i not in used:
                        chosen.append(i); used.add(i)
                        if len(chosen) >= keep:
                            break
            return prefix[torch.as_tensor(chosen[:keep], dtype=torch.long, device=self.device)]

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

    def _initial_shared_raw(self, template: dict, starts_override: int | None = None) -> torch.Tensor:
        nlat = len(template["spec"]); ncoord = sum(template["site_dofs"])
        nsite = len(template["site_labels"]); nblock = int(template["nblocks"])
        npose = nsite if self.construction_symmetry == "full" else nblock
        extra = 3*nblock if self.construction_symmetry == "off" else 0
        max_cn = max(self.model.template_cn.values(), default=0)
        pair_mus = [x.mu for x in self.model.channels.values()]
        base = float(np.mean(pair_mus)) * max(nblock, 1) ** (1/3) * 1.8
        nstarts = int(self.starts if starts_override is None else max(1, starts_override))
        raw = torch.randn((nstarts, nlat+ncoord+5*npose+3*npose*max_cn+extra), device=self.device)
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


    # ------------------------------------------------------------------
    # Floating TiO6 proposal -> exact triplet clustering -> explicit oxygen
    # ------------------------------------------------------------------
    @staticmethod
    def _axis_angle_rotation(w: torch.Tensor) -> torch.Tensor:
        theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1.0e-8)
        axis = w / theta
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        k = torch.stack([zero,-z,y, z,zero,-x, -y,x,zero], dim=-1).reshape(*w.shape[:-1],3,3)
        eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(*w.shape[:-1],3,3)
        s = torch.sin(theta)[...,None]
        c = torch.cos(theta)[...,None]
        return eye + s*k + (1.0-c)*torch.matmul(k,k)

    def _initial_floating_raw(self, nbranch: int, nsite: int) -> torch.Tensor:
        """Hard-bound TiO6 variables per independent Ti site.

        Every oxygen port is always represented as Ti + bounded Ti--O bond
        vector.  Layout: rotation(3), angular port distortions(18), and six
        independent Ti--O stretches(6).  There are no free oxygen positions
        and no independent cage-centre/off-centering variables.
        """
        raw = torch.randn((int(nbranch), int(nsite), 27), dtype=torch.float32, device=self.device)
        raw[..., :3] *= 1.5
        raw[..., 3:21] *= 0.30
        raw[..., 21:27] *= 0.35
        return raw

    def _rigid_tio_distance(self, template: dict) -> float:
        child_label = tuple(self.plan['children'])[0]
        values = [float(self.model.pair(label, child_label).mu) for label in template['site_labels']]
        return float(np.mean(values))

    def _floating_vertices(self, template: dict, framework: torch.Tensor, branch_raw: torch.Tensor):
        """Build symmetry-propagated oxygen ports hard-bound to their Ti owners."""
        abc, angles, cell, z2_raw, ti_frac = self._framework_geometry(template, framework)
        bsz = len(framework)
        inv_cell = torch.linalg.inv(cell)
        radius = self._rigid_tio_distance(template)
        vertex_frac, vertex_unwrapped, owners = [], [], []
        cursor = 0
        for site_id, rotations in enumerate(template['orbit_rot']):
            norbit = rotations.shape[0]
            local = branch_raw[:, site_id]
            rlocal = self._axis_angle_rotation(local[:, :3])
            dirs = torch.einsum('vj,bij->bvi', self._base_oct, rlocal)
            dirs = dirs + 0.14 * torch.tanh(local[:, 3:21].reshape(bsz, 6, 3))
            dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp_min(1.0e-8)
            radii = radius * (1.0 + 0.12 * torch.tanh(local[:, 21:27]))
            # Direct owner-centred bond construction: O_port = Ti + r*u.
            dcart = dirs * radii[:, :, None]
            dfrac = torch.einsum('bvi,bij->bvj', dcart, inv_cell)
            transformed = torch.einsum('oij,bvj->bovi', rotations, dfrac)
            centers = ti_frac[:, cursor:cursor + norbit]
            vu = centers[:, :, None, :] + transformed
            vertex_unwrapped.append(vu.reshape(bsz, norbit * 6, 3))
            vertex_frac.append((vu % 1.0).reshape(bsz, norbit * 6, 3))
            for oid in range(norbit):
                owners.extend([cursor + oid] * 6)
            cursor += norbit
        return (abc, angles, cell, z2_raw, ti_frac,
                torch.cat(vertex_frac, dim=1), torch.cat(vertex_unwrapped, dim=1),
                torch.as_tensor(owners, dtype=torch.long, device=self.device))

    def _floating_vertex_distances(self, vu: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        delta = vu[:, None, :, None, :] + self.shift_t[None, None, None, :, :] - vu[:, :, None, None, :]
        vec = torch.einsum('bijnk,bkl->bijnl', delta, cell)
        return torch.linalg.norm(vec, dim=-1)

    @staticmethod
    def _best_disjoint_triplets(frac, unwrapped, owners, cell, tolerance, target,
                                require_full=False, node_limit=6000):
        """Find disjoint three-owner Ti-bound port triplets.

        ``tolerance`` is used only for strict coincidence detection.  When it
        is ``None`` a complete provisional partition is constructed from the
        minimum-cost three-owner combinations, even when the ports are still
        far apart.  The provisional partition supplies the contraction force
        needed to bootstrap the iterative reduction.
        """
        frac=np.asarray(frac,float)%1.0; unwrapped=np.asarray(unwrapped,float)
        owners=np.asarray(owners,int); cell=np.asarray(cell,float)
        n=len(frac); strict=tolerance is not None and np.isfinite(tolerance)
        candidates=[]
        for i in range(n-2):
            for j in range(i+1,n-1):
                if owners[j]==owners[i]: continue
                dij=frac[j][None,:]+SHIFTS-frac[i][None,:]
                sj=int(np.argmin(np.linalg.norm(dij@cell,axis=1)))
                pj=unwrapped[j]+SHIFTS[sj]
                if strict and np.linalg.norm((pj-unwrapped[i])@cell)>float(tolerance): continue
                for k in range(j+1,n):
                    if owners[k] in (owners[i],owners[j]): continue
                    dik=frac[k][None,:]+SHIFTS-frac[i][None,:]
                    sk=int(np.argmin(np.linalg.norm(dik@cell,axis=1)))
                    pk=unwrapped[k]+SHIFTS[sk]
                    pts=np.asarray([unwrapped[i],pj,pk])
                    cart=pts@cell
                    pd=[np.linalg.norm(cart[0]-cart[1]),np.linalg.norm(cart[0]-cart[2]),
                        np.linalg.norm(cart[1]-cart[2])]
                    if strict and max(pd)>float(tolerance): continue
                    cen=cart.mean(axis=0)
                    var=float(np.mean(np.sum((cart-cen)**2,axis=1)))
                    rms=float(np.sqrt(max(var,0.0)))
                    candidates.append(((i,j,k),((0,0,0),tuple(map(int,SHIFTS[sj])),
                                                  tuple(map(int,SHIFTS[sk]))),rms,var))
        if not candidates:
            return []
        by_vertex=[[] for _ in range(n)]
        for ci,c in enumerate(candidates):
            for v in c[0]: by_vertex[v].append(ci)
        for v in range(n):
            by_vertex[v].sort(key=lambda ci:candidates[ci][3])

        # Provisional assignment must be complete and very fast.  An exact-cover
        # search here stalls every GPU worker before the first heartbeat.  For
        # four Ti owners and six ports per owner, eight OTi3 groups necessarily
        # omit each owner exactly twice.  We therefore use a fixed balanced slot
        # pattern and optimize owner-to-slot permutations by bounded random
        # restarts plus local swaps.  This always covers all 24 ports exactly
        # once and has deterministic bounded runtime.
        if require_full:
            unique_owners=sorted(int(x) for x in np.unique(owners))
            if len(unique_owners)!=4 or n!=24 or target!=8:
                # Generic deterministic greedy fallback for unexpected sizes.
                used=set(); groups=[]
                for c in sorted(candidates,key=lambda x:x[3]):
                    if any(q in used for q in c[0]):
                        continue
                    groups.append(c); used.update(c[0])
                    if len(groups)==target:
                        break
                if len(groups)==target and len(used)==n:
                    return [(c[0],c[1],c[2]) for c in groups]
                return []

            owner_ports={o:[int(i) for i in np.where(owners==o)[0]] for o in unique_owners}
            if any(len(v)!=6 for v in owner_ports.values()):
                return []

            # Slot s omits omit_owner[s].  Each owner therefore appears in six
            # slots, exactly matching its six hard-bound ports.
            omit_owner=[unique_owners[0],unique_owners[0],
                        unique_owners[1],unique_owners[1],
                        unique_owners[2],unique_owners[2],
                        unique_owners[3],unique_owners[3]]
            eligible={o:[s for s in range(8) if omit_owner[s]!=o] for o in unique_owners}

            # Precompute minimum-image pair costs and shifts for all ports.
            pair_cost=np.zeros((n,n),float)
            pair_shift=np.zeros((n,n,3),int)
            for i in range(n):
                for j in range(n):
                    d=frac[j][None,:]+SHIFTS-frac[i][None,:]
                    k=int(np.argmin(np.linalg.norm(d@cell,axis=1)))
                    pair_cost[i,j]=float(np.sum((d[k]@cell)**2))
                    pair_shift[i,j]=np.asarray(SHIFTS[k],int)

            def slot_cost(slot_members):
                vals=[]
                for members in slot_members:
                    a,b,c=members
                    vals.append(pair_cost[a,b]+pair_cost[a,c]+pair_cost[b,c])
                return float(sum(vals))

            rng=np.random.default_rng(104729 + int(np.round(np.sum(frac)*1e6))%1000003)
            best_assign=None; best_cost=float('inf')
            nrestart=48
            for restart in range(nrestart):
                slots=[[] for _ in range(8)]
                for oi,o in enumerate(unique_owners):
                    ports=list(owner_ports[o])
                    if restart==0:
                        order=ports
                    else:
                        order=list(np.asarray(ports)[rng.permutation(6)])
                    for p,slt in zip(order,eligible[o]):
                        slots[slt].append(int(p))
                cost=slot_cost(slots)
                # Bounded pair-swap descent within each owner assignment.
                improved=True; sweeps=0
                while improved and sweeps<3:
                    improved=False; sweeps+=1
                    for o in unique_owners:
                        sl=eligible[o]
                        for ia in range(5):
                            for ib in range(ia+1,6):
                                sa,sb=sl[ia],sl[ib]
                                pa=next(p for p in slots[sa] if owners[p]==o)
                                pb=next(p for p in slots[sb] if owners[p]==o)
                                old=(pair_cost[slots[sa][0],slots[sa][1]]+pair_cost[slots[sa][0],slots[sa][2]]+pair_cost[slots[sa][1],slots[sa][2]]+
                                     pair_cost[slots[sb][0],slots[sb][1]]+pair_cost[slots[sb][0],slots[sb][2]]+pair_cost[slots[sb][1],slots[sb][2]])
                                na=[pb if p==pa else p for p in slots[sa]]
                                nb=[pa if p==pb else p for p in slots[sb]]
                                newc=(pair_cost[na[0],na[1]]+pair_cost[na[0],na[2]]+pair_cost[na[1],na[2]]+
                                      pair_cost[nb[0],nb[1]]+pair_cost[nb[0],nb[2]]+pair_cost[nb[1],nb[2]])
                                if newc+1e-12<old:
                                    slots[sa],slots[sb]=na,nb
                                    cost += float(newc-old); improved=True
                if cost<best_cost:
                    best_cost=cost; best_assign=[list(x) for x in slots]

            groups=[]
            for members in best_assign:
                members=sorted(members)
                i,j,k=members
                sj=tuple(map(int,pair_shift[i,j]))
                sk=tuple(map(int,pair_shift[i,k]))
                pts=np.asarray([unwrapped[i],unwrapped[j]+np.asarray(sj),unwrapped[k]+np.asarray(sk)])
                cart=pts@cell; cen=cart.mean(axis=0)
                rms=float(np.sqrt(np.mean(np.sum((cart-cen)**2,axis=1))))
                groups.append(((i,j,k),((0,0,0),sj,sk),rms))
            return groups

        # Strict detector: maximize the number of compact disjoint triplets,
        # then minimize their total centroid RMS.
        best_groups=[]; best_score=(-1,float('inf'))
        def dfs_partial(used,skipped,groups,rms_sum):
            nonlocal best_groups,best_score
            covered=3*len(groups)
            if covered>best_score[0] or (covered==best_score[0] and rms_sum<best_score[1]):
                best_score=(covered,rms_sum); best_groups=list(groups)
            if len(groups)>=target: return
            available=n-len(used)-len(skipped)
            optimistic=covered+3*min(target-len(groups),available//3)
            if optimistic<best_score[0]: return
            remaining=[v for v in range(n) if v not in used and v not in skipped]
            if not remaining: return
            def nviable(x):
                return sum(1 for ci in by_vertex[x]
                           if not any(q in used or q in skipped for q in candidates[ci][0]))
            v=min(remaining,key=nviable)
            viable=[ci for ci in by_vertex[v]
                    if not any(q in used or q in skipped for q in candidates[ci][0])]
            viable.sort(key=lambda ci:candidates[ci][3])
            for ci in viable:
                c=candidates[ci]
                dfs_partial(used|set(c[0]),skipped,groups+[c],rms_sum+c[2])
            dfs_partial(used,skipped|{v},groups,rms_sum)
        dfs_partial(set(),set(),[],0.0)
        return [(c[0],c[1],c[2]) for c in best_groups]

    def _assignment_tensors(self, assignments, nvert):
        bsz=len(assignments); ng=max((len(x) for x in assignments),default=0)
        if ng==0: return None
        idx=torch.zeros((bsz,ng,3),dtype=torch.long,device=self.device)
        shifts=torch.zeros((bsz,ng,3,3),dtype=torch.float32,device=self.device)
        mask=torch.zeros((bsz,ng),dtype=torch.bool,device=self.device)
        for b,groups in enumerate(assignments):
            for g,(verts,sh,rms) in enumerate(groups):
                idx[b,g]=torch.as_tensor(verts,dtype=torch.long,device=self.device)
                shifts[b,g]=torch.as_tensor(sh,dtype=torch.float32,device=self.device)
                mask[b,g]=True
        return idx,shifts,mask

    def _floating_loss(self, template: dict, framework: torch.Tensor, branch_raw: torch.Tensor,
                       progress: float, mode: str = 'global', assignments=None):
        abc, angles, cell, z2_raw, ti_frac, vf, vu, owners = self._floating_vertices(template, framework, branch_raw)
        bsz, nvert = vu.shape[:2]; nti = ti_frac.shape[1]
        dist_img = self._floating_vertex_distances(vu, cell); ns = dist_img.shape[-1]
        p = float(np.clip(progress, 0.0, 1.0)); smooth = p*p*(3.0-2.0*p)
        sigma_start = max(1.35, 4.0*self.floating_coincidence_sigma)
        sigma = sigma_start*(1.0-smooth) + max(0.32, self.floating_coincidence_sigma)*smooth
        kernel = torch.exp(-0.5*(dist_img/sigma).pow(2))
        owner_class = (owners[:,None] + nti*torch.arange(ns,device=self.device)[None,:]).reshape(-1)
        owner_mass = torch.zeros((bsz,nvert,nti*ns),dtype=kernel.dtype,device=self.device)
        owner_mass.scatter_add_(2,owner_class[None,None,:].expand(bsz,nvert,-1),kernel.reshape(bsz,nvert,-1))
        own_class = owners + nti*ZERO_SHIFT
        own_mask = torch.nn.functional.one_hot(own_class,num_classes=nti*ns).bool()[None].expand(bsz,-1,-1)
        presence=(1.0-torch.exp(-owner_mass)).masked_fill(own_mask,0.0)
        rho=presence.sum(-1)
        occupancy=(rho-2.0).pow(2).mean(1)
        overcoord=torch.relu(rho-2.20).pow(2).mean(1)
        owner_dist=torch.full((bsz,nvert,nti*ns),1.0e6,dtype=dist_img.dtype,device=self.device)
        owner_dist.scatter_reduce_(2,owner_class[None,None,:].expand(bsz,nvert,-1),dist_img.reshape(bsz,nvert,-1),reduce='amin',include_self=True)
        owner_dist=owner_dist.masked_fill(own_mask,1.0e6)
        nearest=torch.topk(owner_dist,k=min(3,nti*ns-1),dim=-1,largest=False).values
        compact=torch.log1p((nearest[:,:,:2]/sigma).pow(2)).mean((1,2))
        fourth=(torch.relu(1.20*sigma-nearest[:,:,2]).pow(2).mean(1)/max(sigma*sigma,1e-5)
                if nearest.shape[-1]>=3 else torch.zeros(bsz,device=self.device))

        angular_dist=torch.tanh(branch_raw[...,3:21]).pow(2).mean((1,2))
        radial_dist=torch.tanh(branch_raw[...,21:27]).pow(2).mean((1,2))

        zero_dist=dist_img[...,ZERO_SHIFT]
        same=owners[None,:,None]==owners[None,None,:]
        eye=torch.eye(nvert,dtype=torch.bool,device=self.device)[None]
        same_mask=same & ~eye
        min_same=zero_dist.masked_fill(~same_mask,1.0e6).amin((1,2))
        radius=self._rigid_tio_distance(template)
        topology_safe=0.62*math.sqrt(2.0)*radius
        collapse=torch.relu(topology_safe-min_same).pow(2)/max(radius*radius,0.1)

        assigned_loss=torch.zeros(bsz,device=self.device)
        assigned_coverage=torch.zeros(bsz,device=self.device)
        at=self._assignment_tensors(assignments,nvert) if assignments is not None else None
        if at is not None:
            idx,ashift,amask=at
            gather_idx=idx[...,None].expand(-1,-1,-1,3)
            pts=torch.gather(vu[:,None,:,:].expand(-1,idx.shape[1],-1,-1),2,gather_idx)+ashift
            cart=torch.einsum('bgpi,bij->bgpj',pts,cell)
            cen=cart.mean(2,keepdim=True)
            var=((cart-cen).pow(2).sum(-1).mean(-1))
            assigned_loss=(var*amask).sum(1)/amask.sum(1).clamp_min(1)
            assigned_coverage=3.0*amask.sum(1).to(var.dtype)/float(nvert)

        child_label=tuple(self.plan['children'])[0]
        cutoffs=[self.model.pair(label,child_label).first_shell_cutoff for label in template['expanded_labels']]
        ti_o_cutoff=float(np.mean(cutoffs))
        tio_delta=vu[:,None,:,None,:]+self.shift_t[None,None,None,:,:]-ti_frac[:,:,None,None,:]
        tio_vec=torch.einsum('btvsk,bkl->btvsl',tio_delta,cell)
        tio_dist=torch.linalg.norm(tio_vec,dim=-1)
        cn_weight=torch.sigmoid((ti_o_cutoff-tio_dist)/0.10)
        ti_cn_img=cn_weight.sum((2,3))/3.0
        ti_cn_loss=(ti_cn_img-6.0).pow(2).mean(1)
        ti_cn_over=torch.relu(ti_cn_img-6.0).pow(2).mean(1)

        framework_loss, fdetail, _ = self._framework_loss(template, framework)
        coincidence_scale=0.7+1.8*smooth
        topology_reg=4.0*angular_dist+3.0*radial_dist+18.0*collapse
        total=(coincidence_scale*(1.4*occupancy+0.8*compact+2.5*overcoord+2.0*fourth)
               +1.2*ti_cn_loss+2.0*ti_cn_over+topology_reg)
        if assignments is not None:
            total=total+(5.0+8.0*smooth)*assigned_loss-1.5*assigned_coverage
        if mode=='global': total=total+0.18*framework_loss
        detail={
            'floating_occupancy_loss':occupancy,'floating_compactness_loss':compact,
            'floating_overcoord_loss':overcoord,'floating_fourth_owner_loss':fourth,
            'floating_same_ti_collapse_loss':collapse,
            'floating_image_resolved_ti_cn_loss':ti_cn_loss,
            'floating_image_resolved_ti_cn_over_loss':ti_cn_over,
            'floating_image_resolved_ti_cn_mean':ti_cn_img.mean(1),
            'floating_image_resolved_ti_cn_q90_error':torch.quantile(torch.abs(ti_cn_img-6.0),0.9,dim=1),
            'floating_local_distortion_loss':angular_dist,
            'floating_radial_distortion_loss':radial_dist,
            'floating_ti_offcenter_loss':torch.zeros_like(radial_dist),
            'floating_assigned_triplet_loss':assigned_loss,
            'floating_assigned_fraction':assigned_coverage,
            'floating_framework_loss':framework_loss,
            'floating_framework_restraint_loss':torch.zeros_like(framework_loss),
            'floating_rho_mean':rho.mean(1),'floating_rho_q10':torch.quantile(rho,0.1,dim=1),
            'floating_rho_q90':torch.quantile(rho,0.9,dim=1),
            'minimum_same_ti_vertex_distance_A':min_same,
            'floating_anneal_progress':torch.full_like(occupancy,p),
            'floating_capture_sigma_A':torch.full_like(occupancy,float(sigma)),
            'floating_rigid_ti_o_distance_A':torch.full_like(occupancy,float(radius)),
        }
        for k,v in fdetail.items(): detail.setdefault(k,v)
        return total,detail,(abc,angles,cell,ti_frac,vf,vu,owners)

    def _optimize_floating_block(self, template, framework, branch_raw, steps, lr, mode, cycle,
                                 assignments=None, heartbeat=None):
        framework=framework.detach().clone().requires_grad_(mode=='global')
        branch_raw=branch_raw.detach().clone().requires_grad_(True)
        params=[branch_raw]+([framework] if mode=='global' else [])
        opt=torch.optim.Adam(params,lr=float(lr))
        last=time.time()
        for step in range(int(steps)):
            progress=(cycle + (step+1)/max(int(steps),1))/max(self._floating_cycles,1)
            opt.zero_grad(set_to_none=True)
            loss,_,_=self._floating_loss(template,framework,branch_raw,progress,mode=mode,assignments=assignments)
            loss.mean().backward(); torch.nn.utils.clip_grad_norm_(params,10.0); opt.step()
            now=time.time()
            if heartbeat is not None and (step+1==int(steps) or now-last>=10):
                heartbeat(f'octahedron_{mode}',step+1,int(steps),float(loss.min().detach().cpu())); last=now
        return framework.detach(),branch_raw.detach()

    def _assign_floating_triplets(self, vf, vu, owners, cell, tolerance, provisional=False):
        target=2*len(np.unique(owners))
        tol=None if provisional else float(tolerance)
        groups=self._best_disjoint_triplets(vf,vu,owners,cell,tol,target,require_full=bool(provisional))
        covered=3*len(groups)
        rms=[g[2] for g in groups]
        diag={'floating_cluster_success':len(groups)==target,
              'floating_n_clusters':len(groups),'floating_target_clusters':target,
              'floating_exact_triplets':len(groups),'floating_assigned_vertices':covered,
              'floating_triplet_rms_mean_A':float(np.mean(rms)) if rms else float('inf'),
              'floating_triplet_rms_max_A':float(np.max(rms)) if rms else float('inf')}
        topology=None
        if len(groups)==target:
            oxygen=[]; out_groups=[]
            for verts,sh,rmsv in groups:
                pts=np.asarray([np.asarray(vu[verts[q]])+np.asarray(sh[q]) for q in range(3)])
                oxygen.append(np.mean(pts,axis=0)%1.0)
                out_groups.append(tuple((int(owners[verts[q]]),np.asarray(sh[q],dtype=int)) for q in range(3)))
            oxygen=np.asarray(oxygen,float)
            dd=oxygen[None,:,None,:]-oxygen[:,None,None,:]+SHIFTS[None,None,:,:]
            od=np.linalg.norm(np.einsum('...i,ij->...j',dd,np.asarray(cell,float)),axis=-1)
            for i in range(len(oxygen)): od[i,i,ZERO_SHIFT]=np.inf
            omin=float(np.min(od))
            diag['floating_initial_oo_min_A']=omin
            if omin>=max(0.45,0.45*self.minimum_distance):
                topology={'groups':out_groups,'child_label':tuple(self.plan['children'])[0],
                          'oxygen_frac_init':oxygen}
            else:
                diag['floating_cluster_success']=False
        return groups,topology,diag

    def _make_floating_o_topologies(self, template: dict, frameworks: torch.Tensor, heartbeat=None):
        out_framework=[]; out_topology=[]; rows=[]
        self._floating_cycles=5
        local_steps=max(8,int(self.octahedron_prepare_steps)//2)
        global_steps=max(12,int(self.octahedron_match_steps)//2)
        for fi in range(len(frameworks)):
            nbranch=self.octahedral_branches
            fw=frameworks[fi:fi+1].repeat(nbranch,1)
            br=self._initial_floating_raw(nbranch,len(template['site_labels']))
            assignments=[[] for _ in range(nbranch)]
            # Bootstrap with a complete minimum-cost provisional partition of
            # all 24 hard-bound ports.  These are optimization guides only;
            # strict OTi3 coincidence is evaluated separately.
            with torch.no_grad():
                _,_,geom0=self._floating_loss(template,fw,br,0.0,mode='local')
                _,_,cell0,_,vf0,vu0,owners0=geom0
                for b in range(nbranch):
                    groups,_,_=self._assign_floating_triplets(
                        vf0[b].cpu().numpy(),vu0[b].cpu().numpy(),owners0.cpu().numpy(),
                        cell0[b].cpu().numpy(),self.floating_cluster_tolerance,provisional=True)
                    assignments[b]=groups
            best_rank=[(-1,float('inf'),float('inf')) for _ in range(nbranch)]
            best_state=[None]*nbranch
            for cyc in range(self._floating_cycles):
                fw,br=self._optimize_floating_block(template,fw,br,local_steps,0.20*self.lr,'local',cyc,
                                                    assignments=assignments,heartbeat=heartbeat)
                with torch.no_grad():
                    _,_,geom=self._floating_loss(template,fw,br,(cyc+0.45)/self._floating_cycles,mode='local')
                    _,_,cell,_,vf,vu,owners=geom
                    new_assign=[]
                    for b in range(nbranch):
                        groups,_,_=self._assign_floating_triplets(
                            vf[b].cpu().numpy(),vu[b].cpu().numpy(),owners.cpu().numpy(),
                            cell[b].cpu().numpy(),self.floating_cluster_tolerance,provisional=True)
                        new_assign.append(groups if groups else assignments[b])
                    assignments=new_assign
                fw,br=self._optimize_floating_block(template,fw,br,global_steps,0.16*self.lr,'global',cyc,
                                                    assignments=assignments,heartbeat=heartbeat)
                with torch.no_grad():
                    loss,detail,geom=self._floating_loss(template,fw,br,(cyc+1)/self._floating_cycles,
                                                         mode='global',assignments=assignments)
                    abc,angles,cell,ti_frac,vf,vu,owners=geom
                    next_assign=[]
                    # Strict coincidence threshold is annealed independently
                    # from the always-complete provisional assignment.
                    start_tol=max(1.20,3.0*self.floating_cluster_tolerance)
                    frac=(cyc+1)/self._floating_cycles
                    eval_tol=start_tol*(1.0-frac)+self.floating_cluster_tolerance*frac
                    for b in range(nbranch):
                        strict_groups,topology,diag=self._assign_floating_triplets(
                            vf[b].cpu().numpy(),vu[b].cpu().numpy(),owners.cpu().numpy(),
                            cell[b].cpu().numpy(),eval_tol,provisional=False)
                        provisional_groups,_,_=self._assign_floating_triplets(
                            vf[b].cpu().numpy(),vu[b].cpu().numpy(),owners.cpu().numpy(),
                            cell[b].cpu().numpy(),self.floating_cluster_tolerance,provisional=True)
                        next_assign.append(provisional_groups if provisional_groups else assignments[b])
                        diag['floating_strict_triplet_tolerance_A']=float(eval_tol)
                        groups=strict_groups
                        rank=(int(diag.get('floating_assigned_vertices',0)),
                              -float(diag.get('floating_triplet_rms_max_A',1e9)),
                              -float(loss[b]))
                        if rank>best_rank[b]:
                            best_rank[b]=rank
                            best_state[b]=(fw[b].detach().clone(),br[b].detach().clone(),topology,diag,
                                           {k:float(v[b]) for k,v in detail.items()},float(loss[b]),cyc)
                    assignments=next_assign
            for b,state in enumerate(best_state):
                if state is None: continue
                fwb,brb,topology,diag,det,lossv,cyc=state
                row={'stage':'o_topology_screen','branch':int(fi*nbranch+b),'optimization_cycle':int(cyc+1),
                     'construction_symmetry':self.construction_symmetry,'ti_framework_finite':True,
                     'ti_framework_accepted':True,'o_topology_complete':topology is not None,
                     'o_topology_accepted':topology is not None,'o_attachment_accepted':False,
                     'strict_valid':False,'loss':lossv,**diag,**det}
                rows.append(row)
                if topology is not None: out_framework.append(fwb); out_topology.append(topology)
        if not out_framework: return torch.empty((0,frameworks.shape[1]),device=self.device),[],rows
        return torch.stack(out_framework),out_topology,rows

    @staticmethod
    def _slice_explicit_topology(topologies, selector):
        if torch.is_tensor(selector): selector = selector.detach().cpu().numpy()
        arr = np.asarray(selector)
        indices = np.flatnonzero(arr).tolist() if arr.dtype == bool else arr.astype(int).reshape(-1).tolist()
        return [topologies[i] for i in indices]

    def _explicit_o_raw(self, framework: torch.Tensor, topologies: list[dict]) -> torch.Tensor:
        """Build explicit-O variables using *unwrapped* fractional coordinates.

        v65 used sigmoid(logit(frac)) for the physical O coordinates.  That is
        numerically pathological at crystallographic boundaries: an O starting
        at x=0 or 1 has a sigmoid derivative of order 1e-6 and is effectively
        pinned exactly where the nonbonded repair most needs it to move.

        Here the O variables are direct unwrapped fractional coordinates.  The
        assigned periodic-image shifts therefore remain continuous if an O moves
        through a cell boundary; coordinates are wrapped only for final output.
        """
        psize = framework.shape[1]
        nchild = int(self.target_counts[topologies[0]["child_label"]])
        rows = []
        for i, topology in enumerate(topologies):
            ofrac = np.asarray(topology["oxygen_frac_init"], float).reshape(-1)
            rows.append(torch.cat([framework[i], torch.as_tensor(ofrac, dtype=framework.dtype, device=self.device)]))
        return torch.stack(rows).reshape(len(rows), psize + 3*nchild)

    def _explicit_o_loss(self, template: dict, raw: torch.Tensor, topologies: list[dict], phase: str):
        psize = self._shared_prefix_size(template)
        prefix = raw[:, :psize]
        abc, angles, cell, z2_raw, parent_frac = self._framework_geometry(template, prefix)
        bsz, nparent, _ = parent_frac.shape
        child_label = topologies[0]["child_label"]
        nchild = int(self.target_counts[child_label])
        # Direct, unwrapped fractional O coordinates.  Do not sigmoid: the
        # periodic topology already provides the relevant image shifts and a
        # sigmoid would freeze O atoms initialized on x/y/z = 0 or 1.
        child_frac = raw[:, psize:psize+3*nchild].reshape(bsz,nchild,3)
        parent_cart = torch.einsum('bni,bij->bnj', parent_frac, cell)
        child_cart = torch.einsum('bni,bij->bnj', child_frac, cell)
        assigned_window = torch.zeros(bsz, device=self.device)
        assigned_radial = torch.zeros_like(assigned_window)
        assigned_fraction = torch.zeros_like(assigned_window)
        child_angle = torch.zeros_like(assigned_window)
        coverage = torch.zeros_like(assigned_window)
        assigned_mask = torch.zeros((bsz,nparent,nchild),dtype=torch.bool,device=self.device)
        parent_vectors = [[[] for _ in range(nparent)] for _ in range(bsz)]
        edge_dist_rows=[]
        for b, topology in enumerate(topologies):
            edge_dist=[]; edge_lo=[]; edge_hi=[]; edge_mu=[]; edge_sig=[]
            for g, images in enumerate(topology["groups"]):
                reverse=[]
                for parent, shift_np in images:
                    shift=torch.as_tensor(shift_np,dtype=cell.dtype,device=self.device)
                    pimage=parent_cart[b,parent] + shift @ cell[b]
                    vec=child_cart[b,g]-pimage
                    d=torch.linalg.norm(vec)
                    ch=self.model.pair(template["expanded_labels"][parent],child_label)
                    edge_dist.append(d); edge_lo.append(ch.sampling_min); edge_hi.append(ch.sampling_max)
                    edge_mu.append(ch.mu); edge_sig.append(max(ch.sigma,0.02))
                    assigned_mask[b,parent,g]=True
                    parent_vectors[b][parent].append(vec)
                    reverse.append(-vec)
                vv=torch.stack(reverse)
                uu=vv/torch.linalg.norm(vv,dim=1,keepdim=True).clamp_min(1e-6)
                tri=torch.triu_indices(len(images),len(images),1,device=self.device)
                obs=torch.sort(torch.rad2deg(torch.acos((uu@uu.T).clamp(-1+1e-6,1-1e-6)[tri[0],tri[1]]))).values
                target=torch.as_tensor(self.model.template_angles[child_label],dtype=obs.dtype,device=self.device)
                sig=torch.as_tensor(self.model.template_angle_sigma[child_label],dtype=obs.dtype,device=self.device)
                if len(obs)==len(target): child_angle[b]+=torch.mean(((obs-target)/sig.clamp_min(1e-3))**2)
            d=torch.stack(edge_dist)
            lo=torch.as_tensor(edge_lo,dtype=d.dtype,device=self.device); hi=torch.as_tensor(edge_hi,dtype=d.dtype,device=self.device)
            mu=torch.as_tensor(edge_mu,dtype=d.dtype,device=self.device); sig=torch.as_tensor(edge_sig,dtype=d.dtype,device=self.device)
            assigned_window[b]=(torch.relu((lo-d)/sig).pow(2)+torch.relu((d-hi)/sig).pow(2)).mean()
            assigned_radial[b]=((d-mu)/sig).pow(2).mean()
            assigned_fraction[b]=((d>=lo)&(d<=hi)).float().mean()
            child_angle[b]/=max(nchild,1)
            # Global spherical coverage at each Ti: small vector centroid and no
            # angular crowding. Detailed learned Ti angles enter only in polish.
            for p,label in enumerate(template["expanded_labels"]):
                vecs=parent_vectors[b][p]
                if len(vecs)!=self.model.template_cn[label]:
                    coverage[b]+=100.; continue
                vv=torch.stack(vecs); uu=vv/torch.linalg.norm(vv,dim=1,keepdim=True).clamp_min(1e-6)
                gram=uu@uu.T
                tri=torch.triu_indices(len(vecs),len(vecs),1,device=self.device)
                crowd=torch.nn.functional.softplus((gram[tri[0],tri[1]]-0.35)/0.10).pow(2).mean()
                centroid=uu.mean(0).pow(2).sum()
                coverage[b]+=crowd+4.0*centroid
            coverage[b]/=max(nparent,1)
            edge_dist_rows.append(d)
        # Unassigned Ti--O shell exclusion using minimum periodic distance.
        delta=child_frac[:,None,:,None,:]-parent_frac[:,:,None,None,:]+self.shift_t[None,None,None,:,:]
        dpc=torch.linalg.norm(torch.einsum('bpksi,bij->bpksj',delta,cell),dim=-1).amin(3)
        shell=torch.zeros_like(dpc)
        for p,label in enumerate(template["expanded_labels"]):
            shell[:,p,:]=float(self.model.pair(label,child_label).first_shell_cutoff)
        unmask=~assigned_mask
        penetration=torch.relu((shell+self.nonbonded_margin-dpc)/self.nonbonded_width).pow(2)
        unselected=(penetration*unmask).sum((1,2))/unmask.sum((1,2)).clamp_min(1)
        unselected_count=(unmask&(dpc<shell)).sum((1,2))

        # Learned physical nonbonded exclusions.  These are distinct from the
        # first-shell topology rule above: the latter says that an unassigned
        # Ti--O contact may not enter the bonded shell, while these walls encode
        # how closely *physical nonbonded atoms* approached in accepted references.
        tio_soft=torch.zeros((nparent,nchild),dtype=cell.dtype,device=self.device)
        for p,label in enumerate(template["expanded_labels"]):
            tio_soft[p,:]=float(self.model.nonbonded_soft_min(label,child_label))
        tio_nb_pen=torch.relu((tio_soft[None]-dpc)/self.nonbonded_width).pow(2)
        # Normalize per physical O, not per possible Ti--O pair.  Otherwise a
        # single severe clash is diluted as the cell grows and can lose against
        # the radial/topology terms.
        tio_nb_pen=(tio_nb_pen*unmask).sum((1,2))/max(nchild,1)
        min_unassigned_tio=dpc.masked_fill(~unmask,1.0e6).amin((1,2))

        oo_delta=child_frac[:,None,:,None,:]-child_frac[:,:,None,None,:]+self.shift_t[None,None,None,:,:]
        oo_all=torch.linalg.norm(torch.einsum('bmnsi,bij->bmnsj',oo_delta,cell),dim=-1)
        oo_diag=torch.arange(nchild,device=self.device)
        oo_zero_mask=torch.zeros((nchild,nchild,len(SHIFTS)),dtype=torch.bool,device=self.device)
        oo_zero_mask[oo_diag,oo_diag,ZERO_SHIFT]=True
        oo_all=oo_all.masked_fill(oo_zero_mask[None],1.0e6)
        # Keep diagonal entries after removing only the zero image: O_i--O_i+T
        # is a real periodic nonbonded contact and must obey the learned wall.
        oo_dist=oo_all.amin(3)
        oo_mask=torch.ones((1,nchild,nchild),dtype=torch.bool,device=self.device)
        oo_soft=float(self.model.nonbonded_soft_min(child_label,child_label))
        oo_pen=torch.relu((oo_soft-oo_dist)/self.nonbonded_width).pow(2)
        # Normalize per physical O rather than n_O^2 pair slots.  Each real
        # violating contact must remain expensive independent of system size.
        oo_pen=(oo_pen*oo_mask).sum((1,2))/max(nchild,1)
        min_oo=oo_dist.amin((1,2))

        tt_delta=parent_frac[:,None,:,None,:]-parent_frac[:,:,None,None,:]+self.shift_t[None,None,None,:,:]
        tt_all=torch.linalg.norm(torch.einsum('bmnsi,bij->bmnsj',tt_delta,cell),dim=-1)
        tt_diag=torch.arange(nparent,device=self.device)
        tt_zero_mask=torch.zeros((nparent,nparent,len(SHIFTS)),dtype=torch.bool,device=self.device)
        tt_zero_mask[tt_diag,tt_diag,ZERO_SHIFT]=True
        tt_all=tt_all.masked_fill(tt_zero_mask[None],1.0e6)
        # Likewise preserve Ti_i--Ti_i+T periodic contacts.
        tt_dist=tt_all.amin(3)
        tt_mask=torch.ones((1,nparent,nparent),dtype=torch.bool,device=self.device)
        tt_floor=torch.zeros((nparent,nparent),dtype=cell.dtype,device=self.device)
        for i,li in enumerate(template["expanded_labels"]):
            for j,lj in enumerate(template["expanded_labels"]):
                tt_floor[i,j]=float(self.model.nonbonded_soft_min(li,lj))
        tt_pen=torch.relu((tt_floor[None]-tt_dist)/self.nonbonded_width).pow(2)
        # Same size-independent normalization for the Ti sublattice.
        tt_pen=(tt_pen*tt_mask).sum((1,2))/max(nparent,1)
        min_tt=tt_dist.masked_fill(~tt_mask,1.0e6).amin((1,2))
        learned_nonbonded=tio_nb_pen+oo_pen+tt_pen

        # v74 crystallographic O-orbit restraint.  Direct hypergraph
        # covers built from complete SG orbits carry relations (rep, member, R, t).
        # The independent O coordinates remain trainable, but symmetry-related O
        # atoms are softly constrained to move together in fractional space.
        oxygen_symmetry_loss = torch.zeros(bsz, dtype=cell.dtype, device=self.device)
        for b, topology in enumerate(topologies):
            rels = topology.get("symmetry_relations", [])
            if not rels:
                continue
            vals = []
            for rep, member, rot_np, trans_np in rels:
                rot = torch.as_tensor(rot_np, dtype=cell.dtype, device=self.device)
                trans = torch.as_tensor(trans_np, dtype=cell.dtype, device=self.device)
                target = rot @ child_frac[b, int(rep)] + trans
                df = child_frac[b, int(member)] - target
                df = df - torch.round(df)
                vals.append(torch.sum(df * df))
            if vals:
                oxygen_symmetry_loss[b] = torch.stack(vals).mean()

        # Generic all-physical fallback remains as a last-resort sanity wall.
        final_frac=torch.cat([parent_frac,child_frac],1)
        _,all_dist=self._center_geometry(final_frac,cell)
        min_physical=all_dist.amin((1,2,3))
        overlap=torch.relu((self.minimum_distance-min_physical)/0.05).pow(2)
        # v73: once explicit O exists, the realized local chemistry itself sets
        # the cell scale.  Do not add any Ti-only density/volume objective or raw
        # framework restraint.  Retain only loose numerical cell conditioning.
        aspect = abc.amax(1) / abc.amin(1).clamp_min(1.0e-4)
        qdet = torch.abs(torch.linalg.det(cell)) / torch.prod(abc, dim=1).clamp_min(1.0e-8)
        cell_guard = 0.01 * torch.relu(aspect - 8.0).pow(2) + torch.relu(0.12 - qdet).pow(2) * 100.0
        cell_guard = cell_guard + torch.relu(1.0e-4 - z2_raw / abc[:,2].square().clamp_min(1.0e-8)).pow(2) * 1.0e6
        framework_loss = cell_guard
        framework_restraint = torch.zeros_like(cell_guard)
        framework_detail = {
            "framework_cell_guard_loss": cell_guard,
            "framework_aspect_ratio": aspect,
            "framework_volume_per_parent_A3": torch.abs(torch.linalg.det(cell)) / max(nparent, 1),
            "framework_normalized_determinant": qdet,
        }
        # Detailed Ti angle matching is delayed and weak.
        parent_angle=torch.zeros_like(coverage)
        if phase=='o_polish':
            for b in range(bsz):
                for p,label in enumerate(template["expanded_labels"]):
                    vv=torch.stack(parent_vectors[b][p]); uu=vv/torch.linalg.norm(vv,dim=1,keepdim=True).clamp_min(1e-6)
                    tri=torch.triu_indices(len(vv),len(vv),1,device=self.device)
                    obs=torch.sort(torch.rad2deg(torch.acos((uu@uu.T).clamp(-1+1e-6,1-1e-6)[tri[0],tri[1]]))).values
                    target=torch.as_tensor(self.model.template_angles[label],dtype=obs.dtype,device=self.device)
                    sig=torch.as_tensor(self.model.template_angle_sigma[label],dtype=obs.dtype,device=self.device)
                    if len(obs)==len(target): parent_angle[b]+=torch.mean(((obs-target)/sig.clamp_min(1e-3))**2)
                parent_angle[b]/=max(nparent,1)
        if phase=='o_place':
            wwin,wrad,wcov,wang,wex,wrest=12.0,1.0,3.0,0.15,5.0,0.0
        elif phase=='o_contact':
            wwin,wrad,wcov,wang,wex,wrest=18.0,1.5,5.0,0.35,12.0,0.0
        else:
            wwin,wrad,wcov,wang,wex,wrest=20.0,2.0,6.0,1.0,16.0,0.0
        # During o_contact the first priority is to push the realized physical
        # structure out of forbidden learned nonbonded regions.  o_polish then
        # restores the full radial/angular balance without weakening the wall.
        learned_nb_scale = 3.0 if phase == 'o_contact' else 2.0
        symmetry_weight = 18.0 if phase == 'o_polish' else 10.0
        total=(wwin*assigned_window+wrad*assigned_radial+wcov*coverage
               +wang*(child_angle+0.35*parent_angle)+wex*self.nonbonded_weight*unselected
               +learned_nb_scale*self.nonbonded_weight*learned_nonbonded
               +symmetry_weight*oxygen_symmetry_loss
               +self.overlap_weight*overlap + framework_loss)
        total=torch.nan_to_num(total,nan=1e9,posinf=1e9,neginf=1e9)
        details={"topology_complete":torch.ones(bsz,dtype=torch.bool,device=self.device),
                 "assigned_bond_window_loss":assigned_window,"assigned_radial_loss":assigned_radial,
                 "assigned_bond_fraction":assigned_fraction,"ti_spherical_coverage_loss":coverage,
                 "parent_angular_loss":parent_angle,"child_angular_loss":child_angle,
                 "unselected_contact_loss":unselected,
                 "unselected_first_shell_contact_count_soft":unselected_count,
                 "learned_nonbonded_exclusion_loss":learned_nonbonded,
                 "oxygen_symmetry_loss":oxygen_symmetry_loss,
                 "minimum_unassigned_ti_o_A":min_unassigned_tio,
                 "minimum_o_o_A":min_oo,"minimum_ti_ti_A":min_tt,
                 "minimum_physical_distance_A":min_physical,
                 "framework_loss":framework_loss,"framework_restraint_loss":framework_restraint,
                 **framework_detail}
        return total,details,(abc,angles,cell,parent_frac,child_frac,final_frac)

    def _optimize_explicit_o(self, template, raw, topologies, steps, phase, lr, heartbeat=None):
        variable=raw.detach().clone().requires_grad_(True)
        best=variable.detach().clone(); best_loss=torch.full((len(raw),),float('inf'),device=self.device)
        opt=torch.optim.Adam([variable],lr=float(lr)); last=time.perf_counter()
        for step in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            loss,_,_=self._explicit_o_loss(template,variable,topologies,phase)
            finite=torch.isfinite(loss)&torch.isfinite(variable).all(1)
            if not bool(finite.any()): break
            torch.where(finite,loss,torch.zeros_like(loss)).sum().div(finite.sum().clamp_min(1)).backward()
            torch.nn.utils.clip_grad_norm_([variable],10.0); opt.step()
            with torch.no_grad():
                # O variables are unwrapped fractional coordinates.  A generous
                # range preserves smooth crossing through cell boundaries while
                # preventing an unstable branch from wandering multiple cells.
                psize = self._shared_prefix_size(template)
                variable[:, psize:].clamp_(-0.75, 1.75)
                post,_,_=self._explicit_o_loss(template,variable,topologies,phase)
                improved=torch.isfinite(post)&(post<best_loss)
                best_loss=torch.where(improved,post,best_loss); best[improved]=variable.detach()[improved]
                bad=~torch.isfinite(variable).all(1)
                if bool(bad.any()): variable[bad]=best[bad]
            now=time.perf_counter()
            if heartbeat is not None and (step==0 or now-last>=10 or step+1==int(steps)):
                heartbeat(phase,step+1,int(steps),float(best_loss.min().detach().cpu())); last=now
        return best

    def _strict_explicit_o(self, template: dict, cell: np.ndarray, parent_frac: np.ndarray,
                           child_frac: np.ndarray, topology: dict) -> dict:
        labels=tuple(template["expanded_labels"]); child_label=topology["child_label"]
        nparent=len(labels); nchild=len(child_frac)
        parent_cart=parent_frac@cell; child_cart=child_frac@cell
        assigned=set(); radial=[]; child_z=[]; parent_vec=[[] for _ in range(nparent)]
        assigned_ok=True
        for g,images in enumerate(topology["groups"]):
            reverse=[]
            for p,shift in images:
                vec=child_cart[g]-(parent_cart[p]+np.asarray(shift)@cell)
                d=float(np.linalg.norm(vec)); ch=self.model.pair(labels[p],child_label)
                assigned.add((p,g)); assigned_ok &= ch.sampling_min-1e-7<=d<=ch.sampling_max+1e-7
                radial.append(abs(d-ch.mu)); parent_vec[p].append(vec); reverse.append(-vec)
            obs=_angles_np(np.asarray(reverse)); target=self.model.template_angles[child_label]
            sig=self.model.template_angle_sigma[child_label]
            if len(obs)!=len(target): child_z.append(float('inf'))
            else: child_z.extend((np.abs(obs-target)/np.maximum(sig,1e-3)).tolist())
        parent_z=[]
        exact_parent=True
        for p,label in enumerate(labels):
            exact_parent &= len(parent_vec[p])==self.model.template_cn[label]
            obs=_angles_np(np.asarray(parent_vec[p])); target=self.model.template_angles[label]
            sig=self.model.template_angle_sigma[label]
            if len(obs)!=len(target): parent_z.append(float('inf'))
            else: parent_z.extend((np.abs(obs-target)/np.maximum(sig,1e-3)).tolist())
        # rigorous minimum-image audit over [-2,2]^3
        shifts=np.asarray([[i,j,k] for i in range(-2,3) for j in range(-2,3) for k in range(-2,3)],float)
        final=np.vstack([parent_frac,child_frac])
        delta=final[None,:,None,:]-final[:,None,None,:]+shifts[None,None,:,:]
        dist=np.linalg.norm(np.einsum('...i,ij->...j',delta,cell),axis=-1)
        zero=int(np.flatnonzero(np.all(shifts==0,axis=1))[0])
        for i in range(len(final)): dist[i,i,zero]=np.inf
        min_physical=float(np.min(dist))
        unselected_count=0; minimum_unselected=float('inf')
        for p,label in enumerate(labels):
            ch=self.model.pair(label,child_label)
            for g in range(nchild):
                if (p,g) in assigned: continue
                d=float(np.min(np.linalg.norm((child_frac[g][None,:]-parent_frac[p][None,:]+shifts)@cell,axis=1)))
                minimum_unselected=min(minimum_unselected,d)
                if d<ch.first_shell_cutoff-1e-7: unselected_count+=1
        parent_zmax=float(np.max(parent_z)) if parent_z else 0.0
        child_zmax=float(np.max(child_z)) if child_z else 0.0
        valid=bool(assigned_ok and exact_parent and unselected_count==0
                   and min_physical>=self.minimum_distance
                   and parent_zmax<=self.angular_vector_z_max
                   and child_zmax<=self.angular_vector_z_max)
        return {"exact_target_cn_fraction":1.0 if exact_parent else 0.0,
                "assigned_bond_fraction":1.0 if assigned_ok else 0.0,
                "local_radial_mae_A":float(np.mean(radial)) if radial else 0.0,
                "local_radial_vector_max_A":float(np.max(radial)) if radial else 0.0,
                "parent_angular_vector_z_max":parent_zmax,
                "child_angular_vector_z_max":child_zmax,
                "minimum_physical_distance_A":min_physical,
                "minimum_unselected_distance_A":minimum_unselected,
                "unselected_first_shell_contact_count":int(unselected_count),
                "full_contact_shell_valid":unselected_count==0,
                "strict_valid":valid}

    # ------------------------------------------------------------------
    # v48 hybrid: modern chemistry-based Ti framework + literal old-school O
    # ------------------------------------------------------------------
    @staticmethod
    def _geometry_descriptor(frac: np.ndarray, cell: np.ndarray, owners=None) -> np.ndarray:
        frac = np.asarray(frac, float) % 1.0
        cell = np.asarray(cell, float)
        owners = None if owners is None else np.asarray(owners, int)
        same, other = [], []
        for i in range(len(frac)):
            for j in range(i + 1, len(frac)):
                delta = frac[j][None, :] + SHIFTS - frac[i][None, :]
                d = float(np.min(np.linalg.norm(delta @ cell, axis=1)))
                (same if owners is not None and owners[i] == owners[j] else other).append(d)
        return np.asarray(sorted(same) + sorted(other), dtype=np.float32)

    def _select_diverse_oldschool_branches(self, template, framework_one, candidates, nkeep):
        with torch.no_grad():
            fw = framework_one.repeat(len(candidates), 1)
            _abc, _angles, cell, _z2, _ti, vf, _vu, owners = self._oldschool_vertices(template, fw, candidates)
            loss, _, _ = self._oldschool_loss(template, fw, candidates)
            owner_np = owners.cpu().numpy()
            desc = [self._geometry_descriptor(vf[i].cpu().numpy(), cell[i].cpu().numpy(), owner_np)
                    for i in range(len(candidates))]
            scores = loss.detach().cpu().numpy()
        if len(candidates) <= nkeep:
            return candidates
        selected = [int(np.argmin(scores))]
        mind = np.full(len(candidates), np.inf)
        while len(selected) < min(int(nkeep), len(candidates)):
            last = desc[selected[-1]]
            for i, d in enumerate(desc):
                if i in selected:
                    mind[i] = -np.inf
                else:
                    mind[i] = min(mind[i], float(np.sqrt(np.mean((d-last)**2))))
            best = int(np.argmax(mind))
            if not np.isfinite(mind[best]) or mind[best] < self.oxygen_proposal_descriptor_tol:
                break
            selected.append(best)
        return candidates[torch.as_tensor(selected, dtype=torch.long, device=self.device)]


    @staticmethod
    def _triangle_circumcenter_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray):
        """Return circumcenter/radius for a non-collinear 3D triangle."""
        a = np.asarray(a, float); b = np.asarray(b, float); c = np.asarray(c, float)
        u = b - a; v = c - a
        w = np.cross(u, v)
        w2 = float(np.dot(w, w))
        if w2 <= 1.0e-12:
            return None, float("inf")
        center = a + (
            float(np.dot(u, u)) * np.cross(v, w)
            + float(np.dot(v, v)) * np.cross(w, u)
        ) / (2.0 * w2)
        return center, float(np.linalg.norm(center - a))

    @staticmethod
    def _periodic_cart_distance(frac_a: np.ndarray, frac_b: np.ndarray, cell: np.ndarray) -> float:
        d = np.asarray(frac_b, float)[None, :] + SHIFTS - np.asarray(frac_a, float)[None, :]
        return float(np.min(np.linalg.norm(d @ np.asarray(cell, float), axis=1)))

    def _framework_feasibility_descriptor(self, template: dict, framework_one: torch.Tensor) -> dict:
        """Cheap Ti-only descriptor derived from the required shared-site chemistry.

        This deliberately does *not* force the learned Ti--Ti polymorph fingerprint.
        Its primary signal is whether triples of parent centres can geometrically host
        a child site while keeping all parent--child bonds inside the learned window.
        The learned Ti-framework statistics are retained only as a weak diagnostic.
        """
        with torch.no_grad():
            abc, _angles, cell_t, _z2, frac_t = self._framework_geometry(template, framework_one)
        cell = cell_t[0].detach().cpu().numpy()
        frac = frac_t[0].detach().cpu().numpy() % 1.0
        abc_np = abc[0].detach().cpu().numpy()
        nparent = len(frac)
        child_labels = tuple(self.plan.get("children", ()))
        if len(child_labels) != 1:
            return {
                "shared_o_cover_fraction": 0.0,
                "shared_o_candidate_count": 0.0,
                "shared_o_o_angle_z_q50": 99.0,
                "shared_o_ti_angle_z_max": 99.0,
                "framework_prior_q90": 99.0,
                "minimum_ti_distance_A": 0.0,
                "volume_per_parent_A3": float(abs(np.linalg.det(cell)) / max(nparent, 1)),
                "aspect_ratio": float(np.max(abc_np) / max(np.min(abc_np), 1.0e-8)),
            }
        child = child_labels[0]
        channels = [self.model.pair(label, child) for label in template["expanded_labels"]]
        r_mu = float(np.mean([x.mu for x in channels]))
        r_min = float(np.min([x.sampling_min for x in channels]))
        r_max = float(np.max([x.sampling_max for x in channels]))

        # Generic Ti packing descriptors from physical periodic neighbours.
        nearest_rows = []
        min_ti = float("inf")
        for i in range(nparent):
            ds = []
            for j in range(nparent):
                for sid, sh in enumerate(SHIFTS):
                    if i == j and sid == ZERO_SHIFT:
                        continue
                    d = float(np.linalg.norm((frac[j] + sh - frac[i]) @ cell))
                    ds.append(d); min_ti = min(min_ti, d)
            ds.sort()
            nearest_rows.append(ds[:6])
        nearest_mean = np.mean(np.asarray(nearest_rows, float), axis=0) if nearest_rows else np.zeros(6)

        # v75 learned-framework diagnostics are site resolved.  The historical
        # ``framework_prior_q90`` is retained for continuity, while explicit radial
        # and shell-pair angular q90 values expose where a framework leaves the
        # learned chemistry-valid Ti manifold.
        prior_z = []
        radial_site_z = []
        radial_rank_z = defaultdict(list)
        radial_rank_d = defaultdict(list)
        angular_site_z = []
        angular_shell_z = defaultdict(list)
        thresholds = []
        for i, label in enumerate(template["expanded_labels"]):
            fm = self.model.framework_models.get(label, {})
            target = np.asarray(fm.get("radial_mean_A", []), float)
            sigma = np.asarray(fm.get("radial_sigma_A", []), float)
            k = int(fm.get("neighbor_count", len(target)) or 0)
            if k <= 0 or len(target) != k or len(sigma) != k or len(nearest_rows[i]) < k:
                continue
            thresholds.append(float(fm.get("score_q90_max", 3.5)))
            nd = np.asarray(nearest_rows[i][:k], float)
            rz = np.abs(nd - target) / np.maximum(sigma, 0.06)
            radial_site_z.extend(rz.tolist())
            for rank in range(k):
                radial_rank_z[int(rank)].append(float(rz[rank]))
                radial_rank_d[int(rank)].append(float(nd[rank]))

            # Recover the same ordered neighbour vectors used by the learner.
            neigh = []
            for j in range(nparent):
                for sid, sh in enumerate(SHIFTS):
                    if i == j and sid == ZERO_SHIFT:
                        continue
                    vec = (frac[j] + sh - frac[i]) @ cell
                    neigh.append((float(np.linalg.norm(vec)), np.asarray(vec, float)))
            neigh.sort(key=lambda x: x[0])
            if len(neigh) < k:
                continue
            vec = np.vstack([x[1] for x in neigh[:k]])
            unit = vec / np.maximum(np.linalg.norm(vec, axis=1)[:, None], EPS)
            for group in fm.get("angular_shell_pair_groups", []):
                pairs = [(int(a), int(b)) for a, b in group.get("neighbor_rank_pairs", [])]
                am = np.asarray(group.get("angular_mean_deg", []), float)
                sg = np.asarray(group.get("angular_sigma_deg", []), float)
                if not pairs or len(am) != len(pairs) or len(sg) != len(pairs):
                    continue
                obs = []
                valid_group = True
                for a, b in pairs:
                    if a >= k or b >= k:
                        valid_group = False; break
                    obs.append(float(np.degrees(np.arccos(np.clip(np.dot(unit[a], unit[b]), -1.0, 1.0)))))
                if not valid_group:
                    continue
                obs = np.sort(np.asarray(obs, float))
                az = np.abs(obs - am) / np.maximum(sg, 1.0e-3)
                angular_site_z.extend(az.tolist())
                sp = group.get("shell_pair", [0, 0])
                angular_shell_z[f"{int(sp[0])}{int(sp[1])}"].extend(az.tolist())

            # Historical prior uses the mean ranked distances over all Ti sites.
            if len(target) == len(nearest_mean):
                prior_z.extend((np.abs(nearest_mean - target) / np.maximum(sigma, 0.06)).tolist())
        framework_prior_q90 = float(np.quantile(prior_z, 0.90)) if prior_z else 0.0
        framework_prior_radial_q90 = float(np.quantile(radial_site_z, 0.90)) if radial_site_z else 0.0
        framework_prior_radial_z_max = float(np.max(radial_site_z)) if radial_site_z else 0.0
        framework_prior_angular_q90 = float(np.quantile(angular_site_z, 0.90)) if angular_site_z else 0.0
        framework_prior_angular_z_max = float(np.max(angular_site_z)) if angular_site_z else 0.0
        prior_threshold = float(np.median(thresholds)) if thresholds else 3.5
        prior_threshold = max(prior_threshold, 1.0e-3)
        prior_envelope_excess = (max(0.0, (framework_prior_radial_q90-prior_threshold)/prior_threshold)**2
                                 + float(self.framework_prior_angular_factor)
                                 * max(0.0, (framework_prior_angular_q90-prior_threshold)/prior_threshold)**2)

        # v60 first implementation targets the currently validated Ti4/O8 shared-site case.
        # Each child omits exactly one of four parents; a balanced Ti4O8 cover therefore
        # needs two geometrically plausible child sites in each omission class.
        candidate_by_omission = defaultdict(list)
        if nparent == 4 and int(self.target_counts.get(child, 0)) == 8:
            target_o_angles = np.asarray(self.model.template_angles[child], float)
            target_o_sigma = np.maximum(np.asarray(self.model.template_angle_sigma[child], float), 1.0e-3)
            cart0 = frac @ cell
            for combo in itertools.combinations(range(nparent), 3):
                omitted = int(next(x for x in range(nparent) if x not in combo))
                ia, ib, ic = combo
                a = cart0[ia]
                image_b = sorted(
                    [(float(np.linalg.norm((frac[ib] + sh - frac[ia]) @ cell)), sh.copy()) for sh in SHIFTS],
                    key=lambda x: x[0],
                )[:8]
                image_c = sorted(
                    [(float(np.linalg.norm((frac[ic] + sh - frac[ia]) @ cell)), sh.copy()) for sh in SHIFTS],
                    key=lambda x: x[0],
                )[:8]
                for db, sb in image_b:
                    if db > 2.0 * r_max + 0.15:
                        continue
                    b = (frac[ib] + sb) @ cell
                    for dc, sc in image_c:
                        if dc > 2.0 * r_max + 0.15:
                            continue
                        c = (frac[ic] + sc) @ cell
                        dbc = float(np.linalg.norm(c - b))
                        if dbc > 2.0 * r_max + 0.15:
                            continue
                        center, radius = self._triangle_circumcenter_3d(a, b, c)
                        if center is None or radius > r_max + 1.0e-8:
                            continue
                        # Use the least strained equal-radius point still inside the learned bond window.
                        r_eff = float(np.clip(max(r_mu, radius + 1.0e-6), r_min, r_max))
                        if radius > r_eff + 1.0e-8:
                            continue
                        normal = np.cross(b - a, c - a)
                        nn = float(np.linalg.norm(normal))
                        if nn <= 1.0e-10:
                            continue
                        normal /= nn
                        h = math.sqrt(max(r_eff * r_eff - radius * radius, 0.0))
                        for sign in (-1.0, 1.0):
                            opos = center + sign * h * normal
                            reverse = np.asarray([a - opos, b - opos, c - opos], float)
                            obs = _angles_np(reverse)
                            if len(obs) == len(target_o_angles):
                                oz = np.abs(obs - target_o_angles) / target_o_sigma
                                o_z = float(np.mean(oz))
                            else:
                                o_z = 99.0
                            ofrac = (opos @ np.linalg.inv(cell)) % 1.0
                            entry = {
                                "frac": ofrac,
                                "o_z": o_z,
                                "vectors": {
                                    int(ia): opos - a,
                                    int(ib): opos - b,
                                    int(ic): opos - c,
                                },
                            }
                            # Periodically cluster equivalent candidate child sites and retain
                            # the member with the best learned child-angle score.
                            replaced = False
                            for q, old in enumerate(candidate_by_omission[omitted]):
                                if self._periodic_cart_distance(old["frac"], ofrac, cell) < 0.18:
                                    if o_z < old["o_z"]:
                                        candidate_by_omission[omitted][q] = entry
                                    replaced = True
                                    break
                            if not replaced:
                                candidate_by_omission[omitted].append(entry)

        counts = [len(candidate_by_omission.get(i, [])) for i in range(nparent)]
        if nparent == 4:
            cover_fraction = float(sum(min(x, 2) for x in counts) / 8.0)
            all_o_z = [x["o_z"] for rows in candidate_by_omission.values() for x in rows]
            o_z_q50 = float(np.median(all_o_z)) if all_o_z else 99.0
            ti_angle_z_max = 99.0
        else:
            # The balanced omission descriptor is specific to four crystallographic
            # parents.  For other native cell sizes it must be neutral rather than
            # falsely declaring the framework chemically impossible.  The actual
            # floating-port/analytic-O stages remain the authoritative feasibility test.
            cover_fraction = 0.5
            o_z_q50 = 4.0
            ti_angle_z_max = 6.0
        if nparent == 4 and all(x >= 2 for x in counts):
            selected = []
            for omitted in range(4):
                selected.extend(sorted(candidate_by_omission[omitted], key=lambda x: x["o_z"])[:2])
            ti_site_z = []
            for i, label in enumerate(template["expanded_labels"]):
                vecs = [entry["vectors"][i] for entry in selected if i in entry["vectors"]]
                cn = int(self.model.template_cn[label])
                if len(vecs) != cn:
                    ti_site_z.append(99.0); continue
                obs = _angles_np(np.asarray(vecs, float))
                target = np.asarray(self.model.template_angles[label], float)
                sig = np.maximum(np.asarray(self.model.template_angle_sigma[label], float), 1.0e-3)
                if len(obs) != len(target):
                    ti_site_z.append(99.0)
                else:
                    ti_site_z.append(float(np.max(np.abs(obs - target) / sig)))
            ti_angle_z_max = float(max(ti_site_z)) if ti_site_z else 99.0

        return {
            "shared_o_cover_fraction": cover_fraction,
            "shared_o_candidate_count": float(sum(counts)),
            "shared_o_o_angle_z_q50": o_z_q50,
            "shared_o_ti_angle_z_max": ti_angle_z_max,
            "framework_prior_q90": framework_prior_q90,
            "framework_prior_radial_q90": framework_prior_radial_q90,
            "framework_prior_radial_z_max": framework_prior_radial_z_max,
            "framework_prior_angular_q90": framework_prior_angular_q90,
            "framework_prior_angular_z_max": framework_prior_angular_z_max,
            "framework_prior_envelope_threshold": prior_threshold,
            "framework_prior_inside_envelope": bool(framework_prior_radial_q90 <= prior_threshold
                                                     and framework_prior_angular_q90 <= prior_threshold),
            "framework_prior_envelope_excess": float(prior_envelope_excess),
            **{f"framework_prior_radial_rank{rank+1}_z_mean":
               float(np.mean(radial_rank_z.get(rank, [0.0]))) for rank in range(6)},
            **{f"framework_prior_radial_rank{rank+1}_distance_A":
               float(np.mean(radial_rank_d.get(rank, [0.0]))) for rank in range(6)},
            "framework_prior_angular_shell00_q90": float(np.quantile(angular_shell_z["00"], 0.90)) if angular_shell_z.get("00") else 0.0,
            "framework_prior_angular_shell01_q90": float(np.quantile(angular_shell_z["01"], 0.90)) if angular_shell_z.get("01") else 0.0,
            "framework_prior_angular_shell11_q90": float(np.quantile(angular_shell_z["11"], 0.90)) if angular_shell_z.get("11") else 0.0,
            "minimum_ti_distance_A": float(min_ti),
            "volume_per_parent_A3": float(abs(np.linalg.det(cell)) / max(nparent, 1)),
            "aspect_ratio": float(np.max(abc_np) / max(np.min(abc_np), 1.0e-8)),
        }

    @staticmethod
    def _framework_memory_vector(features: dict) -> np.ndarray:
        """Low-dimensional chemistry-feasibility vector, not a structure identity."""
        vals = np.asarray([
            float(features.get("shared_o_cover_fraction", 0.0)),
            min(float(features.get("shared_o_candidate_count", 0.0)), 32.0) / 16.0,
            min(float(features.get("shared_o_o_angle_z_q50", 99.0)), 12.0) / 4.0,
            min(float(features.get("shared_o_ti_angle_z_max", 99.0)), 20.0) / 6.0,
            min(float(features.get("framework_prior_q90", 99.0)), 20.0) / 6.0,
            float(features.get("minimum_ti_distance_A", 0.0)) / 3.0,
            float(features.get("volume_per_parent_A3", 0.0)) / 20.0,
            min(float(features.get("aspect_ratio", 10.0)), 8.0) / 3.0,
        ], dtype=float)
        return np.nan_to_num(vals, nan=10.0, posinf=10.0, neginf=-10.0)

    def _read_framework_memory(self) -> list[dict]:
        """Incrementally refresh strict-chemistry search memory from append-only JSONL."""
        if not self.framework_memory_path:
            return []
        path = Path(self.framework_memory_path)
        if not path.exists():
            return self._chem_memory_rows
        import fcntl
        try:
            size = path.stat().st_size
        except OSError:
            return self._chem_memory_rows
        if size < self._chem_memory_offset:
            self._chem_memory_rows = []
            self._chem_memory_offset = 0
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            handle.seek(self._chem_memory_offset)
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row.get("features"), dict):
                        self._chem_memory_rows.append(row)
                except Exception:
                    continue
            self._chem_memory_offset = handle.tell()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return self._chem_memory_rows

    def _framework_memory_score(self, features: dict) -> dict:
        """Predict strict chemistry and topology separately; never hard-reject here."""
        rows = self._read_framework_memory()
        x = self._framework_memory_vector(features)
        if not rows:
            return {
                "chemistry_memory_samples": 0,
                "chemistry_predicted_strict": 0.02,
                "chemistry_predicted_topology": 0.10,
                "chemistry_memory_novelty": 9.0,
                "chemistry_memory_uncertainty": 1.0,
            }
        vectors = np.asarray([self._framework_memory_vector(row["features"]) for row in rows], float)
        distances = np.sqrt(np.mean((vectors - x[None, :]) ** 2, axis=1))
        novelty = float(np.min(distances))
        k = min(self.framework_memory_k, len(rows))
        ids = np.argsort(distances)[:k]
        w = np.exp(-0.5 * (distances[ids] / 0.55) ** 2) + 1.0e-6
        y_strict = np.asarray([float(bool(rows[i].get("strict_success", False))) for i in ids], float)
        y_top = np.asarray([float(bool(rows[i].get("o_topology_success", False))) for i in ids], float)
        # Weak Beta-like priors avoid all-zero collapse during the first few successes.
        p_strict = float((0.02 + np.sum(w * y_strict)) / (1.0 + np.sum(w)))
        p_top = float((0.10 + np.sum(w * y_top)) / (1.0 + np.sum(w)))
        uncertainty = float(1.0 / math.sqrt(1.0 + np.sum(w)))
        return {
            "chemistry_memory_samples": int(len(rows)),
            "chemistry_predicted_strict": p_strict,
            "chemistry_predicted_topology": p_top,
            "chemistry_memory_novelty": novelty,
            "chemistry_memory_uncertainty": uncertainty,
        }

    def _append_framework_memory(self, features: dict, outcome: dict):
        if not self.framework_memory_path:
            return
        import fcntl
        path = Path(self.framework_memory_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"version": 2, "features": {k: float(v) for k, v in features.items()}, **outcome}
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush(); os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _framework_structure_vector(self, template: dict, framework_one: torch.Tensor) -> np.ndarray:
        """Permutation/rotation-insensitive Ti-only descriptor for attraction-basin memory.

        It deliberately describes structure, not chemical fitness: periodic nearest-Ti
        distance spectrum, local angular spectrum, and cell-metric invariants.  The
        exact proper-rotation matcher remains authoritative after full refinement.
        """
        with torch.no_grad():
            _abc, _angles, cell_t, _z2, frac_t = self._framework_geometry(template, framework_one)
        cell = np.asarray(cell_t[0].detach().cpu().numpy(), float)
        frac = np.asarray(frac_t[0].detach().cpu().numpy(), float) % 1.0
        all_dist = []
        all_ang = []
        for i in range(len(frac)):
            neigh = []
            for j in range(len(frac)):
                for sid, sh in enumerate(SHIFTS):
                    if i == j and sid == ZERO_SHIFT:
                        continue
                    vec = (frac[j] + sh - frac[i]) @ cell
                    dist = float(np.linalg.norm(vec))
                    if dist > 1.0e-8:
                        neigh.append((dist, vec))
            neigh.sort(key=lambda x: x[0])
            nearest = neigh[:6]
            all_dist.extend([x[0] for x in nearest])
            vecs = np.asarray([x[1] for x in nearest], float)
            if len(vecs) >= 2:
                unit = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1.0e-12)
                for a in range(len(unit)):
                    for b in range(a + 1, len(unit)):
                        c = float(np.clip(np.dot(unit[a], unit[b]), -1.0, 1.0))
                        all_ang.append(math.degrees(math.acos(c)))
        dist_spec = np.sort(np.asarray(all_dist, float))
        if len(dist_spec) < 24:
            dist_spec = np.pad(dist_spec, (0, 24 - len(dist_spec)), constant_values=9.0)
        dist_spec = dist_spec[:24] / 4.0
        ang = np.asarray(all_ang, float)
        if len(ang):
            ang_q = np.quantile(ang, np.linspace(0.1, 0.9, 9)) / 180.0
        else:
            ang_q = np.ones(9, float)
        lengths = np.linalg.norm(cell, axis=1)
        volume = max(float(abs(np.linalg.det(cell))), 1.0e-8)
        scale = volume ** (1.0 / 3.0)
        shape = np.sort(lengths / max(scale, 1.0e-8))
        cosang = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            cosang.append(float(np.dot(cell[i], cell[j]) /
                                max(np.linalg.norm(cell[i]) * np.linalg.norm(cell[j]), 1.0e-12)))
        metric = np.concatenate([shape, np.sort(np.asarray(cosang, float)),
                                 np.asarray([volume / max(len(frac), 1) / 20.0])])
        vec = np.concatenate([dist_spec, ang_q, metric]).astype(float)
        return np.nan_to_num(vec, nan=9.0, posinf=9.0, neginf=-9.0)

    @staticmethod
    def _construction_context_key(template: dict) -> str:
        """Exact crystallographic construction context for Ti-basin memory/dedup.

        The same Ti point set reached through a different space group or Wyckoff/orbit
        decomposition is *not* treated as the same construction state, because the
        symmetry relations among attached local templates differ.
        """
        return "spg={}|wps={}|labels={}".format(
            int(template["spg"]),
            ",".join(str(int(x)) for x in template["wps"]),
            ",".join(str(x) for x in template["site_labels"]),
        )

    def _refresh_basin_memory(self):
        if not self.framework_basin_memory_path:
            return
        path = Path(self.framework_basin_memory_path)
        if not path.exists():
            return
        import fcntl
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < self._basin_memory_offset:
            self._basin_memory_offset = 0
            self._basin_stats = {}
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            handle.seek(self._basin_memory_offset)
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    context = str(row["context"])
                    basin_id = int(row["basin_id"])
                    vec = np.asarray(row["structural_vector"], float)
                except Exception:
                    continue
                key = (context, basin_id)
                stat = self._basin_stats.get(key)
                if stat is None:
                    self._basin_stats[key] = {
                        "count": 1, "sum": vec.copy(), "sumsq": vec * vec,
                        "duplicates": int(bool(row.get("duplicate", False))),
                    }
                elif len(stat["sum"]) == len(vec):
                    stat["count"] += 1
                    stat["sum"] += vec
                    stat["sumsq"] += vec * vec
                    stat["duplicates"] += int(bool(row.get("duplicate", False)))
            self._basin_memory_offset = handle.tell()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _basin_memory_score(self, structural_vector: np.ndarray, template: dict) -> dict:
        self._refresh_basin_memory()
        context = self._construction_context_key(template)
        keys = [key for key in self._basin_stats if key[0] == context]
        if not keys:
            return {
                "basin_memory_count": 0, "basin_nearest_id": -1,
                "basin_nearest_observations": 0, "basin_novelty": 9.0,
                "basin_nearest_spread": 0.0, "basin_skip_threshold": 0.0,
                "basin_skip": False,
            }
        keys.sort(key=lambda x: x[1])
        centroids = np.asarray([self._basin_stats[k]["sum"] / self._basin_stats[k]["count"] for k in keys], float)
        x = np.asarray(structural_vector, float)
        distances = np.sqrt(np.mean((centroids - x[None, :]) ** 2, axis=1))
        kidx = int(np.argmin(distances))
        key = keys[kidx]
        basin_id = int(key[1])
        stat = self._basin_stats[key]
        mean = stat["sum"] / stat["count"]
        var = np.maximum(stat["sumsq"] / stat["count"] - mean * mean, 0.0)
        spread = float(math.sqrt(float(np.mean(var))))
        threshold = float(np.clip(0.020 + 2.5 * spread, 0.035, 0.10))
        novelty = float(distances[kidx])
        skip = bool(stat["count"] >= 3 and novelty <= threshold)
        return {
            "basin_memory_count": int(len(keys)),
            "basin_nearest_id": basin_id,
            "basin_nearest_observations": int(stat["count"]),
            "basin_novelty": novelty,
            "basin_nearest_spread": spread,
            "basin_skip_threshold": threshold,
            "basin_skip": skip,
        }

    def _append_basin_memory(self, structural_vector: np.ndarray, basin_id: int,
                             duplicate: bool, template: dict):
        if not self.framework_basin_memory_path:
            return
        import fcntl
        path = Path(self.framework_basin_memory_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "version": 2,
            "context": self._construction_context_key(template),
            "basin_id": int(basin_id),
            "duplicate": bool(duplicate),
            "structural_vector": [round(float(x), 6) for x in np.asarray(structural_vector, float)],
        }
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush(); os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _diversify_known_basin(self, template: dict, framework_one: torch.Tensor, trial: int) -> torch.Tensor:
        """Move a cheap Ti proposal away from a learned attraction basin without changing symmetry.

        Only lattice/Wyckoff free variables are perturbed.  The candidate then receives
        a very short packing re-optimization, so this is far cheaper than spending a Ti
        token on full refinement.
        """
        trial = max(1, int(trial))
        x = framework_one.detach().clone()
        nlat = len(template["spec"])
        with torch.no_grad():
            if nlat:
                x[:, :nlat] += (0.10 + 0.04 * trial) * torch.randn_like(x[:, :nlat])
            if x.shape[1] > nlat:
                # Wyckoff coordinates live as logits; a larger perturbation is needed
                # to produce a meaningful fractional-coordinate displacement.
                x[:, nlat:] += (0.65 + 0.25 * trial) * torch.randn_like(x[:, nlat:])
        return self._optimize_framework(template, x, steps=6, heartbeat=None)

    def _select_intelligent_frameworks(self, template: dict, framework: torch.Tensor, keep: int):
        """Breadth-first Ti preselection before expensive refinement.

        The old Ti4 omission descriptor is still computed and logged, but v78 does
        not use it to decide which starts receive Ti tokens.  One retained start is
        chosen for generic near-hypergraph chemistry; the remaining slots maximize
        Ti-framework structural novelty.  Search-memory predictions are diagnostic
        only here because their feature vector contains historical Ti4-specific terms.
        """
        raw_features = []
        chemistry = []
        structural = []
        basin = []
        for i in range(len(framework)):
            feat = self._framework_feasibility_descriptor(template, framework[i:i + 1])
            svec = self._framework_structure_vector(template, framework[i:i + 1])
            raw_features.append(feat)
            chemistry.append(self._framework_memory_score(feat))
            structural.append(svec)
            bdiag = self._basin_memory_score(svec, template)
            bdiag["basin_diversification_attempted"] = False
            bdiag["basin_diversification_trials"] = 0
            bdiag["basin_original_novelty"] = float(bdiag.get("basin_novelty", 9.0))
            basin.append(bdiag)

        # Divert exact/context-local attraction-basin repeats before spending a token.
        for i in range(len(framework)):
            if not bool(basin[i].get("basin_skip", False)):
                continue
            original_novelty = float(basin[i].get("basin_novelty", 0.0))
            best = (original_novelty, framework[i:i + 1].detach().clone(),
                    raw_features[i], chemistry[i], structural[i], basin[i], 0)
            for trial in (1, 2):
                candidate = self._diversify_known_basin(template, framework[i:i + 1], trial)
                feat = self._framework_feasibility_descriptor(template, candidate)
                svec = self._framework_structure_vector(template, candidate)
                cdiag = self._framework_memory_score(feat)
                bdiag = self._basin_memory_score(svec, template)
                novelty = float(bdiag.get("basin_novelty", 0.0))
                if novelty > best[0]:
                    best = (novelty, candidate, feat, cdiag, svec, bdiag, trial)
                if not bool(bdiag.get("basin_skip", False)):
                    best = (novelty, candidate, feat, cdiag, svec, bdiag, trial)
                    break
            _novelty, candidate, feat, cdiag, svec, bdiag, trials = best
            framework[i:i + 1] = candidate
            raw_features[i] = feat
            chemistry[i] = cdiag
            structural[i] = svec
            bdiag = dict(bdiag)
            bdiag["basin_diversification_attempted"] = True
            bdiag["basin_diversification_trials"] = int(trials)
            bdiag["basin_original_novelty"] = original_novelty
            basin[i] = bdiag

        eligible = [i for i, b in enumerate(basin) if not bool(b["basin_skip"])]
        if not eligible:
            return framework[:0], [], [], [], basin, raw_features, chemistry, structural

        # Evaluate the same generic near-hypergraph surrogate used in full Ti
        # refinement, with the exploratory branch mode and no batch-diversity term.
        with torch.no_grad():
            modes = torch.full((len(framework),), 2, dtype=torch.long, device=self.device)
            gloss, gdetail, _ = self._framework_loss(
                template, framework, branch_modes=modes, diversity_weight=0.0)
        generic = []
        for i in range(len(framework)):
            generic.append({
                "preselect_generic_loss": float(gloss[i]),
                "preselect_near_triplet_loss": float(gdetail["framework_near_triplet_loss"][i]),
                "preselect_incidence_deficit_loss": float(gdetail["framework_incidence_deficit_loss"][i]),
                "preselect_min_good_triplets": float(gdetail["framework_min_near_triplet_good_count"][i]),
            })

        # First branch: generic chemistry anchor.  Crucially this does not use the
        # Ti4 omission-class cover fraction/candidate count.
        def chemistry_key(i):
            g = generic[i]
            return (g["preselect_near_triplet_loss"]
                    + 1.6 * g["preselect_incidence_deficit_loss"]
                    + 0.08 * math.log1p(max(g["preselect_generic_loss"], 0.0)))

        selected = [min(eligible, key=chemistry_key)]

        # Additional branches: max-min Ti-framework novelty.  A small generic
        # chemistry term breaks ties but cannot overwhelm geometric breadth.
        while len(selected) < min(int(keep), len(eligible)):
            best = None
            for i in eligible:
                if i in selected:
                    continue
                within_batch_novelty = min(
                    float(np.sqrt(np.mean((structural[i] - structural[j]) ** 2))) for j in selected
                )
                basin_novelty = min(float(basin[i].get("basin_novelty", 0.0)), 1.5)
                score = (2.4 * min(within_batch_novelty, 0.8)
                         + 0.35 * basin_novelty
                         - 0.08 * math.log1p(max(chemistry_key(i), 0.0)))
                if best is None or score > best[0]:
                    best = (score, i)
            if best is None:
                break
            selected.append(int(best[1]))

        sel = torch.as_tensor(selected, dtype=torch.long, device=self.device)
        combined = [
            {**raw_features[i], **chemistry[i], **basin[i], **generic[i],
             "preselection_ti4_omission_score_used": False}
            for i in selected
        ]
        return framework[sel], combined, selected, [structural[i] for i in selected], basin, raw_features, chemistry, structural

    def _claim_ti_tokens(self, requested: int) -> tuple[int, int, int]:
        """Atomically reserve full-refinement Ti tokens across all builder workers."""
        requested = max(0, int(requested))
        if requested == 0:
            used = int(self.ti_token_counter.value) if self.ti_token_counter is not None else 0
            return 0, used, used
        if self.ti_token_counter is None:
            return requested, 0, requested
        with self.ti_token_counter.get_lock():
            before = int(self.ti_token_counter.value)
            remaining = max(int(self.ti_token_budget) - before, 0)
            claimed = min(requested, remaining)
            self.ti_token_counter.value = before + claimed
            after = int(self.ti_token_counter.value)
        return int(claimed), int(before), int(after)

    def _oxygen_branch_budget(self, features: dict, chemistry: dict) -> int:
        """Allocate O exploration effort without converting low likelihood into a hard ban."""
        cover = float(features.get("shared_o_cover_fraction", 0.0))
        ps = float(chemistry.get("chemistry_predicted_strict", 0.02))
        pt = float(chemistry.get("chemistry_predicted_topology", 0.10))
        novelty = float(chemistry.get("chemistry_memory_novelty", 9.0))
        uncertainty = float(chemistry.get("chemistry_memory_uncertainty", 1.0))
        if cover < 0.75:
            n = 4
        elif ps >= 0.18:
            n = 32
        elif pt >= 0.35:
            n = 24
        elif novelty >= 0.80 or uncertainty >= 0.55:
            n = 16
        else:
            n = 8
        return max(1, min(int(self.octahedral_branches), int(n)))

    def _claim_ti_framework(self, template, framework_one):
        """Claim an exact Ti basin *within the same crystallographic construction context*.

        Geometry-only Ti identity is insufficient here: identical Ti point sets reached
        through different SG/Wyckoff decompositions generate different symmetry-related
        local-template orientations.  Cross-context structures therefore remain eligible
        for O construction and are only compared later at the full-structure dedup stage.
        """
        if not self.ti_registry_path:
            return True, {"ti_duplicate": False, "ti_match_checks": 0, "ti_basin_id": -1,
                          "ti_construction_context": self._construction_context_key(template)}
        import fcntl
        with torch.no_grad():
            _abc, _angles, cell, _z2, frac = self._framework_geometry(template, framework_one)
        symbols = tuple(self.model.block_atoms[x][0][0] for x in template["expanded_labels"])
        context = self._construction_context_key(template)
        current = {"cell": cell[0].cpu().numpy(), "frac": frac[0].cpu().numpy(), "symbols": symbols}
        path = Path(self.ti_registry_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checks = 0
        valid_records = 0
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            for line in handle:
                if not line.strip():
                    continue
                try:
                    previous = json.loads(line)
                except Exception:
                    continue
                basin_id = valid_records
                valid_records += 1
                if str(previous.get("context", "")) != context:
                    continue
                checks += 1
                previous_structure = previous.get("structure", previous)
                match = _strict_proper_exact_match(current, previous_structure,
                    lattice_rel_tol=0.02, angle_tol_deg=1.0, rms_tol_A=0.05, max_tol_A=0.10)
                if match is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    return False, {"ti_duplicate": True, "ti_match_checks": checks,
                                   "ti_basin_id": int(basin_id), "ti_construction_context": context, **match}
            basin_id = valid_records
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps({
                "version": 2,
                "context": context,
                "spg": int(template["spg"]),
                "wps": [int(x) for x in template["wps"]],
                "site_labels": list(template["site_labels"]),
                "structure": {
                    "cell": np.asarray(current["cell"]).tolist(),
                    "frac": (np.asarray(current["frac"]) % 1.0).tolist(),
                    "symbols": list(symbols),
                },
            }, separators=(",", ":")) + "\n")
            handle.flush(); os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return True, {"ti_duplicate": False, "ti_match_checks": checks,
                      "ti_basin_id": int(basin_id), "ti_construction_context": context}

    def _oldschool_initial_branches(self, template: dict, nbranch: int) -> torch.Tensor:
        nsite = len(template["site_labels"])
        raw = torch.randn((int(nbranch), nsite, 27), dtype=torch.float32, device=self.device)
        raw[..., :3] *= 1.5
        raw[..., 3:21] *= 0.35
        raw[..., 21:27] *= 0.5
        return raw

    def _oldschool_vertices(self, template: dict, framework: torch.Tensor, branch_raw: torch.Tensor):
        """Floating TiO6 geometry on a symmetry-parameterized Ti framework."""
        abc, angles, cell, z2_raw, ti_frac = self._framework_geometry(template, framework)
        bsz = len(branch_raw)
        inv_cell = torch.linalg.inv(cell)
        child_label = tuple(self.plan["children"])[0]
        vertex_frac, vertex_unwrapped, owners = [], [], []
        cursor = 0
        for site_id, rotations in enumerate(template["orbit_rot"]):
            norbit = rotations.shape[0]
            parent_label = template["site_labels"][site_id]
            channel = self.model.pair(parent_label, child_label)
            local = branch_raw[:, site_id]
            rlocal = self._axis_angle_rotation(local[:, :3])
            dirs = torch.einsum("vj,bij->bvi", self._base_oct, rlocal)
            dirs = dirs + 0.16 * torch.tanh(local[:, 3:21].reshape(bsz, 6, 3))
            dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp_min(1.0e-8)
            radii = float(channel.mu) + max(float(channel.sigma), 0.05) * torch.tanh(local[:, 21:27])
            dcart = dirs * radii[..., None]
            dfrac = torch.einsum("bvi,bij->bvj", dcart, inv_cell)
            transformed = torch.einsum("oij,bvj->bovi", rotations, dfrac)
            centers = ti_frac[:, cursor:cursor + norbit]
            vu = centers[:, :, None, :] + transformed
            vertex_unwrapped.append(vu.reshape(bsz, norbit * 6, 3))
            vertex_frac.append((vu % 1.0).reshape(bsz, norbit * 6, 3))
            for oid in range(norbit):
                owners.extend([cursor + oid] * 6)
            cursor += norbit
        return (abc, angles, cell, z2_raw, ti_frac,
                torch.cat(vertex_frac, 1), torch.cat(vertex_unwrapped, 1),
                torch.as_tensor(owners, dtype=torch.long, device=self.device))

    def _oldschool_loss(self, template: dict, framework: torch.Tensor, branch_raw: torch.Tensor):
        abc, angles, cell, z2_raw, ti_frac, vf, vu, owners = self._oldschool_vertices(
            template, framework, branch_raw
        )
        bsz, nvert = vu.shape[:2]
        nti = ti_frac.shape[1]
        dist_img = self._floating_vertex_distances(vu, cell)
        nshift = dist_img.shape[-1]
        sigma = self.floating_coincidence_sigma
        kernel = torch.exp(-0.5 * (dist_img / sigma).pow(2))
        owner_class = (owners[:, None] + nti * torch.arange(nshift, device=self.device)[None, :]).reshape(-1)
        owner_mass = torch.zeros((bsz, nvert, nti * nshift), dtype=kernel.dtype, device=self.device)
        owner_mass.scatter_add_(2, owner_class[None, None, :].expand(bsz, nvert, -1), kernel.reshape(bsz, nvert, -1))
        own_class = owners + nti * ZERO_SHIFT
        own_mask = torch.nn.functional.one_hot(own_class, num_classes=nti * nshift).bool()[None].expand(bsz, -1, -1)
        presence = (1.0 - torch.exp(-owner_mass)).masked_fill(own_mask, 0.0)
        rho = presence.sum(-1)
        occupancy = (rho - 2.0).pow(2).mean(1)
        overcoord = torch.relu(rho - 2.15).pow(2).mean(1)

        owner_dist = torch.full((bsz, nvert, nti * nshift), 1.0e6, dtype=dist_img.dtype, device=self.device)
        owner_dist.scatter_reduce_(2, owner_class[None, None, :].expand(bsz, nvert, -1),
                                   dist_img.reshape(bsz, nvert, -1), reduce="amin", include_self=True)
        owner_dist = owner_dist.masked_fill(own_mask, 1.0e6)
        nearest = torch.topk(owner_dist, k=min(3, nti * nshift - 1), dim=-1, largest=False).values
        compact = nearest[:, :, :2].pow(2).mean((1, 2)) / max(sigma ** 2, 1.0e-4)
        overcollapse = torch.relu(1.5 * sigma - nearest[:, :, 2]).pow(2).mean(1) / max(sigma ** 2, 1.0e-4)

        zero_dist = dist_img[..., ZERO_SHIFT]
        same = owners[None, :, None] == owners[None, None, :]
        eye = torch.eye(nvert, dtype=torch.bool, device=self.device)[None]
        min_same = zero_dist.masked_fill(~same | eye, 1.0e6).amin((1, 2))
        child_label = tuple(self.plan["children"])[0]
        cutoff = float(np.mean([self.model.pair(label, child_label).first_shell_cutoff
                                for label in template["expanded_labels"]]))
        same_collapse = torch.relu(0.65 * cutoff - min_same).pow(2) / max(cutoff ** 2, 0.1)

        tio_delta = vu[:, None, :, None, :] + self.shift_t[None, None, None, :, :] - ti_frac[:, :, None, None, :]
        tio_cart = torch.einsum("btvsk,bkl->btvsl", tio_delta, cell)
        tio_dist = torch.linalg.norm(tio_cart, dim=-1)
        cn_weight = torch.sigmoid((cutoff - tio_dist) / 0.08)
        target_cn = float(self.model.template_cn[template["expanded_labels"][0]])
        child_cn = float(self.model.template_cn[child_label])
        ti_cn = cn_weight.sum((2, 3)) / child_cn
        ti_cn_loss = (ti_cn - target_cn).pow(2).mean(1)
        ti_cn_over = torch.relu(ti_cn - target_cn).pow(2).mean(1)
        local_penalty = torch.tanh(branch_raw[..., 3:21]).pow(2).mean((1, 2))
        radial_penalty = torch.tanh(branch_raw[..., 21:27]).pow(2).mean((1, 2))
        total = (1.8 * occupancy + 1.2 * compact + 2.5 * overcoord + 2.0 * overcollapse
                 + 6.0 * same_collapse + 1.2 * ti_cn_loss + 2.0 * ti_cn_over
                 + 0.08 * local_penalty + 0.04 * radial_penalty)
        detail = {
            "oldschool_occupancy_loss": occupancy,
            "oldschool_compactness_loss": compact,
            "oldschool_overcoord_loss": overcoord,
            "oldschool_overcollapse_loss": overcollapse,
            "oldschool_same_ti_collapse_loss": same_collapse,
            "oldschool_ti_cn_loss": ti_cn_loss,
            "oldschool_ti_cn_mean": ti_cn.mean(1),
            "oldschool_rho_mean": rho.mean(1),
            "oldschool_rho_q10": torch.quantile(rho, 0.1, dim=1),
            "oldschool_rho_q90": torch.quantile(rho, 0.9, dim=1),
            "minimum_same_ti_vertex_distance_A": min_same,
        }
        return total, detail, (abc, angles, cell, ti_frac, vf, vu, owners)

    def _oldschool_optimize(self, template, framework, branch_raw, steps, heartbeat=None):
        """O warm-up followed by joint symmetry-constrained Ti/cell/port relaxation.

        ``framework`` is the raw crystallographic parameter vector, so optimizing it
        keeps the lattice in its crystal family and Ti atoms on their selected Wyckoff
        manifolds.  Each O branch receives its own framework copy after the warm-up.
        """
        anchor = framework.detach().clone()
        fw_variable = framework.detach().clone().requires_grad_(True)
        variable = branch_raw.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([
            {"params": [variable], "lr": self.lr},
            {"params": [fw_variable], "lr": 0.18 * self.lr},
        ])
        warmup = min(30, max(8, int(steps) // 5))
        last = time.perf_counter(); pruned = 0
        for step in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            if step < warmup:
                o_loss, _, _ = self._oldschool_loss(template, anchor, variable)
                total = o_loss
            else:
                o_loss, _, _ = self._oldschool_loss(template, fw_variable, variable)
                fw_loss, _, _ = self._framework_loss(template, fw_variable)
                # A weak raw-coordinate anchor stabilizes the attraction basin but does
                # not freeze any lattice or Wyckoff degree of freedom.
                scale = 1.0 + torch.abs(anchor)
                anchor_loss = ((fw_variable - anchor) / scale).pow(2).mean(1)
                total = o_loss + 0.12 * self.framework_weight * fw_loss + 0.025 * anchor_loss
            total.mean().backward()
            torch.nn.utils.clip_grad_norm_([variable, fw_variable], 10.0)
            opt.step()
            if (self.oxygen_basin_prune_every > 0 and len(variable) > 1
                    and (step + 1) % self.oxygen_basin_prune_every == 0
                    and step + 1 < int(steps)):
                with torch.no_grad():
                    fw_eval = anchor if step < warmup else fw_variable
                    cur_o, _, geom = self._oldschool_loss(template, fw_eval, variable)
                    if step < warmup:
                        cur_loss = cur_o
                    else:
                        cur_fw, _, _ = self._framework_loss(template, fw_variable)
                        scale = 1.0 + torch.abs(anchor)
                        cur_anchor = ((fw_variable - anchor) / scale).pow(2).mean(1)
                        cur_loss = cur_o + 0.12 * self.framework_weight * cur_fw + 0.025 * cur_anchor
                    cell, vf, owners = geom[2], geom[4], geom[6]
                    owner_np = owners.cpu().numpy()
                    desc = [self._geometry_descriptor(vf[i].cpu().numpy(), cell[i].cpu().numpy(), owner_np)
                            for i in range(len(variable))]
                    order = np.argsort(cur_loss.detach().cpu().numpy())
                    keep = []
                    for idx in order:
                        idx = int(idx)
                        if all(float(np.sqrt(np.mean((desc[idx]-desc[j])**2))) >= self.oxygen_proposal_descriptor_tol
                               for j in keep):
                            keep.append(idx)
                    if not keep:
                        keep = [int(order[0])]
                    pruned += len(variable) - len(keep)
                    sel = torch.as_tensor(keep, dtype=torch.long, device=self.device)
                    anchor = anchor[sel].detach().clone()
                    fw_variable = fw_variable.detach()[sel].clone().requires_grad_(True)
                    variable = variable.detach()[sel].clone().requires_grad_(True)
                    opt = torch.optim.Adam([
                        {"params": [variable], "lr": self.lr},
                        {"params": [fw_variable], "lr": 0.18 * self.lr},
                    ])
            now = time.perf_counter()
            if heartbeat is not None and (step + 1 == int(steps) or now - last >= 10.0):
                heartbeat("oldschool_joint_ti_o", step + 1, int(steps), float(total.min().detach().cpu()))
                last = now
        return variable.detach(), fw_variable.detach(), pruned, anchor.detach()

    def _joint_framework_relaxation_diagnostics(self, template, anchor, relaxed):
        """Per-branch Ti/cell displacement diagnostics for the joint O stage."""
        with torch.no_grad():
            a0, ang0, c0, _z0, f0 = self._framework_geometry(template, anchor)
            a1, ang1, c1, _z1, f1 = self._framework_geometry(template, relaxed)
            df = f1 - f0
            df = df - torch.round(df)
            dcart = torch.einsum("bni,bij->bnj", df, c1)
            ti_rms = torch.sqrt(torch.mean(torch.sum(dcart * dcart, dim=-1), dim=1))
            cell_rel = torch.linalg.norm(c1 - c0, dim=(1, 2)) / torch.linalg.norm(c0, dim=(1, 2)).clamp_min(1.0e-8)
            length_rel = torch.max(torch.abs(a1 - a0) / a0.clamp_min(1.0e-8), dim=1).values
            angle_max = torch.max(torch.abs(ang1 - ang0), dim=1).values * (180.0 / math.pi)
        return {
            "joint_ti_rms_displacement_A": ti_rms,
            "joint_cell_relative_change": cell_rel,
            "joint_length_relative_max": length_rel,
            "joint_angle_change_max_deg": angle_max,
        }

    @staticmethod
    def _oldschool_cluster(vf, vu, owners, cell, tolerance):
        vf = np.asarray(vf, float) % 1.0
        vu = np.asarray(vu, float)
        owners = np.asarray(owners, int)
        cell = np.asarray(cell, float)
        n = len(vf)
        delta = vf[None, :, None, :] - vf[:, None, None, :] + SHIFTS[None, None, :, :]
        dist = np.linalg.norm(np.einsum("...i,ij->...j", delta, cell), axis=-1).min(-1)
        parent = np.arange(n)
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] <= float(tolerance): union(i, j)
        comps = {}
        for i in range(n): comps.setdefault(find(i), []).append(i)
        clusters = list(comps.values())
        shifts = np.zeros((n, 3), dtype=int)
        exact = 0
        for cluster in clusters:
            anchor = cluster[0]; owner_ids = []
            for idx in cluster:
                d = vu[idx][None, :] + SHIFTS - vu[anchor][None, :]
                sid = int(np.argmin(np.linalg.norm(d @ cell, axis=1)))
                shifts[idx] = SHIFTS[sid].astype(int)
                owner_ids.append((int(owners[idx]), *map(int, shifts[idx])))
            if len(cluster) == 3 and len(set(owner_ids)) == 3: exact += 1
        target = 2 * len(np.unique(owners))
        hist = {str(k): int(sum(len(c) == k for c in clusters)) for k in sorted(set(map(len, clusters)))}
        diag = {"cluster_success": bool(len(clusters) == target and exact == target),
                "n_clusters": int(len(clusters)), "target_clusters": int(target),
                "exact_triplet_clusters": int(exact),
                "cluster_size_histogram_json": json.dumps(hist, separators=(",", ":"))}
        if not diag["cluster_success"]: return None, diag, None
        oxygen = []
        topology_groups = []
        for cluster in clusters:
            oxygen.append(np.mean([vu[idx] + shifts[idx] for idx in cluster], axis=0) % 1.0)
            topology_groups.append([
                (int(owners[idx]), tuple(int(x) for x in shifts[idx])) for idx in cluster
            ])
        return np.asarray(oxygen, float), diag, topology_groups

    @staticmethod
    def _periodic_point_distance_np(frac_a, frac_b, cell):
        """Minimum 27-image distance between two fractional points."""
        delta = np.asarray(frac_a, float) - np.asarray(frac_b, float)
        return float(np.min(np.linalg.norm((delta[None, :] + SHIFTS) @ np.asarray(cell, float), axis=1)))

    def _analytic_triplet_candidates(self, template: dict, ti_frac: np.ndarray, cell: np.ndarray, group):
        """Return the two chemistry-derived O positions for one periodic Ti triplet.

        The three Ti images define a triangle.  For the current X3 child chemistry we
        place O on the two normals through the triangle circumcenter.  A common Ti--O
        radius is chosen as close as possible to the learned mean while remaining in a
        small near-feasible extension of the learned radial window.  No global O--O
        decision is made here; that is handled by the assignment DFS below.
        """
        child_label = tuple(self.plan["children"])[0]
        ti_frac = np.asarray(ti_frac, float)
        cell = np.asarray(cell, float)
        inv_cell = np.linalg.inv(cell)
        centers=[]; mus=[]; los=[]; his=[]
        for parent, shift in group:
            parent=int(parent); sh=np.asarray(shift, int)
            centers.append((ti_frac[parent] + sh) @ cell)
            ch=self.model.pair(template["expanded_labels"][parent], child_label)
            mus.append(float(ch.mu)); los.append(float(ch.sampling_min)); his.append(float(ch.sampling_max))
        if len(centers) != 3:
            return [], {"analytic_triplet_reason":"not_three_owners"}
        p1,p2,p3=map(np.asarray, centers)
        v12=p2-p1; d=float(np.linalg.norm(v12))
        if not np.isfinite(d) or d < 1.0e-8:
            return [], {"analytic_triplet_reason":"degenerate_ti_pair"}
        ex=v12/d
        v13=p3-p1; ii=float(np.dot(ex,v13))
        ey_raw=v13-ii*ex; jj=float(np.linalg.norm(ey_raw))
        if not np.isfinite(jj) or jj < 1.0e-8:
            return [], {"analytic_triplet_reason":"collinear_ti_triplet"}
        ey=ey_raw/jj; ez=np.cross(ex,ey); ez/=max(float(np.linalg.norm(ez)),1.0e-12)
        # Circumcenter in the Ti triangle plane for equal O--Ti distances.
        x=0.5*d
        y=(ii*ii + jj*jj - ii*d)/(2.0*jj)
        r_circ=float(np.hypot(x,y))
        lower=max(0.0, max(los)-self.analytic_o_radial_slack_A, r_circ+1.0e-6)
        upper=min(his)+self.analytic_o_radial_slack_A
        if lower > upper:
            return [], {"analytic_triplet_reason":"sphere_no_intersection",
                        "analytic_triplet_circumradius_A":r_circ,
                        "analytic_triplet_radius_upper_A":float(upper)}
        radius=float(np.clip(float(np.mean(mus)), lower, upper))
        h2=max(radius*radius-r_circ*r_circ,0.0); h=float(np.sqrt(h2))
        plane=p1+x*ex+y*ey
        points=[plane+h*ez]
        if h > 1.0e-5:
            points.append(plane-h*ez)
        target=np.sort(np.asarray(self.model.template_angles[child_label],float))
        sig=np.asarray(self.model.template_angle_sigma[child_label],float)
        sig=np.maximum(sig,1.0e-3)
        candidates=[]
        assigned={(int(parent),tuple(int(z) for z in shift)) for parent,shift in group}
        for sign,point in enumerate(points):
            ofrac=point @ inv_cell
            vec=np.stack(centers)-point[None,:]
            unit=vec/np.maximum(np.linalg.norm(vec,axis=1,keepdims=True),1.0e-12)
            dots=np.clip(unit@unit.T,-1.0,1.0)
            ang=np.sort(np.degrees(np.arccos(dots[np.triu_indices(3,1)])))
            angle_z=float(np.max(np.abs((ang-target)/sig))) if len(target)==3 else 0.0
            # Check all nearby *periodic Ti image instances* except the three assigned
            # owners.  This avoids the old crystallographic-index masking loophole.
            min_unassigned=float('inf'); min_clear=float('inf')
            for parent,label in enumerate(template["expanded_labels"]):
                base=np.rint(ofrac-ti_frac[parent]).astype(int)
                floor=float(self.model.nonbonded_hard_min(label,child_label))
                for ds in SHIFTS.astype(int):
                    sh=tuple(int(z) for z in (base+ds))
                    if (int(parent),sh) in assigned:
                        continue
                    pos=(ti_frac[parent]+np.asarray(sh,int))@cell
                    dd=float(np.linalg.norm(point-pos))
                    min_unassigned=min(min_unassigned,dd)
                    min_clear=min(min_clear,dd-floor)
            candidates.append({"frac":np.asarray(ofrac,float),"cart":np.asarray(point,float),
                               "sign":int(sign),"radius_A":radius,
                               "circumradius_A":r_circ,"child_angle_zmax":angle_z,
                               "minimum_unassigned_ti_o_A":float(min_unassigned),
                               "unassigned_clearance_A":float(min_clear)})
        return candidates, {"analytic_triplet_reason":"ok","analytic_triplet_circumradius_A":r_circ,
                            "analytic_triplet_radius_A":radius,"analytic_triplet_height_A":h,
                            "analytic_triplet_child_angle_zmax":float(candidates[0]["child_angle_zmax"])}

    def _analytic_o_assignments(self, template: dict, ti_frac: np.ndarray, cell: np.ndarray, topology_groups):
        """Solve the global +/- sphere-intersection assignment with pruning.

        For N Ti and 2N O, each child triplet has at most two sphere-intersection
        choices.  The DFS is pruned aggressively by learned O--O exclusions rather than
        blindly enumerating the full 2^(2N) space: impossible individual O candidates are removed first, then a DFS
        rejects an assignment immediately when a newly added O violates the learned
        periodic O--O wall.  Only the best few physically viable seeds are returned.
        """
        child_label=tuple(self.plan["children"])[0]
        oo_hard=float(self.model.nonbonded_hard_min(child_label,child_label))
        slack=float(self.analytic_o_nonbonded_slack_A)
        cell=np.asarray(cell,float); ti_frac=np.asarray(ti_frac,float)
        # Periodic self images are independent of O position.  If the cell itself is
        # too short for O--O chemistry no sign assignment can repair it.
        sh=np.asarray(SHIFTS,float)
        nz=np.any(np.abs(sh)>0,axis=1)
        self_image=float(np.min(np.linalg.norm(sh[nz]@cell,axis=1)))
        if self_image < oo_hard-slack:
            return [], {"analytic_assignment_success":False,"analytic_assignment_reason":"o_self_image_wall",
                        "analytic_o_self_image_min_A":self_image,"analytic_assignments_feasible":0}
        groups=[]; triplet_diag=[]
        raw_choices=1
        for gi,group in enumerate(topology_groups):
            cand,diag=self._analytic_triplet_candidates(template,ti_frac,cell,group)
            # Individual unassigned Ti--O wall.  A tiny slack is reserved for the
            # subsequent symmetry-constrained polish, but grossly invalid seeds die here.
            cand=[c for c in cand if c["unassigned_clearance_A"] >= -slack]
            triplet_diag.append(diag)
            if not cand:
                return [], {"analytic_assignment_success":False,
                            "analytic_assignment_reason":f"triplet_{gi}_no_physical_candidate",
                            "analytic_triplets_feasible":int(gi),
                            "analytic_o_self_image_min_A":self_image,
                            "analytic_assignments_feasible":0}
            groups.append(cand); raw_choices*=len(cand)
        # Most constrained groups first improves pruning; restore original ordering at output.
        order=sorted(range(len(groups)), key=lambda i:(len(groups[i]),
                    min(c["unassigned_clearance_A"] for c in groups[i])))
        chosen=[None]*len(groups); feasible=[]; nodes=0; pruned_oo=0
        def visit(depth):
            nonlocal nodes,pruned_oo
            if depth==len(order):
                fr=np.stack([chosen[i]["frac"] for i in range(len(groups))])
                min_oo=float('inf')
                for i in range(len(fr)):
                    for j in range(i+1,len(fr)):
                        min_oo=min(min_oo,self._periodic_point_distance_np(fr[i],fr[j],cell))
                min_tio=min(c["minimum_unassigned_ti_o_A"] for c in chosen)
                max_ang=max(c["child_angle_zmax"] for c in chosen)
                # Lower score is better.  Clearances are rewarded but capped so the
                # child angular likelihood still breaks ties between topology variants.
                score=max_ang-0.40*min(max(min_oo-oo_hard,0.0),0.5)-0.25*min(max(
                    min(c["unassigned_clearance_A"] for c in chosen),0.0),0.5)
                feasible.append((float(score),fr,float(min_oo),float(min_tio),float(max_ang),
                                 [int(chosen[i]["sign"]) for i in range(len(groups))]))
                return
            gi=order[depth]
            for c in groups[gi]:
                nodes+=1; ok=True
                for gj in order[:depth]:
                    prev=chosen[gj]
                    if self._periodic_point_distance_np(c["frac"],prev["frac"],cell) < oo_hard-slack:
                        ok=False; pruned_oo+=1; break
                if not ok: continue
                chosen[gi]=c; visit(depth+1); chosen[gi]=None
        visit(0)
        feasible.sort(key=lambda x:x[0])
        out=[]
        for idx,(score,fr,min_oo,min_tio,max_ang,signs) in enumerate(feasible[:self.analytic_o_max_assignments]):
            out.append({"oxygen_frac_init":fr,"analytic_assignment_score":score,
                        "analytic_seed_minimum_o_o_A":min_oo,
                        "analytic_seed_minimum_unassigned_ti_o_A":min_tio,
                        "analytic_seed_child_angle_zmax":max_ang,
                        "analytic_signs":signs,"analytic_assignment_rank":idx})
        return out,{"analytic_assignment_success":bool(out),
                    "analytic_assignment_reason":"ok" if out else "global_o_o_incompatible",
                    "analytic_triplets_feasible":len(groups),"analytic_raw_sign_choices":int(raw_choices),
                    "analytic_dfs_nodes":int(nodes),"analytic_dfs_oo_pruned":int(pruned_oo),
                    "analytic_assignments_feasible":int(len(feasible)),
                    "analytic_assignments_retained":int(len(out)),
                    "analytic_o_self_image_min_A":self_image,
                    "analytic_triplet_circumradius_max_A":float(max(d.get("analytic_triplet_circumradius_A",0.0) for d in triplet_diag)),
                    "analytic_triplet_child_angle_zmax":float(max(d.get("analytic_triplet_child_angle_zmax",0.0) for d in triplet_diag))}

    def _oldschool_audit(self, template: dict, ti_frac: np.ndarray, o_frac: np.ndarray,
                         cell: np.ndarray) -> dict:
        """Strict realized-coordinate audit of the learned local labels.

        v59 accepted exact CN even when the six parent contacts had collapsed to a
        planar or strongly split geometry.  v60 reconstructs the actual first-shell
        neighbour instances, requires every selected Ti--O bond to lie inside the
        learned sampling window, and enforces the learned Ti-centred angular label.
        Low-CN child angles remain diagnostic because the training model explicitly
        treats strict angular filtering only from CN >= 4.
        """
        labels = tuple(template["expanded_labels"])
        child_label = tuple(self.plan["children"])[0]
        nti, no = len(ti_frac), len(o_frac)
        delta = o_frac[None, :, None, :] - ti_frac[:, None, None, :] + SHIFTS[None, None, :, :]
        vec = np.einsum("...i,ij->...j", delta, cell)
        dist = np.linalg.norm(vec, axis=-1)
        cutoffs = np.asarray([self.model.pair(label, child_label).first_shell_cutoff for label in labels])
        ti_cn = np.asarray([np.sum(dist[i] <= cutoffs[i] + 1.0e-8) for i in range(nti)], int)
        o_cn = np.zeros(no, int)
        for j in range(no):
            o_cn[j] = sum(int(np.sum(dist[i, j] <= cutoffs[i] + 1.0e-8)) for i in range(nti))
        target_ti_cn = np.asarray([self.model.template_cn[x] for x in labels], int)
        target_o_cn = int(self.model.template_cn[child_label])

        radial = []
        parent_ang = []
        child_ang = []
        parent_site_mean_z = []
        parent_vector_z = []
        child_vector_z = []
        bond_window_valid = True
        selected_bond_count = 0

        # Parent-centred labels: use the actual first-shell contacts, not merely
        # the N nearest distances.  Periodic images are distinct physical neighbours.
        for i, label in enumerate(labels):
            ch = self.model.pair(label, child_label)
            mask = dist[i] <= ch.first_shell_cutoff + 1.0e-8
            ds = dist[i][mask]
            vs = vec[i][mask]
            if len(ds) != self.model.template_cn[label]:
                continue
            selected_bond_count += len(ds)
            radial.extend(np.abs(ds - ch.mu).tolist())
            bond_window_valid &= bool(np.all(ds >= ch.sampling_min - 1.0e-8)
                                      and np.all(ds <= ch.sampling_max + 1.0e-8))
            obs = _angles_np(vs)
            target = np.asarray(self.model.template_angles[label], float)
            sig = np.maximum(np.asarray(self.model.template_angle_sigma[label], float), 1.0e-3)
            if len(obs) == len(target):
                err = np.abs(obs - target)
                z = err / sig
                parent_ang.extend(err.tolist())
                parent_site_mean_z.append(float(np.mean(z)))
                parent_vector_z.extend(z.tolist())
            else:
                parent_site_mean_z.append(float("inf")); parent_vector_z.append(float("inf"))

        # Child-centred geometry is reconstructed independently.  Radial windows
        # are still hard constraints; CN3 angular deviations are diagnostics only.
        rdelta = ti_frac[None, :, None, :] - o_frac[:, None, None, :] + SHIFTS[None, None, :, :]
        rvec = np.einsum("...i,ij->...j", rdelta, cell)
        rdist = np.linalg.norm(rvec, axis=-1)
        child_target = np.asarray(self.model.template_angles[child_label], float)
        child_sig = np.maximum(np.asarray(self.model.template_angle_sigma[child_label], float), 1.0e-3)
        for j in range(no):
            contacts = []
            for i, label in enumerate(labels):
                ch = self.model.pair(label, child_label)
                for sid in range(len(SHIFTS)):
                    if rdist[j, i, sid] <= ch.first_shell_cutoff + 1.0e-8:
                        contacts.append((float(rdist[j, i, sid]), rvec[j, i, sid], ch))
            if len(contacts) != target_o_cn:
                continue
            for d, _v, ch in contacts:
                bond_window_valid &= bool(ch.sampling_min - 1.0e-8 <= d <= ch.sampling_max + 1.0e-8)
            obs = _angles_np(np.asarray([x[1] for x in contacts], float))
            if len(obs) == len(child_target):
                err = np.abs(obs - child_target)
                child_ang.extend(err.tolist())
                child_vector_z.extend((err / child_sig).tolist())

        # Physical nonbonded distances.  These are audited separately from the
        # bonded Ti--O shell so a valid Ti--O bond is never mistaken for a clash.
        oo = o_frac[None, :, None, :] - o_frac[:, None, None, :] + SHIFTS[None, None, :, :]
        ood = np.linalg.norm(np.einsum("...i,ij->...j", oo, cell), axis=-1)
        for i in range(no):
            ood[i, i, ZERO_SHIFT] = np.inf
        minimum_oo = float(np.min(ood))
        oo_floor = float(self.model.nonbonded_hard_min(child_label, child_label))

        tt = ti_frac[None, :, None, :] - ti_frac[:, None, None, :] + SHIFTS[None, None, :, :]
        ttd = np.linalg.norm(np.einsum("...i,ij->...j", tt, cell), axis=-1)
        for i in range(nti):
            ttd[i, i, ZERO_SHIFT] = np.inf
        minimum_tt = float(np.min(ttd))
        tt_floors = []
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                if i == j:
                    continue
                tt_floors.append(float(self.model.nonbonded_hard_min(li, lj)))
        tt_floor = max(tt_floors) if tt_floors else 0.0

        unassigned_tio = []
        unassigned_tio_floors = []
        for i, label in enumerate(labels):
            ch = self.model.pair(label, child_label)
            floor = float(self.model.nonbonded_hard_min(label, child_label))
            vals = dist[i][dist[i] > ch.first_shell_cutoff + 1.0e-8]
            if len(vals):
                unassigned_tio.extend(vals.tolist())
                unassigned_tio_floors.extend([floor] * len(vals))
        minimum_unassigned_tio = float(np.min(unassigned_tio)) if unassigned_tio else float("inf")
        tio_nonbonded_valid = bool(all(d + 1.0e-8 >= f for d, f in zip(unassigned_tio, unassigned_tio_floors)))
        nonbonded_valid = bool(minimum_oo + 1.0e-8 >= oo_floor
                               and minimum_tt + 1.0e-8 >= tt_floor
                               and tio_nonbonded_valid)

        final = np.vstack([ti_frac, o_frac])
        fd = final[None, :, None, :] - final[:, None, None, :] + SHIFTS[None, None, :, :]
        fdist = np.linalg.norm(np.einsum("...i,ij->...j", fd, cell), axis=-1)
        for i in range(len(final)):
            fdist[i, i, ZERO_SHIFT] = np.inf

        parent_site_z_max = float(np.max(parent_site_mean_z)) if parent_site_mean_z else float("inf")
        parent_vector_z_max = float(np.max(parent_vector_z)) if parent_vector_z else float("inf")
        child_vector_z_max = float(np.max(child_vector_z)) if child_vector_z else 0.0
        exact_parent = bool(np.all(ti_cn == target_ti_cn))
        exact_child = bool(np.all(o_cn == target_o_cn))
        angular_valid = bool(parent_site_z_max <= self.angular_site_z_max
                             and parent_vector_z_max <= self.angular_vector_z_max)
        hard = bool(exact_parent and exact_child and bond_window_valid and angular_valid
                    and nonbonded_valid and float(np.min(fdist)) >= self.minimum_distance)
        return {
            "exact_target_cn_fraction": float(np.mean(ti_cn == target_ti_cn)),
            "exact_child_cn_fraction": float(np.mean(o_cn == target_o_cn)),
            "mean_parent_cn": float(np.mean(ti_cn)), "mean_child_cn": float(np.mean(o_cn)),
            "selected_bond_count": int(selected_bond_count),
            "bond_window_valid": bool(bond_window_valid),
            "local_radial_mae_A": float(np.mean(radial)) if radial else 0.0,
            "local_radial_vector_max_A": float(np.max(radial)) if radial else 0.0,
            "local_angular_site_max_deg": float(max(np.mean(parent_ang) if parent_ang else 0.0,
                                                       np.mean(child_ang) if child_ang else 0.0)),
            "parent_angular_mae_deg": float(np.mean(parent_ang)) if parent_ang else 0.0,
            "child_angular_mae_deg": float(np.mean(child_ang)) if child_ang else 0.0,
            "parent_angular_site_z_max": parent_site_z_max,
            "parent_angular_vector_z_max": parent_vector_z_max,
            "child_angular_vector_z_max_diagnostic": child_vector_z_max,
            "angular_label_valid": bool(angular_valid),
            "nonbonded_exclusion_valid": bool(nonbonded_valid),
            "minimum_physical_distance_A": float(np.min(fdist)),
            "minimum_o_o_A": minimum_oo, "learned_o_o_hard_min_A": oo_floor,
            "minimum_ti_ti_A": minimum_tt, "learned_ti_ti_hard_min_A": tt_floor,
            "minimum_unassigned_ti_o_A": minimum_unassigned_tio,
            "learned_unassigned_ti_o_hard_min_A": float(max(unassigned_tio_floors) if unassigned_tio_floors else 0.0),
            "strict_valid": hard,
        }


    @staticmethod
    def _canonical_hyperedge(frac, group):
        """Translate an O hyperedge into a unique central-cell representation."""
        f = np.asarray(frac, float)
        shift_o = np.floor(f + 1.0e-10).astype(int)
        f0 = (f - shift_o) % 1.0
        owners = []
        for parent, shift in group:
            sh = np.asarray(shift, int) - shift_o
            owners.append((int(parent), tuple(int(x) for x in sh)))
        owners = tuple(sorted(owners))
        return f0, owners

    def _hypergraph_candidate_sites(self, template: dict, ti_frac: np.ndarray, cell: np.ndarray):
        """Enumerate locally feasible periodic Ti3->O hyperedges directly.

        One owner is translated to the central Ti cell, so every periodic triplet
        has a representation with an anchor shift of (0,0,0).  Analytic local O
        geometry and learned nonbonded walls remove impossible hyperedges before
        the global CN exact-cover search sees them.
        """
        ti_frac = np.asarray(ti_frac, float) % 1.0
        cell = np.asarray(cell, float)
        nparent = len(ti_frac)
        child_label = tuple(self.plan["children"])[0]
        labels = tuple(template["expanded_labels"])
        oo_hard = float(self.model.nonbonded_hard_min(child_label, child_label))
        slack = float(self.analytic_o_nonbonded_slack_A)

        # A periodic O cannot satisfy its learned O--O self-image wall if the cell
        # itself is shorter than the wall.
        nz = np.any(np.abs(SHIFTS) > 0, axis=1)
        self_image = float(np.min(np.linalg.norm(SHIFTS[nz] @ cell, axis=1)))
        if self_image < oo_hard - slack:
            return [], {"hypergraph_candidate_reason": "o_self_image_wall",
                        "hypergraph_o_self_image_min_A": self_image,
                        "hypergraph_candidate_count": 0}

        pair_max = 0.0
        for li in labels:
            chi = self.model.pair(li, child_label)
            for lj in labels:
                chj = self.model.pair(lj, child_label)
                pair_max = max(pair_max, float(chi.sampling_max + chj.sampling_max))
        pair_max += float(self.hypergraph_pair_slack_A)

        candidates = {}
        triplets_tested = 0
        analytic_points = 0
        for anchor in range(nparent):
            anchor_cart = ti_frac[anchor] @ cell
            neighbours = []
            for j in range(nparent):
                for shf in SHIFTS.astype(int):
                    sh = tuple(int(x) for x in shf)
                    if j == anchor and sh == (0, 0, 0):
                        continue
                    cart = (ti_frac[j] + shf) @ cell
                    d = float(np.linalg.norm(cart - anchor_cart))
                    if d <= pair_max:
                        neighbours.append((d, int(j), sh, cart))
            neighbours.sort(key=lambda x: x[0])
            neighbours = neighbours[:int(self.hypergraph_neighbor_cap)]
            for a in range(len(neighbours)):
                _, j, sj, cj = neighbours[a]
                for b in range(a + 1, len(neighbours)):
                    _, k, sk, ck = neighbours[b]
                    if j == k and sj == sk:
                        continue
                    if float(np.linalg.norm(cj - ck)) > pair_max:
                        continue
                    group = [(int(anchor), (0, 0, 0)), (j, sj), (k, sk)]
                    if len(set(group)) != 3:
                        continue
                    triplets_tested += 1
                    local, _diag = self._analytic_triplet_candidates(template, ti_frac, cell, group)
                    for cand in local:
                        analytic_points += 1
                        if float(cand.get("unassigned_clearance_A", -1.0e9)) < -slack:
                            continue
                        if float(cand.get("child_angle_zmax", 1.0e9)) > float(self.hypergraph_angle_z_soft_max):
                            continue
                        f0, owners = self._canonical_hyperedge(cand["frac"], group)
                        # Canonicalization can expose a duplicated periodic owner.
                        if len(set(owners)) != 3:
                            continue
                        incidence = np.zeros(nparent, dtype=np.int16)
                        for parent, _shift in owners:
                            incidence[int(parent)] += 1
                        # Same owner index may appear through different periodic images;
                        # incidence multiplicity is intentional (e.g. small rutile cells).
                        key = (owners, tuple(np.round(f0 / 0.015).astype(int).tolist()))
                        radial_term = abs(float(cand.get("radius_A", 0.0)) -
                                          float(np.mean([self.model.pair(labels[p], child_label).mu
                                                         for p, _ in owners])))
                        score = (float(cand.get("child_angle_zmax", 0.0)) ** 2
                                 + 2.0 * (max(0.0, -float(cand.get("unassigned_clearance_A", 0.0))) /
                                          max(slack, 0.02)) ** 2
                                 + (radial_term / 0.08) ** 2)
                        rec = {"frac": f0, "groups": [list(owners)], "group": list(owners),
                               "incidence": incidence, "score": float(score),
                               "minimum_unassigned_ti_o_A": float(cand.get("minimum_unassigned_ti_o_A", np.inf)),
                               "child_angle_zmax": float(cand.get("child_angle_zmax", np.inf))}
                        old = candidates.get(key)
                        if old is None or rec["score"] < old["score"]:
                            candidates[key] = rec

        rows = list(candidates.values())
        # Balanced cap: preserve good candidates incident on every Ti before filling
        # the remaining budget globally by chemistry score.
        if len(rows) > int(self.hypergraph_candidate_cap):
            keep = set()
            for p in range(nparent):
                idx = [i for i, r in enumerate(rows) if int(r["incidence"][p]) > 0]
                idx.sort(key=lambda i: rows[i]["score"])
                keep.update(idx[:int(self.hypergraph_per_parent_cap)])
            rest = sorted((i for i in range(len(rows)) if i not in keep), key=lambda i: rows[i]["score"])
            for i in rest:
                if len(keep) >= int(self.hypergraph_candidate_cap):
                    break
                keep.add(i)
            rows = [rows[i] for i in sorted(keep)]
        rows.sort(key=lambda r: r["score"])
        for i, r in enumerate(rows):
            r["candidate_id"] = int(i)
        incident = np.zeros(nparent, dtype=int)
        for r in rows:
            incident += (r["incidence"] > 0).astype(int)
        return rows, {"hypergraph_candidate_reason": "ok" if rows else "no_local_ti3_o_sites",
                      "hypergraph_triplets_tested": int(triplets_tested),
                      "hypergraph_analytic_points_tested": int(analytic_points),
                      "hypergraph_candidate_count": int(len(rows)),
                      "hypergraph_parent_candidate_min": int(incident.min()) if len(incident) else 0,
                      "hypergraph_parent_candidate_max": int(incident.max()) if len(incident) else 0,
                      "hypergraph_pair_upper_A": float(pair_max),
                      "hypergraph_o_self_image_min_A": float(self_image)}

    def _hypergraph_site_records(self, candidates, nparent):
        out = []
        for c in candidates:
            out.append({"member_ids": frozenset([int(c["candidate_id"])]),
                        "n_o": 1, "incidence": np.asarray(c["incidence"], dtype=np.int16),
                        "score": float(c["score"]), "groups": [c["group"]],
                        "oxygen_frac": [np.asarray(c["frac"], float)],
                        "symmetry_relations": []})
        return out

    def _hypergraph_symmetry_records(self, template, candidates, cell, nchild):
        """Build complete O-site orbits and all representative symmetry constraints."""
        if not candidates:
            return []
        try:
            ops = template["group"][0].ops
        except Exception:
            return []
        cell = np.asarray(cell, float)
        fr = np.asarray([c["frac"] for c in candidates], float)
        orbit_map = {}
        for base_idx, base in enumerate(candidates):
            members = {}
            member_ops = defaultdict(list)
            failed = False
            for op in ops:
                rot = np.asarray(op.rotation_matrix, float)
                trans = np.asarray(op.translation_vector, float)
                target = (rot @ np.asarray(base["frac"], float) + trans) % 1.0
                d = np.asarray([self._periodic_point_distance_np(target, x, cell) for x in fr], float)
                j = int(np.argmin(d))
                if float(d[j]) > float(self.hypergraph_symmetry_match_A):
                    failed = True
                    break
                member_ops[j].append((target, rot, trans))
                if j not in members:
                    members[j] = (target, rot, trans)
            if failed or not members or len(members) > int(nchild):
                continue
            ids = tuple(sorted(members))
            if base_idx != ids[0] or ids in orbit_map:
                continue
            oxygen = []
            groups = []
            incidence = np.zeros(len(base["incidence"]), dtype=np.int16)
            relations = []
            rep_global = ids[0]
            rep_local = 0
            local_of = {j: local for local, j in enumerate(ids)}
            for local_idx, j in enumerate(ids):
                target, _rot, _trans = members[j]
                oxygen.append(np.asarray(target, float) % 1.0)
                groups.append(candidates[j]["group"])
                incidence += np.asarray(candidates[j]["incidence"], dtype=np.int16)
            # Every operation applied to the representative is retained.  Relations
            # with member==representative are not redundant: they enforce special-
            # position/site-symmetry constraints during continuous O polish.
            for j in ids:
                local_idx = local_of[j]
                for target, rot, trans in member_ops[j]:
                    relations.append((rep_local, local_idx, rot.tolist(), trans.tolist()))
            oo_floor = float(self.model.nonbonded_hard_min(tuple(self.plan["children"])[0],
                                                           tuple(self.plan["children"])[0]))
            valid = True
            for i in range(len(oxygen)):
                for j in range(i + 1, len(oxygen)):
                    if self._periodic_point_distance_np(oxygen[i], oxygen[j], cell) < oo_floor - self.analytic_o_nonbonded_slack_A:
                        valid = False; break
                if not valid: break
            if not valid:
                continue
            orbit_map[ids] = {"member_ids": frozenset(ids), "n_o": int(len(ids)),
                              "incidence": incidence,
                              "score": float(sum(candidates[j]["score"] for j in ids)),
                              "groups": groups, "oxygen_frac": oxygen,
                              "symmetry_relations": relations,
                              "symmetry_operation_count": int(sum(len(member_ops[j]) for j in ids))}
        return sorted(orbit_map.values(), key=lambda r: (r["score"] / max(r["n_o"], 1), -r["n_o"]))

    @staticmethod
    def _cover_topology_signature(groups):
        """Canonical within-framework signature of a Ti3->O exact cover.

        The signature intentionally ignores the O coordinates/sign choice and keeps
        only the periodic owner hyperedges.  It is used to diversify covers from the
        *same* Ti framework, where the parent indexing and lattice basis are common.
        It is therefore a ranking fingerprint, not a global duplicate criterion.
        """
        edges = []
        for group in groups:
            owners = []
            for parent, shift in group:
                sh = tuple(int(x) for x in shift)
                owners.append((int(parent), sh))
            # Translation-gauge normalize each hyperedge by subtracting the first
            # lexicographic owner's image shift.  This avoids distinguishing a
            # periodic copy of the same Ti3 owner relation.
            owners = sorted(owners, key=lambda x: (x[0], x[1]))
            anchor = np.asarray(owners[0][1], dtype=int)
            norm = tuple(sorted((p, tuple((np.asarray(sh, dtype=int) - anchor).tolist()))
                                for p, sh in owners))
            edges.append(norm)
        return tuple(sorted(edges))

    @staticmethod
    def _cover_topology_distance(sig_a, sig_b):
        """Jaccard distance between periodic Ti3 hyperedge sets."""
        a, b = set(sig_a), set(sig_b)
        if not a and not b:
            return 0.0
        return 1.0 - len(a & b) / max(len(a | b), 1)

    def _select_topology_diverse_covers(self, covers, max_keep):
        """Keep chemistry-valid covers by max-min connectivity diversity.

        The best local-chemistry cover is retained first.  Each subsequent cover
        maximizes its minimum Ti3-hyperedge Jaccard distance to the selected set;
        chemistry score breaks ties.  Identical connectivity signatures are merged
        before selection.
        """
        if not covers:
            return []
        unique = {}
        for c in covers:
            sig = self._cover_topology_signature(c.get("groups", []))
            score = float(c.get("hypergraph_cover_score", float("inf")))
            old = unique.get(sig)
            if old is None or score < float(old.get("hypergraph_cover_score", float("inf"))):
                cc = dict(c)
                cc["hypergraph_topology_signature"] = sig
                cc["hypergraph_topology_hash"] = hashlib.sha1(repr(sig).encode("utf-8")).hexdigest()[:16]
                unique[sig] = cc
        pool = sorted(unique.values(), key=lambda c: float(c.get("hypergraph_cover_score", float("inf"))))
        if len(pool) <= int(max_keep):
            for c in pool:
                c["hypergraph_topology_min_distance"] = 1.0 if len(pool) == 1 else 0.0
            return pool
        selected = [pool.pop(0)]
        selected[0]["hypergraph_topology_min_distance"] = 1.0
        while pool and len(selected) < int(max_keep):
            best_i = None
            best_key = None
            best_d = None
            for i, cand in enumerate(pool):
                sig = cand["hypergraph_topology_signature"]
                dmin = min(self._cover_topology_distance(sig, s["hypergraph_topology_signature"])
                           for s in selected)
                # Diversity is primary; chemistry only resolves equal/near-equal
                # connectivity choices.  Rounded dmin prevents tiny float noise from
                # turning chemistry into a hidden primary selector.
                key = (round(float(dmin), 8), -float(cand.get("hypergraph_cover_score", 0.0)))
                if best_key is None or key > best_key:
                    best_key = key; best_i = i; best_d = float(dmin)
            cand = pool.pop(int(best_i))
            cand["hypergraph_topology_min_distance"] = float(best_d)
            selected.append(cand)
        return selected

    def _solve_hypergraph_records(self, records, target_cn, nchild, cell, child_label, max_solutions):
        """Bounded exact-cover search over candidate O sites or full O orbits."""
        if not records:
            return [], {"hypergraph_search_nodes": 0, "hypergraph_record_count": 0}
        target = np.asarray(target_cn, dtype=np.int16)
        nparent = len(target)
        oo_floor = float(self.model.nonbonded_hard_min(child_label, child_label))
        oo_cut = oo_floor - float(self.analytic_o_nonbonded_slack_A)
        # Remove records that individually overfill a parent or the O count.
        records = [r for r in records if r["n_o"] <= nchild and np.all(r["incidence"] <= target)]
        if not records:
            return [], {"hypergraph_search_nodes": 0, "hypergraph_record_count": 0}
        conflicts = [set() for _ in records]
        for i, ri in enumerate(records):
            for j in range(i + 1, len(records)):
                rj = records[j]
                bad = bool(ri["member_ids"] & rj["member_ids"])
                if not bad:
                    for oi in ri["oxygen_frac"]:
                        if bad: break
                        for oj in rj["oxygen_frac"]:
                            if self._periodic_point_distance_np(oi, oj, cell) < oo_cut:
                                bad = True; break
                if bad:
                    conflicts[i].add(j); conflicts[j].add(i)
        options_by_parent = [[] for _ in range(nparent)]
        for i, r in enumerate(records):
            for p in range(nparent):
                if int(r["incidence"][p]) > 0:
                    options_by_parent[p].append(i)
        solutions = []
        seen = set()
        nodes = 0
        node_cap = int(self.hypergraph_search_node_cap)

        def recurse(counts, used_o, chosen, blocked, score):
            nonlocal nodes
            if nodes >= node_cap or len(solutions) >= int(max_solutions):
                return
            nodes += 1
            if used_o == int(nchild):
                if np.array_equal(counts, target):
                    key = tuple(sorted(chosen))
                    if key not in seen:
                        seen.add(key); solutions.append((float(score), key))
                return
            if used_o > int(nchild) or np.any(counts > target):
                return
            need = target - counts
            deficient = np.flatnonzero(need > 0)
            if len(deficient) == 0:
                return
            feasible_by_parent = {}
            for p in deficient:
                opts = []
                for rid in options_by_parent[int(p)]:
                    if rid in blocked or rid in chosen:
                        continue
                    r = records[rid]
                    if used_o + int(r["n_o"]) > int(nchild):
                        continue
                    if np.any(counts + r["incidence"] > target):
                        continue
                    opts.append(rid)
                if not opts:
                    return
                feasible_by_parent[int(p)] = opts
            # Fail-fast parent: smallest compatible branching factor relative to need.
            p = min(feasible_by_parent, key=lambda q: (len(feasible_by_parent[q]), -int(need[q])))
            opts = feasible_by_parent[p]
            opts.sort(key=lambda rid: (records[rid]["score"] / max(records[rid]["n_o"], 1),
                                       -int(records[rid]["incidence"][p])))
            for rid in opts:
                r = records[rid]
                new_counts = counts + r["incidence"]
                new_blocked = set(blocked)
                new_blocked.add(rid); new_blocked.update(conflicts[rid])
                recurse(new_counts, used_o + int(r["n_o"]), chosen + (rid,),
                        new_blocked, score + float(r["score"]))
                if nodes >= node_cap or len(solutions) >= int(max_solutions):
                    break

        recurse(np.zeros(nparent, dtype=np.int16), 0, tuple(), set(), 0.0)
        solutions.sort(key=lambda x: x[0])
        out = []
        for score, ids in solutions[:int(max_solutions)]:
            groups = []; oxygen = []; relations = []
            offset = 0
            member_ids = set()
            for rid in ids:
                r = records[rid]
                groups.extend(r["groups"])
                oxygen.extend(r["oxygen_frac"])
                member_ids.update(r["member_ids"])
                for rep, member, rot, trans in r.get("symmetry_relations", []):
                    relations.append((int(rep) + offset, int(member) + offset, rot, trans))
                offset += int(r["n_o"])
            out.append({"child_label": child_label, "groups": groups,
                        "oxygen_frac_init": np.asarray(oxygen, float),
                        "symmetry_relations": relations,
                        "hypergraph_cover_score": float(score),
                        "hypergraph_member_ids": sorted(int(x) for x in member_ids)})
        return out, {"hypergraph_search_nodes": int(nodes),
                     "hypergraph_record_count": int(len(records)),
                     "hypergraph_search_truncated": bool(nodes >= node_cap)}

    def _direct_hypergraph_o_topologies(self, template: dict, ti_frac: np.ndarray, cell: np.ndarray):
        """Construct CN6/CN3 O networks and retain topology-diverse exact covers."""
        child_label = tuple(self.plan["children"])[0]
        nchild = int(self.target_counts[child_label])
        target_cn = np.asarray([int(self.model.template_cn[x]) for x in template["expanded_labels"]], dtype=np.int16)
        candidates, cdiag = self._hypergraph_candidate_sites(template, ti_frac, cell)
        if not candidates:
            return [], {**cdiag, "hypergraph_symmetry_orbit_count": 0,
                        "hypergraph_cover_count": 0, "hypergraph_cover_enumerated_count": 0,
                        "hypergraph_cover_unique_topology_count": 0,
                        "hypergraph_symmetry_cover_count": 0,
                        "hypergraph_site_fallback_used": False, "hypergraph_search_nodes": 0}
        orbit_records = self._hypergraph_symmetry_records(template, candidates, cell, nchild)
        enumerated, sdiag = self._solve_hypergraph_records(
            orbit_records, target_cn, nchild, cell, child_label,
            int(self.hypergraph_enumerate_covers))
        symmetry_enumerated = len(enumerated)
        fallback = False
        if not enumerated:
            fallback = True
            site_records = self._hypergraph_site_records(candidates, len(target_cn))
            enumerated, sdiag = self._solve_hypergraph_records(
                site_records, target_cn, nchild, cell, child_label,
                int(self.hypergraph_site_enumerate_covers))
        topology_unique = len({self._cover_topology_signature(c.get("groups", [])) for c in enumerated})
        keep_cap = int(self.hypergraph_max_covers)
        if fallback:
            keep_cap = min(keep_cap, int(self.hypergraph_site_fallback_max_covers))
        covers = self._select_topology_diverse_covers(enumerated, keep_cap)
        for rank, c in enumerate(covers):
            # Orbit-based covers are symmetry-preserving by construction even when
            # every selected O orbit has multiplicity one.  Site-symmetry self
            # relations are retained separately for continuous-polish restraint.
            c["hypergraph_symmetry_preserved"] = bool(not fallback)
            c["hypergraph_site_fallback_used"] = bool(fallback)
            c["hypergraph_topology_selection_rank"] = int(rank)
        return covers, {**cdiag, **sdiag,
                        "hypergraph_symmetry_orbit_count": int(len(orbit_records)),
                        "hypergraph_cover_count": int(len(covers)),
                        "hypergraph_cover_enumerated_count": int(len(enumerated)),
                        "hypergraph_cover_unique_topology_count": int(topology_unique),
                        "hypergraph_symmetry_cover_count": int(symmetry_enumerated if not fallback else 0),
                        "hypergraph_site_fallback_used": bool(fallback),
                        "hypergraph_site_fallback_cover_cap": int(self.hypergraph_site_fallback_max_covers),
                        "hypergraph_topology_diverse_selection": True}

    @staticmethod
    def _export_validity(audit: dict) -> tuple[bool, dict]:
        """Single authoritative export gate assembled from explicit audit fields."""
        coordination_valid = bool(float(audit.get("exact_target_cn_fraction", 0.0)) >= 1.0 - 1.0e-12
                                  and float(audit.get("exact_child_cn_fraction", 0.0)) >= 1.0 - 1.0e-12)
        reciprocal_topology_valid = bool(audit.get("selected_bond_count", 0) > 0
                                         and coordination_valid)
        checks = {
            "coordination_valid": coordination_valid,
            "reciprocal_topology_valid": reciprocal_topology_valid,
            "bond_window_valid": bool(audit.get("bond_window_valid", False)),
            "angular_label_valid": bool(audit.get("angular_label_valid", False)),
            "nonbonded_exclusion_valid": bool(audit.get("nonbonded_exclusion_valid", False)),
            "physical_distance_valid": bool(float(audit.get("minimum_physical_distance_A", 0.0)) > 0.0),
            "strict_valid": bool(audit.get("strict_valid", False)),
        }
        export_valid = bool(all(checks.values()))
        return export_valid, {**checks, "export_valid": export_valid}

    def _build_shared(self, template: dict, heartbeat=None,
                      entry_first_pass: bool = False, entry_visit_number: int = 1):
        """Scout an entrance cheaply, then spend Ti tokens only on salvageable basins.

        v78 keeps the broad unseen-Wyckoff-first sampler but makes the first
        visit to an entrance deliberately cheap.  An entrance is never rejected from
        crystallographic identity alone: early escape is based only on the best cheap
        Ti geometry found for that entrance, and a fixed rescue fraction preserves
        rare false negatives.  Revisits use the deeper broad screen.
        """
        attempts = []; candidates = []
        first_pass = bool(entry_first_pass or int(entry_visit_number) <= 1)
        scout_starts = min(int(self.starts), 16) if first_pass else int(self.starts)
        scout_steps = min(int(self.screen_steps), 8) if first_pass else int(self.screen_steps)
        scout_keep = min(max(1, int(self.framework_keep)), 6) if first_pass else max(self.framework_keep, 1)
        raw = self._initial_shared_raw(template, starts_override=scout_starts)
        psize = self._shared_prefix_size(template)
        framework = raw[:, :psize]
        # Cheap first-pass screening stays exploratory.  The rescue/escape decision is
        # made before any Ti token is reserved.
        screen_modes = torch.full((len(framework),), 2, dtype=torch.long, device=self.device)
        framework = self._optimize_framework(
            template, framework, scout_steps, heartbeat=heartbeat,
            branch_modes=screen_modes, diversity_weight=0.16 if first_pass else 0.20)
        framework = self._prune_framework_prefix(template, framework, keep=scout_keep)
        attempts.append({
            "stage": "ti_entrance_scout", "branch": -1,
            "entry_first_pass": bool(first_pass),
            "entry_visit_number_local": int(entry_visit_number),
            "scout_starts": int(scout_starts), "scout_steps": int(scout_steps),
            "scout_keep": int(scout_keep), "scout_survivors": int(len(framework)),
            "strict_valid": False,
        })
        if len(framework) == 0:
            attempts.append({"stage": "ti_early_escape", "branch": -1,
                             "early_escape": True, "early_escape_reason": "no_finite_scout_survivor",
                             "early_escape_rescued": False, "strict_valid": False})
            return None, attempts

        pre_count = len(framework)
        (framework, pre_features, pre_selected, pre_structural,
         all_basin, all_features, all_chemistry, _all_structural) = self._select_intelligent_frameworks(
            template, framework, keep=min(self.framework_intelligent_keep, len(framework)))

        # Conservative first-pass early escape.  This uses only the *generic*
        # periodic Ti3->O surrogate from the actual v70+ hypergraph geometry.  It
        # explicitly does NOT use the old Ti4 omission descriptor, the learned pooled
        # Ti-framework prior, the requested SG, or the Wyckoff class itself.
        #
        # The previous broad run showed that very large best near-triplet loss is an
        # excellent negative predictor after screening.  Because the scout is intentionally short, the threshold remains conservative
        # and stochastic rescue is retained as a false-negative sentinel.
        early_escape = False
        early_rescued = False
        escape_reason = ""
        best_near = float("inf")
        best_inc = float("inf")
        best_good = 0.0
        if pre_features:
            best_near = min(float(x.get("preselect_near_triplet_loss", 1.0e9)) for x in pre_features)
            best_inc = min(float(x.get("preselect_incidence_deficit_loss", 1.0e9)) for x in pre_features)
            best_good = max(float(x.get("preselect_min_good_triplets", 0.0)) for x in pre_features)

        # Two negative-confidence tiers.  No proposal is deterministically forbidden:
        # 15% of ordinary unlikely scouts and 5% of catastrophic scouts are still
        # promoted to full refinement as false-negative sentinels.
        catastrophic = bool(first_pass and pre_features
                            and best_near > 7.0 and best_inc > 0.95)
        unlikely = bool(first_pass and pre_features
                        and best_near > 4.9 and best_inc > 0.95 and best_good < 0.5)
        if catastrophic or unlikely:
            rescue_fraction = 0.02 if catastrophic else 0.05
            early_rescued = bool(np.random.random() < rescue_fraction)
            early_escape = not early_rescued
            escape_reason = "catastrophic_generic_hypergraph_mismatch" if catastrophic else "unlikely_generic_hypergraph_mismatch"
        attempts.append({
            "stage": "ti_early_escape_check", "branch": -1,
            "early_escape": bool(early_escape), "early_escape_rescued": bool(early_rescued),
            "early_escape_reason": escape_reason,
            "early_escape_best_near_triplet_loss": float(best_near),
            "early_escape_best_incidence_deficit_loss": float(best_inc),
            "early_escape_best_min_good_triplets": float(best_good),
            "early_escape_catastrophic": bool(catastrophic),
            "early_escape_unlikely": bool(unlikely),
            "strict_valid": False,
        })
        if early_escape:
            return None, attempts

        # Log every cheap basin decision, including candidates skipped before a Ti token is spent.
        selected_set = set(int(x) for x in pre_selected)
        for i in range(pre_count):
            attempts.append({
                "stage": "ti_basin_precheck", "branch": int(i),
                "basin_precheck_retained": bool(i in selected_set),
                "ti_framework_accepted": bool(not all_basin[i].get("basin_skip", False)),
                "strict_valid": False,
                **{k: float(v) if not isinstance(v, bool) else bool(v) for k, v in all_features[i].items()},
                **all_chemistry[i], **all_basin[i],
            })
        if len(framework) == 0:
            return None, attempts

        # One token means exactly one branch admitted to expensive full Ti refinement.
        claimed, token_before, token_after = self._claim_ti_tokens(len(framework))
        if claimed <= 0:
            attempts.append({"stage": "ti_token_exhausted", "branch": -1,
                             "ti_token_total": int(token_after), "strict_valid": False})
            return None, attempts
        if claimed < len(framework):
            framework = framework[:claimed]
            pre_features = pre_features[:claimed]
            pre_selected = pre_selected[:claimed]
            pre_structural = pre_structural[:claimed]
        # v78 spends three expensive Ti tokens per promoted entrance by default.
        # All are exploratory: v77 showed the strong branch had ~4x lower exact-cover
        # productivity despite producing more local O candidates.  The three starts
        # themselves are max-min diverse, and the batch diversity term keeps them apart.
        framework_modes = torch.full(
            (len(framework),), 2, dtype=torch.long, device=self.device)
        for rank, (source_index, feat) in enumerate(zip(pre_selected, pre_features)):
            attempts.append({
                "stage": "ti_token_admit", "branch": int(source_index),
                "ti_token_id": int(token_before + rank + 1),
                "ti_token_total_after_claim": int(token_after),
                "pre_refine_rank": int(rank), "pre_refine_pool": int(pre_count),
                "pre_refine_retained": int(len(framework)),
                "framework_refine_mode": int(framework_modes[rank].item()),
                "framework_refine_mode_name": self.framework_branch_mode_names[int(framework_modes[rank].item())],
                "ti_framework_accepted": True, "strict_valid": False,
                **{k: float(v) if not isinstance(v, bool) else bool(v) for k, v in feat.items()},
            })

        for steps in (max(1, self.refine_steps // 3),
                      max(1, self.refine_steps // 3),
                      max(1, self.refine_steps - 2 * (self.refine_steps // 3))):
            framework = self._optimize_framework(
                template, framework, steps, heartbeat=heartbeat,
                branch_modes=framework_modes,
                diversity_weight=float(self.framework_branch_diversity_weight))
        with torch.no_grad():
            floss, fdetail, _ = self._framework_loss(
                template, framework, branch_modes=framework_modes,
                diversity_weight=float(self.framework_branch_diversity_weight))
            finite = torch.isfinite(floss) & torch.isfinite(framework).all(1)
            accepted = (finite
                        & (fdetail["framework_min_nonbonded_margin_A"] >= -1.0e-6)
                        & (fdetail["framework_min_ti_contact_count"] >= 2.0))
            for i in range(len(framework)):
                attempts.append({"stage": "ti_framework_screen", "branch": int(i),
                    "construction_symmetry": self.construction_symmetry,
                    "framework_refine_mode": int(framework_modes[i].item()),
                    "framework_refine_mode_name": self.framework_branch_mode_names[int(framework_modes[i].item())],
                    "ti_framework_finite": bool(finite[i]), "ti_framework_accepted": bool(accepted[i]),
                    "o_topology_complete": False, "o_topology_accepted": False,
                    "o_attachment_accepted": False, "strict_valid": False, "loss": float(floss[i]),
                    **{k: float(v[i]) for k, v in fdetail.items()}})
            mask = accepted.detach().cpu().numpy().astype(bool)
            framework = framework[accepted]
            framework_modes = framework_modes[accepted]
            pre_structural = [x for x, keep in zip(pre_structural, mask) if keep]
        if len(framework) == 0:
            return None, attempts

        # Generic CN6/CN3 shared-octahedra path.  Stoichiometric incidence requires
        # 6*N_parent == 3*N_child, i.e. N_child = 2*N_parent for TiO2-like chemistry.
        # The direct-hypergraph path is cell-size generic (e.g. Z=2, 4, or 8); the
        # production benchmark may still choose TiO2=4 for throughput comparisons.
        nparent = int(template["nblocks"])
        children = tuple(self.plan["children"])
        if len(children) != 1:
            raise ValueError("v75 direct-hypergraph shared-O path currently requires exactly one child construction label")
        child_label = children[0]
        nchild = int(self.target_counts[child_label])
        parent_cn = [int(self.model.template_cn[x]) for x in template["expanded_labels"]]
        child_cn = int(self.model.template_cn[child_label])
        if nparent < 1 or any(x != 6 for x in parent_cn) or child_cn != 3 \
                or sum(parent_cn) != nchild * child_cn:
            raise ValueError(
                "v75 direct-hypergraph shared-O path requires CN6 parents, one CN3 child label, "
                "and exact incidence balance sum(parent_CN)=Nchild*3; "
                f"got Nparent={nparent}, Nchild={nchild}, parent_CN={parent_cn}, child_CN={child_cn}"
            )

        for fi in range(len(framework)):
            ffeat = self._framework_feasibility_descriptor(template, framework[fi:fi + 1])
            cdiag = self._framework_memory_score(ffeat)
            attempts.append({
                "stage": "ti_chemistry_memory", "branch": int(fi), "framework_rank": int(fi),
                "ti_framework_accepted": True, "strict_valid": False,
                **{k: float(v) for k, v in ffeat.items()}, **cdiag,
            })
            claimed, tdiag = self._claim_ti_framework(template, framework[fi:fi + 1])
            if int(tdiag.get("ti_basin_id", -1)) >= 0:
                self._append_basin_memory(pre_structural[fi], int(tdiag["ti_basin_id"]),
                                          bool(tdiag.get("ti_duplicate", False)), template)
            if not claimed:
                attempts.append({"stage": "ti_framework_duplicate_reject", "branch": int(fi),
                                 "framework_rank": int(fi), "ti_framework_accepted": False,
                                 "strict_valid": False, **tdiag})
                continue

            framework_o_topology_success = False
            framework_strict_success = False
            framework_best_o_loss = float("inf")
            framework_best_cluster_count = 0
            # v73: direct chemistry-derived Ti3->O hypergraph construction.
            # No random floating octahedra are used to discover connectivity.
            with torch.no_grad():
                _abc_h, _ang_h, hcell_t, _z_h, hti_t = self._framework_geometry(template, framework[fi:fi + 1])
                hcell = hcell_t[0].cpu().numpy()
                hti = hti_t[0].cpu().numpy()
            topologies, hdiag = self._direct_hypergraph_o_topologies(template, hti, hcell)
            attempts.append({"stage": "o_hypergraph_candidate_screen", "branch": -1,
                             "framework_rank": int(fi),
                             "framework_refine_mode": int(framework_modes[fi].item()),
                             "framework_refine_mode_name": self.framework_branch_mode_names[int(framework_modes[fi].item())],
                             "ti_framework_accepted": True,
                             "o_topology_complete": bool(topologies),
                             "o_topology_accepted": bool(topologies),
                             "o_attachment_accepted": False, "strict_valid": False,
                             **hdiag, **tdiag})
            framework_o_topology_success = bool(topologies)
            framework_best_cluster_count = int(nchild if topologies else 0)
            if not topologies:
                framework_best_o_loss = min(framework_best_o_loss, 1.0e9)

            for hi, topology in enumerate(topologies):
                hscore = float(topology.get("hypergraph_cover_score", 0.0))
                framework_best_o_loss = min(framework_best_o_loss, hscore)
                row = {"stage": "o_topology_screen", "branch": int(hi), "framework_rank": int(fi),
                       "construction_symmetry": self.construction_symmetry,
                       "ti_framework_finite": True, "ti_framework_accepted": True,
                       "o_topology_complete": True, "o_topology_accepted": True,
                       "o_attachment_accepted": False, "strict_valid": False,
                       "loss": hscore, "exact_triplet_clusters": int(nchild),
                       "hypergraph_direct": True,
                       "hypergraph_symmetry_preserved": bool(topology.get("hypergraph_symmetry_preserved", False)),
                       "hypergraph_site_fallback_used": bool(topology.get("hypergraph_site_fallback_used", hdiag.get("hypergraph_site_fallback_used", False))),
                       "hypergraph_cover_score": hscore,
                       "hypergraph_topology_hash": topology.get("hypergraph_topology_hash"),
                       "hypergraph_topology_min_distance": topology.get("hypergraph_topology_min_distance"),
                       "hypergraph_topology_selection_rank": topology.get("hypergraph_topology_selection_rank"),
                       **hdiag}
                attempts.append(row)
                # Preserve the existing Oanalytic progress semantics: a direct
                # hypergraph cover already contains physical analytic O coordinates.
                adiag = {"analytic_assignment_success": True,
                         "analytic_assignment_reason": "direct_hypergraph_cover",
                         "analytic_assignments_feasible": 1,
                         "hypergraph_direct": True,
                         "hypergraph_symmetry_preserved": bool(topology.get("hypergraph_symmetry_preserved", False))}
                attempts.append({**row, "stage": "analytic_o_assignment_screen",
                                 "analytic_assignment_accepted": True, **adiag})

                explicit_raw = self._explicit_o_raw(framework[fi:fi + 1].detach(), [topology])
                old_reference = self._framework_reference
                self._framework_reference = framework[fi:fi + 1].detach().clone()
                try:
                    with torch.no_grad():
                        preloss, predetail, _ = self._explicit_o_loss(template, explicit_raw, [topology], "o_polish")
                    polished = self._optimize_explicit_o(
                        template, explicit_raw, [topology],
                        steps=max(18, int(self.polish_steps)), phase="o_polish",
                        lr=0.10 * self.lr, heartbeat=None)
                    with torch.no_grad():
                        ploss, pdetail, pgeom = self._explicit_o_loss(template, polished, [topology], "o_polish")
                        pabc, pangles, pcell, pti, po, _pfinal = pgeom
                        refined_ti = pti[0].cpu().numpy()
                        refined_o = po[0].cpu().numpy() % 1.0
                        refined_cell = pcell[0].cpu().numpy()
                        audit = self._oldschool_audit(template, refined_ti, refined_o, refined_cell)
                        export_valid, export_diag = self._export_validity(audit)
                        audit.update(export_diag)
                        relax_detail = self._joint_framework_relaxation_diagnostics(
                            template, framework[fi:fi + 1].detach(), polished[:, :self._shared_prefix_size(template)])
                        postdiag = {
                            "hypergraph_cover_index": int(hi),
                            "hypergraph_cover_score": hscore,
                            "hypergraph_symmetry_preserved": bool(topology.get("hypergraph_symmetry_preserved", False)),
                            "prepolish_joint_loss": float(preloss[0]),
                            "prepolish_nonbonded_loss": float(predetail["learned_nonbonded_exclusion_loss"][0]),
                            "prepolish_minimum_o_o_A": float(predetail["minimum_o_o_A"][0]),
                            "prepolish_minimum_ti_ti_A": float(predetail["minimum_ti_ti_A"][0]),
                            "prepolish_minimum_unassigned_ti_o_A": float(predetail["minimum_unassigned_ti_o_A"][0]),
                            "prepolish_volume_per_parent_A3": float(predetail["framework_volume_per_parent_A3"][0]),
                            "prepolish_cell_aspect_ratio": float(predetail["framework_aspect_ratio"][0]),
                            "prepolish_oxygen_symmetry_loss": float(predetail.get("oxygen_symmetry_loss", torch.zeros(1,device=self.device))[0]),
                            "postcluster_joint_loss": float(ploss[0]),
                            "postcluster_nonbonded_loss": float(pdetail["learned_nonbonded_exclusion_loss"][0]),
                            "postcluster_minimum_o_o_A": float(pdetail["minimum_o_o_A"][0]),
                            "postcluster_minimum_ti_ti_A": float(pdetail["minimum_ti_ti_A"][0]),
                            "postcluster_minimum_unassigned_ti_o_A": float(pdetail["minimum_unassigned_ti_o_A"][0]),
                            "postpolish_volume_per_parent_A3": float(pdetail["framework_volume_per_parent_A3"][0]),
                            "postpolish_cell_aspect_ratio": float(pdetail["framework_aspect_ratio"][0]),
                            "postpolish_oxygen_symmetry_loss": float(pdetail.get("oxygen_symmetry_loss", torch.zeros(1,device=self.device))[0]),
                            **{k: float(v[0]) for k, v in relax_detail.items()},
                        }
                finally:
                    self._framework_reference = old_reference

                attach = {**row, "stage": "o_attachment_screen",
                          "o_attachment_accepted": bool(audit["strict_valid"]),
                          **adiag, **postdiag, **audit}
                attempts.append(attach)
                final_row = {**attach, "stage": "strict_final_audit"}
                attempts.append(final_row)
                framework_strict_success = framework_strict_success or bool(audit.get("export_valid", False))
                if not audit.get("export_valid", False):
                    continue
                symbols = tuple(self.model.block_atoms[x][0][0] for x in template["expanded_labels"]) + \
                          tuple([self.model.block_atoms[child_label][0][0]] * nchild)
                candidates.append({**final_row,
                    "cell": refined_cell,
                    "lattice": np.concatenate([pabc[0].cpu().numpy(), pangles[0].cpu().numpy()]),
                    "frac": np.vstack([refined_ti, refined_o]),
                    "symbols": symbols,
                    "center_frac": refined_ti,
                    "free": np.zeros((len(template["wps"]), 3), float),
                    # Internal shadow-standardization metadata.  These fields are
                    # carried to the manager but do not alter raw scientific output.
                    "_topology_groups": topology.get("groups", []),
                    "_parent_labels": tuple(template["expanded_labels"]),
                    "_child_label": child_label,
                    "hypergraph_topology_hash": topology.get("hypergraph_topology_hash"),
                    "hypergraph_topology_min_distance": topology.get("hypergraph_topology_min_distance"),
                    "hypergraph_topology_selection_rank": topology.get("hypergraph_topology_selection_rank")})
            self._append_framework_memory(ffeat, {
                "o_topology_success": bool(framework_o_topology_success),
                "strict_success": bool(framework_strict_success),
                "best_o_loss": float(framework_best_o_loss if np.isfinite(framework_best_o_loss) else 1.0e9),
                "best_exact_triplet_clusters": int(framework_best_cluster_count),
            })
        candidates.sort(key=lambda x: (x.get("local_radial_mae_A", 999.0),
                                      x.get("local_angular_site_max_deg", 999.0), x["loss"]))
        return candidates, attempts

    def build(self, spg: int, wp_token: str, species_token: str, heartbeat=None, task_meta: dict | None = None):
        template = self._template(spg, wp_token, species_token)
        # Repeated zero-DOF copies of the same Wyckoff orbit are necessarily
        # coincident and cannot be repaired by continuous optimization.
        seen_fixed = set()
        for wp, label, dof in zip(template["wps"], template["site_labels"], template["site_dofs"]):
            key = (int(wp), str(label))
            if int(dof) == 0 and key in seen_fixed:
                return None, [{
                    "stage": "proposal_reject_duplicate_parent_orbit",
                    "branch": -1, "construction_symmetry": self.construction_symmetry,
                    "ti_framework_finite": False, "ti_framework_accepted": False,
                    "o_topology_complete": False, "o_topology_accepted": False,
                    "o_attachment_accepted": False, "strict_valid": False,
                }]
            if int(dof) == 0:
                seen_fixed.add(key)
        if self.plan["mode"] == "shared_site":
            task_meta = {} if task_meta is None else dict(task_meta)
            return self._build_shared(
                template, heartbeat=heartbeat,
                entry_first_pass=bool(task_meta.get("entry_first_pass", False)),
                entry_visit_number=int(task_meta.get("entry_visit_number", 1) or 1))
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


def _detect_space_group(result: dict, symprec: float = 0.10, angle_tolerance: float = 5.0):
    """Return detected final space-group number/symbol using pymatgen."""
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        structure = Structure(
            lattice=np.asarray(result["cell"], float),
            species=list(result["symbols"]),
            coords=np.asarray(result["frac"], float),
            coords_are_cartesian=False,
        )
        analyzer = SpacegroupAnalyzer(structure, symprec=float(symprec), angle_tolerance=float(angle_tolerance))
        return int(analyzer.get_space_group_number()), str(analyzer.get_space_group_symbol())
    except Exception:
        return 0, "unknown"


def _parent_space_group_retention(result: dict, parent_spg: int,
                                  tolerance_A: float = 0.12) -> tuple[bool, float, int]:
    """Audit whether every operation of the parent SG still maps the final structure.

    The check is performed in the same conventional fractional basis used to build
    the parent Wyckoff entrance.  Species are matched with a Hungarian assignment
    under 27-image periodic distances.  This correctly credits accidental relaxation
    into a higher-symmetry supergroup because all parent operations remain valid.
    If the parent operation table cannot be constructed, callers may safely fall back
    to exact parent/final SG-number equality.
    """
    try:
        from pyxtal.symmetry import Group
        group = Group(int(parent_spg))
        ops = list(group[0].ops)
        cell = np.asarray(result["cell"], float)
        frac = np.asarray(result["frac"], float) % 1.0
        symbols = np.asarray(list(map(str, result["symbols"])), dtype=object)
        max_error = 0.0
        checked = 0
        for op in ops:
            rot = np.asarray(op.rotation_matrix, float)
            trans = np.asarray(op.translation_vector, float)
            moved = (frac @ rot.T + trans[None, :]) % 1.0
            for symbol in sorted(set(symbols.tolist())):
                ids = np.flatnonzero(symbols == symbol)
                if len(ids) == 0:
                    continue
                a = moved[ids]
                b = frac[ids]
                # Pairwise minimum over explicit neighbouring lattice images;
                # this remains robust for non-orthogonal cells.
                delta = b[None, :, None, :] + SHIFTS[None, None, :, :] - a[:, None, None, :]
                cart = np.einsum("...i,ij->...j", delta, cell)
                cost = np.linalg.norm(cart, axis=-1).min(axis=-1)
                rr, cc = linear_sum_assignment(cost)
                err = float(cost[rr, cc].max()) if len(rr) else 0.0
                max_error = max(max_error, err)
                if err > float(tolerance_A):
                    return False, max_error, checked + 1
            checked += 1
        return True, max_error, checked
    except Exception:
        return False, float("nan"), 0



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



class DiagnosticFamilyTracker:
    """Moderate-tolerance motif clustering for reporting only; never rejects output."""
    def __init__(self, primitive_cell=False, ltol=0.12, stol=0.25, angle_tol=5.0):
        self.representatives = []
        self.members = defaultdict(list)
        self.primitive_cell = bool(primitive_cell)
        self.ltol = float(ltol); self.stol = float(stol); self.angle_tol = float(angle_tol)

    @staticmethod
    def _structure(result):
        from pymatgen.core import Structure
        return Structure(
            lattice=np.asarray(result["cell"], float),
            species=list(result["symbols"]),
            coords=np.asarray(result["frac"], float) % 1.0,
            coords_are_cartesian=False,
        )

    def assign(self, candidate_id: int, result: dict) -> dict:
        try:
            from pymatgen.analysis.structure_matcher import StructureMatcher
            current = self._structure(result)
            matcher = StructureMatcher(
                ltol=self.ltol, stol=self.stol, angle_tol=self.angle_tol,
                primitive_cell=self.primitive_cell, scale=True,
                attempt_supercell=False, allow_subset=False,
            )
            for family_id, record in enumerate(self.representatives):
                if Counter(record["symbols"]) != Counter(result["symbols"]):
                    continue
                previous = self._structure(record)
                if not matcher.fit(current, previous):
                    continue
                rms = None; max_dist = None
                try:
                    distances = matcher.get_rms_dist(current, previous)
                    if distances is not None:
                        rms, max_dist = map(float, distances)
                except Exception:
                    pass
                self.members[family_id].append(int(candidate_id))
                return {"family_id": int(family_id), "family_is_new": False,
                        "family_rms_normalized": rms, "family_max_normalized": max_dist}
        except Exception:
            # Diagnostics must never interfere with accepted-structure generation.
            pass
        family_id = len(self.representatives)
        self.representatives.append({
            "cell": np.asarray(result["cell"], float).tolist(),
            "frac": (np.asarray(result["frac"], float) % 1.0).tolist(),
            "symbols": list(map(str, result["symbols"])),
            "candidate_id": int(candidate_id),
        })
        self.members[family_id].append(int(candidate_id))
        return {"family_id": int(family_id), "family_is_new": True,
                "family_rms_normalized": None, "family_max_normalized": None}

    def rows(self):
        out = []
        for family_id, rep in enumerate(self.representatives):
            members = list(self.members.get(family_id, []))
            out.append({
                "family_id": int(family_id),
                "representative_candidate_id": int(rep["candidate_id"]),
                "member_count": int(len(members)),
                "candidate_ids": ",".join(str(x) for x in members),
            })
        return out

class ShadowCanonicalizer:
    """Diagnostic fixed-connectivity chemistry normalization + aggressive symmetry.

    This class never changes raw acceptance.  It operates only on already strict,
    exact-unique candidates and returns a shadow representation used for family
    counting and reporting.
    """
    def __init__(self, model: XNModel, symprec_A: float = 0.10, angle_tolerance_deg: float = 5.0):
        self.model = model
        self.symprec_A = float(symprec_A)
        self.angle_tolerance_deg = float(angle_tolerance_deg)
        # v74: canonicalization is a nearby projection, never a second structure search.
        # The v73 run showed a sharp bimodality: useful projections were <~0.15 A RMS
        # and <~0.11 strain, whereas pathological reconstructions jumped to
        # 0.58--0.85 A RMS and ~0.26--0.30 strain.  Bounds and post-fit guards
        # therefore sit safely inside that gap.
        self.max_frac_shift = 0.06
        self.max_strain = 0.06
        self.max_nfev = 90
        self.chem_rms_guard_A = 0.20
        self.chem_max_guard_A = 0.35
        self.chem_strain_guard = 0.15
        self.sym_rms_guard_A = 0.15
        self.sym_max_guard_A = 0.30
        self.sym_same_basis_cell_guard = 0.08

    @staticmethod
    def _strain_matrix(v):
        e0,e1,e2,e3,e4,e5 = map(float, v)
        return np.asarray([[e0,e3,e4],[e3,e1,e5],[e4,e5,e2]], float)

    @staticmethod
    def _min_periodic_distance(frac_a, frac_b, cell):
        d = np.asarray(frac_b, float)[None,:] + SHIFTS - np.asarray(frac_a, float)[None,:]
        return float(np.min(np.linalg.norm(d @ np.asarray(cell, float), axis=1)))

    def _residual(self, x, cell0, frac0, parent_labels, child_label, groups):
        nat = len(frac0); nparent = len(parent_labels); nchild = nat - nparent
        df = np.asarray(x[:3*nat], float).reshape(nat,3)
        strain = self._strain_matrix(x[3*nat:3*nat+6])
        cell = np.asarray(cell0, float) @ (np.eye(3) + strain)
        frac = np.asarray(frac0, float) + df
        parent_frac = frac[:nparent]; child_frac = frac[nparent:]
        parent_cart = parent_frac @ cell; child_cart = child_frac @ cell
        res = []
        assigned = set(); parent_vec = [[] for _ in range(nparent)]
        # Standardize the fixed selected Ti--O hypergraph.
        for g, images in enumerate(groups):
            reverse = []
            for p, shift in images:
                p = int(p); shift = np.asarray(shift, float)
                vec = child_cart[int(g)] - (parent_cart[p] + shift @ cell)
                d = float(np.linalg.norm(vec))
                ch = self.model.pair(parent_labels[p], child_label)
                res.append((d - float(ch.mu)) / 0.020)
                assigned.add((p, int(g)))
                parent_vec[p].append(vec); reverse.append(-vec)
            obs = _angles_np(np.asarray(reverse, float))
            target = np.asarray(self.model.template_angles[child_label], float)
            sig = np.maximum(np.asarray(self.model.template_angle_sigma[child_label], float), 2.0)
            if len(obs) == len(target):
                # Child CN3 angular statistics are broad/multimodal in the learned
                # data, so canonicalization uses a deliberately weaker pull.
                res.extend((0.45 * (obs-target) / sig).tolist())
        for p, label in enumerate(parent_labels):
            obs = _angles_np(np.asarray(parent_vec[p], float))
            target = np.asarray(self.model.template_angles[label], float)
            sig = np.maximum(np.asarray(self.model.template_angle_sigma[label], float), 2.0)
            if len(obs) == len(target):
                res.extend(((obs-target)/sig).tolist())
            else:
                res.extend([20.0] * max(len(target),1))

        # Fixed-length nonbonded/topology wall residuals.  They prevent the shadow
        # normalization from obtaining a pretty local geometry by changing the
        # selected first-shell graph.
        for p, label in enumerate(parent_labels):
            ch = self.model.pair(label, child_label)
            wall = max(float(ch.first_shell_cutoff), float(self.model.nonbonded_hard_min(label, child_label)))
            for g in range(nchild):
                if (p,g) in assigned:
                    res.append(0.0); continue
                d = self._min_periodic_distance(parent_frac[p], child_frac[g], cell)
                res.append(max(0.0, wall-d)/0.035)
        oo_wall = float(self.model.nonbonded_hard_min(child_label, child_label))
        for i in range(nchild):
            for j in range(i+1,nchild):
                d = self._min_periodic_distance(child_frac[i], child_frac[j], cell)
                res.append(max(0.0, oo_wall-d)/0.035)
        for i, li in enumerate(parent_labels):
            for j in range(i+1,nparent):
                d = self._min_periodic_distance(parent_frac[i], parent_frac[j], cell)
                wall = float(self.model.nonbonded_hard_min(li, parent_labels[j]))
                res.append(max(0.0, wall-d)/0.035)

        # Weak proximity/gauge regularizers choose the nearest standardized member
        # of the same connectivity basin rather than allowing a wholesale rebuild.
        cart_disp = df @ np.asarray(cell0, float)
        res.extend((cart_disp.reshape(-1)/0.30).tolist())
        res.extend((np.asarray(x[3*nat:3*nat+6], float)/0.10).tolist())
        return np.asarray(res, float)

    @staticmethod
    def _result_from_structure(structure):
        return {"cell": np.asarray(structure.lattice.matrix, float),
                "frac": np.asarray(structure.frac_coords, float) % 1.0,
                "symbols": [str(site.specie.symbol) for site in structure]}

    @staticmethod
    def _species_periodic_rms(a, b):
        if len(a["symbols"]) != len(b["symbols"]) or Counter(a["symbols"]) != Counter(b["symbols"]):
            return float("inf"), float("inf")
        cell = np.asarray(b["cell"], float)
        fa = np.asarray(a["frac"], float) % 1.0
        fb = np.asarray(b["frac"], float) % 1.0
        vals=[]
        for sp in sorted(set(a["symbols"])):
            ia=[i for i,x in enumerate(a["symbols"]) if x==sp]
            ib=[i for i,x in enumerate(b["symbols"]) if x==sp]
            cost=np.zeros((len(ia),len(ib)),float)
            for ii,i in enumerate(ia):
                for jj,j in enumerate(ib):
                    d=fb[j][None,:]+SHIFTS-fa[i][None,:]
                    cost[ii,jj]=float(np.min(np.linalg.norm(d@cell,axis=1)))
            r,c=linear_sum_assignment(cost)
            vals.extend(cost[r,c].tolist())
        arr=np.asarray(vals,float)
        return (float(np.sqrt(np.mean(arr*arr))) if len(arr) else 0.0,
                float(np.max(arr)) if len(arr) else 0.0)

    @staticmethod
    def _reorder_like(reference, candidate):
        """Reorder candidate sites to reference species/site order by periodic assignment."""
        if len(reference["symbols"]) != len(candidate["symbols"]) or Counter(reference["symbols"]) != Counter(candidate["symbols"]):
            return None
        cell=np.asarray(candidate["cell"],float)
        fr=np.asarray(reference["frac"],float)%1.0
        fc=np.asarray(candidate["frac"],float)%1.0
        order=[None]*len(fr)
        for sp in sorted(set(reference["symbols"])):
            ia=[i for i,x in enumerate(reference["symbols"]) if x==sp]
            ib=[i for i,x in enumerate(candidate["symbols"]) if x==sp]
            cost=np.zeros((len(ia),len(ib)),float)
            for ii,i in enumerate(ia):
                for jj,j in enumerate(ib):
                    d=fc[j][None,:]+SHIFTS-fr[i][None,:]
                    cost[ii,jj]=float(np.min(np.linalg.norm(d@cell,axis=1)))
            r,c=linear_sum_assignment(cost)
            for rr,cc in zip(r,c): order[ia[int(rr)]]=ib[int(cc)]
        if any(x is None for x in order): return None
        return {"cell":cell,"frac":fc[np.asarray(order,int)],
                "symbols":[candidate["symbols"][int(j)] for j in order]}

    def _fixed_topology_preserved(self, result, parent_labels, child_label, groups):
        """Verify the *periodic* selected Ti--O owner graph in the current basis.

        v73 only tracked (parent_index, child_index), which could miss an extra
        first-shell contact to a different periodic image of an already assigned
        parent.  v74 compares the full (parent_index, lattice_shift) owner set for
        every O.  This routine is used only where the lattice basis is unchanged.
        """
        frac=np.asarray(result["frac"],float)%1.0; cell=np.asarray(result["cell"],float)
        nparent=len(parent_labels); child=frac[nparent:]; parent=frac[:nparent]
        if len(child) != len(groups):
            return False
        expected_by_child=[]
        for g,images in enumerate(groups):
            expected=set()
            for p,shift in images:
                p=int(p); sh=tuple(int(x) for x in np.asarray(shift,int).tolist())
                if p < 0 or p >= nparent:
                    return False
                expected.add((p,sh))
                d=float(np.linalg.norm((child[int(g)]-(parent[p]+np.asarray(sh,float)))@cell))
                ch=self.model.pair(parent_labels[p],child_label)
                if not (float(ch.sampling_min)-0.08 <= d <= float(ch.sampling_max)+0.08):
                    return False
            expected_by_child.append(expected)

        # No unselected periodic Ti image may enter the first shell.
        for g in range(len(child)):
            expected=expected_by_child[g]
            for p,label in enumerate(parent_labels):
                ch=self.model.pair(label,child_label)
                cutoff=float(ch.first_shell_cutoff)-0.04
                for shift in SHIFTS:
                    sh=tuple(int(x) for x in np.asarray(shift,int).tolist())
                    if (p,sh) in expected:
                        continue
                    d=float(np.linalg.norm((child[g]-(parent[p]+np.asarray(sh,float)))@cell))
                    if d < cutoff:
                        return False

        try:
            oo_wall=float(self.model.nonbonded_hard_min(child_label,child_label))-0.06
            for i in range(len(child)):
                for j in range(i+1,len(child)):
                    if self._min_periodic_distance(child[i],child[j],cell) < oo_wall: return False
            for i,li in enumerate(parent_labels):
                for j in range(i+1,nparent):
                    wall=float(self.model.nonbonded_hard_min(li,parent_labels[j]))-0.06
                    if self._min_periodic_distance(parent[i],parent[j],cell) < wall: return False
        except Exception:
            pass
        return True

    def canonicalize(self, result: dict):
        raw = {"cell": np.asarray(result["cell"], float),
               "frac": np.asarray(result["frac"], float) % 1.0,
               "symbols": list(map(str,result["symbols"]))}
        groups = result.get("_topology_groups")
        parent_labels = tuple(result.get("_parent_labels", ()))
        child_label = result.get("_child_label")
        diag = {"canonicalization_success": False,
                "canonical_symprec_A": self.symprec_A,
                "canonical_chemistry_used": False,
                "canonical_aggressive_symmetry_used": False}
        if not groups or not parent_labels or not child_label:
            diag["canonicalization_reason"]="missing_fixed_topology_metadata"
            return raw, diag
        nat=len(raw["frac"])
        if nat <= len(parent_labels):
            diag["canonicalization_reason"]="invalid_parent_child_count"
            return raw, diag

        # First create a *nearby* chemistry-normalized shadow.  A fit that reaches a
        # different geometric basin is rejected wholesale and the raw strict structure
        # becomes the canonical base.  The raw scientific candidate is never modified.
        base = raw
        x0=np.zeros(3*nat+6,float)
        lo=np.concatenate([np.full(3*nat,-self.max_frac_shift),np.full(6,-self.max_strain)])
        hi=np.concatenate([np.full(3*nat, self.max_frac_shift),np.full(6, self.max_strain)])
        try:
            fit=least_squares(self._residual,x0,bounds=(lo,hi),max_nfev=self.max_nfev,
                              xtol=2e-5,ftol=2e-5,gtol=2e-5,
                              args=(raw["cell"],raw["frac"],parent_labels,str(child_label),groups))
            x=np.asarray(fit.x,float)
            df=x[:3*nat].reshape(nat,3); strain=self._strain_matrix(x[3*nat:])
            cell1=raw["cell"]@(np.eye(3)+strain)
            frac1=(raw["frac"]+df)%1.0
            chem={"cell":cell1,"frac":frac1,"symbols":list(raw["symbols"])}
            dcart=(df-np.round(df))@raw["cell"]
            disp=np.linalg.norm(dcart,axis=1)
            rms=float(np.sqrt(np.mean(disp*disp)))
            maxd=float(np.max(disp))
            strain_fro=float(np.linalg.norm(strain))
            topology_ok=bool(self._fixed_topology_preserved(chem,parent_labels,str(child_label),groups))
            chem_guard_ok=bool(np.isfinite(rms) and np.isfinite(maxd) and np.isfinite(strain_fro)
                               and rms <= self.chem_rms_guard_A
                               and maxd <= self.chem_max_guard_A
                               and strain_fro <= self.chem_strain_guard
                               and topology_ok)
            diag.update({"canonical_optimizer_nfev":int(getattr(fit,"nfev",0)),
                         "canonical_optimizer_cost":float(getattr(fit,"cost",float("nan"))),
                         "canonical_chemistry_rms_move_A":rms,
                         "canonical_chemistry_max_move_A":maxd,
                         "canonical_cell_strain_fro":strain_fro,
                         "canonical_chemistry_topology_preserved":topology_ok,
                         "canonical_chemistry_guard_passed":chem_guard_ok})
            if chem_guard_ok:
                base=chem
                diag["canonical_chemistry_used"]=True
                diag["canonicalization_reason"]="chemistry_projection_ok"
            else:
                diag["canonicalization_reason"]="chemistry_projection_rejected_by_nearby_guard"
        except Exception as exc:
            diag["canonicalization_reason"]="chemistry_normalization_failed:"+type(exc).__name__

        canonical=base
        # Aggressive symmetry *detection* remains at symprec=0.10 A, but the refined
        # coordinates are used only when they are a small same-basis displacement and
        # preserve the fixed Ti--O topology.  StructureMatcher is deliberately NOT an
        # override: crystallographic equivalence alone cannot prove topology retention.
        try:
            from pymatgen.core import Structure
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            st=Structure(lattice=base["cell"],species=base["symbols"],coords=base["frac"],coords_are_cartesian=False)
            sga=SpacegroupAnalyzer(st,symprec=self.symprec_A,angle_tolerance=self.angle_tolerance_deg)
            diag["canonical_pre_sym_spg"]=int(sga.get_space_group_number())
            diag["canonical_pre_sym_spg_symbol"]=str(sga.get_space_group_symbol())
            refined=sga.get_refined_structure()
            sym0=self._result_from_structure(refined)
            denom=max(float(np.linalg.norm(np.asarray(base["cell"],float))), EPS)
            same_basis_cell_change=float(np.linalg.norm(np.asarray(sym0["cell"],float)-np.asarray(base["cell"],float))/denom)
            sym=self._reorder_like(base,sym0) if same_basis_cell_change <= self.sym_same_basis_cell_guard else None
            if sym is None:
                rms,maxd=float("inf"),float("inf"); topology_ok=False
            else:
                rms,maxd=self._species_periodic_rms(base,sym)
                topology_ok=self._fixed_topology_preserved(sym,parent_labels,str(child_label),groups)
            accept=bool(sym is not None and np.isfinite(rms) and np.isfinite(maxd)
                        and rms <= self.sym_rms_guard_A
                        and maxd <= self.sym_max_guard_A
                        and same_basis_cell_change <= self.sym_same_basis_cell_guard
                        and topology_ok)
            diag.update({"canonical_aggressive_symmetry_rms_A":float(rms),
                         "canonical_aggressive_symmetry_max_A":float(maxd),
                         "canonical_aggressive_symmetry_same_basis_cell_change":same_basis_cell_change,
                         "canonical_aggressive_symmetry_topology_preserved":bool(topology_ok),
                         "canonical_aggressive_symmetry_matcher_equivalent":False,
                         "canonical_aggressive_symmetry_used":accept})
            if accept:
                canonical=sym
            sga2=SpacegroupAnalyzer(Structure(lattice=canonical["cell"],species=canonical["symbols"],
                                              coords=canonical["frac"],coords_are_cartesian=False),
                                    symprec=0.02,angle_tolerance=3.0)
            diag["canonical_spg"]=int(sga2.get_space_group_number())
            diag["canonical_spg_symbol"]=str(sga2.get_space_group_symbol())
            diag["canonicalization_success"]=True
            if accept:
                diag["canonicalization_reason"] += "+safe_symmetry_projection"
            elif not diag.get("canonical_chemistry_used",False):
                diag["canonicalization_reason"] += "+raw_shadow"
        except Exception as exc:
            diag["canonical_symmetry_reason"]="symmetry_refinement_failed:"+type(exc).__name__
            diag["canonicalization_success"]=True
            # Safe fallback: base is either the nearby chemistry projection or raw.
        return canonical, diag


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
    wyckoff_dof = [int(group[i].get_dof()) for i in range(len(group))]
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

        # The set of already-used fixed Wyckoff positions affects which
        # continuations are valid, so it must be part of the memoization state.
        used_fixed = tuple(sorted(wp for wp in wps if wyckoff_dof[wp] == 0))
        state = (start_pos, remaining, used_fixed)
        if state in dead:
            return False
        before = len(rows)

        # Nondecreasing Wyckoff indices avoid permutation duplicates while still
        # allowing repeated use of a Wyckoff position when PyXtal permits it.
        for pos in range(start_pos, len(allowed)):
            wp = allowed[pos]
            # Reusing a zero-DOF Wyckoff position reproduces the exact same
            # crystallographic orbit.  Such proposals are irreparable and must
            # never enter the GPU builder queue.  Positive-DOF positions may be
            # repeated because independent free parameters can separate them.
            if wyckoff_dof[wp] == 0 and wp in wps:
                continue
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


def _wyckoff_entry_metadata(spg: int, wp_token: str) -> dict:
    """Compact crystallographic entrance descriptors for coverage diagnostics."""
    from pyxtal.symmetry import Group
    group = Group(int(spg))
    wps = decode_int_token(wp_token)
    mult = [int(group[i].multiplicity) for i in wps]
    dof = [int(group[i].get_dof()) for i in wps]
    pattern = "+".join(str(x) for x in sorted(mult, reverse=True))
    repeated_free = 0
    for wp in set(wps):
        if int(group[wp].get_dof()) > 0:
            repeated_free += max(0, int(wps.count(wp)) - 1)
    return {
        "entry_multiplicity_pattern": pattern,
        "entry_orbit_count": int(len(wps)),
        "entry_total_wyckoff_dof": int(sum(dof)),
        "entry_fixed_orbit_count": int(sum(x == 0 for x in dof)),
        "entry_repeated_free_orbit_count": int(repeated_free),
    }


def _entrance_count_worker(payload):
    spg, labels, counts, cap = payload
    rows = _exact_entries_for_group(int(spg), tuple(labels), dict(counts), int(cap))
    return int(spg), int(len(rows))


def _precompute_compatible_space_groups(space_groups, labels, counts, cap, nworkers):
    """Enumerate target-compatible SGs once; stochastic sampling never sees impossible SGs."""
    groups = [int(x) for x in space_groups]
    payloads = [(spg, tuple(labels), dict(counts), int(cap)) for spg in groups]
    counts_out = {}
    workers = max(1, min(int(nworkers), len(groups)))
    if workers == 1:
        results = map(_entrance_count_worker, payloads)
        for spg, n in results:
            if n > 0:
                counts_out[int(spg)] = int(n)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for spg, n in pool.imap_unordered(_entrance_count_worker, payloads, chunksize=1):
                if n > 0:
                    counts_out[int(spg)] = int(n)
    return sorted(counts_out), counts_out


def _proposal_worker(worker_id, request_queue, result_queue, labels, counts, cap, seed, space_groups):
    _set_thread_limits()
    cache = {}
    space_groups = tuple(int(x) for x in space_groups)
    while True:
        request = request_queue.get()
        if request is None:
            break
        if isinstance(request, dict):
            task_id = int(request["task_id"])
            requested_spg = request.get("spg")
            requested_entry = request.get("entry_index")
        else:
            task_id = int(request)
            requested_spg = None
            requested_entry = None
        try:
            if requested_spg is None:
                if not space_groups:
                    raise RuntimeError("No compatible crystallographic space groups")
                spg = int(space_groups[deterministic_seed(seed, task_id, worker_id) % len(space_groups)])
            else:
                spg = int(requested_spg)
                if spg not in space_groups:
                    raise RuntimeError(f"Requested SG {spg} is not in the precomputed compatible set")
            if spg not in cache:
                cache[spg] = _exact_entries_for_group(spg, labels, counts, cap)
            rows = cache[spg]
            if not rows:
                # This must never silently fall back to another group.  A mismatch
                # between startup enumeration and a proposal worker is a hard error.
                raise RuntimeError(f"Requested SG {spg} has no exact target-compatible entrance")
            if requested_entry is not None:
                entry_index = int(requested_entry) % len(rows)
            else:
                entry_index = int(deterministic_seed(seed, task_id, spg, worker_id) % len(rows))
            wp, species = rows[entry_index]
            emeta = _wyckoff_entry_metadata(spg, wp)
            result_queue.put({"kind":"proposal", "task_id":task_id, "spg":spg,
                              "requested_spg":spg if requested_spg is None else int(requested_spg),
                              "entry_index":int(entry_index), "entry_count":int(len(rows)),
                              "wp_token":wp, "species_token":species, **emeta, "error":None})
        except Exception as exc:
            result_queue.put({"kind":"proposal", "task_id":task_id,
                              "requested_spg":requested_spg,
                              "requested_entry":requested_entry,
                              "error":f"{type(exc).__name__}: {exc}"})


class ProposalPool:
    def __init__(self, nworkers, labels, counts, cap, seed, space_groups):
        self.ctx = mp.get_context("spawn")
        self.requests = self.ctx.Queue(maxsize=max(4, 2 * nworkers))
        self.results = self.ctx.Queue()
        self.processes = []
        for wid in range(nworkers):
            p = self.ctx.Process(target=_proposal_worker,
                args=(wid, self.requests, self.results, labels, counts, cap, seed, tuple(space_groups)), daemon=True)
            p.start(); self.processes.append(p)
    def request(self, task_id, spg=None, entry_index=None):
        self.requests.put({"task_id":int(task_id),
                           "spg":None if spg is None else int(spg),
                           "entry_index":None if entry_index is None else int(entry_index)})
    def close(self):
        for _ in self.processes: self.requests.put(None)
        for p in self.processes: p.join()


def _builder_worker(worker_id, device_id, tasks, results, model_path, config, ti_token_counter, ti_token_budget):
    _set_thread_limits(); torch.set_num_threads(1)
    if device_id is None: device = "cpu"
    else:
        torch.cuda.set_device(int(device_id)); device = f"cuda:{int(device_id)}"
    model = XNModel(model_path)
    builder = XNBuilder(model=model, device=device, **config)
    builder.ti_token_counter = ti_token_counter
    builder.ti_token_budget = int(ti_token_budget)
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
                                                task["species_token"], heartbeat=heartbeat,
                                                task_meta=task)
            if selected is None:
                selected_list = []
            elif isinstance(selected, list):
                selected_list = selected
            else:
                selected_list = [selected]
            results.put({"kind":"result", "worker_id":worker_id, "task":task,
                         "selected":selected_list, "attempts":attempts, "error":None})
        except Exception as exc:
            results.put({"kind":"result", "worker_id":worker_id, "task":task,
                         "selected":None, "attempts":[],
                         "error":f"{type(exc).__name__}: {exc}"})


class BuilderPool:
    def __init__(self, ngpu, queue_depth, model_path, config, ti_token_budget):
        self.ctx = mp.get_context("spawn")
        devices = list(range(ngpu)) if ngpu > 0 else [None]
        self.tasks = self.ctx.Queue(maxsize=max(2, queue_depth * len(devices)))
        self.results = self.ctx.Queue()
        self.ti_token_budget = int(ti_token_budget)
        self.ti_token_counter = self.ctx.Value("q", 0)
        self.processes = []
        for wid, did in enumerate(devices):
            p = self.ctx.Process(target=_builder_worker,
                args=(wid, did, self.tasks, self.results, model_path, config,
                      self.ti_token_counter, self.ti_token_budget), daemon=True)
            p.start(); self.processes.append(p)
    @property
    def workers(self): return len(self.processes)
    @property
    def ti_tokens_used(self):
        with self.ti_token_counter.get_lock():
            return int(self.ti_token_counter.value)
    def close(self):
        for _ in self.processes: self.tasks.put(None)
        for p in self.processes: p.join()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate crystals with efficient unseen-first Ti entrances, conservative early escape, and exploratory Ti refinement")
    p.add_argument("--chemistry-model", default="data/xn_templates/chemistry_model.json")
    p.add_argument("--target", action="append", default=[],
                   help="Requested construction label or named/formula recipe as LABEL=COUNT")
    p.add_argument("--ti-token-budget", type=int, default=20000,
                   help="Global number of Ti-framework branches admitted to full refinement")
    p.add_argument("--max-runtime-minutes", type=float, default=115.0)
    p.add_argument("--max-entries-per-group", type=int, default=5000)
    p.add_argument("--space-groups", default="17-230",
                   help="Crystallographic proposal domain, e.g. 17-230, all, or 17-74,123,225")
    p.add_argument("--proposal-workers", type=int, default=0)
    p.add_argument("--ngpu", type=int, default=0)
    p.add_argument("--gpu-queue-depth", type=int, default=2)
    p.add_argument("--starts", type=int, default=48)
    p.add_argument("--screen-steps", type=int, default=20)
    p.add_argument("--refine-steps", type=int, default=105)
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
    p.add_argument("--framework-keep", type=int, default=12)
    p.add_argument("--framework-patience", type=int, default=12,
                   help="Stop a Ti-framework branch after this many non-improving steps")
    p.add_argument("--octahedral-branches", type=int, default=32,
                   help="Rigid rotating TiO6 proposal branches per accepted Ti framework")
    p.add_argument("--oxygen-proposal-oversample", type=int, default=4)
    p.add_argument("--oxygen-proposal-descriptor-tol", type=float, default=0.025)
    p.add_argument("--oxygen-basin-prune-every", type=int, default=25)
    p.add_argument("--octahedron-prepare-steps", type=int, default=40)
    p.add_argument("--octahedron-match-steps", type=int, default=100)
    p.add_argument("--octahedron-cluster-steps", type=int, default=50)
    p.add_argument("--floating-coincidence-sigma", type=float, default=0.24,
                   help="Final narrow capture width; rigid search starts automatically at >=1.4 A")
    p.add_argument("--floating-cluster-tolerance", type=float, default=0.38)
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
                   help="Write one compact manager status line every N completed proposal tasks")
    p.add_argument("--verbose-workers", action="store_true",
                   help="Print per-worker optimization-stage heartbeats")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-folder", default="generated_xn_v78_efficient_breadth")
    return p.parse_args()



def _append_jsonl_rows(path: Path, rows: list[dict]) -> None:
    """Append one event batch with O(batch) I/O; safe for very long token-budget runs."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")


def _jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> None:
    """Materialize the append-only diagnostic log as one rectangular CSV."""
    if not jsonl_path.exists():
        pd.DataFrame().to_csv(csv_path, index=False)
        return
    fields = []
    seen = set()
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            for key in row:
                if key not in seen:
                    seen.add(key); fields.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    writer.writerow(json.loads(line))
                except Exception:
                    continue


def _sampling_score(stat: Counter, total_tokens: int, n_groups: int) -> float:
    """Coverage-first stochastic reward with crystallographic diversity feedback.

    Final-SG/family/symmetry information is deliberately soft: it changes only the
    probability of future entrances and never becomes an acceptance/rejection rule.
    """
    completed = int(stat.get("completed", 0))
    tokens = float(stat.get("ti_tokens", 0))
    new = float(stat.get("new_ti_basins", 0))
    dup = float(stat.get("duplicate_ti_basins", 0))
    seen = new + dup
    novelty_yield = (new + 1.0) / (seen + 4.0)
    duplicate_rate = (dup + 0.5) / (seen + 2.0)
    topo_tasks = float(stat.get("topology_productive_tasks", 0))
    strict_tasks = float(stat.get("strict_productive_tasks", 0))
    productive_rate = (topo_tasks + 0.5) / (completed + 2.0)
    strict_rate = (strict_tasks + 0.25) / (completed + 2.0)
    sym_topo = float(stat.get("symmetry_topology_productive_tasks", 0))
    fallback_topo = float(stat.get("site_fallback_topology_productive_tasks", 0))
    sym_topo_rate = (sym_topo + 0.20) / (completed + 2.0)
    fallback_topo_rate = fallback_topo / max(topo_tasks + 1.0, 1.0)
    new_families = float(stat.get("new_families", 0))
    family_variants = float(stat.get("existing_family_variants", 0))
    family_rate = (new_families + 0.20) / (completed + 2.0)
    family_variant_fraction = family_variants / max(new_families + family_variants + 2.0, 1.0)
    retained = float(stat.get("symmetry_retained_exact", 0))
    broken = float(stat.get("symmetry_broken_exact", 0))
    retention_rate = (retained + 0.15) / max(retained + broken + 1.0, 1.0)
    final_div_credit = float(stat.get("final_spg_diversity_credit", 0.0))
    final_div_rate = (final_div_credit + 0.10) / (completed + 2.0)
    target_exposure = float(total_tokens) / max(int(n_groups), 1)
    coverage = math.log((target_exposure + 12.0) / (tokens + 12.0))
    return (1.75 * coverage + 0.55 * novelty_yield + 0.70 * productive_rate
            + 0.20 * strict_rate + 1.00 * sym_topo_rate + 1.15 * family_rate
            + 0.80 * retention_rate + 1.00 * final_div_rate
            - 0.25 * duplicate_rate - 0.55 * family_variant_fraction
            - 0.40 * fallback_topo_rate)


def _softmax_choice(items, scores, rng: np.random.Generator, temperature: float = 0.70):
    if not items:
        return None
    a = np.asarray(scores, float) / max(float(temperature), 1.0e-6)
    a -= np.max(a)
    p = np.exp(np.clip(a, -40.0, 40.0))
    p /= np.sum(p)
    return items[int(rng.choice(len(items), p=p))]


def _stochastic_pick_space_group(space_groups, stats, rng: np.random.Generator,
                                 total_tokens: int, uniform_floor: float = 0.30) -> int:
    """Breadth-first SG draw, then soft adaptive allocation.

    Every compatible SG is touched once before productivity feedback is allowed to
    dominate.  Thereafter a substantial uniform floor preserves exploration.
    """
    groups = [int(x) for x in space_groups]
    unseen = [spg for spg in groups if int(stats[spg].get("attempts", 0)) <= 0]
    if unseen:
        return int(unseen[int(rng.integers(0, len(unseen)))])
    if rng.random() < float(uniform_floor):
        return int(groups[int(rng.integers(0, len(groups)))])
    scores = [_sampling_score(stats[spg], total_tokens, len(groups)) for spg in groups]
    return int(_softmax_choice(groups, scores, rng, temperature=0.85))


def _stochastic_pick_entry(spg: int, entry_counts: dict, entry_stats: dict,
                           rng: np.random.Generator, total_tokens: int,
                           uniform_floor: float = 0.30, reserved_entries=None):
    """Sample unseen Wyckoff entrances before revisiting known ones.

    ``reserved_entries`` contains manager-side in-flight proposal reservations so a
    shallow look-ahead queue cannot accidentally request the same unseen entrance
    several times before its first result reaches the GPU queue.
    """
    n = int(entry_counts.get(int(spg), 0))
    if n <= 0:
        return None
    reserved_entries = reserved_entries or set()
    unseen = [idx for idx in range(n)
              if int(entry_stats[(int(spg), int(idx))].get("attempts", 0)) <= 0
              and (int(spg), int(idx)) not in reserved_entries]
    if unseen:
        return int(unseen[int(rng.integers(0, len(unseen)))])

    # Once basic entry coverage is exhausted, keep a large random floor and only
    # then use productivity/family feedback. Avoid currently reserved entries when
    # another choice exists.
    available = [idx for idx in range(n) if (int(spg), int(idx)) not in reserved_entries]
    if not available:
        available = list(range(n))
    if rng.random() < float(uniform_floor):
        return int(available[int(rng.integers(0, len(available)))])
    random_n = min(len(available), 128)
    if random_n == len(available):
        candidates = set(available)
    else:
        candidates = set(int(x) for x in rng.choice(available, size=random_n, replace=False))
    known = [(key[1], stat) for key, stat in entry_stats.items()
             if int(key[0]) == int(spg) and int(key[1]) in available]
    spg_tokens = int(sum(int(stat.get("ti_tokens", 0)) for _idx, stat in known))
    if known:
        known.sort(key=lambda item: _sampling_score(item[1], spg_tokens, max(n, 1)), reverse=True)
        candidates.update(int(idx) for idx, _stat in known[:24])
    items = sorted(candidates)
    scores = [_sampling_score(entry_stats[(int(spg), int(idx))], spg_tokens, max(n, 1)) for idx in items]
    return int(_softmax_choice(items, scores, rng, temperature=0.80))



def _pick_global_unseen_entrance(space_groups, entry_counts: dict, entry_stats: dict,
                                 rng: np.random.Generator, reserved_entries=None):
    """Pick one globally unseen entrance without letting low-symmetry catalog size dominate.

    Space groups with at least one unseen, unreserved entrance are sampled uniformly;
    then one unseen entrance is drawn uniformly inside that SG.  Thus a group with
    hundreds of legal decompositions does not automatically monopolize the breadth
    budget.  Returns ``(spg, entry_index)`` or ``None`` when basic coverage is exhausted.
    """
    reserved_entries = reserved_entries or set()
    groups = []
    unseen_by_group = {}
    for spg in (int(x) for x in space_groups):
        n = int(entry_counts.get(spg, 0))
        if n <= 0:
            continue
        unseen = [idx for idx in range(n)
                  if int(entry_stats[(spg, int(idx))].get("attempts", 0)) <= 0
                  and (spg, int(idx)) not in reserved_entries]
        if unseen:
            groups.append(spg)
            unseen_by_group[spg] = unseen
    if not groups:
        return None
    spg = int(groups[int(rng.integers(0, len(groups)))])
    unseen = unseen_by_group[spg]
    eidx = int(unseen[int(rng.integers(0, len(unseen)))])
    return spg, eidx

def _update_sampler_stat(stat: Counter, *, tokens: int, new_basins: int, duplicates: int,
                         topology_productive: bool, symmetry_topology_productive: bool,
                         site_fallback_topology_productive: bool, strict_successes: int) -> None:
    stat["completed"] += 1
    stat["ti_tokens"] += int(tokens)
    stat["new_ti_basins"] += int(new_basins)
    stat["duplicate_ti_basins"] += int(duplicates)
    stat["topology_productive_tasks"] += int(bool(topology_productive))
    stat["symmetry_topology_productive_tasks"] += int(bool(symmetry_topology_productive))
    stat["site_fallback_topology_productive_tasks"] += int(bool(site_fallback_topology_productive))
    stat["strict_successes"] += int(strict_successes)
    stat["strict_productive_tasks"] += int(int(strict_successes) > 0)


def _update_family_sampler_stat(stat: Counter, *, new_families: int, existing_variants: int) -> None:
    """Feed diagnostic family novelty back into sampling without making it a rejection rule."""
    stat["new_families"] += int(new_families)
    stat["existing_family_variants"] += int(existing_variants)


def _update_crystal_sampler_stat(stat: Counter, *, symmetry_retained: int,
                                 symmetry_broken: int, final_spg_diversity_credit: float,
                                 new_final_spgs: int) -> None:
    """Soft feedback from detected final crystallography; never an export gate."""
    stat["symmetry_retained_exact"] += int(symmetry_retained)
    stat["symmetry_broken_exact"] += int(symmetry_broken)
    stat["final_spg_diversity_credit"] += float(final_spg_diversity_credit)
    stat["new_final_spgs"] += int(new_final_spgs)


def main() -> None:
    args = parse_args()
    if int(args.ti_token_budget) <= 0:
        raise ValueError("--ti-token-budget must be positive")
    _set_thread_limits()
    model = XNModel(args.chemistry_model)
    requested_space_groups = parse_space_groups(args.space_groups)
    space_groups = list(requested_space_groups)
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
    output = Path(args.output_folder)
    run_state = [output / "attempts.jsonl", output / "ti_framework_registry.jsonl",
                 output / "ti_basin_memory.jsonl", output / "framework_chemistry_memory.jsonl",
                 output / "dedup_index.jsonl", output / "summary.json"]
    stale = [str(x) for x in run_state if x.exists() and x.stat().st_size > 0]
    if stale:
        raise RuntimeError(
            "v78 efficient-breadth entrance benchmarks require a fresh output folder; existing run state: "
            + ", ".join(stale)
        )
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
        oxygen_screen_max_max=args.oxygen_screen_max_max,
        octahedral_branches=args.octahedral_branches,
        octahedron_prepare_steps=args.octahedron_prepare_steps,
        octahedron_match_steps=args.octahedron_match_steps,
        octahedron_cluster_steps=args.octahedron_cluster_steps,
        floating_coincidence_sigma=args.floating_coincidence_sigma,
        floating_cluster_tolerance=args.floating_cluster_tolerance,
        ti_registry_path=str(output / "ti_framework_registry.jsonl"),
        oxygen_proposal_oversample=args.oxygen_proposal_oversample,
        oxygen_proposal_descriptor_tol=args.oxygen_proposal_descriptor_tol,
        oxygen_basin_prune_every=args.oxygen_basin_prune_every,
        framework_memory_path=str(output / "framework_chemistry_memory.jsonl"),
        framework_basin_memory_path=str(output / "ti_basin_memory.jsonl"),
        framework_intelligent_keep=3)
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

    # Enumerate exact target-compatible crystallographic entrances once.  Impossible
    # SGs never enter the stochastic sampler and proposal workers are forbidden to
    # fall back to another group.
    catalog_workers = max(1, min(ncpu, 8))
    space_groups, precomputed_entry_counts = _precompute_compatible_space_groups(
        requested_space_groups, model.labels, construction_counts,
        args.max_entries_per_group, catalog_workers)
    if not space_groups:
        raise RuntimeError("No target-compatible crystallographic entrances in the requested SG domain")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"spg": int(spg), "entry_count": int(precomputed_entry_counts[spg])}
                  for spg in space_groups]).to_csv(output / "compatible_space_groups.csv", index=False)

    proposal_pool = ProposalPool(proposal_workers, model.labels, construction_counts,
                                 args.max_entries_per_group, args.seed + 211, space_groups)
    builder_pool = BuilderPool(ngpu, args.gpu_queue_depth, args.chemistry_model, config, args.ti_token_budget)
    pool_dir = output / "candidate_pool"
    canonical_pool_dir = output / "canonical_pool"
    family_rep_dir = output / "family_representatives"
    pool_dir.mkdir(parents=True, exist_ok=True)
    canonical_pool_dir.mkdir(parents=True, exist_ok=True)
    family_rep_dir.mkdir(parents=True, exist_ok=True)
    attempts_jsonl = output / "attempts.jsonl"
    progress_csv = output / "progress.csv"
    active_labels = tuple(label for label, count in counts.items() if count > 0)
    deduplicator = ConservativeDeduplicator(output / "dedup_index.jsonl", active_labels, model)
    # v74 family semantics are quotient-like: first cluster raw strict candidates,
    # canonicalize only one representative per raw family, then allow different
    # raw families to merge in canonical space.  A raw family can never split,
    # so N_canonical <= N_raw by construction.
    raw_family_tracker = DiagnosticFamilyTracker(primitive_cell=False)
    family_tracker = DiagnosticFamilyTracker(primitive_cell=True)
    canonicalizer = ShadowCanonicalizer(model, symprec_A=0.10, angle_tolerance_deg=5.0)
    raw_family_canonical_cache = {}
    final_spg_counts = Counter()
    canonical_spg_counts = Counter()
    accepted_rows=[]; duplicate_rows=[]; export_audit_rows=[]
    stage_counts=Counter(ti_tested=0,ti_accepted=0,ti_entrance_scouted=0,
                         ti_early_escape_checked=0,ti_early_escaped=0,ti_early_rescued=0,
                         o_topology_tested=0,o_topology_accepted=0,
                         o_analytic_tested=0,o_analytic_accepted=0,o_attachment_tested=0,o_attachment_accepted=0,
                         strict_tested=0,strict_accepted=0)
    existing_candidate_ids = [int(r.get("candidate_id", -1)) for r in deduplicator.records]
    next_candidate_id = max(existing_candidate_ids, default=-1) + 1
    spg_stats=defaultdict(lambda: Counter(attempts=0, exact_unique=0, duplicates=0))
    entrance_stats=defaultdict(Counter)
    entry_counts=dict(precomputed_entry_counts)
    entrance_reserved=set()
    sampler_rng=np.random.default_rng(args.seed + 909)
    start=time.perf_counter(); deadline=start + args.max_runtime_minutes*60.0
    last_progress_token = -1
    last_progress_time = start
    last_unique_token = 0
    last_family_token = 0

    def write_entrance_coverage():
        rows = []
        for spg in space_groups:
            n = int(entry_counts.get(int(spg), 0))
            visits = [int(entrance_stats[(int(spg), idx)].get("attempts", 0)) for idx in range(n)]
            unique = int(sum(v > 0 for v in visits))
            total_visits = int(sum(visits))
            rows.append({
                "spg": int(spg),
                "entry_count": n,
                "unique_entries_sampled": unique,
                "entry_coverage_fraction": unique / max(n, 1),
                "total_entry_visits": total_visits,
                "repeated_entry_visits": max(total_visits - unique, 0),
                "all_entries_sampled": bool(n > 0 and unique >= n),
            })
        pd.DataFrame(rows).to_csv(output / "entrance_coverage.csv", index=False)

    def write_progress_snapshot(force=False):
        nonlocal last_progress_token, last_progress_time
        now = time.perf_counter()
        tokens = int(builder_pool.ti_tokens_used)
        if not force and tokens - last_progress_token < 250 and now - last_progress_time < 120.0:
            return
        new_ti = int(sum(x.get("new_ti_basins", 0) for x in spg_stats.values()))
        elapsed_min = (now - start) / 60.0
        row = {
            "stage": "progress_snapshot",
            "elapsed_min": elapsed_min,
            "ti_tokens_used": tokens,
            "ti_token_fraction": tokens / max(int(args.ti_token_budget), 1),
            "proposals_completed": int(completed),
            "new_ti_count": new_ti,
            "o_topology_count": int(stage_counts["o_topology_accepted"]),
            "o_analytic_count": int(stage_counts["o_analytic_accepted"]),
            "strict_valid_count": int(stage_counts["strict_accepted"]),
            "exact_unique_count": int(len(accepted_rows)),
            "family_count": int(len(family_tracker.representatives)),
            "canonical_family_count": int(len(family_tracker.representatives)),
            "raw_family_count": int(len(raw_family_tracker.representatives)),
            "unique_entries_sampled": int(sum(1 for stat in entrance_stats.values() if int(stat.get("attempts", 0)) > 0)),
            "total_compatible_entries": int(sum(int(v) for v in entry_counts.values())),
            "entry_coverage_fraction": (
                sum(1 for stat in entrance_stats.values() if int(stat.get("attempts", 0)) > 0)
                / max(sum(int(v) for v in entry_counts.values()), 1)
            ),
            "repeated_entry_visits": int(sum(max(int(stat.get("attempts", 0)) - 1, 0) for stat in entrance_stats.values())),
            "spg_with_any_entry_sampled": int(sum(any(int(entrance_stats[(int(spg), idx)].get("attempts", 0)) > 0
                                                       for idx in range(int(entry_counts.get(int(spg), 0))))
                                                   for spg in space_groups)),
            "spg_all_entries_sampled": int(sum(int(entry_counts.get(int(spg), 0)) > 0 and
                                                  all(int(entrance_stats[(int(spg), idx)].get("attempts", 0)) > 0
                                                      for idx in range(int(entry_counts.get(int(spg), 0))))
                                              for spg in space_groups)),
            "distinct_final_spg_count": int(len(final_spg_counts)),
            "distinct_canonical_spg_count": int(len(canonical_spg_counts)),
            "symmetry_retained_exact_count": int(sum(bool(r.get("final_symmetry_retained", False)) for r in accepted_rows)),
            "symmetry_orbit_exact_count": int(sum(bool(r.get("hypergraph_symmetry_preserved", False)) for r in accepted_rows)),
            "site_fallback_exact_count": int(sum(bool(r.get("hypergraph_site_fallback_used", False)) for r in accepted_rows)),
            "exact_unique_per_1000_tokens": 1000.0 * len(accepted_rows) / max(tokens, 1),
            "families_per_1000_tokens": 1000.0 * len(family_tracker.representatives) / max(tokens, 1),
            "exact_unique_per_hour": len(accepted_rows) / max(elapsed_min / 60.0, 1.0e-9),
            "families_per_hour": len(family_tracker.representatives) / max(elapsed_min / 60.0, 1.0e-9),
            "tokens_since_last_unique": max(tokens - int(last_unique_token), 0),
            "tokens_since_last_family": max(tokens - int(last_family_token), 0),
        }
        _append_jsonl_rows(attempts_jsonl, [row])
        exists = progress_csv.exists() and progress_csv.stat().st_size > 0
        pd.DataFrame([row]).to_csv(progress_csv, mode="a", header=not exists, index=False)
        # Materialize the full attempts.csv only at sparse progress snapshots, not
        # after every proposal.  This keeps a live inspectable CSV without recreating
        # the v59/v60 I/O bottleneck.
        _jsonl_to_csv(attempts_jsonl, output / "attempts.csv")
        write_entrance_coverage()
        last_progress_token = tokens
        last_progress_time = now
    next_id=0; requested=0; submitted=0; completed=0; proposal_pending=set(); gpu_inflight=set(); worker_errors=0
    request_sampling_channel = {}
    max_gpu_inflight=max(1, builder_pool.workers * args.gpu_queue_depth)
    # Keep the proposal look-ahead shallow so stochastic sampling sees recent outcomes
    # instead of queuing hundreds of stale choices ahead of GPU feedback.
    max_proposals=max(4, max_gpu_inflight*2)
    stop_new=False
    last_manager_status = start
    print("--- Juliette v78 efficient unseen-first Wyckoff breadth / explore-only Ti / topology-diverse O ---", flush=True)
    print(
        f"Targets: {args.target}; resolved={counts}; construction={construction_counts}; "
        f"mode={construction_plan['mode']}; symmetry={args.construction_symmetry}; "
        f"atoms={model.physical_count(counts)}; Ti_token_budget={args.ti_token_budget}; compatible_SG={len(space_groups)}; global_unseen_fraction=0.95; stochastic_uniform_floor=0.30; "
        f"allocated_CPU={ncpu}; proposal_workers={proposal_workers}; GPU={builder_pool.workers}",
        flush=True,
    )
    try:
        while True:
            now=time.perf_counter()
            if now >= deadline or builder_pool.ti_tokens_used >= args.ti_token_budget:
                stop_new=True
            while (not stop_new and len(proposal_pending) < max_proposals):
                sampled_tokens = int(builder_pool.ti_tokens_used)
                # v78: while unseen entrances remain, 95% of requests are true breadth
                # draws.  The remaining 5% is an explicit exploitation/revisit channel.
                breadth_pick = None
                if sampler_rng.random() < 0.95:
                    breadth_pick = _pick_global_unseen_entrance(
                        space_groups, entry_counts, entrance_stats, sampler_rng,
                        reserved_entries=entrance_reserved)
                if breadth_pick is not None:
                    requested_spg, requested_entry = breadth_pick
                    sampling_channel = "global_unseen"
                else:
                    requested_spg = _stochastic_pick_space_group(
                        space_groups, spg_stats, sampler_rng, sampled_tokens, uniform_floor=0.30)
                    requested_entry = _stochastic_pick_entry(
                        requested_spg, entry_counts, entrance_stats, sampler_rng, sampled_tokens,
                        uniform_floor=0.30, reserved_entries=entrance_reserved)
                    sampling_channel = "adaptive_exploit"
                request_sampling_channel[int(next_id)] = sampling_channel
                proposal_pool.request(next_id, requested_spg, requested_entry)
                if requested_entry is not None:
                    entrance_reserved.add((int(requested_spg), int(requested_entry)))
                proposal_pending.add(next_id)
                next_id += 1; requested += 1
            # Drain prepared proposals into GPU queue while capacity exists.
            while len(gpu_inflight) < max_gpu_inflight:
                try: msg=proposal_pool.results.get_nowait()
                except queue.Empty: break
                tid=int(msg["task_id"]); proposal_pending.discard(tid)
                _rsg = msg.get("requested_spg")
                _rei = msg.get("entry_index", msg.get("requested_entry"))
                if _rsg is not None and _rei is not None:
                    entrance_reserved.discard((int(_rsg), int(_rei)))
                if stop_new:
                    request_sampling_channel.pop(tid, None)
                    continue
                if msg.get("error"):
                    request_sampling_channel.pop(tid, None)
                    _append_jsonl_rows(attempts_jsonl, [{"attempt_id":tid,"stage":"proposal_error","error":msg["error"]}])
                    completed += 1; continue
                actual_spg = int(msg["spg"])
                requested_spg = int(msg.get("requested_spg", actual_spg))
                if actual_spg != requested_spg:
                    raise RuntimeError(
                        f"Proposal SG mismatch: requested {requested_spg}, generated {actual_spg}; fallback is forbidden"
                    )
                entry_counts[actual_spg] = int(msg.get("entry_count", entry_counts.get(actual_spg, 0)))
                eidx = int(msg["entry_index"]) if msg.get("entry_index") is not None else -1
                visit_before = int(entrance_stats[(actual_spg, eidx)].get("attempts", 0)) if eidx >= 0 else 0
                task={**msg,
                      "entrance_sampling_channel": request_sampling_channel.pop(tid, "unknown"),
                      "entry_visit_number": int(visit_before + 1),
                      "entry_first_pass": bool(visit_before == 0),
                      "seed":deterministic_seed(args.seed, tid, msg["spg"], msg["wp_token"], msg["species_token"])}
                builder_pool.tasks.put(task); gpu_inflight.add(tid); submitted += 1
                spg_stats[actual_spg]["attempts"] += 1
                if eidx >= 0:
                    entrance_stats[(actual_spg, eidx)]["attempts"] += 1
            try:
                event=builder_pool.results.get(timeout=1.0)
            except queue.Empty:
                now = time.perf_counter()
                if now - last_manager_status >= 30.0:
                    print(
                        f"Manager: exact_unique={len(accepted_rows)} families={len(family_tracker.representatives)}; proposals={completed}; "
                        f"Ti_tokens={builder_pool.ti_tokens_used}/{args.ti_token_budget}; "
                        f"Ti={stage_counts['ti_accepted']}/{stage_counts['ti_tested']}; "
                        f"newTi={sum(x.get('new_ti_basins',0) for x in spg_stats.values())}; "
                        f"Otop={stage_counts['o_topology_accepted']}/{stage_counts['o_topology_tested']}; "
                        f"Oanalytic={stage_counts['o_analytic_accepted']}/{stage_counts['o_analytic_tested']}; "
                        f"Oattach={stage_counts['o_attachment_accepted']}/{stage_counts['o_attachment_tested']}; "
                        f"proposal_pending={len(proposal_pending)}; gpu_inflight={len(gpu_inflight)}; "
                        f"elapsed={(now-start)/60.0:.1f} min",
                        flush=True,
                    )
                    last_manager_status = now
                if stop_new and not gpu_inflight: break
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
            if builder_pool.ti_tokens_used >= args.ti_token_budget:
                stop_new = True
            event_attempt_rows = []
            event_elapsed_min = (time.perf_counter() - start) / 60.0
            event_tokens = int(builder_pool.ti_tokens_used)
            for row in event.get("attempts",[]):
                event_attempt_rows.append({"attempt_id":tid,"spg":task["spg"],
                    "elapsed_min": event_elapsed_min, "ti_tokens_used": event_tokens,
                    "requested_spg":task.get("requested_spg"),
                    "entry_index":task.get("entry_index"), "entry_count":task.get("entry_count"),
                    "entrance_sampling_channel":task.get("entrance_sampling_channel"),
                    "entry_visit_number":task.get("entry_visit_number"),
                    "entry_first_pass":task.get("entry_first_pass"),
                    "entry_multiplicity_pattern":task.get("entry_multiplicity_pattern"),
                    "entry_orbit_count":task.get("entry_orbit_count"),
                    "entry_total_wyckoff_dof":task.get("entry_total_wyckoff_dof"),
                    "entry_fixed_orbit_count":task.get("entry_fixed_orbit_count"),
                    "entry_repeated_free_orbit_count":task.get("entry_repeated_free_orbit_count"),
                    "wp_token":task["wp_token"], "species_token":task["species_token"], **row})
                stage=row.get("stage")
                if stage=="ti_framework_screen":
                    stage_counts["ti_tested"] += 1
                    stage_counts["ti_accepted"] += int(bool(row.get("ti_framework_accepted",False)))
                elif stage=="ti_entrance_scout":
                    stage_counts["ti_entrance_scouted"] += 1
                elif stage=="ti_early_escape_check":
                    stage_counts["ti_early_escape_checked"] += 1
                    stage_counts["ti_early_escaped"] += int(bool(row.get("early_escape", False)))
                    stage_counts["ti_early_rescued"] += int(bool(row.get("early_escape_rescued", False)))
                elif stage=="ti_early_escape":
                    stage_counts["ti_early_escaped"] += 1
                elif stage=="ti_basin_precheck":
                    stage_counts["ti_basin_prechecked"] += 1
                    stage_counts["ti_basin_skipped"] += int(bool(row.get("basin_skip",False)))
                elif stage=="ti_token_admit":
                    stage_counts["ti_token_admitted"] += 1
                elif stage=="ti_chemistry_memory":
                    stage_counts["ti_chemistry_memory_tested"] += 1
                elif stage=="ti_framework_duplicate_reject":
                    stage_counts["ti_registry_duplicate_rejected"] += 1
                elif stage=="o_topology_screen":
                    stage_counts["o_topology_tested"] += 1
                    stage_counts["o_topology_accepted"] += int(bool(row.get("o_topology_accepted",False)))
                elif stage=="analytic_o_assignment_screen":
                    stage_counts["o_analytic_tested"] += 1
                    stage_counts["o_analytic_accepted"] += int(bool(row.get("analytic_assignment_accepted",False)))
                elif stage=="o_attachment_screen":
                    stage_counts["o_attachment_tested"] += 1
                    stage_counts["o_attachment_accepted"] += int(bool(row.get("o_attachment_accepted",False)))
                elif stage=="strict_final_audit":
                    stage_counts["strict_tested"] += 1
                    stage_counts["strict_accepted"] += int(bool(row.get("strict_valid",False)))
            # Feed the exact post-refinement outcome back into the proposal sampler.
            task_tokens = sum(1 for row in event.get("attempts", []) if row.get("stage") == "ti_token_admit")
            task_registry_seen = sum(1 for row in event.get("attempts", []) if row.get("stage") == "ti_chemistry_memory")
            task_duplicate_basins = sum(1 for row in event.get("attempts", [])
                                        if row.get("stage") == "ti_framework_duplicate_reject")
            task_new_basins = max(0, task_registry_seen - task_duplicate_basins)
            task_topology_productive = any(bool(row.get("analytic_assignment_accepted", False))
                                           for row in event.get("attempts", [])
                                           if row.get("stage") == "analytic_o_assignment_screen")
            task_symmetry_topology_productive = any(
                bool(row.get("hypergraph_symmetry_preserved", False))
                for row in event.get("attempts", []) if row.get("stage") == "o_topology_screen")
            task_site_fallback_topology_productive = any(
                bool(row.get("hypergraph_site_fallback_used", False))
                for row in event.get("attempts", []) if row.get("stage") == "o_topology_screen")
            task_strict_successes = sum(int(bool(row.get("strict_valid", False)))
                                        for row in event.get("attempts", [])
                                        if row.get("stage") == "strict_final_audit")
            sstat = spg_stats[int(task["spg"])]
            _update_sampler_stat(sstat, tokens=task_tokens, new_basins=task_new_basins,
                                 duplicates=task_duplicate_basins,
                                 topology_productive=task_topology_productive,
                                 symmetry_topology_productive=task_symmetry_topology_productive,
                                 site_fallback_topology_productive=task_site_fallback_topology_productive,
                                 strict_successes=task_strict_successes)
            if task.get("entry_index") is not None:
                estat = entrance_stats[(int(task["spg"]), int(task["entry_index"]))]
                _update_sampler_stat(estat, tokens=task_tokens, new_basins=task_new_basins,
                                     duplicates=task_duplicate_basins,
                                     topology_productive=task_topology_productive,
                                     symmetry_topology_productive=task_symmetry_topology_productive,
                                     site_fallback_topology_productive=task_site_fallback_topology_productive,
                                     strict_successes=task_strict_successes)

            if event.get("error"):
                worker_errors += 1
                err = str(event["error"])
                event_attempt_rows.append({"attempt_id":tid,"spg":task["spg"],"wp_token":task.get("wp_token"),
                    "species_token":task.get("species_token"),"stage":"worker_error","error":err})
                print(f"Worker error on attempt {tid} (spg={task['spg']}): {err}", flush=True)
                if worker_errors >= max(8, 2 * builder_pool.workers) and stage_counts["ti_tested"] == 0:
                    raise RuntimeError(
                        f"Aborting after {worker_errors} worker errors before any Ti framework was evaluated. "
                        f"First inspect {output/'attempts.jsonl'} for the exception."
                    )
            _append_jsonl_rows(attempts_jsonl, event_attempt_rows)
            selected_list = event.get("selected") or []
            task_new_families = 0
            task_existing_family_variants = 0
            task_symmetry_retained = 0
            task_symmetry_broken = 0
            task_final_spg_diversity_credit = 0.0
            task_new_final_spgs = 0
            for selected in selected_list:
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
                    continue

                if not bool(selected.get("export_valid", False)):
                    raise RuntimeError("Internal export-gate violation: a non-export-valid candidate reached candidate_pool")
                cid = next_candidate_id + len(accepted_rows)
                detected_spgr, detected_spgr_symbol = _detect_space_group(selected)
                parent_spg = int(task["spg"])
                final_spg = int(detected_spgr) if detected_spgr is not None else 0
                prior_final_count = int(final_spg_counts.get(final_spg, 0)) if final_spg > 0 else 0
                final_diversity_credit = (1.0 / math.sqrt(1.0 + prior_final_count)) if final_spg > 0 else 0.0
                new_final_spg = bool(final_spg > 0 and prior_final_count == 0)
                if final_spg > 0:
                    final_spg_counts[final_spg] += 1
                symmetry_orbit_cover = bool(selected.get("hypergraph_symmetry_preserved", False))
                parent_final_spg_match = bool(final_spg > 0 and final_spg == parent_spg)
                parent_symmetry_retained, parent_symmetry_max_error_A, parent_symmetry_ops_checked = \
                    _parent_space_group_retention(selected, parent_spg, tolerance_A=0.12)
                # Fail conservative if the explicit parent-operation audit is unavailable:
                # exact SG equality is sufficient, while an arbitrary detected SG is not.
                if int(parent_symmetry_ops_checked) <= 0:
                    parent_symmetry_retained = bool(parent_final_spg_match)
                symmetry_retained = bool(parent_symmetry_retained)
                site_fallback = bool(selected.get("hypergraph_site_fallback_used", False))
                cif = pool_dir / f"candidate_{cid:06d}_parentSG{parent_spg}_finalSG{final_spg}.cif"
                _write_cif(selected, cif)
                deduplicator.add(cid, selected, task["species_token"], dedup["fingerprint"])

                # v74: canonicalization is a quotient of the raw family partition.
                # Only the first member of a raw family is normalized/symmetrized.
                # Later members inherit its canonical family, guaranteeing that
                # standardization can merge raw families but can never split one.
                raw_family_diag = raw_family_tracker.assign(cid, selected)
                raw_family_id = int(raw_family_diag["family_id"])
                raw_family_is_new = bool(raw_family_diag.get("family_is_new", False))
                if raw_family_is_new:
                    canonical_result, canonical_diag = canonicalizer.canonicalize(selected)
                    canonical_diag = dict(canonical_diag)
                    canonical_diag["canonicalization_inherited"] = False
                    canonical_diag["canonicalization_source_candidate_id"] = int(cid)
                    canonical_spg = int(canonical_diag.get("canonical_spg", 0) or 0)
                    canonical_cif = canonical_pool_dir / (f"canonical_rawfamily_{raw_family_id:04d}_rep_{cid:06d}_"
                                                            f"rawSG{final_spg}_canonicalSG{canonical_spg}.cif")
                    _write_cif(canonical_result, canonical_cif)
                    family_diag = family_tracker.assign(cid, canonical_result)
                    raw_family_canonical_cache[raw_family_id] = {
                        "family_id": int(family_diag["family_id"]),
                        "source_candidate_id": int(cid),
                        "canonical_result": {
                            "cell": np.asarray(canonical_result["cell"], float).copy(),
                            "frac": np.asarray(canonical_result["frac"], float).copy(),
                            "symbols": list(canonical_result["symbols"]),
                        },
                        "canonical_diag": dict(canonical_diag),
                        "canonical_spg": int(canonical_spg),
                        "canonical_cif": str(canonical_cif),
                    }
                    is_new_family = bool(family_diag.get("family_is_new", False))
                    if is_new_family:
                        task_new_families += 1
                        rep_cif = family_rep_dir / (f"family_{int(family_diag['family_id']):04d}_candidate_{cid:06d}_"
                                                    f"canonicalSG{canonical_spg}.cif")
                        _write_cif(canonical_result, rep_cif)
                    else:
                        task_existing_family_variants += 1
                else:
                    cache = raw_family_canonical_cache.get(raw_family_id)
                    if cache is None:
                        raise RuntimeError(f"Missing canonical representative for raw family {raw_family_id}")
                    canonical_result = cache["canonical_result"]
                    canonical_spg = int(cache["canonical_spg"])
                    canonical_cif = Path(cache["canonical_cif"])
                    source_diag = dict(cache["canonical_diag"])
                    canonical_diag = {
                        "canonicalization_success": True,
                        "canonicalization_reason": "inherited_from_raw_family_representative",
                        "canonicalization_inherited": True,
                        "canonicalization_source_candidate_id": int(cache["source_candidate_id"]),
                        "canonical_symprec_A": source_diag.get("canonical_symprec_A"),
                        "canonical_spg": int(canonical_spg),
                        "canonical_spg_symbol": source_diag.get("canonical_spg_symbol"),
                        "canonical_chemistry_used": source_diag.get("canonical_chemistry_used"),
                        "canonical_aggressive_symmetry_used": source_diag.get("canonical_aggressive_symmetry_used"),
                        "canonical_chemistry_guard_passed": source_diag.get("canonical_chemistry_guard_passed"),
                        "canonical_chemistry_topology_preserved": source_diag.get("canonical_chemistry_topology_preserved"),
                        "canonical_aggressive_symmetry_topology_preserved": source_diag.get("canonical_aggressive_symmetry_topology_preserved"),
                    }
                    family_id = int(cache["family_id"])
                    family_tracker.members[family_id].append(int(cid))
                    family_diag = {"family_id": family_id, "family_is_new": False,
                                   "family_rms_normalized": None, "family_max_normalized": None}
                    is_new_family = False
                    task_existing_family_variants += 1

                if canonical_spg > 0:
                    canonical_spg_counts[canonical_spg] += 1
                if len(family_tracker.representatives) > len(raw_family_tracker.representatives):
                    raise RuntimeError(
                        "Canonical family quotient invariant violated: "
                        f"canonical={len(family_tracker.representatives)} > raw={len(raw_family_tracker.representatives)}"
                    )
                effective_final_diversity_credit = float(final_diversity_credit) * (1.0 if is_new_family else 0.12)
                task_symmetry_retained += int(symmetry_retained)
                task_symmetry_broken += int(not symmetry_retained)
                task_final_spg_diversity_credit += float(effective_final_diversity_credit)
                task_new_final_spgs += int(new_final_spg)
                export_audit_rows.append({
                    "candidate_id": cid, "attempt_id": tid, "spg": parent_spg,
                    "parent_spg": parent_spg, "final_spg": final_spg,
                    "final_spg_symbol": detected_spgr_symbol,
                    "hypergraph_symmetry_preserved": bool(selected.get("hypergraph_symmetry_preserved", False)),
                    "hypergraph_site_fallback_used": site_fallback,
                    "final_symmetry_retained": symmetry_retained,
                    "parent_final_spg_match": parent_final_spg_match,
                    "parent_symmetry_max_error_A": parent_symmetry_max_error_A,
                    "parent_symmetry_ops_checked": int(parent_symmetry_ops_checked),
                    "family_id": int(family_diag["family_id"]),
                    "canonical_family_id": int(family_diag["family_id"]),
                    "raw_family_id": int(raw_family_diag["family_id"]),
                    "canonical_family_is_new": bool(family_diag.get("family_is_new", False)),
                    "raw_family_is_new": bool(raw_family_diag.get("family_is_new", False)),
                    "canonical_spg": int(canonical_spg),
                    "canonical_spg_symbol": canonical_diag.get("canonical_spg_symbol"),
                    "canonicalization_success": bool(canonical_diag.get("canonicalization_success", False)),
                    "canonicalization_inherited": bool(canonical_diag.get("canonicalization_inherited", False)),
                    "canonicalization_source_candidate_id": canonical_diag.get("canonicalization_source_candidate_id"),
                    "canonical_chemistry_used": canonical_diag.get("canonical_chemistry_used"),
                    "canonical_chemistry_guard_passed": canonical_diag.get("canonical_chemistry_guard_passed"),
                    "canonical_chemistry_topology_preserved": canonical_diag.get("canonical_chemistry_topology_preserved"),
                    "canonical_aggressive_symmetry_used": bool(canonical_diag.get("canonical_aggressive_symmetry_used", False)),
                    "canonical_chemistry_rms_move_A": canonical_diag.get("canonical_chemistry_rms_move_A"),
                    "canonical_chemistry_max_move_A": canonical_diag.get("canonical_chemistry_max_move_A"),
                    "canonical_aggressive_symmetry_rms_A": canonical_diag.get("canonical_aggressive_symmetry_rms_A"),
                    "canonical_aggressive_symmetry_max_A": canonical_diag.get("canonical_aggressive_symmetry_max_A"),
                    "canonical_aggressive_symmetry_topology_preserved": canonical_diag.get("canonical_aggressive_symmetry_topology_preserved"),
                    "hypergraph_topology_hash": selected.get("hypergraph_topology_hash"),
                    "coordination_valid": bool(selected.get("coordination_valid", False)),
                    "reciprocal_topology_valid": bool(selected.get("reciprocal_topology_valid", False)),
                    "bond_window_valid": bool(selected.get("bond_window_valid", False)),
                    "angular_label_valid": bool(selected.get("angular_label_valid", False)),
                    "nonbonded_exclusion_valid": bool(selected.get("nonbonded_exclusion_valid", False)),
                    "minimum_o_o_A": selected.get("minimum_o_o_A"),
                    "learned_o_o_hard_min_A": selected.get("learned_o_o_hard_min_A"),
                    "minimum_unassigned_ti_o_A": selected.get("minimum_unassigned_ti_o_A"),
                    "learned_unassigned_ti_o_hard_min_A": selected.get("learned_unassigned_ti_o_hard_min_A"),
                    "minimum_ti_ti_A": selected.get("minimum_ti_ti_A"),
                    "learned_ti_ti_hard_min_A": selected.get("learned_ti_ti_hard_min_A"),
                    "local_radial_mae_A": selected.get("local_radial_mae_A"),
                    "local_angular_site_max_deg": selected.get("local_angular_site_max_deg"),
                    "strict_valid": bool(selected.get("strict_valid", False)),
                    "export_valid": bool(selected.get("export_valid", False)),
                    "cif": str(cif),
                    "canonical_cif": str(canonical_cif),
                })
                accepted_rows.append({
                    "candidate_id": cid, "attempt_id": tid, "spg": parent_spg,
                    "parent_spg": parent_spg, "final_spg": final_spg,
                    "spgr": detected_spgr, "spgr_symbol": detected_spgr_symbol,
                    "hypergraph_symmetry_preserved": bool(selected.get("hypergraph_symmetry_preserved", False)),
                    "hypergraph_site_fallback_used": site_fallback,
                    "final_symmetry_retained": symmetry_retained,
                    "parent_final_spg_match": parent_final_spg_match,
                    "parent_symmetry_max_error_A": parent_symmetry_max_error_A,
                    "parent_symmetry_ops_checked": int(parent_symmetry_ops_checked),
                    "final_spg_count_before": prior_final_count,
                    "final_spg_diversity_credit": final_diversity_credit,
                    "effective_final_spg_diversity_credit": effective_final_diversity_credit,
                    **family_diag,
                    "canonical_family_id": int(family_diag["family_id"]),
                    "raw_family_id": int(raw_family_diag["family_id"]),
                    "raw_family_is_new": bool(raw_family_diag.get("family_is_new", False)),
                    "canonical_spg": int(canonical_spg),
                    "canonical_spg_symbol": canonical_diag.get("canonical_spg_symbol"),
                    "canonicalization_success": bool(canonical_diag.get("canonicalization_success", False)),
                    "canonicalization_reason": canonical_diag.get("canonicalization_reason"),
                    "canonicalization_inherited": bool(canonical_diag.get("canonicalization_inherited", False)),
                    "canonicalization_source_candidate_id": canonical_diag.get("canonicalization_source_candidate_id"),
                    "canonical_chemistry_used": canonical_diag.get("canonical_chemistry_used"),
                    "canonical_chemistry_guard_passed": canonical_diag.get("canonical_chemistry_guard_passed"),
                    "canonical_chemistry_topology_preserved": canonical_diag.get("canonical_chemistry_topology_preserved"),
                    "canonical_aggressive_symmetry_used": bool(canonical_diag.get("canonical_aggressive_symmetry_used", False)),
                    "canonical_chemistry_rms_move_A": canonical_diag.get("canonical_chemistry_rms_move_A"),
                    "canonical_chemistry_max_move_A": canonical_diag.get("canonical_chemistry_max_move_A"),
                    "canonical_cell_strain_fro": canonical_diag.get("canonical_cell_strain_fro"),
                    "canonical_aggressive_symmetry_rms_A": canonical_diag.get("canonical_aggressive_symmetry_rms_A"),
                    "canonical_aggressive_symmetry_max_A": canonical_diag.get("canonical_aggressive_symmetry_max_A"),
                    "canonical_aggressive_symmetry_topology_preserved": canonical_diag.get("canonical_aggressive_symmetry_topology_preserved"),
                    "hypergraph_topology_hash": selected.get("hypergraph_topology_hash"),
                    "hypergraph_topology_min_distance": selected.get("hypergraph_topology_min_distance"),
                    "hypergraph_topology_selection_rank": selected.get("hypergraph_topology_selection_rank"),
                    "entry_index": task.get("entry_index"),
                    "entrance_sampling_channel": task.get("entrance_sampling_channel"),
                    "entry_visit_number": task.get("entry_visit_number"),
                    "entry_first_pass": task.get("entry_first_pass"),
                    "entry_multiplicity_pattern": task.get("entry_multiplicity_pattern"),
                    "entry_orbit_count": task.get("entry_orbit_count"),
                    "entry_total_wyckoff_dof": task.get("entry_total_wyckoff_dof"),
                    "wp_token": task["wp_token"], "species_token": task["species_token"], "cif": str(cif),
                    "canonical_cif": str(canonical_cif),
                    "loss": selected["loss"],
                    "local_radial_mae_A": selected["local_radial_mae_A"],
                    "local_angular_site_max_deg": selected["local_angular_site_max_deg"],
                    "minimum_physical_distance_A": selected["minimum_physical_distance_A"],
                    "minimum_o_o_A": selected.get("minimum_o_o_A"),
                    "learned_o_o_hard_min_A": selected.get("learned_o_o_hard_min_A"),
                    "minimum_unassigned_ti_o_A": selected.get("minimum_unassigned_ti_o_A"),
                    "learned_unassigned_ti_o_hard_min_A": selected.get("learned_unassigned_ti_o_hard_min_A"),
                    "minimum_ti_ti_A": selected.get("minimum_ti_ti_A"),
                    "learned_ti_ti_hard_min_A": selected.get("learned_ti_ti_hard_min_A"),
                    "nonbonded_exclusion_valid": selected.get("nonbonded_exclusion_valid"),
                    "fingerprint_alerts": dedup["fingerprint_alerts"],
                    "exact_match_checks": dedup["exact_match_checks"],
                })
                _append_jsonl_rows(attempts_jsonl, [{
                    "attempt_id": tid, "stage": "canonicalization_shadow",
                    "elapsed_min": (time.perf_counter() - start) / 60.0,
                    "ti_tokens_used": int(builder_pool.ti_tokens_used),
                    "candidate_id": int(cid), "parent_spg": parent_spg,
                    "raw_final_spg": final_spg, "canonical_spg": int(canonical_spg),
                    "raw_family_id": int(raw_family_diag["family_id"]),
                    "canonical_family_id": int(family_diag["family_id"]),
                    "canonical_family_is_new": bool(is_new_family),
                    "hypergraph_topology_hash": selected.get("hypergraph_topology_hash"),
                    **canonical_diag,
                }])
                _append_jsonl_rows(attempts_jsonl, [{
                    "attempt_id": tid, "stage": "accepted_exact_unique",
                    "elapsed_min": (time.perf_counter() - start) / 60.0,
                    "ti_tokens_used": int(builder_pool.ti_tokens_used),
                    "candidate_id": int(cid), "parent_spg": parent_spg,
                    "final_spg": final_spg, "final_spg_symbol": detected_spgr_symbol,
                    "hypergraph_symmetry_preserved": bool(selected.get("hypergraph_symmetry_preserved", False)),
                    "hypergraph_site_fallback_used": site_fallback,
                    "final_symmetry_retained": symmetry_retained,
                    "parent_final_spg_match": parent_final_spg_match,
                    "parent_symmetry_max_error_A": parent_symmetry_max_error_A,
                    "parent_symmetry_ops_checked": int(parent_symmetry_ops_checked),
                    "family_id": int(family_diag["family_id"]),
                    "canonical_family_id": int(family_diag["family_id"]),
                    "raw_family_id": int(raw_family_diag["family_id"]),
                    "family_is_new": bool(is_new_family),
                    "canonical_spg": int(canonical_spg),
                    "canonical_aggressive_symmetry_used": bool(canonical_diag.get("canonical_aggressive_symmetry_used", False)),
                    "hypergraph_topology_hash": selected.get("hypergraph_topology_hash"),
                    "final_spg_count_before": int(prior_final_count),
                    "final_spg_diversity_credit": float(final_diversity_credit),
                    "effective_final_spg_diversity_credit": float(effective_final_diversity_credit),
                }])
                spg_stats[int(task["spg"])]["exact_unique"] += 1
                last_unique_token = int(builder_pool.ti_tokens_used)
                if is_new_family:
                    last_family_token = int(builder_pool.ti_tokens_used)
                    print(
                        f"New family {len(family_tracker.representatives)}: exact_unique={len(accepted_rows)} "
                        f"parentSG={parent_spg} rawSG={final_spg} canonicalSG={canonical_spg} "
                        f"raw_families={len(raw_family_tracker.representatives)} "
                        f"Ti_tokens={builder_pool.ti_tokens_used} proposal={completed}",
                        flush=True,
                    )

            # Family classification is deliberately diagnostic. It only tilts future
            # stochastic allocation; it never rejects or deletes an exact-unique candidate.
            _update_family_sampler_stat(sstat, new_families=task_new_families,
                                        existing_variants=task_existing_family_variants)
            if task.get("entry_index") is not None:
                _update_family_sampler_stat(estat, new_families=task_new_families,
                                            existing_variants=task_existing_family_variants)
            _update_crystal_sampler_stat(
                sstat, symmetry_retained=task_symmetry_retained,
                symmetry_broken=task_symmetry_broken,
                final_spg_diversity_credit=task_final_spg_diversity_credit,
                new_final_spgs=task_new_final_spgs)
            if task.get("entry_index") is not None:
                _update_crystal_sampler_stat(
                    estat, symmetry_retained=task_symmetry_retained,
                    symmetry_broken=task_symmetry_broken,
                    final_spg_diversity_credit=task_final_spg_diversity_credit,
                    new_final_spgs=task_new_final_spgs)
            # attempts.jsonl is the live append-only diagnostic. Small accepted/duplicate
            # tables remain checkpointed without rewriting the full attempt history.
            pd.DataFrame(accepted_rows).to_csv(output/"accepted.csv",index=False)
            pd.DataFrame(accepted_rows).to_csv(output/"summary.csv",index=False)
            pd.DataFrame(duplicate_rows).to_csv(output/"duplicates.csv",index=False)
            pd.DataFrame(family_tracker.rows()).to_csv(output/"family_statistics.csv", index=False)
            pd.DataFrame(raw_family_tracker.rows()).to_csv(output/"raw_family_statistics.csv", index=False)
            pd.DataFrame([r for r in accepted_rows if bool(r.get("family_is_new", False))]).to_csv(
                output/"family_representatives.csv", index=False)
            pd.DataFrame(export_audit_rows).to_csv(output/"export_audit.csv", index=False)
            pd.DataFrame([{"final_spg": int(k), "exact_unique_count": int(v)}
                          for k, v in sorted(final_spg_counts.items())]).to_csv(
                              output/"final_space_group_statistics.csv", index=False)
            pd.DataFrame([{"canonical_spg": int(k), "exact_unique_count": int(v)}
                          for k, v in sorted(canonical_spg_counts.items())]).to_csv(
                              output/"canonical_space_group_statistics.csv", index=False)
            write_progress_snapshot(force=False)
            if completed % args.progress_every == 0:
                elapsed_min = (time.perf_counter() - start) / 60.0
                print(
                    f"Status: exact_unique={len(accepted_rows)} families={len(family_tracker.representatives)}; proposals={completed}; "
                    f"Ti_tokens={builder_pool.ti_tokens_used}/{args.ti_token_budget}; "
                    f"Ti={stage_counts['ti_accepted']}/{stage_counts['ti_tested']}; "
                    f"newTi={sum(x.get('new_ti_basins',0) for x in spg_stats.values())}; "
                    f"Otop={stage_counts['o_topology_accepted']}/{stage_counts['o_topology_tested']}; "
                    f"Oanalytic={stage_counts['o_analytic_accepted']}/{stage_counts['o_analytic_tested']}; "
                    f"strict={stage_counts['strict_accepted']}/{stage_counts['strict_tested']}; "
                    f"active={len(gpu_inflight)}; elapsed={elapsed_min:.1f} min",
                    flush=True,
                )
        if builder_pool.ti_tokens_used >= args.ti_token_budget:
            print("Ti-token budget reached: no new proposals; all in-flight GPU jobs were drained.", flush=True)
        elif time.perf_counter() >= deadline:
            print("Runtime deadline reached: no new proposals; all in-flight GPU jobs were drained.", flush=True)
    finally:
        proposal_pool.close(); builder_pool.close()
    write_progress_snapshot(force=True)
    _jsonl_to_csv(attempts_jsonl, output/"attempts.csv")
    pd.DataFrame(accepted_rows).to_csv(output/"accepted.csv",index=False)
    pd.DataFrame(accepted_rows).to_csv(output/"summary.csv",index=False)
    pd.DataFrame(duplicate_rows).to_csv(output/"duplicates.csv",index=False)
    pd.DataFrame(family_tracker.rows()).to_csv(output/"family_statistics.csv", index=False)
    pd.DataFrame(raw_family_tracker.rows()).to_csv(output/"raw_family_statistics.csv", index=False)
    pd.DataFrame([r for r in accepted_rows if bool(r.get("family_is_new", False))]).to_csv(
        output/"family_representatives.csv", index=False)
    pd.DataFrame([{"final_spg": int(k), "exact_unique_count": int(v)}
                  for k, v in sorted(final_spg_counts.items())]).to_csv(
                      output/"final_space_group_statistics.csv", index=False)
    pd.DataFrame([{"canonical_spg": int(k), "exact_unique_count": int(v)}
                  for k, v in sorted(canonical_spg_counts.items())]).to_csv(
                      output/"canonical_space_group_statistics.csv", index=False)
    pd.DataFrame(export_audit_rows).to_csv(output/"export_audit.csv", index=False)
    ti_tokens_used = int(builder_pool.ti_tokens_used)
    spg_rows=[]
    for spg in sorted(space_groups):
        row=dict(spg_stats[spg]); row["spg"]=spg
        row["exact_unique_rate"]=row.get("exact_unique",0)/max(row.get("attempts",0),1)
        row["new_ti_per_token"]=row.get("new_ti_basins",0)/max(row.get("ti_tokens",0),1)
        row["duplicate_fraction"]=row.get("duplicate_ti_basins",0)/max(row.get("new_ti_basins",0)+row.get("duplicate_ti_basins",0),1)
        row["stochastic_score_final"]=_sampling_score(spg_stats[spg], ti_tokens_used, len(space_groups))
        spg_rows.append(row)
    pd.DataFrame(spg_rows).to_csv(output/"space_group_statistics.csv",index=False)
    entrance_rows=[]
    for (spg, entry_index), stat in sorted(entrance_stats.items()):
        row=dict(stat); row["spg"]=int(spg); row["entry_index"]=int(entry_index)
        row["new_ti_per_token"]=row.get("new_ti_basins",0)/max(row.get("ti_tokens",0),1)
        row["duplicate_fraction"]=row.get("duplicate_ti_basins",0)/max(row.get("new_ti_basins",0)+row.get("duplicate_ti_basins",0),1)
        spg_entry_tokens = sum(int(x.get("ti_tokens", 0)) for (g, _e), x in entrance_stats.items() if int(g) == int(spg))
        row["stochastic_score_final"]=_sampling_score(stat, spg_entry_tokens, max(int(entry_counts.get(spg, 1)), 1))
        entrance_rows.append(row)
    pd.DataFrame(entrance_rows).to_csv(output/"entrance_sampling_statistics.csv",index=False)
    summary={"model":str(Path(args.chemistry_model).resolve()),"targets":resolved_targets,"resolved_counts":counts,
        "exact_unique_candidates":len(accepted_rows),
        "diagnostic_families":len(family_tracker.representatives),
        "canonical_families":len(family_tracker.representatives),
        "raw_families":len(raw_family_tracker.representatives),
        "distinct_raw_final_space_groups":len(final_spg_counts),
        "distinct_canonical_space_groups":len(canonical_spg_counts),
        "duplicates_rejected":len(duplicate_rows),
        "strict_candidates_emitted":int(stage_counts["strict_accepted"]),
        "ti_token_budget":int(args.ti_token_budget),"ti_tokens_consumed":ti_tokens_used,
        "exact_unique_per_ti_token":len(accepted_rows)/max(ti_tokens_used,1),
        "diagnostic_families_per_ti_token":len(family_tracker.representatives)/max(ti_tokens_used,1),
        "strict_valid_per_ti_token":stage_counts["strict_accepted"]/max(ti_tokens_used,1),
        "proposal_tasks_completed":completed,"runtime_seconds":time.perf_counter()-start,
        "runtime_limit_reached":time.perf_counter()>=deadline,
        "ti_token_budget_reached":ti_tokens_used>=int(args.ti_token_budget),
        "cpu_proposal_workers":proposal_workers,"gpu_workers":builder_pool.workers,"visible_gpus":visible,
        "stage_counts":dict(stage_counts),
        "semantics":"v78_efficient_global_unseen_wyckoff_breadth_explore_only_ti_early_escape_topology_diverse_safe_canonical_family_count",
        "ti_token_semantics":"one token per Ti-framework branch admitted to expensive full refinement",
        "parallelization":"dynamic CPU proposal workers feeding persistent one-process-per-GPU optimization workers",
        "loss_kernel":"fully_vectorized_forward_and_reciprocal_X_port_matching",
        "deduplication":"fingerprint_alert_then_strict_proper_rotation_exact_match",
        "live_attempt_log":str(attempts_jsonl),
        "final_attempt_csv":str(output / "attempts.csv"),
        "live_progress_csv":str(output / "progress.csv"),
        "export_audit_csv":str(output / "export_audit.csv"),
        "cell_scale_policy":"no preferred volume or artificial pressure; pre-O shared-O chemistry tension and post-O realized Ti-O/nonbonded chemistry determine symmetry-allowed cell DOFs",
        "learned_framework_envelope":{
            "policy":"diagnostic only in v78; zero optimization weight for all Ti branches",
            "branch_weights":[0.0,0.0,0.0,0.0],
            "diagnostics":"site-resolved ranked Ti-neighbour radial z plus shell-pair-sorted Ti-Ti-Ti angular z"},
        "framework_chemistry_memory":str(output / "framework_chemistry_memory.jsonl"),
        "framework_basin_memory":str(output / "ti_basin_memory.jsonl"),
        "stochastic_sampling":{
            "requested_space_group_count":int(len(requested_space_groups)),
            "compatible_space_group_count":int(len(space_groups)),
            "compatible_space_groups":str(output / "compatible_space_groups.csv"),
            "global_unseen_probability":0.95,
            "uniform_probability_floor":0.30,
            "space_group_temperature":0.85,
            "entrance_uniform_probability_floor":0.30,
            "entrance_temperature":0.80,
            "policy":"95% global unseen-entry channel while unseen compatible Wyckoff entrances remain, 5% adaptive exploit/revisit channel; SG choice inside unseen channel is balanced across groups with unseen entries; in-flight entries are reserved",
            "new_ti_basins":int(sum(x.get("new_ti_basins",0) for x in spg_stats.values())),
            "duplicate_ti_basins":int(sum(x.get("duplicate_ti_basins",0) for x in spg_stats.values())),
            "entrance_statistics":str(output / "entrance_sampling_statistics.csv"),
            "entrance_coverage":str(output / "entrance_coverage.csv")},
        "framework_memory_policy":"construction-context-aware exact Ti basins plus active pre-refine diversion; transferable chemistry kNN and adaptive O budget",
        "framework_refinement_policy":"three exploratory Ti tokens per promoted entrance, chosen from max-min-diverse scout survivors; first visits use 16-start x 8-step scout; learned framework prior and Ti4 omission descriptor do not control admission",
        "ti_early_escape_policy":"pre-token first-pass escape only when best generic near-triplet loss >4.9, incidence deficit >0.95, and min-good-triplets <0.5; 5% unlikely / 2% catastrophic sentinel rescue",
        "joint_relaxation":"direct Ti3->O hypergraph exact-cover seed followed by short Ti/cell/O chemistry polish with symmetry-orbit restraint when available",
        "cover_diversity_policy":"enumerate up to 48 symmetry / 64 site exact covers, merge identical periodic Ti3 hyperedge signatures, retain up to 4 by max-min Jaccard topology diversity with chemistry score as tie-break; topology never rejects",
        "canonicalization_policy":"raw strict candidate preserved; only one representative per raw family is shadow-standardized; chemistry projection requires RMS<=0.20 A, max<=0.35 A, strain_F<=0.15 and fixed topology; symprec=0.10 A symmetry projection requires same-basis cell change<=0.08, RMS<=0.15 A, max<=0.30 A and fixed topology; no StructureMatcher override",
        "canonical_family_policy":"quotient of raw-family partition: canonicalization may merge different raw families but never split one; invariant N_canonical<=N_raw is enforced at runtime",
        "canonical_pool":str(output / "canonical_pool"),
        "raw_family_statistics":str(output / "raw_family_statistics.csv"),
        "canonical_space_group_statistics":str(output / "canonical_space_group_statistics.csv"),
        "family_diagnostics":"raw families use pymatgen StructureMatcher primitive_cell=False; only raw-family representatives enter canonical clustering with primitive_cell=True, ltol=0.12 stol=0.25 angle_tol=5; all members inherit the representative canonical family; neither family label nor canonicalization rejects raw output",
        "dedup_label_safe":bool(deduplicator.label_safe)}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(
        f"Finished: exact_unique={len(accepted_rows)} canonical_families={len(family_tracker.representatives)} "
        f"raw_families={len(raw_family_tracker.representatives)}; "
        f"Ti_tokens={ti_tokens_used}/{args.ti_token_budget}; "
        f"exact/token={len(accepted_rows)/max(ti_tokens_used,1):.6g}; "
        f"family/token={len(family_tracker.representatives)/max(ti_tokens_used,1):.6g}; proposals={completed}; "
        f"Ti={stage_counts['ti_accepted']}/{stage_counts['ti_tested']}; "
        f"Otop={stage_counts['o_topology_accepted']}/{stage_counts['o_topology_tested']}; "
        f"Oanalytic={stage_counts['o_analytic_accepted']}/{stage_counts['o_analytic_tested']}; "
        f"Oattach={stage_counts['o_attachment_accepted']}/{stage_counts['o_attachment_tested']}; "
        f"runtime={summary['runtime_seconds'] / 60.0:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()

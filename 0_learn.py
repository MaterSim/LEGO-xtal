#!/usr/bin/env python3
"""Learn Juliette building-block + Xn construction templates.

This is a clean replacement for the previous chemistry builders.  It writes one
system-independent schema:

    construction species = physical building block + external Xn template

The first supported sources are:
  * elemental CIF prototypes (graphite/diamond style);
  * binary ASE databases (TiO2 style).

X vertices are expected-neighbour positions relative to the block reference
centre.  X0 is represented by an empty external template and is not special-
cased as a molecular role.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

EPS = 1.0e-12


@dataclass
class LocalEnvironment:
    center_element: str
    neighbor_element: str
    vectors: np.ndarray
    source: str

    @property
    def distances(self) -> np.ndarray:
        return np.linalg.norm(self.vectors, axis=1)

    @property
    def angles(self) -> np.ndarray:
        return _angles(self.vectors)


def _angles(vectors: np.ndarray) -> np.ndarray:
    """Return sorted pair angles in degrees for an ``(n, 3)`` vector set."""
    arr = np.asarray(vectors, dtype=float).reshape(-1, 3)
    out: list[float] = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            ni = float(np.linalg.norm(arr[i]))
            nj = float(np.linalg.norm(arr[j]))
            if ni <= EPS or nj <= EPS:
                raise ValueError("Template vectors must have nonzero length")
            cosine = float(np.dot(arr[i], arr[j]) / (ni * nj))
            out.append(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))
    return np.sort(np.asarray(out, dtype=float))


def _require_runtime_packages():
    try:
        from ase.db import connect  # noqa: F401
        from ase.io import read  # noqa: F401
        from pymatgen.io.ase import AseAtomsAdaptor  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "This script requires ASE and pymatgen in the Juliette runtime environment"
        ) from exc


def _periodic_neighbors(structure, site_id: int, species: str, radius: float) -> list[dict]:
    center = structure[site_id]
    out = []
    for nn in structure.get_neighbors(center, radius, include_index=True, include_image=True):
        if str(nn.specie.symbol) != str(species):
            continue
        out.append(
            {
                "distance": float(nn.nn_distance),
                "vector": np.asarray(nn.coords - center.coords, dtype=float),
                "index": int(nn.index),
                "image": tuple(int(x) for x in nn.image),
            }
        )
    out.sort(key=lambda x: (x["distance"], x["index"], x["image"]))
    return out


def _first_shell_cutoff(per_site_distances: list[list[float]], r_search: float,
                        grid_size: int = 1600, smooth_sigma: float = 7.0,
                        prominence_fraction: float = 0.025,
                        valley_fraction: float = 0.06,
                        min_valley_width: float = 0.08) -> dict:
    """Find the first persistent low-density valley after the first RDF peak."""
    edges = np.linspace(0.0, float(r_search), int(grid_size) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(grid_size, dtype=float)
    for distances in per_site_distances:
        arr = np.asarray(distances, dtype=float)
        arr = arr[(arr > EPS) & (arr <= r_search)]
        hist += np.histogram(arr, bins=edges)[0]
    hist /= max(len(per_site_distances), 1)
    smooth = gaussian_filter1d(hist, sigma=float(smooth_sigma), mode="nearest")
    maximum = float(np.max(smooth))
    peaks, props = find_peaks(smooth, prominence=max(maximum * prominence_fraction, EPS))
    if len(peaks) == 0:
        peaks = np.asarray([int(np.argmax(smooth))])
    first_peak = int(np.min(peaks))
    threshold = max(float(smooth[first_peak]) * float(valley_fraction), EPS)
    dr = float(edges[1] - edges[0])
    minimum_bins = max(1, int(math.ceil(min_valley_width / dr)))
    selected = None
    run_start = None
    for idx in range(first_peak + 1, len(smooth)):
        if smooth[idx] <= threshold and run_start is None:
            run_start = idx
        if (smooth[idx] > threshold or idx == len(smooth) - 1) and run_start is not None:
            run_end = idx - 1 if smooth[idx] > threshold else idx
            if run_end - run_start + 1 >= minimum_bins:
                selected = run_start
                break
            run_start = None
    if selected is None:
        # Robust fallback: minimum between first and second significant peaks.
        later = [int(x) for x in peaks if int(x) > first_peak]
        right = later[0] if later else len(smooth) - 1
        if right <= first_peak + 1:
            raise RuntimeError("Could not separate the first coordination shell")
        selected = first_peak + int(np.argmin(smooth[first_peak:right + 1]))
    return {
        "cutoff_A": float(edges[selected]),
        "first_peak_A": float(centers[first_peak]),
        "selection": "first_persistent_valley_after_first_peak",
        "grid": {
            "r_A": centers.tolist(),
            "density_raw": hist.tolist(),
            "density_smooth": smooth.tolist(),
        },
    }


def _rotation_invariant_signature(env: LocalEnvironment) -> np.ndarray:
    distances = np.sort(env.distances)
    angles = env.angles
    dscale = max(float(np.mean(distances)), 1.0e-6)
    return np.concatenate([distances / dscale, angles / 180.0])


def _medoid_environment(environments: list[LocalEnvironment]) -> LocalEnvironment:
    if not environments:
        raise ValueError("No environments supplied")
    signatures = np.vstack([_rotation_invariant_signature(x) for x in environments])
    center = np.median(signatures, axis=0)
    scale = np.median(np.abs(signatures - center), axis=0) + 1.0e-6
    scores = np.mean(np.abs((signatures - center) / scale), axis=1)
    return environments[int(np.argmin(scores))]


def _template_statistics(environments: list[LocalEnvironment], radial_sigma_floor: float,
                         angular_sigma_floor: float, n_width: float,
                         max_radial_reconstruction_error: float,
                         max_angular_reconstruction_error: float) -> dict:
    """Represent Xn by one real medoid environment, never an averaged vector set.

    Radial/angular distributions are learned from all accepted environments, but
    the canonical Cartesian realization is copied byte-for-value from a physical
    local environment. This avoids vertex cancellation in symmetric X3/X4/X6 sets.
    """
    medoid = _medoid_environment(environments)
    canonical = np.asarray(medoid.vectors, dtype=float).copy()
    radial = np.concatenate([x.distances for x in environments])
    angular_rows = np.vstack([x.angles for x in environments])
    radial_mean = float(np.mean(radial))
    radial_sigma = max(float(np.std(radial)), float(radial_sigma_floor))
    angular_mean = np.median(angular_rows, axis=0)
    angular_mad = np.median(np.abs(angular_rows - angular_mean[None, :]), axis=0)
    angular_sigma = np.maximum(1.4826 * angular_mad, float(angular_sigma_floor))

    q01 = float(np.quantile(radial, 0.01))
    q99 = float(np.quantile(radial, 0.99))
    radial_min = max(0.0, min(q01, radial_mean - float(n_width) * radial_sigma))
    radial_max = max(q99, radial_mean + float(n_width) * radial_sigma)

    canonical_radii = np.linalg.norm(canonical, axis=1)
    canonical_angles = np.sort(_angles(canonical))
    target_angles = np.sort(np.asarray(angular_mean, dtype=float))
    radial_reconstruction_mae = float(np.mean(np.abs(canonical_radii - radial_mean)))
    angular_reconstruction_mae = float(np.mean(np.abs(canonical_angles - target_angles)))
    if radial_reconstruction_mae > float(max_radial_reconstruction_error):
        raise RuntimeError(
            f"Canonical medoid radial reconstruction MAE {radial_reconstruction_mae:.6f} A "
            f"exceeds {max_radial_reconstruction_error:.6f} A; source={medoid.source}"
        )
    if angular_reconstruction_mae > float(max_angular_reconstruction_error):
        raise RuntimeError(
            f"Canonical medoid angular reconstruction MAE {angular_reconstruction_mae:.6f} deg "
            f"exceeds {max_angular_reconstruction_error:.6f} deg; source={medoid.source}"
        )

    return {
        "coordination_number": int(len(canonical)),
        "canonical_vectors_A": canonical.tolist(),
        "radial_mean_A": radial_mean,
        "radial_sigma_A": radial_sigma,
        "radial_min_A": radial_min,
        "radial_max_A": radial_max,
        "angular_mean_deg": angular_mean.tolist(),
        "angular_sigma_deg": angular_sigma.tolist(),
        "template_source": medoid.source,
        "environment_count": int(len(environments)),
        "canonical_self_consistency": {
            "radius_mean_A": float(np.mean(canonical_radii)),
            "radius_spread_A": float(np.std(canonical_radii)),
            "radial_reconstruction_mae_A": radial_reconstruction_mae,
            "angular_reconstruction_mae_deg": angular_reconstruction_mae,
        },
    }


def _atomic_block(element: str) -> dict:
    return {
        "kind": "atomic",
        "atoms": [{"element": str(element), "position_A": [0.0, 0.0, 0.0]}],
        "reference_center_A": [0.0, 0.0, 0.0],
        "internal_degrees_of_freedom": [],
    }


def _species_record(label: str, element: str, partner: str, template: dict,
                    source: str) -> dict:
    return {
        "label": str(label),
        "final_formula": str(element),
        "building_block": _atomic_block(element),
        "external_template": {
            "kind": f"X{template['coordination_number']}",
            "coordination_number": int(template["coordination_number"]),
            "canonical_vectors_A": template["canonical_vectors_A"],
            "allowed_partner_labels": [],
            "allowed_partner_elements": [str(partner)],
            "radial_mean_A": float(template["radial_mean_A"]),
            "radial_sigma_A": float(template["radial_sigma_A"]),
            "radial_sampling_min_A": float(template["radial_min_A"]),
            "radial_sampling_max_A": float(template["radial_max_A"]),
            "first_shell_cutoff_A": float(template["first_shell_cutoff_A"]),
            "angular_mean_deg": template["angular_mean_deg"],
            "angular_sigma_deg": template["angular_sigma_deg"],
            "deformation": {
                "allow_rotation": True,
                "allow_uniform_radial_scale": True,
                "allow_vector_distortion": True,
            },
        },
        "source": str(source),
        "diagnostics": {
            "environment_count": int(template["environment_count"]),
            "template_source": template["template_source"],
            "canonical_self_consistency": template["canonical_self_consistency"],
        },
    }


def _load_prototype(path: str, label: str, final_element: str, cn: int,
                    r_search: float) -> tuple[list[LocalEnvironment], dict]:
    from ase.io import read
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = read(path)
    structure = AseAtomsAdaptor.get_structure(atoms)
    symbols = [str(site.specie.symbol) for site in structure]
    if len(set(symbols)) != 1:
        raise ValueError(f"Prototype {path} must be elemental")
    if symbols[0] != final_element:
        raise ValueError(
            f"Prototype {path} contains {symbols[0]}, expected final element {final_element}"
        )
    out = []
    bonded_outer = []
    next_shell_inner = []
    for site_id in range(len(structure)):
        neighbors = _periodic_neighbors(structure, site_id, final_element, r_search)
        if len(neighbors) < cn:
            raise RuntimeError(f"{path}: site {site_id} has fewer than CN={cn} neighbours")
        local = neighbors[:cn]
        bonded_outer.append(float(local[-1]["distance"]))
        if len(neighbors) > cn:
            next_shell_inner.append(float(neighbors[cn]["distance"]))
        out.append(
            LocalEnvironment(
                center_element=final_element,
                neighbor_element=final_element,
                vectors=np.vstack([x["vector"] for x in local]),
                source=f"prototype:{Path(path).resolve()}:{label}:site{site_id}",
            )
        )
    bonded_q99 = float(np.quantile(bonded_outer, 0.99))
    if next_shell_inner:
        next_q01 = float(np.quantile(next_shell_inner, 0.01))
        if next_q01 <= bonded_q99 + 1.0e-6:
            raise RuntimeError(
                f"{path}: bonded and next-neighbour shells overlap: "
                f"bonded_q99={bonded_q99:.6f}, next_q01={next_q01:.6f}"
            )
        cutoff = 0.5 * (bonded_q99 + next_q01)
        selection = "midpoint_between_CN_and_CN_plus_1_shells"
    else:
        next_q01 = None
        cutoff = bonded_q99 + 0.25
        selection = "bonded_outer_plus_0.25A_fallback"
    shell = {
        "cutoff_A": float(cutoff),
        "bonded_outer_q99_A": bonded_q99,
        "next_shell_inner_q01_A": next_q01,
        "selection": selection,
    }
    return out, shell


def _environment_geometry_error(env: LocalEnvironment, reference: LocalEnvironment) -> dict:
    """Rotation-invariant radial/angular mismatch to a reference environment."""
    radial = np.sort(np.asarray(env.distances, dtype=float))
    ref_radial = np.sort(np.asarray(reference.distances, dtype=float))
    angular = np.sort(np.asarray(env.angles, dtype=float))
    ref_angular = np.sort(np.asarray(reference.angles, dtype=float))
    if radial.shape != ref_radial.shape or angular.shape != ref_angular.shape:
        return {
            "radial_mae_A": float("inf"),
            "radial_max_A": float("inf"),
            "angular_mae_deg": float("inf"),
            "angular_max_deg": float("inf"),
        }
    return {
        "radial_mae_A": float(np.mean(np.abs(radial - ref_radial))),
        "radial_max_A": float(np.max(np.abs(radial - ref_radial))),
        "angular_mae_deg": float(np.mean(np.abs(angular - ref_angular))),
        "angular_max_deg": float(np.max(np.abs(angular - ref_angular))),
    }



def _learn_same_element_frameworks(structures_by_id: dict[int, object], accepted_rows: list[int],
                                   elements: list[str], r_search: float,
                                   radial_sigma_floor: float = 0.06,
                                   angular_sigma_floor: float = 5.0) -> dict[str, dict]:
    """Learn same-element parent-framework fingerprints.

    Neighbours are ranked by distance.  Radial ranks are partitioned into
    robust shells from gaps in their learned median distances.  Ti--Ti--Ti
    angles are then learned separately for every radial-shell pair, rather than
    pooled into one globally sorted angular vector.  This preserves the
    relation between an angle and the two Ti--Ti shells that define it.
    """
    result: dict[str, dict] = {}
    for element in elements:
        available_counts: list[int] = []
        site_vectors: list[np.ndarray] = []

        for row_id in accepted_rows:
            structure = structures_by_id[int(row_id)]
            symbols = [str(site.specie.symbol) for site in structure]
            for site_id, symbol in enumerate(symbols):
                if symbol != element:
                    continue
                neigh = [
                    x for x in _periodic_neighbors(structure, site_id, element, r_search)
                    if float(x["distance"]) > 1.0e-6
                ]
                available_counts.append(len(neigh))

        if not available_counts:
            continue
        k = max(1, min(6, int(np.median(available_counts))))

        for row_id in accepted_rows:
            structure = structures_by_id[int(row_id)]
            symbols = [str(site.specie.symbol) for site in structure]
            for site_id, symbol in enumerate(symbols):
                if symbol != element:
                    continue
                neigh = [
                    x for x in _periodic_neighbors(structure, site_id, element, r_search)
                    if float(x["distance"]) > 1.0e-6
                ]
                if len(neigh) < k:
                    continue
                site_vectors.append(np.vstack([x["vector"] for x in neigh[:k]]))

        if not site_vectors:
            continue

        radial = np.vstack([
            np.linalg.norm(vectors, axis=1) for vectors in site_vectors
        ])
        rmed = np.median(radial, axis=0)
        rmad = np.median(np.abs(radial - rmed[None]), axis=0)
        rsig = np.maximum(1.4826 * rmad, float(radial_sigma_floor))

        # Split only on a clearly resolved radial gap.  The adaptive threshold
        # avoids turning the ordinary spread of one shell into artificial
        # sub-shells, while retaining the ~3.0/~3.55 A Ti-shell separation.
        gap_threshold = max(0.18, 2.5 * float(np.median(rsig)))
        shell_ids = np.zeros(k, dtype=int)
        shell_id = 0
        for rank in range(1, k):
            if float(rmed[rank] - rmed[rank - 1]) > gap_threshold:
                shell_id += 1
            shell_ids[rank] = shell_id
        shell_count = int(shell_id + 1)
        shell_centres = [
            float(np.mean(rmed[shell_ids == sid])) for sid in range(shell_count)
        ]

        pair_groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for left in range(k):
            for right in range(left + 1, k):
                key = tuple(sorted((int(shell_ids[left]), int(shell_ids[right]))))
                pair_groups[key].append((left, right))

        angular_groups = []
        for shell_pair in sorted(pair_groups):
            rank_pairs = pair_groups[shell_pair]
            observations = []
            for vectors in site_vectors:
                norms = np.linalg.norm(vectors, axis=1)
                unit = vectors / np.maximum(norms[:, None], EPS)
                angles = []
                for left, right in rank_pairs:
                    cosine = float(np.clip(np.dot(unit[left], unit[right]), -1.0, 1.0))
                    angles.append(float(np.degrees(np.arccos(cosine))))
                observations.append(np.sort(np.asarray(angles, dtype=float)))
            observed = np.vstack(observations)
            amed = np.median(observed, axis=0)
            amad = np.median(np.abs(observed - amed[None]), axis=0)
            asig = np.maximum(1.4826 * amad, float(angular_sigma_floor))
            angular_groups.append({
                "shell_pair": [int(shell_pair[0]), int(shell_pair[1])],
                "neighbor_rank_pairs": [[int(a), int(b)] for a, b in rank_pairs],
                "angular_mean_deg": amed.tolist(),
                "angular_sigma_deg": asig.tolist(),
                "angle_count": int(len(rank_pairs)),
            })

        nearest = radial[:, 0]
        farthest = radial[:, -1]
        result[element] = {
            "element": element,
            "neighbor_count": int(k),
            "radial_mean_A": rmed.tolist(),
            "radial_sigma_A": rsig.tolist(),
            "radial_shell_ids": shell_ids.tolist(),
            "radial_shell_centers_A": shell_centres,
            "radial_shell_gap_threshold_A": float(gap_threshold),
            "radial_lower_bound_A": float(max(0.5, np.quantile(nearest, 0.01) - 0.10)),
            "connectivity_upper_A": float(np.quantile(farthest, 0.99) + 0.20),
            "angular_shell_pair_groups": angular_groups,
            "site_count": int(len(site_vectors)),
            "score_q90_reference": 1.0,
            "score_q90_max": 3.5,
            "source": "accepted_binary_rows_same_element_periodic_neighbours_shell_pair_resolved",
        }
    return result

def _load_binary_database(database: str, r_search: float,
                          min_structure_pass_fraction: float,
                          max_site_radial_mae: float,
                          max_site_radial_max: float,
                          max_site_angular_mae: float,
                          max_site_angular_max: float,
                          strict_angular_min_cn: int,
                          low_cn_min_angle: float,
                          low_cn_max_angle: float,
                          max_exclusion_shell_margin: float) -> tuple[dict, dict]:
    from ase.db import connect
    from pymatgen.io.ase import AseAtomsAdaptor

    rows = list(connect(database, serial=True).select())
    structures = []
    species_set = set()
    for row in rows:
        structure = AseAtomsAdaptor.get_structure(row.toatoms())
        species = sorted({str(site.specie.symbol) for site in structure})
        if len(species) != 2:
            raise ValueError(f"ASE DB row {row.id} is not binary: {species}")
        species_set.update(species)
        structures.append((int(row.id), structure))
    if len(species_set) != 2:
        raise ValueError(f"Database must contain one binary system; observed={sorted(species_set)}")
    structures_by_id = {int(row_id): structure for row_id, structure in structures}
    a, b = sorted(species_set)
    broad = {(a, b): [], (b, a): []}
    for _, structure in structures:
        symbols = [str(site.specie.symbol) for site in structure]
        for site_id, center in enumerate(symbols):
            partner = b if center == a else a
            broad[(center, partner)].append(
                [x["distance"] for x in _periodic_neighbors(structure, site_id, partner, r_search)]
            )
    shell = {
        pair: _first_shell_cutoff(values, r_search)
        for pair, values in broad.items()
    }
    cn_counts = {pair: Counter() for pair in broad}
    structure_cn = []
    for row_id, structure in structures:
        symbols = [str(site.specie.symbol) for site in structure]
        local = []
        for site_id, center in enumerate(symbols):
            partner = b if center == a else a
            cutoff = shell[(center, partner)]["cutoff_A"]
            cn = sum(
                x["distance"] <= cutoff + 1.0e-8
                for x in _periodic_neighbors(structure, site_id, partner, r_search)
            )
            cn_counts[(center, partner)][int(cn)] += 1
            local.append((center, partner, site_id, int(cn)))
        structure_cn.append((row_id, structure, local))
    target_cn = {
        pair: int(max(counter.items(), key=lambda x: (x[1], -x[0]))[0])
        for pair, counter in cn_counts.items()
    }
    cn_accepted = []
    rejected = []
    provisional_by_row = {}
    provisional_environments = {(a, b): [], (b, a): []}
    environments = {(a, b): [], (b, a): []}
    exclusion_samples = {
        (a, b): {"bonded_outer": [], "next_shell_inner": []},
        (b, a): {"bonded_outer": [], "next_shell_inner": []},
    }
    for row_id, structure, local in structure_cn:
        directional = {}
        reasons = []
        for pair in ((a, b), (b, a)):
            pair_pass = [
                cn == target_cn[pair]
                for center, partner, _, cn in local
                if (center, partner) == pair
            ]
            fraction = float(np.mean(pair_pass)) if pair_pass else 0.0
            directional[pair] = fraction
            if fraction + EPS < min_structure_pass_fraction:
                reasons.append(f"{pair[0]}->{pair[1]}_cn")
        if reasons:
            rejected.append({
                "row_id": row_id,
                "directional_pass_fractions": {
                    f"{x}->{y}": directional[(x, y)] for x, y in directional
                },
                "rejection_reason": ";".join(reasons),
            })
            continue
        cn_accepted.append(row_id)
        row_envs = []
        for center, partner, site_id, _ in local:
            cutoff = shell[(center, partner)]["cutoff_A"]
            all_neighbors = _periodic_neighbors(structure, site_id, partner, r_search)
            neighbors = [x for x in all_neighbors if x["distance"] <= cutoff + 1.0e-8]
            expected = target_cn[(center, partner)]
            if len(neighbors) != expected:
                continue
            env = LocalEnvironment(
                center_element=center,
                neighbor_element=partner,
                vectors=np.vstack([x["vector"] for x in neighbors]),
                source=f"ase_db:{Path(database).resolve()}:row{row_id}:site{site_id}",
            )
            row_envs.append((center, partner, site_id, env, all_neighbors, expected))
            provisional_environments[(center, partner)].append(env)
        provisional_by_row[row_id] = row_envs

    if not cn_accepted:
        raise RuntimeError("No database rows pass exact directional coordination filtering")

    references = {
        pair: _medoid_environment(envs)
        for pair, envs in provisional_environments.items()
        if envs
    }
    accepted = []
    geometry_diagnostics = {}
    for row_id in cn_accepted:
        row_envs = provisional_by_row[row_id]
        row_errors = []
        reasons = []
        for center, partner, site_id, env, all_neighbors, expected in row_envs:
            pair = (center, partner)
            err = _environment_geometry_error(env, references[pair])
            row_errors.append({
                "site_id": int(site_id),
                "direction": f"{center}->{partner}",
                **err,
            })
            if err["radial_mae_A"] > max_site_radial_mae:
                reasons.append(f"{center}->{partner}_radial_mae")
            if err["radial_max_A"] > max_site_radial_max:
                reasons.append(f"{center}->{partner}_radial_max")
            if expected >= strict_angular_min_cn:
                if err["angular_mae_deg"] > max_site_angular_mae:
                    reasons.append(f"{center}->{partner}_angular_mae")
                if err["angular_max_deg"] > max_site_angular_max:
                    reasons.append(f"{center}->{partner}_angular_max")
            else:
                angles = np.asarray(env.angles, dtype=float)
                if (angles.size and
                        (float(np.min(angles)) < low_cn_min_angle or
                         float(np.max(angles)) > low_cn_max_angle)):
                    reasons.append(f"{center}->{partner}_angular_sanity")
        geometry_diagnostics[row_id] = row_errors
        if reasons:
            rejected.append({
                "row_id": row_id,
                "rejection_reason": ";".join(sorted(set(reasons))),
                "geometry_maxima": {
                    "radial_mae_A": max((x["radial_mae_A"] for x in row_errors), default=float("inf")),
                    "radial_max_A": max((x["radial_max_A"] for x in row_errors), default=float("inf")),
                    "angular_mae_deg": max((x["angular_mae_deg"] for x in row_errors), default=float("inf")),
                    "angular_max_deg": max((x["angular_max_deg"] for x in row_errors), default=float("inf")),
                },
            })
            continue
        accepted.append(row_id)
        for center, partner, site_id, env, all_neighbors, expected in row_envs:
            exclusion_samples[(center, partner)]["bonded_outer"].append(
                float(all_neighbors[expected - 1]["distance"])
            )
            if len(all_neighbors) > expected:
                exclusion_samples[(center, partner)]["next_shell_inner"].append(
                    float(all_neighbors[expected]["distance"])
                )
            environments[(center, partner)].append(env)

    if not accepted:
        raise RuntimeError(
            "No database rows pass the combined coordination and local-geometry filters"
        )
    exclusion_shell = {}
    for pair, values in exclusion_samples.items():
        bonded = values["bonded_outer"]
        next_shell = values["next_shell_inner"]
        if not bonded:
            raise RuntimeError(f"No accepted first-shell samples for {pair[0]}->{pair[1]}")
        bonded_q99 = float(np.quantile(bonded, 0.99))
        next_q01 = float(np.quantile(next_shell, 0.01)) if next_shell else None
        conservative_cap = bonded_q99 + max_exclusion_shell_margin
        if next_q01 is not None and next_q01 > bonded_q99 + 1.0e-6:
            raw_cutoff = 0.5 * (bonded_q99 + next_q01)
            cutoff = min(raw_cutoff, conservative_cap)
            selection = (
                "midpoint_between_CN_and_CN_plus_1_shells_on_accepted_rows"
                if raw_cutoff <= conservative_cap + 1.0e-12
                else "midpoint_capped_by_bonded_outer_plus_margin"
            )
        else:
            raw_cutoff = max(float(shell[pair]["cutoff_A"]), bonded_q99 + 0.05)
            cutoff = min(raw_cutoff, conservative_cap)
            selection = (
                "initial_valley_or_bonded_outer_plus_margin_fallback"
                if raw_cutoff <= conservative_cap + 1.0e-12
                else "fallback_capped_by_bonded_outer_plus_margin"
            )
        exclusion_shell[pair] = {
            "cutoff_A": float(cutoff),
            "bonded_outer_q99_A": bonded_q99,
            "next_shell_inner_q01_A": next_q01,
            "selection": selection,
            "max_margin_from_bonded_outer_A": float(max_exclusion_shell_margin),
        }

    max_parent_cn = max(target_cn.values())
    framework_parent_elements = sorted({
        center for (center, _partner), cn in target_cn.items()
        if int(cn) == int(max_parent_cn)
    })
    framework_learning = _learn_same_element_frameworks(
        structures_by_id, accepted, framework_parent_elements, r_search
    )
    diagnostics = {
        "database": str(Path(database).resolve()),
        "framework_learning": framework_learning,
        "input_rows": len(rows),
        "coordination_accepted_rows": cn_accepted,
        "coordination_accepted_row_count": int(len(cn_accepted)),
        "accepted_rows": accepted,
        "accepted_row_count": int(len(accepted)),
        "rejected_rows": rejected,
        "rejected_row_count": int(len(rejected)),
        "shell_learning": {f"{x}->{y}": shell[(x, y)] for x, y in shell},
        "exclusion_shell_learning": {
            f"{x}->{y}": exclusion_shell[(x, y)] for x, y in exclusion_shell
        },
        "target_cn": {f"{x}->{y}": target_cn[(x, y)] for x, y in target_cn},
        "cn_distributions": {
            f"{x}->{y}": dict(sorted(cn_counts[(x, y)].items())) for x, y in cn_counts
        },
        "geometry_filter": {
            "reference_sources": {
                f"{x}->{y}": references[(x, y)].source for x, y in references
            },
            "max_site_radial_mae_A": float(max_site_radial_mae),
            "max_site_radial_max_A": float(max_site_radial_max),
            "max_site_angular_mae_deg": float(max_site_angular_mae),
            "max_site_angular_max_deg": float(max_site_angular_max),
            "strict_angular_min_cn": int(strict_angular_min_cn),
            "low_cn_min_angle_deg": float(low_cn_min_angle),
            "low_cn_max_angle_deg": float(low_cn_max_angle),
            "per_row_site_errors": {str(k): v for k, v in geometry_diagnostics.items()},
        },
    }
    return environments, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn general building-block + Xn templates")
    parser.add_argument(
        "--prototype", nargs=4, action="append", default=[],
        metavar=("CIF", "LABEL", "FINAL_ELEMENT", "CN"),
        help="Repeat for prototype-defined species, e.g. graphite.cif C_sp2 C 3",
    )
    parser.add_argument("--ase-database", action="append", default=[])
    parser.add_argument("--r-search", type=float, default=5.0)
    parser.add_argument("--radial-sigma-floor", type=float, default=0.04)
    parser.add_argument("--angular-sigma-floor", type=float, default=4.0)
    parser.add_argument("--sampling-width-sigma", type=float, default=2.5)
    parser.add_argument("--max-radial-reconstruction-error", type=float, default=0.35)
    parser.add_argument("--max-angular-reconstruction-error", type=float, default=35.0)
    parser.add_argument("--min-structure-pass-fraction", type=float, default=1.0)
    parser.add_argument("--max-site-radial-mae", type=float, default=0.12)
    parser.add_argument("--max-site-radial-max", type=float, default=0.25)
    parser.add_argument("--max-site-angular-mae", type=float, default=12.0)
    parser.add_argument("--max-site-angular-max", type=float, default=25.0)
    parser.add_argument("--strict-angular-min-cn", type=int, default=4)
    parser.add_argument("--low-cn-min-angle", type=float, default=45.0)
    parser.add_argument("--low-cn-max-angle", type=float, default=175.0)
    parser.add_argument("--max-exclusion-shell-margin", type=float, default=0.40)
    parser.add_argument("--output", default="data/xn_templates/chemistry_model.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_runtime_packages()
    if not args.prototype and not args.ase_database:
        raise ValueError("At least one --prototype or --ase-database source is required")
    if (args.r_search <= 0 or args.radial_sigma_floor <= 0 or
            args.angular_sigma_floor <= 0 or args.sampling_width_sigma <= 0):
        raise ValueError("Search radius, sigma floors, and sampling width must be positive")
    if args.max_radial_reconstruction_error <= 0 or args.max_angular_reconstruction_error <= 0:
        raise ValueError("Reconstruction-error limits must be positive")
    if not 0 <= args.min_structure_pass_fraction <= 1:
        raise ValueError("--min-structure-pass-fraction must lie in [0,1]")
    if min(args.max_site_radial_mae, args.max_site_radial_max,
           args.max_site_angular_mae, args.max_site_angular_max) <= 0:
        raise ValueError("All local-geometry filter tolerances must be positive")
    if args.strict_angular_min_cn < 2:
        raise ValueError("--strict-angular-min-cn must be at least 2")
    if not 0.0 < args.low_cn_min_angle < args.low_cn_max_angle < 180.0:
        raise ValueError("Low-CN angle sanity bounds must satisfy 0 < min < max < 180")
    if args.max_exclusion_shell_margin <= 0:
        raise ValueError("--max-exclusion-shell-margin must be positive")

    species_records = []
    source_diagnostics = []
    label_set = set()
    framework_models = {}

    for path, label, element, cn_text in args.prototype:
        if label in label_set:
            raise ValueError(f"Duplicate construction label {label}")
        label_set.add(label)
        envs, prototype_shell = _load_prototype(
            path, label, element, int(cn_text), args.r_search
        )
        stats = _template_statistics(
            envs, args.radial_sigma_floor, args.angular_sigma_floor,
            args.sampling_width_sigma, args.max_radial_reconstruction_error,
            args.max_angular_reconstruction_error,
        )
        stats["first_shell_cutoff_A"] = float(prototype_shell["cutoff_A"])
        species_records.append(_species_record(label, element, element, stats, f"prototype:{Path(path).resolve()}"))
        source_diagnostics.append({
            "type": "prototype",
            "path": str(Path(path).resolve()),
            "label": label,
            "shell_learning": prototype_shell,
        })

    for database in args.ase_database:
        environments, diagnostics = _load_binary_database(
            database, args.r_search, args.min_structure_pass_fraction,
            args.max_site_radial_mae, args.max_site_radial_max,
            args.max_site_angular_mae, args.max_site_angular_max,
            args.strict_angular_min_cn, args.low_cn_min_angle,
            args.low_cn_max_angle, args.max_exclusion_shell_margin,
        )
        elements = sorted({pair[0] for pair in environments})
        for center, partner in sorted(environments):
            label = center
            if label in label_set:
                label = f"{center}_from_{Path(database).stem}"
            if label in label_set:
                raise ValueError(f"Duplicate construction label {label}")
            label_set.add(label)
            stats = _template_statistics(
                environments[(center, partner)],
                args.radial_sigma_floor,
                args.angular_sigma_floor,
                args.sampling_width_sigma,
                args.max_radial_reconstruction_error,
                args.max_angular_reconstruction_error,
            )
            stats["first_shell_cutoff_A"] = float(
                diagnostics["exclusion_shell_learning"][f"{center}->{partner}"]["cutoff_A"]
            )
            species_records.append(
                _species_record(label, center, partner, stats, f"ase_database:{Path(database).resolve()}")
            )
            framework = diagnostics.get("framework_learning", {}).get(center)
            if framework is not None:
                framework_models[label] = dict(framework)
        source_diagnostics.append({"type": "ase_database", **diagnostics, "elements": elements})

    # Resolve element-based partner semantics to construction-label semantics.
    labels_by_element = defaultdict(list)
    for item in species_records:
        labels_by_element[item["final_formula"]].append(item["label"])
    for item in species_records:
        allowed_elements = item["external_template"]["allowed_partner_elements"]
        allowed_labels = []
        for element in allowed_elements:
            allowed_labels.extend(labels_by_element[element])
        item["external_template"]["allowed_partner_labels"] = sorted(set(allowed_labels))

    pair_channels = []
    by_label = {item["label"]: item for item in species_records}
    for label_i in sorted(by_label):
        item_i = by_label[label_i]
        for label_j in sorted(item_i["external_template"]["allowed_partner_labels"]):
            if label_j not in by_label:
                continue
            item_j = by_label[label_j]
            mu_i = float(item_i["external_template"]["radial_mean_A"])
            mu_j = float(item_j["external_template"]["radial_mean_A"])
            sigma = max(
                float(item_i["external_template"]["radial_sigma_A"]),
                float(item_j["external_template"]["radial_sigma_A"]),
            )
            mu = 0.5 * (mu_i + mu_j) if label_i != label_j else mu_i
            sampling_min = max(0.0, mu - float(args.sampling_width_sigma) * sigma)
            sampling_max = mu + float(args.sampling_width_sigma) * sigma
            shell_i = float(item_i["external_template"]["first_shell_cutoff_A"])
            shell_j = float(item_j["external_template"]["first_shell_cutoff_A"])
            first_shell_cutoff = max(shell_i, shell_j, sampling_max + 0.05)
            pair_channels.append(
                {
                    "species_i": label_i,
                    "species_j": label_j,
                    "relation": "external_neighbor",
                    "distance_mu_A": mu,
                    "distance_sigma_A": sigma,
                    "sampling_min_A": sampling_min,
                    "sampling_max_A": sampling_max,
                    "first_shell_cutoff_A": first_shell_cutoff,
                    "source": "same_template" if label_i == label_j else "symmetric_mean",
                    "shell_cutoff_source": "max_directional_first_shell_boundary",
                }
            )

    model = {
        "version": 6,
        "schema": "juliette_building_block_xn_v1",
        "semantics": {
            "construction_species": "building_block_plus_external_template",
            "Xn": "n expected external neighbour positions in the block-local frame",
            "X0": "empty external template; ordinary first-class template",
            "connectivity": "dynamic_reconciliation_not_fixed_graph",
            "symmetry": "all block poses and templates are expanded from Wyckoff-independent variables",
        },
        "construction_species": species_records,
        "pair_channels": pair_channels,
        "framework_models": framework_models,
        "sources": source_diagnostics,
        "parameters": {
            "r_search_A": float(args.r_search),
            "radial_sigma_floor_A": float(args.radial_sigma_floor),
            "angular_sigma_floor_deg": float(args.angular_sigma_floor),
            "sampling_width_sigma": float(args.sampling_width_sigma),
            "max_radial_reconstruction_error_A": float(args.max_radial_reconstruction_error),
            "max_angular_reconstruction_error_deg": float(args.max_angular_reconstruction_error),
            "min_structure_pass_fraction": float(args.min_structure_pass_fraction),
            "max_site_radial_mae_A": float(args.max_site_radial_mae),
            "max_site_radial_max_A": float(args.max_site_radial_max),
            "max_site_angular_mae_deg": float(args.max_site_angular_mae),
            "max_site_angular_max_deg": float(args.max_site_angular_max),
            "strict_angular_min_cn": int(args.strict_angular_min_cn),
            "low_cn_min_angle_deg": float(args.low_cn_min_angle),
            "low_cn_max_angle_deg": float(args.low_cn_max_angle),
            "max_exclusion_shell_margin_A": float(args.max_exclusion_shell_margin),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Juliette Xn template model written")
    print(f"Output: {output}")
    for item in species_records:
        template = item["external_template"]
        print(
            f"  {item['label']}: {item['building_block']['kind']} + {template['kind']} "
            f"partners={template['allowed_partner_labels']} r={template['radial_mean_A']:.5f} A"
        )


if __name__ == "__main__":
    main()


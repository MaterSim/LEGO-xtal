#!/usr/bin/env python3
"""Learn local chemistry directly from physical structures in an ASE database.

This v5 chemistry-training generator is intentionally independent of the old
LEGO-Xtal tabular-representation, SO(3), VAE, and subgroup-augmentation path.

Workflow
--------
PASS 1
    Read raw ASE DB rows directly with row.toatoms().
    Collect broad center-attachment and attachment-center distance spectra up
    to a generous user-defined search radius.
    Learn chemical-shell cutoffs from smoothed dCN/dr and cumulative CN(r),
    using manually specified expected coordination numbers.

PASS 2
    Count per-site coordination at the learned shell cutoffs.
    Reject structures whose center->attachment or attachment->center CN pass
    fraction is below the requested threshold.

PASS 3
    Learn chemistry from accepted physical structures only:
      * center-attachment radial distribution
      * attachment-center-attachment angular distribution
      * attachment-center radial distribution
      * center-attachment-center angular distribution
      * center-center radial distribution
      * center-center-center angular distributions conditioned on the pair of
        learned center-center radial-shell identities

    Each accepted source structure carries total statistical weight 1 within each channel.
    Major modes are detected from a smoothed weighted density and represented
    by Gaussian peak position, width, and probability weight. Sampling bounds
    are stored as mu +/- n_width * sigma.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase.db import connect
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

EPS = 1.0e-12


@dataclass
class WeightedSamples:
    values: list[float]
    weights: list[float]
    site_keys: list[str]
    source_rows: list[int]

    def __init__(self):
        self.values = []
        self.weights = []
        self.site_keys = []
        self.source_rows = []

    def extend(self, values, site_weight_total, site_key, source_row):
        clean = [float(v) for v in values if np.isfinite(v)]
        if not clean:
            return
        weight = float(site_weight_total) / len(clean)
        for value in clean:
            self.values.append(value)
            self.weights.append(weight)
            self.site_keys.append(site_key)
            self.source_rows.append(int(source_row))

    def arrays(self):
        return np.asarray(self.values, dtype=float), np.asarray(self.weights, dtype=float)


def _species_neighbors(structure, center_index, neighbor_species, radius):
    center = structure[center_index]
    neighbors = []
    for neighbor in structure.get_neighbors(center, radius, include_index=True, include_image=True):
        if str(neighbor.specie.symbol) != str(neighbor_species):
            continue
        neighbors.append({
            "distance": float(neighbor.nn_distance),
            "vector": np.asarray(neighbor.coords - center.coords, dtype=float),
            "atom_index": int(neighbor.index),
            "image": tuple(int(x) for x in neighbor.image),
        })
    neighbors.sort(key=lambda item: (item["distance"], item["atom_index"], item["image"]))
    return neighbors


def _angle_deg(v1, v2):
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= EPS or n2 <= EPS:
        return float("nan")
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _weighted_quantile(values, weights, quantiles):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return [float("nan")] * len(quantiles)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return [float(np.interp(float(q), cumulative, values)) for q in quantiles]


def _finite_stats(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {key: None for key in ("min", "q05", "q25", "median", "q75", "q95", "max", "mean")}
    return {
        "min": float(np.min(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _discover_attachment_species(database, building_center):
    db = connect(database, serial=True)
    all_species = set()
    for row in db.select():
        all_species.update(row.toatoms().get_chemical_symbols())
    if building_center not in all_species:
        raise ValueError(f"Building center {building_center!r} absent. Observed={sorted(all_species)}")
    attachments = sorted(all_species - {building_center})
    if len(attachments) != 1:
        raise ValueError(
            "v1 automatically supports one attachment species. "
            f"Building center={building_center}; other species={attachments}."
        )
    return attachments[0]


def _collect_broad_distances(structures, center_species, neighbor_species, radius):
    per_site = []
    for item in structures:
        structure = item["structure"]
        symbols = [str(site.specie.symbol) for site in structure]
        for site_id, symbol in enumerate(symbols):
            if symbol != center_species:
                continue
            neighbors = _species_neighbors(structure, site_id, neighbor_species, radius)
            per_site.append([float(n["distance"]) for n in neighbors])
    return per_site


def _learn_shell_cutoff(per_site_distances, expected_cn, r_search, grid_size,
                        smooth_sigma_bins, peak_prominence_fraction,
                        valley_fraction, valley_min_width_A):
    """Identify the first chemical shell from radial-density separation only.

    Expected CN is recorded for diagnostics but does not influence shell
    selection. The first significant dCN/dr peak defines the first radial
    density group. The shell cutoff is the entrance to the first persistent
    low-density valley after that peak.
    """
    if not per_site_distances:
        raise ValueError("No local environments available for shell learning")

    edges = np.linspace(0.0, r_search, grid_size + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dr = float(edges[1] - edges[0])
    histogram = np.zeros(grid_size, dtype=float)
    cn_curve = np.zeros(grid_size, dtype=float)

    for distances in per_site_distances:
        arr = np.asarray(distances, dtype=float)
        arr = arr[(arr >= 0.0) & (arr <= r_search)]
        counts, _ = np.histogram(arr, bins=edges)
        histogram += counts
        cn_curve += np.searchsorted(np.sort(arr), centers, side="right")

    n_sites = len(per_site_distances)
    histogram /= n_sites
    cn_curve /= n_sites
    dcn_dr = histogram / dr
    smooth = gaussian_filter1d(
        dcn_dr,
        sigma=float(smooth_sigma_bins),
        mode="nearest",
    )

    maximum = float(np.max(smooth)) if len(smooth) else 0.0
    prominence = max(maximum * float(peak_prominence_fraction), EPS)
    peak_ids, peak_props = find_peaks(smooth, prominence=prominence)

    if len(peak_ids) == 0:
        raise RuntimeError(
            "No significant dCN/dr peak found. Increase --r-search, reduce "
            "--shell-smooth-sigma-bins, or reduce "
            "--shell-peak-prominence-fraction."
        )

    first_peak_id = int(np.min(peak_ids))
    first_peak_height = float(smooth[first_peak_id])
    valley_limit = first_peak_height * float(valley_fraction)
    min_bins = max(1, int(math.ceil(float(valley_min_width_A) / dr)))

    selected_id = None
    valley_regions = []
    start = None

    for index in range(first_peak_id + 1, len(smooth)):
        low = bool(smooth[index] <= valley_limit + EPS)
        if low and start is None:
            start = index
        elif not low and start is not None:
            end = index - 1
            width_A = float(edges[end + 1] - edges[start])
            persistent = bool(end - start + 1 >= min_bins)
            valley_regions.append({
                "start_index": int(start),
                "end_index": int(end),
                "start_r_A": float(edges[start]),
                "end_r_A": float(edges[end + 1]),
                "width_A": width_A,
                "persistent": persistent,
            })
            if selected_id is None and persistent:
                selected_id = int(start)
                break
            start = None

    if selected_id is None and start is not None:
        end = len(smooth) - 1
        width_A = float(edges[end + 1] - edges[start])
        persistent = bool(end - start + 1 >= min_bins)
        valley_regions.append({
            "start_index": int(start),
            "end_index": int(end),
            "start_r_A": float(edges[start]),
            "end_r_A": float(edges[end + 1]),
            "width_A": width_A,
            "persistent": persistent,
        })
        if persistent:
            selected_id = int(start)

    if selected_id is None:
        raise RuntimeError(
            "No persistent low-density valley found after the first dCN/dr "
            "peak. Increase --r-search, increase --shell-valley-fraction, or "
            "reduce --shell-valley-min-width."
        )

    prominences = peak_props.get(
        "prominences",
        np.zeros(len(peak_ids), dtype=float),
    )
    peaks = [
        {
            "r_A": float(centers[index]),
            "dcn_dr": float(smooth[index]),
            "prominence": float(prom),
            "mean_cn_at_r": float(cn_curve[index]),
        }
        for index, prom in zip(peak_ids, prominences)
    ]

    return {
        "r_shell_A": float(edges[selected_id]),
        "expected_cn": int(expected_cn),
        "mean_cn_at_shell": float(cn_curve[selected_id]),
        "selection_mode": "first_persistent_low_density_valley_after_first_peak",
        "valley_fraction": float(valley_fraction),
        "valley_min_width_A": float(valley_min_width_A),
        "first_peak": {
            "r_A": float(centers[first_peak_id]),
            "dcn_dr": first_peak_height,
            "valley_limit": float(valley_limit),
        },
        "selected_candidate": {
            "r_A": float(edges[selected_id]),
            "mean_cn": float(cn_curve[selected_id]),
            "dcn_dr": float(smooth[selected_id]),
            "grid_index": int(selected_id),
        },
        "dcn_dr_peaks": peaks,
        "valley_regions": valley_regions,
        "grid": {
            "r_A": centers.tolist(),
            "cn_mean": cn_curve.tolist(),
            "dcn_dr_raw": dcn_dr.tolist(),
            "dcn_dr_smooth": smooth.tolist(),
        },
    }


def _fit_major_gaussian_peaks(samples, n_peak, domain, grid_size,
                              smooth_sigma_bins, prominence_fraction, n_width,
                              exclude_right_boundary=False):
    values, weights = samples.arrays()
    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
        & (values >= domain[0])
        & (values <= domain[1])
    )
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return {
            "status": "empty",
            "sample_count": 0,
            "effective_site_weight": 0.0,
            "peaks": [],
            "retained_probability_mass": 0.0,
            "discarded_probability_mass": 0.0,
            "excluded_boundary_probability_mass": 0.0,
        }

    edges = np.linspace(float(domain[0]), float(domain[1]), int(grid_size) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density, _ = np.histogram(values, bins=edges, weights=weights)
    width = float(edges[1] - edges[0])
    total_weight = float(np.sum(weights))
    if np.sum(density) > 0:
        density = density / (np.sum(density) * width)
    smooth = gaussian_filter1d(
        density,
        sigma=float(smooth_sigma_bins),
        mode="nearest",
    )

    maximum = float(np.max(smooth))
    prominence = max(maximum * float(prominence_fraction), EPS)
    peak_ids, props = find_peaks(smooth, prominence=prominence)
    if len(peak_ids) == 0:
        peak_ids = np.asarray([int(np.argmax(smooth))], dtype=int)
        prominences = np.asarray([maximum], dtype=float)
    else:
        prominences = np.asarray(props["prominences"], dtype=float)

    # Every candidate peak gets a local basin bounded by the minimum-density
    # point between adjacent detected peaks. Edge basins are open at the domain
    # boundary and are explicitly marked.
    ordered = sorted(
        zip(peak_ids.tolist(), prominences.tolist()),
        key=lambda item: item[0],
    )
    boundaries = []
    for (left_id, _), (right_id, _) in zip(ordered[:-1], ordered[1:]):
        local = smooth[left_id:right_id + 1]
        boundaries.append(int(left_id + int(np.argmin(local))))

    candidates = []
    for peak_order, (peak_id, peak_prominence) in enumerate(ordered):
        left_id = 0 if peak_order == 0 else boundaries[peak_order - 1]
        right_id = len(centers) - 1 if peak_order == len(ordered) - 1 else boundaries[peak_order]
        left_open = peak_order == 0
        right_open = peak_order == len(ordered) - 1

        lower = float(edges[left_id])
        upper = float(edges[right_id + 1])
        if peak_order < len(ordered) - 1:
            use = (values >= lower) & (values < upper)
        else:
            use = (values >= lower) & (values <= upper)

        local_values = values[use]
        local_weights = weights[use]
        mass = float(np.sum(local_weights))
        if mass <= EPS:
            continue

        mu = float(np.sum(local_weights * local_values) / mass)
        variance = float(
            np.sum(local_weights * (local_values - mu) ** 2) / mass
        )
        sigma = float(np.sqrt(max(variance, EPS)))
        boundary_truncated = bool(exclude_right_boundary and right_open)

        candidates.append({
            "mu": mu,
            "sigma": sigma,
            "weight": float(mass / total_weight),
            "peak_height": float(smooth[peak_id]),
            "sampling_min": float(max(domain[0], mu - n_width * sigma)),
            "sampling_max": float(min(domain[1], mu + n_width * sigma)),
            "assigned_sample_count": int(np.sum(use)),
            "assigned_effective_weight": mass,
            "initial_peak_location": float(centers[peak_id]),
            "prominence": float(peak_prominence),
            "basin_min": lower,
            "basin_max": upper,
            "left_basin_open": bool(left_open),
            "right_basin_open": bool(right_open),
            "boundary_truncated": boundary_truncated,
        })

    generative_candidates = [
        item for item in candidates if not item["boundary_truncated"]
    ]
    ranked = sorted(
        generative_candidates,
        key=lambda item: (
            item["peak_height"],
            item["prominence"],
        ),
        reverse=True,
    )[: int(n_peak)]
    fitted = list(ranked)

    retained_mass = float(sum(item["weight"] for item in fitted))
    excluded_boundary_mass = float(
        sum(item["weight"] for item in candidates if item["boundary_truncated"])
    )
    discarded_mass = float(max(0.0, 1.0 - retained_mass - excluded_boundary_mass))

    q_names = ["q05", "q25", "median", "q75", "q95"]
    q_values = _weighted_quantile(
        values,
        weights,
        [0.05, 0.25, 0.5, 0.75, 0.95],
    )
    return {
        "status": "ok",
        "sample_count": int(len(values)),
        "effective_site_weight": total_weight,
        "domain": [float(domain[0]), float(domain[1])],
        "peaks": fitted,
        "all_detected_peak_basins": sorted(candidates, key=lambda item: item["mu"]),
        "retained_probability_mass": retained_mass,
        "discarded_probability_mass": discarded_mass,
        "excluded_boundary_probability_mass": excluded_boundary_mass,
        "weighted_quantiles": dict(zip(q_names, q_values)),
    }


def _channel_summary_lines(title, model, unit):
    lines = ["-" * 60, title, "-" * 60]
    if model.get("status") != "ok":
        lines.append("Status                         : EMPTY")
        return lines
    lines.append(f"Samples                        : {model['sample_count']}")
    lines.append(f"Effective site weight          : {model['effective_site_weight']:.3f}")
    q = model["weighted_quantiles"]
    lines.append(
        f"Weighted q05/q25/q50/q75/q95   : {q['q05']:.5f} / {q['q25']:.5f} / "
        f"{q['median']:.5f} / {q['q75']:.5f} / {q['q95']:.5f} {unit}"
    )
    lines.append("")
    lines.append(
        f"{'Peak':>4s}  {'height':>12s}  {'prominence':>12s}  "
        f"{'weight':>10s}  {'mu':>12s}  {'sigma':>12s}  "
        f"{'basin':>25s}  {'sample interval':>27s}"
    )
    for index, peak in enumerate(model["peaks"], start=1):
        lines.append(
            f"{index:4d}  {peak['peak_height']:12.6g}  "
            f"{peak['prominence']:12.6g}  {peak['weight']:10.5f}  "
            f"{peak['mu']:12.5f}  {peak['sigma']:12.5f}  "
            f"[{peak['basin_min']:.5f}, {peak['basin_max']:.5f}]  "
            f"[{peak['sampling_min']:.5f}, {peak['sampling_max']:.5f}] {unit}"
        )
    lines.append(
        f"Retained probability mass      : {model['retained_probability_mass']:.6f}"
    )
    lines.append(
        f"Discarded probability mass     : {model['discarded_probability_mass']:.6f}"
    )
    if model.get("excluded_boundary_probability_mass", 0.0) > EPS:
        lines.append(
            "Boundary-truncated mass         : "
            f"{model['excluded_boundary_probability_mass']:.6f}"
        )
    return lines


def parse_args():
    parser = argparse.ArgumentParser(description="Learn local chemistry from raw physical structures in an ASE DB.")
    parser.add_argument("--database", default="data/source/tio2.db")
    parser.add_argument("--building-center", required=True)
    parser.add_argument("--center-attachment-cn", type=int, required=True)
    parser.add_argument("--attachment-center-cn", type=int, required=True)
    parser.add_argument("--r-search", type=float, required=True)
    parser.add_argument("--n-peak", type=int, default=3)
    parser.add_argument("--n-width", type=float, default=2.0)
    parser.add_argument("--min-cn-pass-fraction", type=float, default=1.0)
    parser.add_argument("--output-dir", default="data/chemistry")
    parser.add_argument("--grid-size", type=int, default=1000)
    parser.add_argument("--shell-smooth-sigma-bins", type=float, default=6.0)
    parser.add_argument("--shell-peak-prominence-fraction", type=float, default=0.03)
    parser.add_argument("--shell-valley-fraction", type=float, default=0.01)
    parser.add_argument(
        "--shell-valley-min-width", type=float, default=0.10,
        help="Minimum persistent low-density valley width in Angstrom.",
    )
    parser.add_argument("--peak-smooth-sigma-bins", type=float, default=5.0)
    parser.add_argument("--peak-prominence-fraction", type=float, default=0.03)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.database):
        raise FileNotFoundError(args.database)
    if args.center_attachment_cn <= 0 or args.attachment_center_cn <= 0:
        raise ValueError("Coordination numbers must be positive")
    if args.r_search <= 0 or args.n_peak <= 0 or args.n_width <= 0:
        raise ValueError("--r-search, --n-peak, and --n-width must be positive")
    if not 0.0 <= args.min_cn_pass_fraction <= 1.0:
        raise ValueError("--min-cn-pass-fraction must lie in [0, 1]")
    if args.grid_size < 100:
        raise ValueError("--grid-size must be at least 100")
    if not 0.0 < args.shell_valley_fraction < 1.0:
        raise ValueError("--shell-valley-fraction must lie in (0, 1)")
    if args.shell_valley_min_width <= 0:
        raise ValueError("--shell-valley-min-width must be positive")

    center_species = str(args.building_center)
    attachment_species = _discover_attachment_species(args.database, center_species)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = connect(args.database, serial=True)
    rows = list(db.select())
    structures = []
    failures = []

    print("--- Chemistry training configuration ---")
    print(f"Database: {args.database}")
    print(f"Input structures: {len(rows)}")
    print(f"Building center: {center_species}")
    print(f"Attachment species: {attachment_species}")
    print(f"{center_species}->{attachment_species} expected CN: {args.center_attachment_cn}")
    print(f"{attachment_species}->{center_species} expected CN: {args.attachment_center_cn}")
    print(f"Broad search radius: {args.r_search:.5f} A")
    print(f"Maximum peaks/channel: {args.n_peak}")
    print(f"Sampling width: +/- {args.n_width:.3f} sigma")
    print(f"Minimum per-structure CN pass fraction: {args.min_cn_pass_fraction:.5f}")
    print("----------------------------------------")

    for completed, row in enumerate(rows, start=1):
        try:
            structure = AseAtomsAdaptor.get_structure(row.toatoms())
            symbols = [str(site.specie.symbol) for site in structure]
            unexpected = sorted(set(symbols) - {center_species, attachment_species})
            if unexpected:
                raise ValueError(f"Unexpected species in row {row.id}: {unexpected}")
            structures.append({"row_id": int(row.id), "structure": structure, "natoms": int(len(structure))})
        except Exception as exc:
            failures.append({"row_id": int(row.id), "error": f"{type(exc).__name__}: {exc}"})
        if completed % args.progress_every == 0 or completed == len(rows):
            print(f"Loaded {completed}/{len(rows)} structures; valid={len(structures)}; failures={len(failures)}")

    if not structures:
        raise RuntimeError("No valid structures were loaded")

    ca_distances = _collect_broad_distances(structures, center_species, attachment_species, args.r_search)
    ac_distances = _collect_broad_distances(structures, attachment_species, center_species, args.r_search)
    ca_shell = _learn_shell_cutoff(ca_distances, args.center_attachment_cn, args.r_search,
                                    args.grid_size, args.shell_smooth_sigma_bins,
                                    args.shell_peak_prominence_fraction,
                                    args.shell_valley_fraction,
                                    args.shell_valley_min_width)
    ac_shell = _learn_shell_cutoff(ac_distances, args.attachment_center_cn, args.r_search,
                                    args.grid_size, args.shell_smooth_sigma_bins,
                                    args.shell_peak_prominence_fraction,
                                    args.shell_valley_fraction,
                                    args.shell_valley_min_width)
    ca_cutoff = float(ca_shell["r_shell_A"])
    ac_cutoff = float(ac_shell["r_shell_A"])

    structure_records = []
    center_site_records = []
    attachment_site_records = []
    accepted_structures = []

    for completed, entry in enumerate(structures, start=1):
        structure = entry["structure"]
        row_id = int(entry["row_id"])
        symbols = [str(site.specie.symbol) for site in structure]
        center_ids = [i for i, s in enumerate(symbols) if s == center_species]
        attachment_ids = [i for i, s in enumerate(symbols) if s == attachment_species]
        center_passes = []
        attachment_passes = []

        for local_index, site_id in enumerate(center_ids):
            neighbors = _species_neighbors(structure, site_id, attachment_species, args.r_search)
            distances = [n["distance"] for n in neighbors]
            cn = int(sum(d <= ca_cutoff + EPS for d in distances))
            passed = cn == args.center_attachment_cn
            center_passes.append(passed)
            nth = float(distances[args.center_attachment_cn - 1]) if len(distances) >= args.center_attachment_cn else np.nan
            nxt = float(distances[args.center_attachment_cn]) if len(distances) > args.center_attachment_cn else np.nan
            gap = float(nxt - nth) if np.isfinite(nth) and np.isfinite(nxt) else np.nan
            center_site_records.append({
                "row_id": row_id, "site_local_index": int(local_index), "atom_index": int(site_id),
                "species": center_species, "neighbor_species": attachment_species,
                "expected_cn": int(args.center_attachment_cn), "learned_shell_cutoff_A": ca_cutoff,
                "observed_cn": cn, "cn_pass": bool(passed), "nth_distance_A": nth,
                "next_distance_A": nxt, "shell_gap_A": gap,
                "distances_within_r_search_A": json.dumps([float(x) for x in distances], separators=(",", ":")),
            })

        for local_index, site_id in enumerate(attachment_ids):
            neighbors = _species_neighbors(structure, site_id, center_species, args.r_search)
            distances = [n["distance"] for n in neighbors]
            cn = int(sum(d <= ac_cutoff + EPS for d in distances))
            passed = cn == args.attachment_center_cn
            attachment_passes.append(passed)
            nth = float(distances[args.attachment_center_cn - 1]) if len(distances) >= args.attachment_center_cn else np.nan
            nxt = float(distances[args.attachment_center_cn]) if len(distances) > args.attachment_center_cn else np.nan
            gap = float(nxt - nth) if np.isfinite(nth) and np.isfinite(nxt) else np.nan
            attachment_site_records.append({
                "row_id": row_id, "site_local_index": int(local_index), "atom_index": int(site_id),
                "species": attachment_species, "neighbor_species": center_species,
                "expected_cn": int(args.attachment_center_cn), "learned_shell_cutoff_A": ac_cutoff,
                "observed_cn": cn, "cn_pass": bool(passed), "nth_distance_A": nth,
                "next_distance_A": nxt, "shell_gap_A": gap,
                "distances_within_r_search_A": json.dumps([float(x) for x in distances], separators=(",", ":")),
            })

        center_fraction = float(np.mean(center_passes)) if center_passes else 0.0
        attachment_fraction = float(np.mean(attachment_passes)) if attachment_passes else 0.0
        accepted = bool(center_fraction + EPS >= args.min_cn_pass_fraction and
                        attachment_fraction + EPS >= args.min_cn_pass_fraction)
        reasons = []
        if center_fraction + EPS < args.min_cn_pass_fraction:
            reasons.append("center_cn")
        if attachment_fraction + EPS < args.min_cn_pass_fraction:
            reasons.append("attachment_cn")
        structure_records.append({
            "row_id": row_id, "natoms": int(len(structure)), "n_center": int(len(center_ids)),
            "n_attachment": int(len(attachment_ids)), "center_cn_pass_fraction": center_fraction,
            "attachment_cn_pass_fraction": attachment_fraction, "accepted": accepted,
            "rejection_reason": ";".join(reasons),
        })
        if accepted:
            accepted_structures.append(entry)
        if completed % args.progress_every == 0 or completed == len(structures):
            print(f"CN audit {completed}/{len(structures)} structures; accepted={len(accepted_structures)}")

    if not accepted_structures:
        raise RuntimeError("No structures passed the CN health filter")

    radial_channels = {
        f"{center_species}-{attachment_species}": WeightedSamples(),
        f"{attachment_species}-{center_species}": WeightedSamples(),
        f"{center_species}-{center_species}": WeightedSamples(),
    }
    angular_channels = {
        f"{attachment_species}-{center_species}-{attachment_species}": WeightedSamples(),
        f"{center_species}-{attachment_species}-{center_species}": WeightedSamples(),
    }
    center_neighbor_cache = []

    for completed, entry in enumerate(accepted_structures, start=1):
        structure = entry["structure"]
        row_id = int(entry["row_id"])
        symbols = [str(site.specie.symbol) for site in structure]
        center_ids = [i for i, s in enumerate(symbols) if s == center_species]
        attachment_ids = [i for i, s in enumerate(symbols) if s == attachment_species]
        center_site_weight = 1.0 / len(center_ids) if center_ids else 0.0
        attachment_site_weight = 1.0 / len(attachment_ids) if attachment_ids else 0.0

        for local_index, site_id in enumerate(center_ids):
            site_key = f"row{row_id}:{center_species}{local_index}"
            attachments = [n for n in _species_neighbors(structure, site_id, attachment_species, ca_cutoff)
                           if n["distance"] <= ca_cutoff + EPS]
            if len(attachments) != args.center_attachment_cn:
                raise RuntimeError(f"Final extraction CN mismatch at row {row_id} center {local_index}")
            radial_channels[f"{center_species}-{attachment_species}"].extend(
                [n["distance"] for n in attachments], center_site_weight, site_key, row_id)
            angles = [_angle_deg(attachments[i]["vector"], attachments[j]["vector"])
                      for i in range(len(attachments)) for j in range(i + 1, len(attachments))]
            angular_channels[f"{attachment_species}-{center_species}-{attachment_species}"].extend(
                angles, center_site_weight, site_key, row_id)
            centers = _species_neighbors(structure, site_id, center_species, args.r_search)
            radial_channels[f"{center_species}-{center_species}"].extend(
                [n["distance"] for n in centers], center_site_weight, site_key, row_id)
            center_neighbor_cache.append({"row_id": row_id, "site_key": site_key, "neighbors": centers})

        for local_index, site_id in enumerate(attachment_ids):
            site_key = f"row{row_id}:{attachment_species}{local_index}"
            centers = [n for n in _species_neighbors(structure, site_id, center_species, ac_cutoff)
                       if n["distance"] <= ac_cutoff + EPS]
            if len(centers) != args.attachment_center_cn:
                raise RuntimeError(f"Final extraction CN mismatch at row {row_id} attachment {local_index}")
            radial_channels[f"{attachment_species}-{center_species}"].extend(
                [n["distance"] for n in centers], attachment_site_weight, site_key, row_id)
            angles = [_angle_deg(centers[i]["vector"], centers[j]["vector"])
                      for i in range(len(centers)) for j in range(i + 1, len(centers))]
            angular_channels[f"{center_species}-{attachment_species}-{center_species}"].extend(
                angles, attachment_site_weight, site_key, row_id)

        if completed % args.progress_every == 0 or completed == len(accepted_structures):
            print(f"Chemistry extraction {completed}/{len(accepted_structures)} accepted structures")

    ca_name = f"{center_species}-{attachment_species}"
    ac_name = f"{attachment_species}-{center_species}"
    cc_name = f"{center_species}-{center_species}"
    aca_name = f"{attachment_species}-{center_species}-{attachment_species}"
    cac_name = f"{center_species}-{attachment_species}-{center_species}"

    models = {}
    models[f"radial:{ca_name}"] = _fit_major_gaussian_peaks(
        radial_channels[ca_name], args.n_peak, (0.0, ca_cutoff), args.grid_size,
        args.peak_smooth_sigma_bins, args.peak_prominence_fraction, args.n_width)
    models[f"radial:{ac_name}"] = _fit_major_gaussian_peaks(
        radial_channels[ac_name], args.n_peak, (0.0, ac_cutoff), args.grid_size,
        args.peak_smooth_sigma_bins, args.peak_prominence_fraction, args.n_width)
    models[f"angular:{aca_name}"] = _fit_major_gaussian_peaks(
        angular_channels[aca_name], args.n_peak, (0.0, 180.0), args.grid_size,
        args.peak_smooth_sigma_bins, args.peak_prominence_fraction, args.n_width)
    models[f"angular:{cac_name}"] = _fit_major_gaussian_peaks(
        angular_channels[cac_name], args.n_peak, (0.0, 180.0), args.grid_size,
        args.peak_smooth_sigma_bins, args.peak_prominence_fraction, args.n_width)
    models[f"radial:{cc_name}"] = _fit_major_gaussian_peaks(
        radial_channels[cc_name], args.n_peak, (0.0, args.r_search), args.grid_size,
        args.peak_smooth_sigma_bins, args.peak_prominence_fraction, args.n_width,
        exclude_right_boundary=True)

    cc_peaks = models[f"radial:{cc_name}"]["peaks"]
    shell_pair_channels = defaultdict(WeightedSamples)
    if cc_peaks:
        cc_mus = np.asarray([float(p["mu"]) for p in cc_peaks], dtype=float)
        per_structure_shell_pair_sites = defaultdict(lambda: defaultdict(list))
        for entry in center_neighbor_cache:
            neighbors = entry["neighbors"]
            if len(neighbors) < 2:
                continue
            shell_ids = [int(np.argmin(np.abs(cc_mus - n["distance"]))) for n in neighbors]
            grouped_angles = defaultdict(list)
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    pair = tuple(sorted((shell_ids[i], shell_ids[j])))
                    grouped_angles[pair].append(_angle_deg(neighbors[i]["vector"], neighbors[j]["vector"]))
            for pair, angles in grouped_angles.items():
                per_structure_shell_pair_sites[int(entry["row_id"])][pair].append(
                    (entry["site_key"], angles)
                )

        for row_id, pair_map in per_structure_shell_pair_sites.items():
            for pair, site_entries in pair_map.items():
                site_weight = 1.0 / len(site_entries)
                for site_key, angles in site_entries:
                    shell_pair_channels[pair].extend(
                        angles, site_weight, site_key, row_id
                    )

    for pair, samples in sorted(shell_pair_channels.items()):
        channel = f"angular:{center_species}-{center_species}-{center_species}:shell_{pair[0]+1}_{pair[1]+1}"
        models[channel] = _fit_major_gaussian_peaks(
            samples, args.n_peak, (0.0, 180.0), args.grid_size,
            args.peak_smooth_sigma_bins, args.peak_prominence_fraction, args.n_width)

    radial_rows = []
    for channel, samples in radial_channels.items():
        for value, weight, site_key, source_row in zip(samples.values, samples.weights, samples.site_keys, samples.source_rows):
            radial_rows.append({"channel": channel, "value_A": float(value), "weight": float(weight),
                                "site_key": site_key, "row_id": int(source_row)})

    all_angular_channels = dict(angular_channels)
    for pair, samples in shell_pair_channels.items():
        all_angular_channels[f"{center_species}-{center_species}-{center_species}:shell_{pair[0]+1}_{pair[1]+1}"] = samples
    angular_rows = []
    for channel, samples in all_angular_channels.items():
        for value, weight, site_key, source_row in zip(samples.values, samples.weights, samples.site_keys, samples.source_rows):
            angular_rows.append({"channel": channel, "value_deg": float(value), "weight": float(weight),
                                 "site_key": site_key, "row_id": int(source_row)})

    peak_rows = []
    for channel, model in models.items():
        for peak_index, peak in enumerate(model.get("peaks", []), start=1):
            peak_rows.append({"channel": channel, "peak_index": peak_index, "mu": peak["mu"],
                              "sigma": peak["sigma"], "weight": peak["weight"],
                              "peak_height": peak["peak_height"],
                              "sampling_min": peak["sampling_min"], "sampling_max": peak["sampling_max"],
                              "assigned_sample_count": peak["assigned_sample_count"],
                              "assigned_effective_weight": peak["assigned_effective_weight"],
                              "basin_min": peak["basin_min"], "basin_max": peak["basin_max"],
                              "prominence": peak["prominence"],
                              "boundary_truncated": peak["boundary_truncated"]})

    structure_df = pd.DataFrame(structure_records).sort_values("row_id", kind="stable")
    center_df = pd.DataFrame(center_site_records).sort_values(["row_id", "site_local_index"], kind="stable")
    attachment_df = pd.DataFrame(attachment_site_records).sort_values(["row_id", "site_local_index"], kind="stable")
    radial_df = pd.DataFrame(radial_rows)
    angular_df = pd.DataFrame(angular_rows)
    peak_df = pd.DataFrame(peak_rows)

    structure_file = output_dir / "structure_summary.csv"
    center_file = output_dir / "center_sites.csv"
    attachment_file = output_dir / "attachment_sites.csv"
    radial_file = output_dir / "radial_samples.csv"
    angular_file = output_dir / "angular_samples.csv"
    peak_file = output_dir / "peak_summary.csv"
    failure_file = output_dir / "load_failures.csv"
    model_file = output_dir / "chemistry_model.json"

    structure_df.to_csv(structure_file, index=False)
    center_df.to_csv(center_file, index=False)
    attachment_df.to_csv(attachment_file, index=False)
    radial_df.to_csv(radial_file, index=False)
    angular_df.to_csv(angular_file, index=False)
    peak_df.to_csv(peak_file, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(failure_file, index=False)

    center_cn_counts = Counter(int(v) for v in center_df["observed_cn"])
    attachment_cn_counts = Counter(int(v) for v in attachment_df["observed_cn"])
    rejection_counter = Counter()
    for reason in structure_df.loc[~structure_df["accepted"], "rejection_reason"]:
        for token in str(reason).split(";"):
            if token:
                rejection_counter[token] += 1

    chemistry_model = {
        "version": 5,
        "database": os.path.abspath(args.database),
        "building_center": center_species,
        "attachment_species": attachment_species,
        "parameters": {
            "center_attachment_cn": int(args.center_attachment_cn),
            "attachment_center_cn": int(args.attachment_center_cn),
            "r_search_A": float(args.r_search),
            "n_peak": int(args.n_peak),
            "n_width_sigma": float(args.n_width),
            "min_cn_pass_fraction": float(args.min_cn_pass_fraction),
            "shell_valley_fraction": float(args.shell_valley_fraction),
            "shell_valley_min_width_A": float(args.shell_valley_min_width),
        },
        "shell_learning": {
            f"{center_species}->{attachment_species}": ca_shell,
            f"{attachment_species}->{center_species}": ac_shell,
        },
        "health_filter": {
            "input_structures": int(len(structures)),
            "accepted_structures": int(len(accepted_structures)),
            "rejected_structures": int(len(structures) - len(accepted_structures)),
            "rejection_counts_nonexclusive": dict(sorted(rejection_counter.items())),
            "center_cn_distribution": dict(sorted(center_cn_counts.items())),
            "attachment_cn_distribution": dict(sorted(attachment_cn_counts.items())),
            "center_shell_gap_A_stats": _finite_stats(center_df["shell_gap_A"].astype(float).tolist()),
            "attachment_shell_gap_A_stats": _finite_stats(attachment_df["shell_gap_A"].astype(float).tolist()),
        },
        "chemistry_channels": models,
        "outputs": {
            "structure_summary": str(structure_file), "center_sites": str(center_file),
            "attachment_sites": str(attachment_file), "radial_samples": str(radial_file),
            "angular_samples": str(angular_file), "peak_summary": str(peak_file),
        },
    }
    model_file.write_text(json.dumps(chemistry_model, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print("CHEMISTRY LEARNING SUMMARY")
    print("=" * 60)
    print(f"Structures analysed             : {len(structures)}")
    print(f"Load failures                  : {len(failures)}")
    print(f"Building center                : {center_species}")
    print(f"Attachment species             : {attachment_species}")
    print(f"{center_species}->{attachment_species} expected CN             : {args.center_attachment_cn}")
    print(f"{attachment_species}->{center_species} expected CN             : {args.attachment_center_cn}")
    print(f"Broad search radius            : {args.r_search:.5f} A")
    print(f"Maximum peaks/channel          : {args.n_peak}")
    print(f"Sampling width                 : +/- {args.n_width:.3f} sigma")

    for label, shell, cn_counts, site_df in (
        (f"{center_species} -> {attachment_species}", ca_shell, center_cn_counts, center_df),
        (f"{attachment_species} -> {center_species}", ac_shell, attachment_cn_counts, attachment_df),
    ):
        print("\n" + "-" * 60)
        print(f"{label} SHELL IDENTIFICATION")
        print("-" * 60)
        print(f"Expected coordination           : {shell['expected_cn']}")
        print(f"Learned shell cutoff            : {shell['r_shell_A']:.5f} A")
        print(f"Mean CN at shell cutoff         : {shell['mean_cn_at_shell']:.5f}")
        print(f"Selection mode                  : {shell['selection_mode']}")
        print("dCN/dr major peaks:")
        for index, peak in enumerate(shell["dcn_dr_peaks"], start=1):
            print(f"  {index:2d}  r={peak['r_A']:.5f} A  prominence={peak['prominence']:.6g}  <CN>={peak['mean_cn_at_r']:.5f}")
        first_peak = shell["first_peak"]
        print(f"First radial-density peak       : {first_peak['r_A']:.5f} A")
        print(f"Low-density threshold dCN/dr    : {first_peak['valley_limit']:.6g}")
        print("Persistent low-density valleys:")
        for region in shell["valley_regions"]:
            marker = " SELECTED" if abs(region["start_r_A"] - shell["r_shell_A"]) < EPS else ""
            print(
                f"  [{region['start_r_A']:.5f}, {region['end_r_A']:.5f}] A  "
                f"width={region['width_A']:.5f} A  "
                f"persistent={region['persistent']}{marker}"
            )
        print(f"Per-site CN distribution       : {dict(sorted(cn_counts.items()))}")
        valid = int(site_df["cn_pass"].sum())
        total = int(len(site_df))
        print(f"CN-valid sites                 : {valid}/{total} ({valid/total:.2%})")
        gap_stats = _finite_stats(site_df["shell_gap_A"].astype(float).tolist())
        print("Shell gap q05/q25/q50/q75/q95  : "
              f"{gap_stats['q05']:.5f} / {gap_stats['q25']:.5f} / {gap_stats['median']:.5f} / "
              f"{gap_stats['q75']:.5f} / {gap_stats['q95']:.5f} A")

    print("\n" + "=" * 60)
    print("STRUCTURE HEALTH FILTER")
    print("=" * 60)
    print(f"Input structures                : {len(structures)}")
    print(f"CN pass requirement             : {args.min_cn_pass_fraction:.2%}")
    print(f"Accepted structures             : {len(accepted_structures)}")
    print(f"Rejected structures             : {len(structures)-len(accepted_structures)}")
    print(f"Rejection counts nonexclusive   : {dict(sorted(rejection_counter.items()))}")

    ordered_channels = [
        (f"radial:{ca_name}", f"{ca_name} RADIAL", "A"),
        (f"angular:{aca_name}", f"{aca_name} ANGULAR", "deg"),
        (f"radial:{ac_name}", f"{ac_name} RADIAL", "A"),
        (f"angular:{cac_name}", f"{cac_name} ANGULAR", "deg"),
        (f"radial:{cc_name}", f"{cc_name} RADIAL", "A"),
    ]
    ordered_channels.extend((channel, channel.replace("angular:", "").upper(), "deg")
                            for channel in models
                            if channel.startswith(f"angular:{center_species}-{center_species}-{center_species}:"))
    for channel, title, unit in ordered_channels:
        print()
        for line in _channel_summary_lines(title, models[channel], unit):
            print(line)

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    print(f"Chemistry model                 : {model_file}")
    print(f"Structure summary               : {structure_file}")
    print(f"Center-site audit               : {center_file}")
    print(f"Attachment-site audit           : {attachment_file}")
    print(f"Raw radial samples              : {radial_file}")
    print(f"Raw angular samples             : {angular_file}")
    print(f"Peak summary                    : {peak_file}")
    if failures:
        print(f"Load failures                   : {failure_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate Juliette building-block + Xn chemistry models (v2).

The validator checks schema consistency, local geometry, partner reciprocity,
pair channels, and approximate Wyckoff/site-symmetry compatibility.  It also
exports isolated template CIFs for visual inspection when ASE is available.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

EPS = 1.0e-12


def _angles(vectors: np.ndarray) -> np.ndarray:
    out = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a, b = vectors[i], vectors[j]
            c = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))
            out.append(math.degrees(math.acos(float(np.clip(c, -1.0, 1.0)))))
    return np.sort(np.asarray(out, float))


def _set_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    if len(a) != len(b):
        return float("inf")
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(np.mean(cost[rows, cols] ** 2)))


def _site_symmetry_residual(vectors: np.ndarray, rotations: list[np.ndarray]) -> float:
    """Return the worst normalized set RMSD for one fixed template orientation."""
    if len(vectors) == 0:
        return 0.0
    scale = max(float(np.mean(np.linalg.norm(vectors, axis=1))), EPS)
    residual = 0.0
    for rotation in rotations:
        transformed = vectors @ np.asarray(rotation, float).T
        residual = max(residual, _set_rmsd(vectors, transformed) / scale)
    return float(residual)


def _rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(rotvec, float)).as_matrix()


def _orientation_optimized_residual(
    vectors: np.ndarray,
    rotations: list[np.ndarray],
    cache: dict,
) -> tuple[float, np.ndarray]:
    """Minimize site-symmetry residual over the free block orientation.

    The stabilizer acts in the crystal Cartesian frame.  The construction
    template may rotate freely, so compatibility is an existence question:
    find Q such that every stabilizer operation maps QX onto QX up to a
    permutation of equivalent X vertices.
    """
    vectors = np.asarray(vectors, float).reshape(-1, 3)
    if len(vectors) == 0 or len(rotations) <= 1:
        return 0.0, np.eye(3)

    norms = np.linalg.norm(vectors, axis=1)
    scale = max(float(np.mean(norms)), EPS)
    normalized = vectors / scale
    rotation_key = tuple(
        tuple(np.round(np.asarray(r, float).reshape(-1), 8))
        for r in rotations
    )
    vector_key = tuple(np.round(normalized.reshape(-1), 8))
    key = (vector_key, rotation_key)
    if key in cache:
        value, matrix = cache[key]
        return float(value), np.asarray(matrix, float)

    from scipy.optimize import minimize

    def objective(rotvec):
        q = _rotation_matrix_from_rotvec(rotvec)
        oriented = normalized @ q.T
        return _site_symmetry_residual(oriented, rotations)

    # Deterministic starts cover identity, principal-axis quarter turns, and
    # several generic orientations.  Powell is derivative-free and robust to
    # the permutation changes in the assignment-based set RMSD.
    starts = [
        np.zeros(3),
        np.array([math.pi / 2, 0.0, 0.0]),
        np.array([0.0, math.pi / 2, 0.0]),
        np.array([0.0, 0.0, math.pi / 2]),
        np.array([math.pi / 3, math.pi / 5, math.pi / 7]),
        np.array([-math.pi / 4, math.pi / 3, math.pi / 6]),
    ]
    best_value = float("inf")
    best_matrix = np.eye(3)
    for x0 in starts:
        result = minimize(
            objective,
            x0,
            method="Powell",
            options={"maxiter": 220, "xtol": 1.0e-6, "ftol": 1.0e-8},
        )
        value = float(result.fun)
        if value < best_value:
            best_value = value
            best_matrix = _rotation_matrix_from_rotvec(result.x)
        if best_value <= 1.0e-8:
            break

    cache[key] = (best_value, best_matrix)
    return best_value, best_matrix


def _representative_position(wp) -> np.ndarray:
    dof = int(wp.get_dof())
    if dof == 0:
        return np.asarray(wp.get_position_from_free_xyzs(np.zeros(0)), float) % 1.0
    values = np.asarray([0.173, 0.287, 0.419][:dof], float)
    return np.asarray(wp.get_position_from_free_xyzs(values), float) % 1.0


def _representative_cell(lattice_type: str) -> np.ndarray:
    """Return a non-degenerate row-vector cell for one crystal system."""
    lt = str(lattice_type).lower()
    if lt == "cubic":
        return np.diag([1.0, 1.0, 1.0])
    if lt == "tetragonal":
        return np.diag([1.0, 1.0, 1.31])
    if lt in {"hexagonal", "trigonal"}:
        return np.asarray(
            [[1.0, 0.0, 0.0], [-0.5, math.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 1.37]],
            float,
        )
    if lt == "orthorhombic":
        return np.diag([1.0, 1.19, 1.43])
    if lt == "monoclinic":
        beta = math.radians(107.0)
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.17, 0.0], [1.31 * math.cos(beta), 0.0, 1.31 * math.sin(beta)]],
            float,
        )
    # Generic triclinic metric, deliberately avoiding accidental symmetry.
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.23, 1.11, 0.0], [0.17, 0.29, 1.27]],
        float,
    )


def _fractional_to_cartesian_rotation(rotation: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Convert a column-vector fractional operation to Cartesian coordinates."""
    cell_t = np.asarray(cell, float).T
    return cell_t @ np.asarray(rotation, float) @ np.linalg.inv(cell_t)


def _stabilizer_rotations(
    group,
    position: np.ndarray,
    cell: np.ndarray,
    tolerance: float = 1.0e-6,
) -> list[np.ndarray]:
    """Return Cartesian point operations that leave a position invariant.

    Iterating over ``Group`` yields Wyckoff-position objects, not symmetry
    operations.  The complete space-group operations are stored on the general
    Wyckoff position ``group[0].ops`` in the PyXtal API used by Juliette.
    """
    rotations = []
    for op in group[0].ops:
        rotation = np.asarray(op.rotation_matrix, float)
        translation = np.asarray(op.translation_vector, float)
        mapped = rotation @ np.asarray(position, float) + translation
        delta = (mapped - position + 0.5) % 1.0 - 0.5
        if float(np.linalg.norm(delta)) <= tolerance:
            cart = _fractional_to_cartesian_rotation(rotation, cell)
            # Remove tiny metric-conversion noise while preserving improper ops.
            u, _, vt = np.linalg.svd(cart)
            orthogonal = u @ vt
            rotations.append(orthogonal)
    if not rotations:
        rotations = [np.eye(3)]
    # Deduplicate operations that differ only by numerical noise.
    unique = []
    for rotation in rotations:
        if not any(np.max(np.abs(rotation - old)) <= 1.0e-7 for old in unique):
            unique.append(rotation)
    return unique


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _model_fingerprint(model: dict, tolerance: float) -> str:
    payload = {
        "validator_cache_version": 1,
        "site_symmetry_tolerance": float(tolerance),
        "construction_species": model.get("construction_species", []),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def _scan_space_group_task(task: dict) -> dict:
    """Scan one space group completely inside one CPU worker."""
    from pyxtal.symmetry import Group

    spg = int(task["spg"])
    tolerance = float(task["tolerance"])
    species_payload = task["species"]
    group = Group(spg)
    cell = _representative_cell(group.lattice_type)
    orientation_cache = {}
    rows = []
    for wp_index in range(len(group)):
        wp = group[wp_index]
        position = _representative_position(wp)
        stabilizer = _stabilizer_rotations(group, position, cell)
        for item in species_payload:
            label = str(item["label"])
            vectors = np.asarray(item["vectors"], float).reshape(-1, 3)
            residual, orientation = _orientation_optimized_residual(
                vectors, stabilizer, orientation_cache
            )
            rows.append(
                {
                    "spg": spg,
                    "wp_index": int(wp_index),
                    "multiplicity": int(wp.multiplicity),
                    "dof": int(wp.get_dof()),
                    "label": label,
                    "stabilizer_order": int(len(stabilizer)),
                    "template_set_residual": float(residual),
                    "best_orientation_matrix_json": json.dumps(
                        np.asarray(orientation, float).round(10).tolist(),
                        separators=(",", ":"),
                    ),
                    "compatible": bool(residual <= tolerance),
                }
            )
    return {"spg": spg, "rows": rows}


def _load_cached_space_group(path: Path, fingerprint: str, spg: int) -> list[dict] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("fingerprint") != fingerprint or int(payload.get("spg", -1)) != int(spg):
        return None
    rows = payload.get("rows")
    return rows if isinstance(rows, list) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Juliette Xn template models")
    parser.add_argument("--model", default="data/xn_templates/chemistry_model.json")
    parser.add_argument("--output-dir", default="data/xn_templates/validation")
    parser.add_argument("--site-symmetry-tolerance", type=float, default=0.12)
    parser.add_argument("--space-groups", default="1-230")
    parser.add_argument("--export-template-cifs", action="store_true")
    parser.add_argument(
        "--n-workers", type=int, default=0,
        help="CPU workers for the site-symmetry scan; 0 uses available CPU affinity, 1 is serial.",
    )
    parser.add_argument(
        "--cache-dir", default="",
        help="Progress-cache directory; default is OUTPUT_DIR/site_symmetry_cache.",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def _parse_space_groups(text: str) -> list[int]:
    out = set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(token))
    result = sorted(x for x in out if 1 <= x <= 230)
    if not result:
        raise ValueError("No valid space groups selected")
    return result


def main() -> None:
    args = parse_args()
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    if model.get("schema") != "juliette_building_block_xn_v1":
        raise ValueError(f"Unsupported schema {model.get('schema')!r}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    species = model.get("construction_species", [])
    labels = [str(x["label"]) for x in species]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate construction-species labels")
    by_label = {str(x["label"]): x for x in species}
    pair_keys = {
        (str(x["species_i"]), str(x["species_j"]))
        for x in model.get("pair_channels", [])
        if str(x.get("relation")) == "external_neighbor"
    }

    species_rows = []
    hard_errors = []
    for item in species:
        label = str(item["label"])
        block = item["building_block"]
        template = item["external_template"]
        n = int(template["coordination_number"])
        vectors = np.asarray(template.get("canonical_vectors_A", []), float).reshape(-1, 3)
        if len(vectors) != n:
            hard_errors.append(f"{label}: vector count {len(vectors)} != CN {n}")
        expected_angles = n * (n - 1) // 2
        angle_mu = np.asarray(template.get("angular_mean_deg", []), float)
        angle_sigma = np.asarray(template.get("angular_sigma_deg", []), float)
        if len(angle_mu) != expected_angles:
            hard_errors.append(f"{label}: angular target count {len(angle_mu)} != {expected_angles}")
        if len(angle_sigma) != expected_angles:
            hard_errors.append(f"{label}: angular sigma count {len(angle_sigma)} != {expected_angles}")
        observed_angles = _angles(vectors) if n else np.empty(0)
        angular_reconstruction_mae = (
            float(np.mean(np.abs(np.sort(angle_mu) - observed_angles)))
            if len(angle_mu) == len(observed_angles) and len(angle_mu) else 0.0
        )
        radii = np.linalg.norm(vectors, axis=1) if n else np.empty(0)
        partner_labels = [str(x) for x in template.get("allowed_partner_labels", [])]
        missing_partners = [x for x in partner_labels if x not in by_label]
        if missing_partners:
            hard_errors.append(f"{label}: missing partner labels {missing_partners}")
        missing_channels = [x for x in partner_labels if (label, x) not in pair_keys]
        if missing_channels:
            hard_errors.append(f"{label}: missing pair channels to {missing_channels}")
        atom_count = len(block.get("atoms", []))
        if atom_count < 1:
            hard_errors.append(f"{label}: building block contains no physical atoms")
        species_rows.append(
            {
                "label": label,
                "block_kind": str(block.get("kind", "")),
                "physical_atoms": atom_count,
                "template_kind": str(template.get("kind", "")),
                "coordination_number": n,
                "partner_labels": "|".join(partner_labels),
                "radius_mean_A_from_vectors": float(np.mean(radii)) if len(radii) else 0.0,
                "radius_spread_A_from_vectors": float(np.std(radii)) if len(radii) else 0.0,
                "angular_reconstruction_mae_deg": angular_reconstruction_mae,
                "radial_sampling_min_A": float(template.get("radial_sampling_min_A", 0.0)),
                "radial_sampling_max_A": float(template.get("radial_sampling_max_A", 0.0)),
            }
        )

    symmetry_rows = []
    symmetry_error = ""
    selected_space_groups = _parse_space_groups(args.space_groups)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output / "site_symmetry_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _model_fingerprint(model, args.site_symmetry_tolerance)
    species_payload = [
        {
            "label": label,
            "vectors": item["external_template"].get("canonical_vectors_A", []),
        }
        for label, item in by_label.items()
    ]
    cached_count = 0
    pending = []
    for spg in selected_space_groups:
        cache_file = cache_dir / f"spg_{int(spg):03d}.json"
        cached = None if args.rebuild_cache else _load_cached_space_group(
            cache_file, fingerprint, spg
        )
        if cached is not None:
            symmetry_rows.extend(cached)
            cached_count += 1
        else:
            pending.append(int(spg))

    if args.n_workers < 0:
        raise ValueError("--n-workers cannot be negative")
    if args.n_workers == 0:
        try:
            n_workers = max(1, len(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            n_workers = max(1, int(os.cpu_count() or 1))
    else:
        n_workers = max(1, int(args.n_workers))
    n_workers = min(n_workers, max(1, len(pending)))

    print(
        f"Site-symmetry scan: selected={len(selected_space_groups)} "
        f"cached={cached_count} pending={len(pending)} workers={n_workers}"
    )
    try:
        tasks = [
            {
                "spg": spg,
                "tolerance": float(args.site_symmetry_tolerance),
                "species": species_payload,
            }
            for spg in pending
        ]
        completed_new = 0
        if n_workers == 1:
            iterator = map(_scan_space_group_task, tasks)
            executor = None
        else:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
            iterator = executor.map(_scan_space_group_task, tasks, chunksize=1)
        try:
            for result in iterator:
                spg = int(result["spg"])
                rows = result["rows"]
                symmetry_rows.extend(rows)
                _atomic_write_json(
                    cache_dir / f"spg_{spg:03d}.json",
                    {
                        "fingerprint": fingerprint,
                        "spg": spg,
                        "rows": rows,
                    },
                )
                completed_new += 1
                if (
                    completed_new % max(1, int(args.progress_every)) == 0
                    or completed_new == len(pending)
                ):
                    print(
                        f"Site-symmetry scan progress: "
                        f"{cached_count + completed_new}/{len(selected_space_groups)} space groups"
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
    except Exception as exc:
        symmetry_error = f"{type(exc).__name__}: {exc}"

    symmetry_rows.sort(key=lambda row: (int(row["spg"]), int(row["wp_index"]), str(row["label"])))

    with (output / "species_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(species_rows[0]) if species_rows else ["label"])
        writer.writeheader()
        writer.writerows(species_rows)
    if symmetry_rows:
        with (output / "wyckoff_template_compatibility.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(symmetry_rows[0]))
            writer.writeheader()
            writer.writerows(symmetry_rows)

    if args.export_template_cifs:
        try:
            from ase import Atoms
            from ase.io import write

            template_dir = output / "template_cifs"
            template_dir.mkdir(exist_ok=True)
            for label, item in by_label.items():
                vectors = np.asarray(item["external_template"].get("canonical_vectors_A", []), float).reshape(-1, 3)
                block = item["building_block"]
                block_positions = np.asarray([x["position_A"] for x in block["atoms"]], float)
                block_symbols = [x["element"] for x in block["atoms"]]
                side = max(10.0, 2.0 * (np.max(np.linalg.norm(vectors, axis=1)) if len(vectors) else 1.0) + 5.0)
                shift = np.asarray([side / 2] * 3)
                # X is visualized as He, never as a chemistry label.
                atoms = Atoms(
                    block_symbols + ["He"] * len(vectors),
                    positions=np.vstack([block_positions, vectors]) + shift,
                    cell=np.eye(3) * side,
                    pbc=False,
                )
                write(template_dir / f"{label}_template.cif", atoms)
        except Exception as exc:
            hard_errors.append(f"Template CIF export failed: {type(exc).__name__}: {exc}")

    compatible_counts = {}
    for label in labels:
        rows = [x for x in symmetry_rows if x["label"] == label and x["compatible"]]
        compatible_counts[label] = len(rows)
    summary = {
        "model": str(Path(args.model).resolve()),
        "schema_valid": not hard_errors,
        "hard_errors": hard_errors,
        "species_count": len(species),
        "pair_channel_count": len(pair_keys),
        "site_symmetry_tolerance": float(args.site_symmetry_tolerance),
        "site_symmetry_scan_error": symmetry_error,
        "site_symmetry_cache_dir": str(cache_dir),
        "site_symmetry_cached_space_groups": int(cached_count),
        "site_symmetry_scanned_space_groups": int(len(selected_space_groups) - cached_count),
        "site_symmetry_workers": int(n_workers),
        "compatible_wyckoff_entries": compatible_counts,
        "outputs": {
            "species_validation": str(output / "species_validation.csv"),
            "wyckoff_template_compatibility": str(output / "wyckoff_template_compatibility.csv"),
        },
    }
    (output / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if hard_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Inspect the 34-parent TiO2 training database with the same LEGO-Xtal SO3
objective used during augmentation, and compare it with periodic-image-aware
TiO6 and OTi3 geometry/topology descriptors.

This script imports the installed LEGO-Xtal/PyXtal code. It does not duplicate
the SO3 implementation.

Required runtime imports
------------------------
from lego.builder import builder
from pyxtal import pyxtal

Example
-------
python inspect_tio2_training_db_so3_v2.py \
    --db tio2.db \
    --reference-tio2 rutile.cif \
    --output-dir tio2_parent_so3_analysis

Outputs
-------
structure_metrics.csv
ti_site_metrics.csv
o_site_metrics.csv
metric_correlations.csv
summary.txt
plots/*.png
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase.db import connect
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from lego.builder import builder
from pyxtal import pyxtal


EPS = 1.0e-12
IDEAL_OCT_ANGLES = np.asarray([90.0] * 12 + [180.0] * 3, dtype=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare LEGO SO3 scores with TiO6/OTi3 descriptors."
    )
    p.add_argument("--db", required=True, help="ASE database containing the parent structures.")
    p.add_argument(
        "--reference-tio2",
        required=True,
        help="Reference TiO2 CIF used by the production LEGO SO3 workflow.",
    )
    p.add_argument(
        "--output-dir",
        default="tio2_parent_so3_analysis",
        help="Output directory.",
    )
    p.add_argument("--begin", type=int, default=1, help="First ASE DB row id, inclusive.")
    p.add_argument("--end", type=int, default=-1, help="Last ASE DB row id, inclusive; -1 means all.")
    p.add_argument("--rcut", type=float, default=3.0, help="SO3 cutoff. Default: 3.0 Å.")
    p.add_argument("--lmax", type=int, default=4, help="SO3 lmax. Default follows builder: 4.")
    p.add_argument("--nmax", type=int, default=2, help="SO3 nmax. Default follows builder: 2.")
    p.add_argument("--alpha", type=float, default=1.5, help="SO3 alpha. Default follows builder: 1.5.")
    p.add_argument(
        "--weight-on",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use species-sign weighting in SO3. Default: true.",
    )
    p.add_argument(
        "--neighbor-search-radius",
        type=float,
        default=6.0,
        help="Radius used to enumerate periodic neighbour images for local metrics.",
    )
    p.add_argument(
        "--ti-o-sanity-max",
        type=float,
        default=3.0,
        help="Diagnostic maximum allowed sixth Ti-O distance.",
    )
    p.add_argument(
        "--o-ti-sanity-max",
        type=float,
        default=3.0,
        help="Diagnostic maximum allowed third O-Ti distance.",
    )
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def finite_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= EPS or n2 <= EPS:
        return math.nan
    c = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def image_key(index: int, image: tuple[int, int, int]) -> tuple[int, int, int, int]:
    return (int(index), int(image[0]), int(image[1]), int(image[2]))


def periodic_neighbours(
    structure: Structure,
    center_index: int,
    species: str,
    radius: float,
) -> list[dict[str, Any]]:
    center = structure[center_index]
    candidates = []
    for neigh in structure.get_neighbors(center, radius, include_index=True, include_image=True):
        if neigh.specie.symbol != species:
            continue
        idx = int(neigh.index)
        image = tuple(int(v) for v in neigh.image)
        vector = np.asarray(neigh.coords - center.coords, dtype=float)
        candidates.append(
            {
                "index": idx,
                "image": image,
                "key": image_key(idx, image),
                "distance": float(neigh.nn_distance),
                "vector": vector,
            }
        )
    candidates.sort(key=lambda item: item["distance"])
    return candidates


def aggregate(values: list[float], prefix: str) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
            f"{prefix}_p90": math.nan,
        }
    return {
        f"{prefix}_mean": float(a.mean()),
        f"{prefix}_std": float(a.std()),
        f"{prefix}_min": float(a.min()),
        f"{prefix}_max": float(a.max()),
        f"{prefix}_p90": float(np.quantile(a, 0.90)),
    }


def ti_metrics(structure: Structure, ti_index: int, radius: float) -> dict[str, Any]:
    candidates = periodic_neighbours(structure, ti_index, "O", radius)
    if len(candidates) < 6:
        return {"valid": False, "failure": "fewer_than_6_periodic_O_images"}

    shell = candidates[:6]
    distances = np.asarray([x["distance"] for x in shell], dtype=float)
    vectors = [x["vector"] for x in shell]
    angles = np.sort(
        np.asarray(
            [angle_deg(vectors[i], vectors[j]) for i in range(6) for j in range(i + 1, 6)],
            dtype=float,
        )
    )
    shell_gap = candidates[6]["distance"] - candidates[5]["distance"] if len(candidates) > 6 else math.nan
    mean_d = float(distances.mean())
    std_d = float(distances.std())

    return {
        "valid": True,
        "neighbor_instances_json": json.dumps(
            [{"index": x["index"], "image": x["image"], "distance": x["distance"]} for x in shell],
            separators=(",", ":"),
        ),
        "ti_o_mean": mean_d,
        "ti_o_std": std_d,
        "ti_o_cv": std_d / max(mean_d, EPS),
        "ti_o_min": float(distances.min()),
        "ti_o_max": float(distances.max()),
        "ti_o_shell_gap_6_7": float(shell_gap),
        "oct_angle_rms_deg": float(np.sqrt(np.mean((angles - IDEAL_OCT_ANGLES) ** 2))),
        "oct_cis_rms_deg": float(np.sqrt(np.mean((angles[:12] - 90.0) ** 2))),
        "oct_trans_rms_deg": float(np.sqrt(np.mean((angles[12:] - 180.0) ** 2))),
        "oct_angle_variance_deg2": float(np.sum((angles[:12] - 90.0) ** 2) / 11.0),
        "oct_quadratic_elongation": float(np.mean((distances / max(mean_d, EPS)) ** 2)),
        "o_ti_o_min_deg": float(angles.min()),
        "o_ti_o_max_deg": float(angles.max()),
    }


def o_metrics(structure: Structure, o_index: int, radius: float) -> dict[str, Any]:
    candidates = periodic_neighbours(structure, o_index, "Ti", radius)
    if len(candidates) < 3:
        return {"valid": False, "failure": "fewer_than_3_periodic_Ti_images"}

    shell = candidates[:3]
    distances = np.asarray([x["distance"] for x in shell], dtype=float)
    vectors = np.asarray([x["vector"] for x in shell], dtype=float)
    angles = np.asarray(
        [angle_deg(vectors[0], vectors[1]),
         angle_deg(vectors[0], vectors[2]),
         angle_deg(vectors[1], vectors[2])],
        dtype=float,
    )
    shell_gap = candidates[3]["distance"] - candidates[2]["distance"] if len(candidates) > 3 else math.nan

    sides = np.asarray(
        [
            np.linalg.norm(vectors[0] - vectors[1]),
            np.linalg.norm(vectors[0] - vectors[2]),
            np.linalg.norm(vectors[1] - vectors[2]),
        ],
        dtype=float,
    )
    normal = np.cross(vectors[1] - vectors[0], vectors[2] - vectors[0])
    normal_norm = float(np.linalg.norm(normal))
    plane_distance = (
        float(abs(np.dot(normal, -vectors[0])) / normal_norm)
        if normal_norm > EPS else math.nan
    )

    mean_d = float(distances.mean())
    return {
        "valid": True,
        "neighbor_instances_json": json.dumps(
            [{"index": x["index"], "image": x["image"], "distance": x["distance"]} for x in shell],
            separators=(",", ":"),
        ),
        "neighbor_keys": [x["key"] for x in shell],
        "o_ti_mean": mean_d,
        "o_ti_std": float(distances.std()),
        "o_ti_cv": float(distances.std()) / max(mean_d, EPS),
        "o_ti_min": float(distances.min()),
        "o_ti_max": float(distances.max()),
        "o_ti_shell_gap_3_4": float(shell_gap),
        "ti_o_ti_mean_deg": float(angles.mean()),
        "ti_o_ti_std_deg": float(angles.std()),
        "ti_o_ti_min_deg": float(angles.min()),
        "ti_o_ti_max_deg": float(angles.max()),
        "ti_triangle_side_mean": float(sides.mean()),
        "ti_triangle_side_cv": float(sides.std()) / max(float(sides.mean()), EPS),
        "ti_triangle_area": 0.5 * normal_norm,
        "o_to_ti3_plane_distance": plane_distance,
        "o_to_ti3_plane_distance_norm": plane_distance / max(mean_d, EPS),
    }


def topology_metrics(o_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_counts: Counter[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = Counter()
    for row in o_rows:
        if not row.get("valid"):
            continue
        keys = row["neighbor_keys"]
        for i in range(3):
            for j in range(i + 1, 3):
                pair = tuple(sorted((keys[i], keys[j])))
                pair_counts[pair] += 1

    histogram = Counter(pair_counts.values())
    corner = histogram.get(1, 0)
    edge = histogram.get(2, 0)
    face = sum(count for shared, count in histogram.items() if shared >= 3)
    total = corner + edge + face
    return {
        "topology_connected_periodic_ti_pairs": total,
        "topology_corner_pairs": corner,
        "topology_edge_pairs": edge,
        "topology_face_or_over_pairs": face,
        "topology_corner_fraction": corner / max(total, 1),
        "topology_edge_fraction": edge / max(total, 1),
        "topology_face_or_over_fraction": face / max(total, 1),
        "topology_max_shared_o": max(pair_counts.values(), default=0),
        "topology_shared_o_histogram_json": json.dumps(dict(sorted(histogram.items())), separators=(",", ":")),
    }


def compute_lego_so3(bu: builder, atoms) -> tuple[float, pyxtal]:
    xtal = pyxtal()
    xtal.from_seed(atoms, tol=0.1)
    xtal.resort_species(["Ti", "O"])
    return float(bu.get_similarity(xtal)), xtal


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    targets = ["so3_total", "so3_per_atom", "so3_per_ti"]
    excluded = {
        "db_row", "natoms", "n_ti", "n_o", "space_group_number",
        *targets,
    }
    metrics = [c for c in numeric.columns if c not in excluded and not c.startswith("so3_")]
    records = []

    for target in targets:
        for metric in metrics:
            pair = numeric[[target, metric]].dropna()
            if len(pair) < 3 or pair[target].nunique() < 2 or pair[metric].nunique() < 2:
                continue
            pearson = pair[target].corr(pair[metric], method="pearson")
            spearman = pair[target].corr(pair[metric], method="spearman")
            records.append(
                {
                    "so3_target": target,
                    "metric": metric,
                    "n": len(pair),
                    "pearson_r": float(pearson),
                    "spearman_r": float(spearman),
                    "abs_spearman_r": abs(float(spearman)),
                }
            )
    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["so3_target", "abs_spearman_r"], ascending=[True, False])
    return out


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    x_metrics = [
        ("ti_oct_angle_rms_deg_mean", "Mean TiO6 octahedral angle RMS (deg)"),
        ("ti_ti_o_cv_mean", "Mean Ti-O bond-length CV"),
        ("o_ti_o_ti_mean_deg_mean", "Mean Ti-O-Ti angle (deg)"),
        ("o_o_to_ti3_plane_distance_norm_mean", "Mean normalized O-to-Ti3-plane distance"),
        ("topology_edge_fraction", "Edge-sharing fraction"),
        ("topology_face_or_over_fraction", "Face/over-sharing fraction"),
    ]

    for x_name, x_label in x_metrics:
        if x_name not in df.columns:
            continue
        pair = df[[x_name, "so3_per_ti"]].dropna()
        if len(pair) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.scatter(pair[x_name], pair["so3_per_ti"])
        ax.set_xlabel(x_label)
        ax.set_ylabel("LEGO SO3 objective per Ti")
        ax.set_title(f"SO3 per Ti vs {x_label}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"so3_per_ti_vs_{x_name}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    ref_path = Path(args.reference_tio2)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    if not ref_path.is_file():
        raise FileNotFoundError(ref_path)

    # Use the same builder and element-specific reference route as production.
    bu = builder(
        ["Ti", "O"],
        [1, 2],
        db_file=str(outdir / "_unused_inspector.db"),
        log_file=str(outdir / "_inspector.log"),
        verbose=False,
    )
    bu.set_descriptor_calculator(
        mykwargs={
            "rcut": args.rcut,
            "lmax": args.lmax,
            "nmax": args.nmax,
            "alpha": args.alpha,
            "weight_on": args.weight_on,
        }
    )
    bu.set_reference_enviroments(str(ref_path))

    adaptor = AseAtomsAdaptor()
    structures = []
    ti_all = []
    o_all = []
    failures = []

    with connect(db_path) as db:
        rows = list(db.select())

    selected = [
        row for row in rows
        if row.id >= args.begin and (args.end < 0 or row.id <= args.end)
    ]
    print(f"Selected parent structures: {len(selected)}")
    print(f"SO3: rcut={args.rcut}, lmax={args.lmax}, nmax={args.nmax}, alpha={args.alpha}, weight_on={args.weight_on}")

    for row in selected:
        try:
            atoms = row.toatoms()
            so3, xtal = compute_lego_so3(bu, atoms)
            structure = adaptor.get_structure(atoms)
            ti_indices = [i for i, site in enumerate(structure) if site.specie.symbol == "Ti"]
            o_indices = [i for i, site in enumerate(structure) if site.specie.symbol == "O"]

            ti_rows = []
            for local_id, idx in enumerate(ti_indices):
                rec = {
                    "db_row": row.id,
                    "label": getattr(row, "label", f"row_{row.id}"),
                    "ti_local_id": local_id,
                    "ti_structure_index": idx,
                    **ti_metrics(structure, idx, args.neighbor_search_radius),
                }
                rec["shell_sanity_ok"] = bool(rec.get("valid")) and rec.get("ti_o_max", math.inf) <= args.ti_o_sanity_max
                ti_rows.append(rec)

            o_rows = []
            for local_id, idx in enumerate(o_indices):
                rec = {
                    "db_row": row.id,
                    "label": getattr(row, "label", f"row_{row.id}"),
                    "o_local_id": local_id,
                    "o_structure_index": idx,
                    **o_metrics(structure, idx, args.neighbor_search_radius),
                }
                rec["shell_sanity_ok"] = bool(rec.get("valid")) and rec.get("o_ti_max", math.inf) <= args.o_ti_sanity_max
                o_rows.append(rec)

            base = {
                "db_row": row.id,
                "label": getattr(row, "label", f"row_{row.id}"),
                "formula": structure.composition.reduced_formula,
                "space_group_number": int(xtal.group.number),
                "natoms": len(structure),
                "n_ti": len(ti_indices),
                "n_o": len(o_indices),
                "volume_per_atom": structure.volume / len(structure),
                "so3_total": so3,
                "so3_per_atom": so3 / len(structure),
                "so3_per_ti": so3 / max(len(ti_indices), 1),
                "so3_per_o": so3 / max(len(o_indices), 1),
                "ti_shell_sanity_fraction": float(np.mean([r["shell_sanity_ok"] for r in ti_rows])),
                "o_shell_sanity_fraction": float(np.mean([r["shell_sanity_ok"] for r in o_rows])),
            }

            for metric in [
                "ti_o_mean", "ti_o_std", "ti_o_cv", "ti_o_shell_gap_6_7",
                "oct_angle_rms_deg", "oct_cis_rms_deg", "oct_trans_rms_deg",
                "oct_angle_variance_deg2", "oct_quadratic_elongation",
            ]:
                base.update(aggregate([finite_float(r.get(metric)) for r in ti_rows], f"ti_{metric}"))

            for metric in [
                "o_ti_mean", "o_ti_std", "o_ti_cv", "o_ti_shell_gap_3_4",
                "ti_o_ti_mean_deg", "ti_o_ti_std_deg",
                "ti_triangle_side_cv", "ti_triangle_area",
                "o_to_ti3_plane_distance", "o_to_ti3_plane_distance_norm",
            ]:
                base.update(aggregate([finite_float(r.get(metric)) for r in o_rows], f"o_{metric}"))

            base.update(topology_metrics(o_rows))
            structures.append(base)

            for r in ti_rows:
                r.pop("neighbor_keys", None)
            for r in o_rows:
                r.pop("neighbor_keys", None)
            ti_all.extend(ti_rows)
            o_all.extend(o_rows)

            print(
                f"row {row.id:3d}: spg={xtal.group.number:3d} "
                f"SO3={so3:12.6f} SO3/Ti={base['so3_per_ti']:10.6f} "
                f"octRMS={base['ti_oct_angle_rms_deg_mean']:8.3f} "
                f"edge={base['topology_edge_fraction']:6.3f} "
                f"face={base['topology_face_or_over_fraction']:6.3f}"
            )

        except Exception as exc:
            failures.append(
                {"db_row": row.id, "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"row {row.id:3d}: FAILED: {type(exc).__name__}: {exc}")

    structure_df = pd.DataFrame(structures)
    ti_df = pd.DataFrame(ti_all)
    o_df = pd.DataFrame(o_all)
    fail_df = pd.DataFrame(failures)

    structure_df.to_csv(outdir / "structure_metrics.csv", index=False)
    ti_df.to_csv(outdir / "ti_site_metrics.csv", index=False)
    o_df.to_csv(outdir / "o_site_metrics.csv", index=False)
    fail_df.to_csv(outdir / "failures.csv", index=False)

    correlations = correlation_table(structure_df)
    correlations.to_csv(outdir / "metric_correlations.csv", index=False)

    if not args.no_plots and not structure_df.empty:
        make_plots(structure_df, outdir)

    lines = [
        f"Structures analyzed: {len(structure_df)}",
        f"Failures: {len(fail_df)}",
        "",
    ]
    if not structure_df.empty:
        for key in ["so3_total", "so3_per_atom", "so3_per_ti"]:
            q = structure_df[key].quantile([0, 0.25, 0.5, 0.75, 1]).to_dict()
            lines.append(
                f"{key}: min={q[0]:.6g}, q25={q[0.25]:.6g}, "
                f"median={q[0.5]:.6g}, q75={q[0.75]:.6g}, max={q[1]:.6g}"
            )
    if not correlations.empty:
        lines += ["", "Top SO3-per-Ti correlations (Spearman):"]
        top = correlations[correlations.so3_target == "so3_per_ti"].head(15)
        for _, r in top.iterrows():
            lines.append(
                f"{r.metric}: rho={r.spearman_r:.4f}, "
                f"Pearson={r.pearson_r:.4f}, n={int(r.n)}"
            )

    (outdir / "summary.txt").write_text("\n".join(lines) + "\n")

    print(f"\nOutputs written to: {outdir}")
    print(f"Main table: {outdir / 'structure_metrics.csv'}")
    print(f"Correlations: {outdir / 'metric_correlations.csv'}")
    print(f"Summary: {outdir / 'summary.txt'}")


if __name__ == "__main__":
    main()


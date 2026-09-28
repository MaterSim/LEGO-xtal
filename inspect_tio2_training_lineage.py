#!/usr/bin/env python3
"""Trace Ti-O coordination lineage through the LEGO-Xtal training-data path.

The script dynamically imports the current 0_make_traindata.py and reuses its
actual helper functions for:
    - configuration loading
    - target-label assignment
    - builder creation
    - tabular representation generation
    - augmentation validation
    - orbit expansion
    - cell reconstruction

The purpose is diagnostic, not training-data production.

Stages traced
-------------
raw_db
    Direct ASE DB row.toatoms(); no PyXtal reconstruction.

loaded_pyxtal
    database_topology.get_pyxtal() / get_all_xtals()-equivalent reconstruction.

optimized_parent
    builder.optimize_xtal(parent).

initial_representation
    Accepted parent get_tabular_representations() rows after
    append_target_coordination().

subgroup_preopt
    Valid subgroup xtal before SO3 optimisation.

subgroup_optimized
    Accepted subgroup xtal after SO3 optimisation.

subgroup_representation
    Accepted subgroup tabular representations after
    append_target_coordination().

Outputs
-------
lineage_stage_records.csv
    One row per source/stage/object.

lineage_source_summary.csv
    Aggregated source-row lineage, including representation-weighted final CN.

lineage_site_records.csv
    One row per Ti site for every traced object.

lineage_summary.json
    Global stage-level CN distributions and transition diagnostics.

mutation_examples/
    CIFs for selected source rows where a CN class appears downstream but was
    absent from the raw DB structure.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase.db import connect
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pyxtal.db import database_topology
from pyxtal.util import new_struc_wo_energy


def _load_training_module(path: str):
    path_obj = Path(path).resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(path_obj)
    spec = importlib.util.spec_from_file_location("juliette_make_traindata", path_obj)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import training module from {path_obj}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _periodic_species_neighbors(
    structure,
    center_index: int,
    neighbor_species: str,
    radius: float,
) -> list[dict[str, Any]]:
    center = structure[center_index]
    neighbors = []
    for neighbor in structure.get_neighbors(
        center,
        radius,
        include_index=True,
        include_image=True,
    ):
        if str(neighbor.specie.symbol) != str(neighbor_species):
            continue
        neighbors.append(
            {
                "distance": float(neighbor.nn_distance),
                "atom_index": int(neighbor.index),
                "image": tuple(int(x) for x in neighbor.image),
            }
        )
    neighbors.sort(
        key=lambda item: (
            item["distance"],
            item["atom_index"],
            item["image"],
        )
    )
    return neighbors


def _analyse_structure(
    structure,
    source_db_id: int,
    source_index: int,
    stage: str,
    object_index: int,
    parent_object_index: int | None,
    center_species: str,
    neighbor_species: str,
    hard_cutoff: float,
    search_radius: float,
    metadata: dict[str, Any] | None = None,
):
    symbols = [str(site.specie.symbol) for site in structure]
    ti_indices = [
        index for index, symbol in enumerate(symbols)
        if symbol == center_species
    ]
    site_records = []
    cn_counter = Counter()

    for local_index, atom_index in enumerate(ti_indices):
        neighbors = _periodic_species_neighbors(
            structure,
            atom_index,
            neighbor_species,
            search_radius,
        )
        distances = [item["distance"] for item in neighbors]
        hard_cn = int(
            sum(distance <= hard_cutoff + 1.0e-12 for distance in distances)
        )
        cn_counter[hard_cn] += 1

        site_records.append(
            {
                "source_db_id": int(source_db_id),
                "source_index": int(source_index),
                "stage": str(stage),
                "object_index": int(object_index),
                "parent_object_index": (
                    int(parent_object_index)
                    if parent_object_index is not None
                    else np.nan
                ),
                "ti_local_index": int(local_index),
                "ti_atom_index": int(atom_index),
                "hard_cn": int(hard_cn),
                "hard_cutoff_A": float(hard_cutoff),
                "o_distances_within_search_A": json.dumps(
                    distances,
                    separators=(",", ":"),
                ),
            }
        )

    record = {
        "source_db_id": int(source_db_id),
        "source_index": int(source_index),
        "stage": str(stage),
        "object_index": int(object_index),
        "parent_object_index": (
            int(parent_object_index)
            if parent_object_index is not None
            else np.nan
        ),
        "natoms": int(len(structure)),
        "n_center": int(len(ti_indices)),
        "n_neighbor": int(symbols.count(neighbor_species)),
        "hard_cn_counts": json.dumps(
            dict(sorted(cn_counter.items())),
            separators=(",", ":"),
        ),
        "hard_cn6_fraction": (
            float(cn_counter.get(6, 0) / len(ti_indices))
            if ti_indices else np.nan
        ),
        "hard_cn_min": min(cn_counter) if cn_counter else np.nan,
        "hard_cn_max": max(cn_counter) if cn_counter else np.nan,
    }
    if metadata:
        record.update(metadata)

    return record, site_records, cn_counter


def _representation_to_structure(
    module,
    representation_with_labels,
    n_wp: int,
    species: list[str],
    config: dict[str, dict[str, Any]],
):
    representation_with_labels = np.asarray(
        representation_with_labels,
        dtype=float,
    ).reshape(-1)
    expected_geom_width = 7 + 4 * n_wp
    expected_total_width = expected_geom_width + n_wp
    if len(representation_with_labels) != expected_total_width:
        raise ValueError(
            f"Representation width {len(representation_with_labels)}; "
            f"expected {expected_total_width}."
        )

    geometry = representation_with_labels[:expected_geom_width]
    labels = np.rint(
        representation_with_labels[expected_geom_width:expected_total_width]
    ).astype(int)

    label_to_species = {
        int(config[symbol]["coordination"]): symbol for symbol in species
    }
    occupied = module._expand_representation_orbits(geometry, n_wp)
    cell = module._cell_matrix_from_representation(geometry)

    frac_coords = []
    atom_species = []
    for slot, frac in occupied:
        label = int(labels[slot])
        if label not in label_to_species:
            raise ValueError(
                f"Occupied slot {slot} has target label {label}; "
                f"known labels={sorted(label_to_species)}"
            )
        symbol = label_to_species[label]
        frac_coords.extend(np.asarray(frac, dtype=float).tolist())
        atom_species.extend([symbol] * len(frac))

    if not frac_coords:
        raise ValueError("Representation expands to zero atoms")

    return Structure(
        Lattice(cell),
        atom_species,
        np.asarray(frac_coords, dtype=float),
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


def _write_structure_cif(structure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(path))


def _parse_cn_counts(value: str) -> Counter:
    raw = json.loads(value)
    return Counter({int(key): int(count) for key, count in raw.items()})


def _counter_json(counter: Counter) -> str:
    return json.dumps(
        dict(sorted((int(k), int(v)) for k, v in counter.items())),
        separators=(",", ":"),
    )


def _stage_global_summary(stage_records: pd.DataFrame):
    output = {}
    for stage, subset in stage_records.groupby("stage", sort=False):
        counter = Counter()
        for value in subset["hard_cn_counts"]:
            counter.update(_parse_cn_counts(value))
        total = sum(counter.values())
        output[str(stage)] = {
            "objects": int(len(subset)),
            "ti_sites": int(total),
            "hard_cn_counts": dict(sorted(counter.items())),
            "hard_cn_fractions": {
                str(cn): float(count / total)
                for cn, count in sorted(counter.items())
            } if total else {},
            "cn6_fraction": float(counter.get(6, 0) / total)
            if total else None,
        }
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trace Ti-O CN lineage through the current 0_make_traindata.py path."
        )
    )
    parser.add_argument("--database", default="data/source/tio2.db")
    parser.add_argument(
        "--make-traindata",
        default="0_make_traindata.py",
        help="Path to the exact current training-data script.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/audit/training_lineage",
    )
    parser.add_argument("--tag", default="lineage-audit")
    parser.add_argument("--max_atoms", type=int, default=500)
    parser.add_argument("--min_spg", type=int, default=0)
    parser.add_argument("--max_dof", type=int, default=24)
    parser.add_argument("--max_wp", type=int, default=8)
    parser.add_argument("--max_energy", type=float, default=float("inf"))
    parser.add_argument("--max_per_struc", type=int, default=500)
    parser.add_argument("--rcut", type=float, default=2.4)
    parser.add_argument("--discrete", type=int)
    parser.add_argument("--discrete_cell", action="store_true")
    parser.add_argument("--subgroup-eps", type=float, default=5.0e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hard-cutoff", type=float, default=3.0)
    parser.add_argument("--neighbor-search-radius", type=float, default=6.0)

    parser.add_argument(
        "--no-integrity-filter",
        action="store_true",
        help="Match training run with integrity filter disabled.",
    )
    parser.add_argument("--integrity-neighbor-radius", type=float, default=6.0)
    parser.add_argument("--center-max-neighbor-distance", type=float, default=2.6)
    parser.add_argument("--center-min-shell-gap", type=float, default=0.15)
    parser.add_argument("--center-max-angle-rms", type=float, default=20.0)
    parser.add_argument("--neighbor-max-center-distance", type=float, default=2.6)
    parser.add_argument("--neighbor-min-shell-gap", type=float, default=0.10)
    parser.add_argument(
        "--neighbor-max-angle-rms",
        type=float,
        default=float("inf"),
    )
    parser.add_argument("--min-center-pass-fraction", type=float, default=1.0)
    parser.add_argument("--min-neighbor-pass-fraction", type=float, default=1.0)
    parser.add_argument("--max-so3-per-center", type=float, default=5.0)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--coord-ref-dict")
    group.add_argument("--coord-ref-file")
    parser.add_argument("--composition", default="1,2")

    parser.add_argument(
        "--export-mutation-sources",
        type=int,
        default=10,
        help=(
            "Export up to this many source lineages where a CN class appears "
            "downstream but is absent in raw_db."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.database):
        raise FileNotFoundError(args.database)

    module = _load_training_module(args.make_traindata)
    config, species = module.load_coord_ref_config(
        args.coord_ref_dict,
        args.coord_ref_file,
    )
    composition = module.parse_composition(args.composition, species)
    center_species, neighbor_species = species

    use_discrete = args.discrete is not None
    if args.discrete_cell and not use_discrete:
        args.discrete_cell = False

    integrity = {
        "enabled": not args.no_integrity_filter,
        "neighbor_search_radius": float(args.integrity_neighbor_radius),
        "center_max_neighbor_distance": float(
            args.center_max_neighbor_distance
        ),
        "center_min_shell_gap": float(args.center_min_shell_gap),
        "center_max_angle_rms": float(args.center_max_angle_rms),
        "neighbor_max_center_distance": float(
            args.neighbor_max_center_distance
        ),
        "neighbor_min_shell_gap": float(args.neighbor_min_shell_gap),
        "neighbor_max_angle_rms": float(args.neighbor_max_angle_rms),
        "min_center_pass_fraction": float(args.min_center_pass_fraction),
        "min_neighbor_pass_fraction": float(args.min_neighbor_pass_fraction),
        "max_so3_per_center": float(args.max_so3_per_center),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_db = connect(args.database, serial=True)
    rows = list(raw_db.select())
    db = database_topology(
        args.database,
        log_file=str(output_dir / "lineage_db.log"),
    )

    print("--- Training lineage audit ---")
    print(f"Database: {args.database}")
    print(f"Training module: {Path(args.make_traindata).resolve()}")
    print(f"Rows: {len(rows)}")
    print(f"Species/composition: {species} / {composition}")
    print(
        "Role labels: "
        f"{ {s: config[s]['coordination'] for s in species} }"
    )
    print(f"SO3 cutoff: {args.rcut}")
    print(f"Hard CN cutoff: {args.hard_cutoff}")
    print(f"Integrity: {integrity}")
    print("------------------------------")

    stage_records = []
    site_records = []
    failures = []
    object_serial = 0

    raw_structures_by_source = {}
    mutation_structures_by_source = defaultdict(list)

    for source_index, row in enumerate(rows):
        source_db_id = int(row.id)
        random.seed(args.seed + source_index)
        np.random.seed(args.seed + source_index)

        print(
            f"\nSource {source_index}/{len(rows)-1} "
            f"(DB row {source_db_id})"
        )

        try:
            # Stage 1: raw ASE DB structure.
            raw_atoms = row.toatoms()
            raw_structure = AseAtomsAdaptor.get_structure(raw_atoms)
            raw_structures_by_source[source_index] = raw_structure
            object_serial += 1
            record, sites, raw_cn = _analyse_structure(
                raw_structure,
                source_db_id,
                source_index,
                "raw_db",
                object_serial,
                None,
                center_species,
                neighbor_species,
                args.hard_cutoff,
                args.neighbor_search_radius,
                {"training_filter_pass": True},
            )
            stage_records.append(record)
            site_records.extend(sites)
            print(f"  raw_db CN={dict(sorted(raw_cn.items()))}")

            # Stage 2: exact database_topology reconstruction used by training.
            xtal = db.get_pyxtal(source_db_id)
            if xtal is None:
                failures.append(
                    {
                        "source_db_id": source_db_id,
                        "source_index": source_index,
                        "stage": "loaded_pyxtal",
                        "error": "database_topology.get_pyxtal returned None",
                    }
                )
                print("  loaded_pyxtal FAILED")
                continue

            loaded_structure = xtal.to_pymatgen()
            object_serial += 1
            record, sites, loaded_cn = _analyse_structure(
                loaded_structure,
                source_db_id,
                source_index,
                "loaded_pyxtal",
                object_serial,
                None,
                center_species,
                neighbor_species,
                args.hard_cutoff,
                args.neighbor_search_radius,
                {},
            )
            stage_records.append(record)
            site_records.extend(sites)
            print(
                "  loaded_pyxtal CN="
                f"{dict(sorted(loaded_cn.items()))}"
            )

            module.assign_configured_templates(xtal, config)
            module.target_coordination_vector(xtal, args.max_wp)

            atom_count = sum(xtal.numIons)
            ff_energy = getattr(xtal, "ff_energy", None)
            filter_by_energy = np.isfinite(args.max_energy)
            training_filter_pass = bool(
                xtal.dof <= args.max_dof
                and 1 <= atom_count <= args.max_atoms
                and (
                    not filter_by_energy
                    or (
                        ff_energy is not None
                        and ff_energy <= args.max_energy
                    )
                )
                and xtal.group.number >= args.min_spg
                and len(xtal.atom_sites) <= args.max_wp
            )
            if not training_filter_pass:
                print("  rejected by parent training filter")
                continue

            bu = module.make_builder(
                config,
                species,
                composition,
                args.rcut,
            )

            # Stage 3: parent after SO3 optimisation.
            xtal_opt, sim_parent, _ = bu.optimize_xtal(
                xtal,
                add_db=False,
            )
            if xtal_opt is None or not xtal_opt.check_validity(bu.criteria):
                failures.append(
                    {
                        "source_db_id": source_db_id,
                        "source_index": source_index,
                        "stage": "optimized_parent",
                        "error": "SO3 optimize_xtal failed or validity failed",
                    }
                )
                print("  optimized_parent FAILED")
                continue

            if integrity["enabled"]:
                accepted, integrity_report = module.evaluate_local_integrity(
                    xtal_opt,
                    config,
                    species,
                    sim_parent,
                    integrity,
                    source_index=source_index,
                    stage="parent",
                )
                if not accepted:
                    print(
                        "  optimized_parent integrity rejected: "
                        f"{integrity_report['rejection_reasons']}"
                    )
                    continue

            parent_structure = xtal_opt.to_pymatgen()
            object_serial += 1
            parent_object_index = object_serial
            record, sites, parent_cn = _analyse_structure(
                parent_structure,
                source_db_id,
                source_index,
                "optimized_parent",
                object_serial,
                None,
                center_species,
                neighbor_species,
                args.hard_cutoff,
                args.neighbor_search_radius,
                {
                    "similarity": (
                        float(sim_parent)
                        if sim_parent is not None
                        else np.nan
                    ),
                },
            )
            stage_records.append(record)
            site_records.extend(sites)
            print(
                "  optimized_parent CN="
                f"{dict(sorted(parent_cn.items()))}"
            )

            # Stage 4: accepted initial parent representations.
            n_wps = len(xtal_opt.atom_sites)
            n_max_initial = max(
                1,
                int(
                    0.6
                    * args.max_per_struc
                    * np.ceil(n_wps / args.max_wp)
                ),
            )
            initial = xtal_opt.get_tabular_representations(
                N_wp=args.max_wp,
                N_max=n_max_initial,
                discrete=use_discrete,
                discrete_cell=args.discrete_cell,
                N_grids=args.discrete if use_discrete else None,
            ) or []
            diagnostics = {}
            initial = module.append_target_coordination(
                initial,
                xtal_opt,
                args.max_wp,
                species,
                config,
                diagnostics,
            )

            initial_cn_total = Counter()
            for rep_index, representation in enumerate(initial):
                rep_structure = _representation_to_structure(
                    module,
                    representation,
                    args.max_wp,
                    species,
                    config,
                )
                object_serial += 1
                record, sites, rep_cn = _analyse_structure(
                    rep_structure,
                    source_db_id,
                    source_index,
                    "initial_representation",
                    object_serial,
                    parent_object_index,
                    center_species,
                    neighbor_species,
                    args.hard_cutoff,
                    args.neighbor_search_radius,
                    {
                        "representation_local_index": int(rep_index),
                        "augmentation_candidates": int(
                            diagnostics.get("augmentation_candidates", 0)
                        ),
                        "augmentation_rejected": int(
                            diagnostics.get("augmentation_rejected", 0)
                        ),
                    },
                )
                stage_records.append(record)
                site_records.extend(sites)
                initial_cn_total.update(rep_cn)

                for cn in rep_cn:
                    if cn not in raw_cn:
                        mutation_structures_by_source[source_index].append(
                            (
                                f"initial_representation_cn{cn}_rep{rep_index}",
                                rep_structure,
                            )
                        )

            print(
                f"  initial reps={len(initial)} "
                f"weighted CN={dict(sorted(initial_cn_total.items()))} "
                f"validation={diagnostics}"
            )

            # Stage 5+: exact subgroup loop from training code.
            max_cell_factor = max(
                args.max_atoms / sum(xtal_opt.numIons),
                1.0,
            )
            trial_cache = [xtal_opt]
            reps_so_far = len(initial)
            subgroup_counter = 0
            subgroup_rep_counter = 0
            stop = False

            for group_type in ("t", "k"):
                if stop:
                    break
                for trial_index in range(20):
                    if reps_so_far >= args.max_per_struc:
                        stop = True
                        break

                    xtal_sub = xtal_opt.subgroup_once(
                        eps=args.subgroup_eps,
                        group_type=group_type,
                        max_cell=max_cell_factor,
                        mut_lat=False,
                    )
                    if xtal_sub is None:
                        xtal0 = xtal_opt.subgroup_once(group_type="t")
                        if xtal0 is not None:
                            xtal_sub = xtal0.subgroup_once(
                                eps=args.subgroup_eps,
                                group_type="t",
                                max_cell=max_cell_factor,
                                mut_lat=False,
                            )
                    if xtal_sub is None:
                        continue

                    para = xtal_sub.lattice.get_para(degree=True)
                    if not (
                        xtal_sub.get_dof() <= args.max_dof
                        and len(xtal_sub.atom_sites) <= args.max_wp
                        and max(para[:3]) < 50
                        and min(para[3:]) > 30
                        and max(para[3:]) < 150
                    ):
                        continue

                    module.target_coordination_vector(
                        xtal_sub,
                        args.max_wp,
                    )
                    if not new_struc_wo_energy(
                        xtal_sub,
                        trial_cache,
                        0.025,
                        0.025,
                        1.0,
                    ):
                        continue

                    subgroup_counter += 1
                    sub_pre_structure = xtal_sub.to_pymatgen()
                    object_serial += 1
                    sub_pre_object_index = object_serial
                    record, sites, sub_pre_cn = _analyse_structure(
                        sub_pre_structure,
                        source_db_id,
                        source_index,
                        "subgroup_preopt",
                        object_serial,
                        parent_object_index,
                        center_species,
                        neighbor_species,
                        args.hard_cutoff,
                        args.neighbor_search_radius,
                        {
                            "group_type": group_type,
                            "subgroup_trial_index": int(trial_index),
                            "subgroup_local_index": int(subgroup_counter - 1),
                        },
                    )
                    stage_records.append(record)
                    site_records.extend(sites)

                    try:
                        xtal_sub_opt, sim_sub, _ = bu.optimize_xtal(
                            xtal_sub,
                            add_db=False,
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "source_db_id": source_db_id,
                                "source_index": source_index,
                                "stage": "subgroup_optimized",
                                "error": (
                                    f"{type(exc).__name__}: {exc}"
                                ),
                            }
                        )
                        continue

                    if (
                        xtal_sub_opt is None
                        or not xtal_sub_opt.check_validity(bu.criteria)
                    ):
                        continue

                    if integrity["enabled"]:
                        accepted, integrity_report = (
                            module.evaluate_local_integrity(
                                xtal_sub_opt,
                                config,
                                species,
                                sim_sub,
                                integrity,
                                source_index=source_index,
                                stage=f"subgroup_{group_type}",
                            )
                        )
                        if not accepted:
                            continue

                    trial_cache.append(xtal_sub_opt)

                    sub_opt_structure = xtal_sub_opt.to_pymatgen()
                    object_serial += 1
                    sub_opt_object_index = object_serial
                    record, sites, sub_opt_cn = _analyse_structure(
                        sub_opt_structure,
                        source_db_id,
                        source_index,
                        "subgroup_optimized",
                        object_serial,
                        sub_pre_object_index,
                        center_species,
                        neighbor_species,
                        args.hard_cutoff,
                        args.neighbor_search_radius,
                        {
                            "group_type": group_type,
                            "subgroup_trial_index": int(trial_index),
                            "subgroup_local_index": int(subgroup_counter - 1),
                            "similarity": (
                                float(sim_sub)
                                if sim_sub is not None
                                else np.nan
                            ),
                        },
                    )
                    stage_records.append(record)
                    site_records.extend(sites)

                    for cn in sub_opt_cn:
                        if cn not in raw_cn:
                            mutation_structures_by_source[
                                source_index
                            ].append(
                                (
                                    (
                                        f"subgroup_optimized_cn{cn}_"
                                        f"{group_type}{trial_index}"
                                    ),
                                    sub_opt_structure,
                                )
                            )

                    n_max_sub = max(
                        1,
                        int(
                            0.2
                            * args.max_per_struc
                            * np.ceil(
                                len(xtal_sub_opt.atom_sites)
                                / args.max_wp
                            )
                        ),
                    )
                    sub = xtal_sub_opt.get_tabular_representations(
                        N_wp=args.max_wp,
                        N_max=n_max_sub,
                        discrete=use_discrete,
                        discrete_cell=args.discrete_cell,
                        N_grids=args.discrete if use_discrete else None,
                    ) or []
                    sub_diagnostics = {}
                    sub = module.append_target_coordination(
                        sub,
                        xtal_sub_opt,
                        args.max_wp,
                        species,
                        config,
                        sub_diagnostics,
                    )

                    for rep_local_index, representation in enumerate(sub):
                        if reps_so_far >= args.max_per_struc:
                            stop = True
                            break

                        rep_structure = _representation_to_structure(
                            module,
                            representation,
                            args.max_wp,
                            species,
                            config,
                        )
                        object_serial += 1
                        record, sites, rep_cn = _analyse_structure(
                            rep_structure,
                            source_db_id,
                            source_index,
                            "subgroup_representation",
                            object_serial,
                            sub_opt_object_index,
                            center_species,
                            neighbor_species,
                            args.hard_cutoff,
                            args.neighbor_search_radius,
                            {
                                "group_type": group_type,
                                "subgroup_trial_index": int(trial_index),
                                "subgroup_local_index": int(
                                    subgroup_counter - 1
                                ),
                                "representation_local_index": int(
                                    rep_local_index
                                ),
                                "subgroup_representation_global_index": int(
                                    subgroup_rep_counter
                                ),
                                "augmentation_candidates": int(
                                    sub_diagnostics.get(
                                        "augmentation_candidates",
                                        0,
                                    )
                                ),
                                "augmentation_rejected": int(
                                    sub_diagnostics.get(
                                        "augmentation_rejected",
                                        0,
                                    )
                                ),
                            },
                        )
                        stage_records.append(record)
                        site_records.extend(sites)
                        subgroup_rep_counter += 1
                        reps_so_far += 1

                        for cn in rep_cn:
                            if cn not in raw_cn:
                                mutation_structures_by_source[
                                    source_index
                                ].append(
                                    (
                                        (
                                            f"subgroup_representation_cn{cn}_"
                                            f"{group_type}{trial_index}_"
                                            f"rep{rep_local_index}"
                                        ),
                                        rep_structure,
                                    )
                                )

            print(
                f"  subgroup accepted candidates={subgroup_counter}; "
                f"subgroup reps={subgroup_rep_counter}; "
                f"total accepted reps={reps_so_far}"
            )

        except Exception as exc:
            failures.append(
                {
                    "source_db_id": source_db_id,
                    "source_index": source_index,
                    "stage": "source_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                "  SOURCE FAILED: "
                f"{type(exc).__name__}: {exc}"
            )

    stage_df = pd.DataFrame(stage_records)
    site_df = pd.DataFrame(site_records)
    failures_df = pd.DataFrame(failures)

    if stage_df.empty:
        raise RuntimeError("No lineage stage records produced")

    stage_df = stage_df.sort_values(
        ["source_index", "object_index"],
        kind="stable",
    )
    site_df = site_df.sort_values(
        [
            "source_index",
            "object_index",
            "ti_local_index",
        ],
        kind="stable",
    )

    stage_file = output_dir / "lineage_stage_records.csv"
    site_file = output_dir / "lineage_site_records.csv"
    failures_file = output_dir / "lineage_failures.csv"
    source_file = output_dir / "lineage_source_summary.csv"
    summary_file = output_dir / "lineage_summary.json"

    stage_df.to_csv(stage_file, index=False)
    site_df.to_csv(site_file, index=False)
    if not failures_df.empty:
        failures_df.to_csv(failures_file, index=False)

    source_rows = []
    final_stages = {
        "initial_representation",
        "subgroup_representation",
    }

    for source_index, subset in stage_df.groupby(
        "source_index",
        sort=True,
    ):
        source_db_id = int(subset["source_db_id"].iloc[0])

        raw_subset = subset[subset["stage"] == "raw_db"]
        loaded_subset = subset[subset["stage"] == "loaded_pyxtal"]
        parent_subset = subset[subset["stage"] == "optimized_parent"]
        final_subset = subset[subset["stage"].isin(final_stages)]

        def combined_counter(frame):
            counter = Counter()
            for value in frame["hard_cn_counts"]:
                counter.update(_parse_cn_counts(value))
            return counter

        raw_counter = combined_counter(raw_subset)
        loaded_counter = combined_counter(loaded_subset)
        parent_counter = combined_counter(parent_subset)
        final_counter = combined_counter(final_subset)
        final_total = sum(final_counter.values())

        new_classes = sorted(set(final_counter) - set(raw_counter))
        lost_classes = sorted(set(raw_counter) - set(final_counter))

        source_rows.append(
            {
                "source_db_id": source_db_id,
                "source_index": int(source_index),
                "raw_cn_counts": _counter_json(raw_counter),
                "loaded_pyxtal_cn_counts": _counter_json(loaded_counter),
                "optimized_parent_cn_counts": _counter_json(parent_counter),
                "accepted_representation_count": int(len(final_subset)),
                "final_weighted_cn_counts": _counter_json(final_counter),
                "final_weighted_cn6_fraction": (
                    float(final_counter.get(6, 0) / final_total)
                    if final_total else np.nan
                ),
                "new_cn_classes_vs_raw": json.dumps(
                    new_classes,
                    separators=(",", ":"),
                ),
                "lost_cn_classes_vs_raw": json.dumps(
                    lost_classes,
                    separators=(",", ":"),
                ),
                "raw_to_loaded_changed": bool(
                    raw_counter != loaded_counter
                ),
                "raw_to_parent_changed": bool(
                    raw_counter != parent_counter
                ),
                "raw_to_final_new_cn_class": bool(new_classes),
            }
        )

    source_df = pd.DataFrame(source_rows).sort_values(
        ["source_index"],
        kind="stable",
    )
    source_df.to_csv(source_file, index=False)

    global_stage_summary = _stage_global_summary(stage_df)
    mutation_sources = source_df[
        source_df["raw_to_final_new_cn_class"]
    ]["source_index"].astype(int).tolist()

    summary = {
        "database": os.path.abspath(args.database),
        "training_module": str(Path(args.make_traindata).resolve()),
        "species": species,
        "composition": composition,
        "hard_cutoff_A": float(args.hard_cutoff),
        "neighbor_search_radius_A": float(args.neighbor_search_radius),
        "integrity": integrity,
        "source_rows": len(rows),
        "analysed_stage_objects": int(len(stage_df)),
        "analysed_site_records": int(len(site_df)),
        "failures": int(len(failures_df)),
        "stage_summary": global_stage_summary,
        "sources_raw_to_loaded_cn_changed": int(
            source_df["raw_to_loaded_changed"].sum()
        ),
        "sources_raw_to_parent_cn_changed": int(
            source_df["raw_to_parent_changed"].sum()
        ),
        "sources_with_new_final_cn_class_vs_raw": int(
            source_df["raw_to_final_new_cn_class"].sum()
        ),
        "mutation_source_indices": mutation_sources,
    }
    summary_file.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.export_mutation_sources > 0:
        mutation_dir = output_dir / "mutation_examples"
        exported_sources = 0

        for source_index in mutation_sources:
            if exported_sources >= args.export_mutation_sources:
                break

            raw_structure = raw_structures_by_source.get(source_index)
            downstream = mutation_structures_by_source.get(
                source_index,
                [],
            )
            if raw_structure is None or not downstream:
                continue

            source_db_id = int(
                source_df.loc[
                    source_df["source_index"] == source_index,
                    "source_db_id",
                ].iloc[0]
            )
            source_dir = mutation_dir / (
                f"source_{source_index:03d}_dbrow_{source_db_id:06d}"
            )
            _write_structure_cif(
                raw_structure,
                source_dir / "raw_db.cif",
            )

            seen_labels = set()
            exported = 0
            for label, structure in downstream:
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                safe_label = (
                    label.replace("/", "_")
                    .replace(";", "_")
                    .replace(" ", "_")
                )
                _write_structure_cif(
                    structure,
                    source_dir / f"{safe_label}.cif",
                )
                exported += 1
                if exported >= 5:
                    break

            exported_sources += 1

    print("\n=== Lineage audit summary ===")
    for stage, info in global_stage_summary.items():
        print(
            f"{stage:26s} objects={info['objects']:6d} "
            f"Ti={info['ti_sites']:8d} "
            f"CN6={info['cn6_fraction']:.2%} "
            f"CN={info['hard_cn_counts']}"
        )

    print(
        "Sources with raw -> loaded PyXtal CN change: "
        f"{summary['sources_raw_to_loaded_cn_changed']}/{len(source_df)}"
    )
    print(
        "Sources with raw -> optimized parent CN change: "
        f"{summary['sources_raw_to_parent_cn_changed']}/{len(source_df)}"
    )
    print(
        "Sources where accepted representations contain a new CN class "
        "absent from raw source: "
        f"{summary['sources_with_new_final_cn_class_vs_raw']}/{len(source_df)}"
    )
    print(f"Stage records: {stage_file}")
    print(f"Source summary: {source_file}")
    print(f"Site records: {site_file}")
    print(f"Summary JSON: {summary_file}")
    if not failures_df.empty:
        print(f"Failures: {failures_file}")
    if args.export_mutation_sources > 0:
        print(f"Mutation CIFs: {output_dir / 'mutation_examples'}")


if __name__ == "__main__":
    main()


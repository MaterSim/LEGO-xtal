#!/usr/bin/env python3
"""Juliette known-phase recovery audit (v4).

Purpose
-------
Diagnose whether the shared-O construction itself can represent canonical TiO2
networks before changing stochastic space-group weights.

The audit is deliberately layered:
  A. Find native conventional-cell references in the ASE database by detected SG.
  B. Run the *production strict chemistry audit* on the exact reference geometry.
  C. Remove the reference O coordinates, retain only the reference Ti connectivity
     (which Ti periodic-image triplet owns each O), and ask the production analytic
     sphere-intersection constructor to rebuild physical O seeds.
  D. Check whether the target/native formula-unit count has exact Wyckoff entrances
     in the requested SG under the current generator.
  E. Optionally launch a small normal-production search restricted to each native SG.

No phase-specific coordinates are hard-coded.  Phase names only map SG numbers to
human-readable labels; structures come from --ase-database.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

PHASES = {136: "rutile", 141: "anatase", 61: "brookite", 87: "hollandite"}


def _load_generator(path: str):
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("juliette_generator_runtime", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import generator from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _spglib_dataset(atoms, symprec: float):
    import spglib
    lattice = np.asarray(atoms.cell.array, float)
    frac = np.asarray(atoms.get_scaled_positions(wrap=True), float)
    numbers = np.asarray(atoms.numbers, int)
    ds = spglib.get_symmetry_dataset((lattice, frac, numbers), symprec=float(symprec))
    if ds is None:
        return None
    # spglib 2.x exposes attributes; older releases are dict-like.
    def get(name):
        return getattr(ds, name) if hasattr(ds, name) else ds[name]
    return {
        "number": int(get("number")),
        "international": str(get("international")),
        "hall": str(get("hall")),
    }


def _standardized_conventional(atoms, symprec: float):
    import spglib
    from ase import Atoms
    cell = (np.asarray(atoms.cell.array, float),
            np.asarray(atoms.get_scaled_positions(wrap=True), float),
            np.asarray(atoms.numbers, int))
    out = spglib.standardize_cell(cell, to_primitive=False, no_idealize=False,
                                  symprec=float(symprec))
    if out is None:
        return atoms.copy()
    lattice, frac, numbers = out
    return Atoms(numbers=np.asarray(numbers, int), cell=np.asarray(lattice, float),
                 scaled_positions=np.asarray(frac, float), pbc=True)


def _formula_counts(atoms) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(atoms.get_chemical_symbols()))


def _accepted_row_ids(model) -> set[int]:
    out: set[int] = set()
    for src in model.raw.get("sources", []):
        for row_id in src.get("accepted_rows", []):
            try:
                out.add(int(row_id))
            except Exception:
                pass
    return out


def _find_references(database: str, wanted: list[int], symprec: float, model):
    from ase.db import connect
    accepted = _accepted_row_ids(model)
    found: dict[int, list[dict[str, Any]]] = {int(x): [] for x in wanted}
    for row in connect(database, serial=True).select():
        atoms = row.toatoms()
        counts = _formula_counts(atoms)
        if counts.get("Ti", 0) <= 0 or counts.get("O", 0) != 2 * counts.get("Ti", 0):
            continue
        ds = _spglib_dataset(atoms, symprec)
        if ds is None or int(ds["number"]) not in found:
            continue
        conv = _standardized_conventional(atoms, symprec)
        cds = _spglib_dataset(conv, symprec)
        if cds is None:
            continue
        ccounts = _formula_counts(conv)
        nti = int(ccounts.get("Ti", 0)); no = int(ccounts.get("O", 0))
        if nti <= 0 or no != 2 * nti:
            continue
        found[int(cds["number"])].append({
            "row_id": int(row.id),
            "accepted_training_row": int(row.id) in accepted,
            "atoms": conv,
            "n_ti": nti,
            "n_o": no,
            "symbol": cds["international"],
        })
    selected = {}
    for spg, rows in found.items():
        if not rows:
            selected[spg] = None
            continue
        # Prefer a chemistry-model accepted source, then the smallest conventional Z.
        rows.sort(key=lambda r: (not r["accepted_training_row"], r["n_ti"], r["row_id"]))
        selected[spg] = rows[0]
    return selected


def _label_for_element(model, element: str) -> str:
    choices = [x for x in model.labels if str(model.final_formula[x]) == str(element)]
    if len(choices) != 1:
        raise ValueError(f"Need exactly one construction label for {element}; found {choices}")
    return choices[0]


def _make_builder(gen, model, counts, device: str = "cpu"):
    # Keep these synchronized with production defaults.  The reference audit uses only
    # strict/analytic methods, so optimization-heavy values do not influence A--D.
    return gen.XNBuilder(
        model=model, device=device, starts=4, screen_steps=20, refine_steps=105,
        polish_steps=20, lr=0.04, minimum_distance=1.0, soft_temperature=0.18,
        port_width=0.18, angular_weight=1.0, radial_weight=2.0,
        overlap_weight=10.0, uniqueness_weight=4.0, nonbonded_weight=8.0,
        nonbonded_margin=0.05, nonbonded_width=0.08,
        angular_site_z_max=3.0, angular_vector_z_max=4.0,
        assignment_refresh=0, target_counts=counts, construction_symmetry="full",
        coincidence_rms_max=0.12, coincidence_max_max=0.20,
        coincidence_weight=6.0, distortion_weight=0.20, distortion_max=0.35,
        framework_weight=3.0, framework_restraint_weight=40.0,
        framework_keep=8, framework_patience=12,
        oxygen_coincidence_steps=80, oxygen_contact_steps=80,
        oxygen_assigned_fraction_min=0.95,
        oxygen_screen_rms_max=0.60, oxygen_screen_max_max=1.00,
        octahedral_branches=32, octahedron_prepare_steps=40,
        octahedron_match_steps=100, octahedron_cluster_steps=50,
        floating_coincidence_sigma=0.24, floating_cluster_tolerance=0.38,
        ti_registry_path="", oxygen_proposal_oversample=4,
        oxygen_proposal_descriptor_tol=0.025, oxygen_basin_prune_every=25,
        framework_memory_path="", framework_basin_memory_path="",
        framework_intelligent_keep=4, framework_memory_k=8,
    )


def _split_ti_o(atoms):
    syms = np.asarray(atoms.get_chemical_symbols())
    frac = np.asarray(atoms.get_scaled_positions(wrap=True), float)
    return frac[syms == "Ti"], frac[syms == "O"], np.asarray(atoms.cell.array, float)


def _reference_topology(gen, builder, template, ti_frac, o_frac, cell):
    child = tuple(builder.plan["children"])[0]
    groups = []
    details = []
    for oi, of in enumerate(np.asarray(o_frac, float)):
        neigh = []
        for i, label in enumerate(template["expanded_labels"]):
            cutoff = float(builder.model.pair(label, child).first_shell_cutoff)
            for sh in np.asarray(gen.SHIFTS, int):
                d = float(np.linalg.norm((of - (ti_frac[i] + sh)) @ cell))
                if d <= cutoff + 1.0e-8:
                    neigh.append((d, int(i), tuple(int(x) for x in sh)))
        neigh.sort(key=lambda x: x[0])
        # Deduplicate identical periodic owner instances, then require exact X3.
        unique = []
        seen = set()
        for row in neigh:
            key = (row[1], row[2])
            if key not in seen:
                seen.add(key); unique.append(row)
        if len(unique) != 3:
            return None, {"reference_topology_valid": False,
                          "reference_topology_failure_o_index": int(oi),
                          "reference_topology_observed_cn": int(len(unique)),
                          "reference_topology_distances_A": [float(x[0]) for x in unique[:8]]}
        groups.append([(x[1], x[2]) for x in unique])
        details.append([float(x[0]) for x in unique])
    return groups, {"reference_topology_valid": True,
                    "reference_topology_bond_min_A": float(min(min(x) for x in details)),
                    "reference_topology_bond_max_A": float(max(max(x) for x in details))}


def _periodic_assignment_rms(gen, ref_o, trial_o, cell):
    from scipy.optimize import linear_sum_assignment
    ref_o = np.asarray(ref_o, float); trial_o = np.asarray(trial_o, float)
    if len(ref_o) != len(trial_o):
        return float("inf"), float("inf")
    cost = np.zeros((len(ref_o), len(trial_o)), float)
    for i, a in enumerate(ref_o):
        for j, b in enumerate(trial_o):
            cost[i, j] = gen.XNBuilder._periodic_point_distance_np(a, b, cell)
    rr, cc = linear_sum_assignment(cost)
    vals = cost[rr, cc]
    return float(np.sqrt(np.mean(vals * vals))), float(np.max(vals))


def _write_atoms(path: Path, ti_frac, o_frac, cell):
    from ase import Atoms
    from ase.io import write
    symbols = ["Ti"] * len(ti_frac) + ["O"] * len(o_frac)
    atoms = Atoms(symbols=symbols, cell=np.asarray(cell, float),
                  scaled_positions=np.vstack([ti_frac, o_frac]) % 1.0, pbc=True)
    write(str(path), atoms, format="cif")


def _audit_phase(gen, model, ref, spg: int, outdir: Path, max_entries: int):
    phase = PHASES.get(int(spg), f"sg{spg}")
    if ref is None:
        return [{"phase": phase, "spg": int(spg), "reference_found": False}]
    atoms = ref["atoms"]
    ti, oo, cell = _split_ti_o(atoms)
    z = len(ti)
    counts, _ = gen.resolve_targets([f"TiO2={z}"], model)
    builder = _make_builder(gen, model, counts, device="cpu")
    ti_label = _label_for_element(model, "Ti")
    template = {"expanded_labels": tuple([ti_label] * z)}

    audit = builder._oldschool_audit(template, ti, oo, cell)
    groups, tdiag = _reference_topology(gen, builder, template, ti, oo, cell)
    plan = model.construction_plan(counts)
    entries = gen._exact_entries_for_group(int(spg), tuple(model.labels),
                                           plan["construction_counts"], int(max_entries))

    base = {
        "phase": phase, "spg": int(spg), "reference_found": True,
        "row_id": int(ref["row_id"]), "accepted_training_row": bool(ref["accepted_training_row"]),
        "native_Z": int(z), "natoms": int(len(atoms)), "international_symbol": ref["symbol"],
        "exact_reference_strict_valid": bool(audit.get("strict_valid", False)),
        "exact_reference_bond_window_valid": bool(audit.get("bond_window_valid", False)),
        "exact_reference_angular_label_valid": bool(audit.get("angular_label_valid", False)),
        "exact_reference_nonbonded_valid": bool(audit.get("nonbonded_exclusion_valid", False)),
        "exact_reference_min_oo_A": float(audit.get("minimum_o_o_A", np.nan)),
        "exact_reference_min_unassigned_tio_A": float(audit.get("minimum_unassigned_ti_o_A", np.nan)),
        "native_exact_entrance_count_capped": int(len(entries)),
        **tdiag,
    }
    phase_dir = outdir / f"{phase}_sg{spg}_Z{z}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    _write_atoms(phase_dir / "reference.cif", ti, oo, cell)

    rows = [dict(base, test="exact_reference")]
    if groups is None:
        rows.append(dict(base, test="analytic_from_reference_topology",
                         analytic_success=False, analytic_reason="reference_topology_invalid"))
        return rows

    for scale in (1.00, 0.98, 1.02):
        scell = np.asarray(cell, float) * float(scale)
        analytic, adiag = builder._analytic_o_assignments(template, ti, scell, groups)
        row = dict(base, test=f"analytic_reference_topology_cell_scale_{scale:.2f}",
                   analytic_success=bool(analytic), **adiag)
        if analytic:
            best = None
            for ai, seed in enumerate(analytic):
                seed_o = np.asarray(seed["oxygen_frac_init"], float)
                aa = builder._oldschool_audit(template, ti, seed_o, scell)
                rms, mx = _periodic_assignment_rms(gen, oo, seed_o, scell)
                candidate = {
                    "analytic_rank": int(ai),
                    "analytic_seed_strict_valid": bool(aa.get("strict_valid", False)),
                    "analytic_seed_bond_window_valid": bool(aa.get("bond_window_valid", False)),
                    "analytic_seed_angular_valid": bool(aa.get("angular_label_valid", False)),
                    "analytic_seed_nonbonded_valid": bool(aa.get("nonbonded_exclusion_valid", False)),
                    "analytic_seed_reference_rms_A": rms,
                    "analytic_seed_reference_max_A": mx,
                    "analytic_seed_min_oo_A": float(aa.get("minimum_o_o_A", np.nan)),
                    "analytic_seed_min_unassigned_tio_A": float(aa.get("minimum_unassigned_ti_o_A", np.nan)),
                }
                score = (not candidate["analytic_seed_strict_valid"], rms)
                if best is None or score < best[0]:
                    best = (score, candidate, seed_o)
            row.update(best[1])
            _write_atoms(phase_dir / f"analytic_seed_scale_{scale:.2f}.cif", ti, best[2], scell)
        rows.append(row)
    return rows


def _run_smoke(args, phase_rows: pd.DataFrame):
    if int(args.smoke_tokens) <= 0:
        return []
    results = []
    for _, row in phase_rows.drop_duplicates(["phase", "spg", "native_Z"]).iterrows():
        if not bool(row.get("reference_found", False)):
            continue
        phase = str(row["phase"]); spg = int(row["spg"]); z = int(row["native_Z"])
        out = Path(args.output_folder) / f"production_{phase}_sg{spg}_Z{z}"
        cmd = [sys.executable, str(Path(args.generator).resolve()),
               "--chemistry-model", str(Path(args.chemistry_model).resolve()),
               "--target", f"TiO2={z}", "--space-groups", str(spg),
               "--ti-token-budget", str(int(args.smoke_tokens)),
               "--max-runtime-minutes", str(float(args.smoke_runtime_minutes)),
               "--output-folder", str(out)]
        print("Production smoke:", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True)
        rec = {"phase": phase, "spg": spg, "native_Z": z,
               "production_returncode": int(proc.returncode),
               "production_output": str(out)}
        summary = out / "summary.json"
        if summary.exists():
            try:
                data = json.loads(summary.read_text())
                for k in ("exact_unique", "families", "strict_valid", "ti_tokens_used"):
                    if k in data: rec[f"production_{k}"] = data[k]
            except Exception:
                pass
        results.append(rec)
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Juliette canonical TiO2 recovery audit")
    p.add_argument("--ase-database", default="tio2.db")
    p.add_argument("--chemistry-model", default="data/xn_templates/chemistry_model.json")
    p.add_argument("--generator", default="1_generate.py")
    p.add_argument("--space-groups", default="61,87,136,141")
    p.add_argument("--symprec", type=float, default=0.05)
    p.add_argument("--max-entries-per-group", type=int, default=5000)
    p.add_argument("--smoke-tokens", type=int, default=0,
                   help="If >0, sequentially launch normal generation at each native SG")
    p.add_argument("--smoke-runtime-minutes", type=float, default=15.0)
    p.add_argument("--output-folder", default="recovery_audit")
    return p.parse_args()


def main():
    args = parse_args()
    gen = _load_generator(args.generator)
    model = gen.XNModel(args.chemistry_model)
    wanted = [int(x.strip()) for x in str(args.space_groups).split(",") if x.strip()]
    outdir = Path(args.output_folder); outdir.mkdir(parents=True, exist_ok=True)
    refs = _find_references(args.ase_database, wanted, args.symprec, model)
    rows = []
    for spg in wanted:
        rows.extend(_audit_phase(gen, model, refs.get(spg), spg, outdir, args.max_entries_per_group))
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "recovery_audit.csv", index=False)

    cols = [c for c in ["phase", "spg", "native_Z", "accepted_training_row",
                        "exact_reference_strict_valid", "native_exact_entrance_count_capped",
                        "test", "analytic_success", "analytic_seed_strict_valid",
                        "analytic_seed_reference_rms_A"] if c in frame.columns]
    print("\n=== Juliette known-phase recovery audit ===")
    if len(frame):
        print(frame[cols].to_string(index=False))
    missing = [PHASES.get(s, str(s)) for s in wanted if refs.get(s) is None]
    if missing:
        print("Missing requested references in ASE database:", ", ".join(missing))

    smoke = _run_smoke(args, frame)
    if smoke:
        pd.DataFrame(smoke).to_csv(outdir / "production_smoke.csv", index=False)
    print(f"\nAudit written to {outdir / 'recovery_audit.csv'}")
    if int(args.smoke_tokens) <= 0:
        print("Normal-production level not run. Add --smoke-tokens 100 (or similar) when desired.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export structures from an ASE database produced by LEGO-Xtal relaxation.

Examples
--------
Export all structures as CIF:
    python export_db_structures.py --db out.db --format cif --out exported_cif

Export all structures as POSCAR:
    python export_db_structures.py --db out.db --format poscar --out exported_poscar

Export only first 100 structures:
    python export_db_structures.py --db out.db --format cif --out exported_cif --max 100

Export structures whose row has relaxed=True, if such a key exists:
    python export_db_structures.py --db out.db --format cif --out exported_cif --query "relaxed=True"
"""

import argparse
import os
import re
from pathlib import Path

from ase.db import connect
from ase.io import write


def safe_name(text: str) -> str:
    """Make a filesystem-safe name."""
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = text.strip("_")
    return text or "structure"


def main():
    parser = argparse.ArgumentParser(
        description="Export structures from an ASE .db file to CIF or POSCAR."
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Input ASE database, e.g. results.db or optimized.db",
    )
    parser.add_argument(
        "--out",
        default="exported_structures",
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        choices=["cif", "poscar", "vasp"],
        default="cif",
        help="Output format. Use 'poscar' or 'vasp' for VASP POSCAR files.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help='Optional ASE db query, e.g. "relaxed=True" or "energy<0".',
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of structures to export.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional filename prefix.",
    )
    parser.add_argument(
        "--include-id",
        action="store_true",
        help="Include ASE database row id in filename.",
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    fmt = "vasp" if args.format in ["poscar", "vasp"] else "cif"
    ext = "vasp" if fmt == "vasp" else "cif"

    db = connect(str(db_path))

    if args.query:
        rows = db.select(args.query)
    else:
        rows = db.select()

    n_exported = 0

    for row in rows:
        if args.max is not None and n_exported >= args.max:
            break

        atoms = row.toatoms()

        # Try to construct a useful filename.
        parts = []

        if args.prefix:
            parts.append(safe_name(args.prefix))

        if args.include_id:
            parts.append(f"id{row.id:05d}")

        # Common possible metadata keys in ASE/LEGO-style databases.
        for key in ["name", "spg", "group", "topology", "ff_energy", "energy"]:
            if key in row.key_value_pairs:
                val = row.key_value_pairs[key]
                parts.append(f"{key}_{safe_name(val)}")

        if not parts:
            parts.append(f"structure_{row.id:05d}")

        filename_base = "_".join(parts)
        filename = outdir / f"{filename_base}.{ext}"

        # Avoid accidental overwrite if metadata creates duplicate names.
        if filename.exists():
            filename = outdir / f"{filename_base}_row{row.id:05d}.{ext}"

        if fmt == "vasp":
            write(str(filename), atoms, format="vasp", direct=True, sort=True)
        else:
            write(str(filename), atoms, format="cif")

        n_exported += 1

    print(f"Exported {n_exported} structures to: {outdir}")


if __name__ == "__main__":
    main()


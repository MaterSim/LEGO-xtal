#!/usr/bin/env python3
"""Build a nearest-neighbor prototype chemistry model for Juliette.

Prototype geometry supplies local radial/angular centers. User settings supply
Gaussian widths. Direct generator species are defined only by nearest-neighbor
chemistry: target CN, local radial roles, and central-species angular geometry.
Outer-shell/environment fingerprints are intentionally not extracted here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

EPS = 1.0e-12


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= EPS or n2 <= EPS:
        return float("nan")
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def site_neighbors(structure: Structure, site_id: int, radius: float) -> list[dict]:
    center = structure[site_id]
    neighbors = []
    for nn in structure.get_neighbors(
        center, radius, include_index=True, include_image=True
    ):
        neighbors.append(
            {
                "distance": float(nn.nn_distance),
                "vector": np.asarray(nn.coords - center.coords, dtype=float),
                "atom_index": int(nn.index),
                "image": tuple(int(x) for x in nn.image),
            }
        )
    neighbors.sort(
        key=lambda item: (item["distance"], item["atom_index"], item["image"])
    )
    return neighbors


def extract_prototype(
    path: str,
    label: str,
    final_element: str,
    cn: int,
    r_search: float,
    radial_sigma: float,
    angular_sigma: float,
) -> dict:
    structure = Structure.from_file(path)
    observed = sorted({str(site.specie.symbol) for site in structure})
    if observed != [str(final_element)]:
        raise ValueError(
            f"Prototype {path!r} for {label} must contain only {final_element}; "
            f"observed={observed}"
        )
    if cn <= 0:
        raise ValueError(f"Target CN must be positive for {label}")

    local_distances: list[float] = []
    local_angle_rows: list[list[float]] = []

    for site_id in range(len(structure)):
        neighbors = site_neighbors(structure, site_id, r_search)
        if len(neighbors) < cn:
            raise RuntimeError(
                f"Prototype {path!r} site {site_id} has only {len(neighbors)} "
                f"neighbors within {r_search} A; CN={cn}"
            )
        local = neighbors[:cn]
        local_distances.extend(float(item["distance"]) for item in local)
        local_angle_rows.append(
            sorted(
                angle_deg(local[i]["vector"], local[j]["vector"])
                for i in range(cn)
                for j in range(i + 1, cn)
            )
        )

    local_mu = float(np.mean(local_distances))
    angle_array = np.asarray(local_angle_rows, dtype=float)
    angle_slots = []
    if angle_array.size:
        slot_centers = np.mean(angle_array, axis=0)
        # Prototype-equivalent angular targets are stored compactly with a
        # multiplicity count, matching the radial-slot representation.
        grouped = []
        for mu in slot_centers:
            mu = float(mu)
            match = next((item for item in grouped if abs(item["mu_deg"] - mu) <= 1.0e-6), None)
            if match is None:
                grouped.append({"mu_deg": mu, "count": 1})
            else:
                match["count"] += 1
        for mode_id, item in enumerate(grouped, start=1):
            angle_slots.append(
                {
                    "role": "angle" if len(grouped) == 1 else f"angle_mode_{mode_id}",
                    "mu_deg": float(item["mu_deg"]),
                    "sigma_deg": float(angular_sigma),
                    "count": int(item["count"]),
                    "source": "prototype",
                }
            )

    return {
        "generator_species": str(label),
        "final_element": str(final_element),
        "prototype": str(Path(path).resolve()),
        "target_local_cn": int(cn),
        "local_radial_slots": [
            {
                "role": "bond",
                "mu_A": local_mu,
                "sigma_A": float(radial_sigma),
                "count": int(cn),
                "source": "prototype",
            }
        ],
        "local_angular_slots": angle_slots,
        "prototype_diagnostics": {
            "natoms": int(len(structure)),
            "local_distance_mean_A": local_mu,
            "local_distance_min_A": float(np.min(local_distances)),
            "local_distance_max_A": float(np.max(local_distances)),
        },
    }


def local_channel(spec_i: dict, spec_j: dict, radial_sigma: float) -> dict:
    self_i = float(spec_i["local_radial_slots"][0]["mu_A"])
    self_j = float(spec_j["local_radial_slots"][0]["mu_A"])
    if spec_i["generator_species"] == spec_j["generator_species"]:
        mu = self_i
        source = "prototype"
    else:
        mu = 0.5 * (self_i + self_j)
        source = "derived_additive_bond_radius"
    return {
        "species_i": spec_i["generator_species"],
        "species_j": spec_j["generator_species"],
        "relation_type": "local",
        "radial_roles": [
            {
                "role": "bond",
                "mu_A": float(mu),
                "sigma_A": float(radial_sigma),
                "sampling_min_A": float(max(0.0, mu - 2.0 * radial_sigma)),
                "sampling_max_A": float(mu + 2.0 * radial_sigma),
                "source": source,
            }
        ],
        "source": source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a nearest-neighbor prototype chemistry model"
    )
    parser.add_argument(
        "--prototype",
        nargs=4,
        action="append",
        metavar=("CIF", "GENERATOR_SPECIES", "FINAL_ELEMENT", "CN"),
        required=True,
        help=(
            "Repeat once per generator species, e.g. "
            "--prototype graphite.cif C_sp2 C 3"
        ),
    )
    parser.add_argument("--radial-sigma", type=float, default=0.08)
    parser.add_argument("--angular-sigma", type=float, default=8.0)
    parser.add_argument("--r-search", type=float, default=5.0)
    parser.add_argument(
        "--output",
        default="data/chemistry/prototype_chemistry/chemistry_model.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.radial_sigma <= 0 or args.angular_sigma <= 0 or args.r_search <= 0:
        raise ValueError("Sigma values and --r-search must be positive")

    species = []
    seen = set()
    for path, label, final_element, cn_text in args.prototype:
        if label in seen:
            raise ValueError(f"Duplicate generator species {label!r}")
        seen.add(label)
        species.append(
            extract_prototype(
                path=path,
                label=label,
                final_element=final_element,
                cn=int(cn_text),
                r_search=float(args.r_search),
                radial_sigma=float(args.radial_sigma),
                angular_sigma=float(args.angular_sigma),
            )
        )

    by_label = {item["generator_species"]: item for item in species}
    labels = sorted(by_label)
    channels = []
    for i, label_i in enumerate(labels):
        for label_j in labels[i:]:
            channels.append(
                local_channel(by_label[label_i], by_label[label_j], args.radial_sigma)
            )

    model = {
        "version": "prototype_user_v5",
        "schema": "juliette_constructive_chemistry_v2",
        "model_type": "prototype_nearest_neighbor_generator_species_chemistry",
        "parameters": {
            "radial_sigma_A": float(args.radial_sigma),
            "angular_sigma_deg": float(args.angular_sigma),
            "r_search_A": float(args.r_search),
        },
        "species_map": {
            item["generator_species"]: item["final_element"] for item in species
        },
        "generator_species": [
            dict(item, construction_role="direct") for item in species
        ],
        "chemistry_channels": channels,
        "channel_key": ["species_i", "species_j", "relation_type"],
        "construction_roles": {
            item["generator_species"]: "direct" for item in species
        },
        "local_angular_semantics": "central_generator_species",
        "local_coordination_semantics": (
            "pair-conditioned local channel sampling_max_A defines the active "
            "coordination cutoff"
        ),
        "center_precedence": [
            "explicit_user_override",
            "observed_cross_species_prototype",
            "derived_additive_bond_radius",
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Prototype nearest-neighbor chemistry model built")
    print(f"Output: {output}")
    print(f"Generator species: {labels}")
    print("Species map:")
    print(json.dumps(model["species_map"], indent=2))
    print("Local pair channels:")
    for channel in channels:
        role = channel["radial_roles"][0]
        print(
            f"  {channel['species_i']} - {channel['species_j']}: "
            f"mu={role['mu_A']:.6f} A sigma={role['sigma_A']:.6f} A "
            f"source={role['source']}"
        )
    print("Central-species nearest-neighbor chemistry:")
    for item in species:
        angles = [
            round(float(slot["mu_deg"]), 6)
            for slot in item["local_angular_slots"]
            for _ in range(int(slot.get("count", 1)))
        ]
        print(
            f"  {item['generator_species']}: CN={item['target_local_cn']} "
            f"local_mu={item['local_radial_slots'][0]['mu_A']:.6f} A "
            f"angles={json.dumps(angles)}"
        )


if __name__ == "__main__":
    main()

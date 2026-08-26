from pathlib import Path
import csv

from mp_api.client import MPRester

key = "5UO5flO6N4HcdiQTTlj2UR1S3OXTIQO1"

SI_O_CUTOFF = 2.15       # Å; deliberately strict tetrahedral first pass
MAX_SITES = 120
MAX_EHULL = 0.30         # eV/atom


def element_symbol(site):
    """Return the element symbol for an ordered pymatgen PeriodicSite."""
    return site.specie.symbol


def analyze_sio2_coordination(structure, cutoff=SI_O_CUTOFF):
    """Check the fixed tetrahedral-silica coordination template.

    Required:
        every Si has exactly 4 O neighbours;
        every O has exactly 2 Si neighbours.

    Same-species neighbours do not contribute to coordination.
    """
    si_counts = []
    o_counts = []

    for index, site in enumerate(structure):
        symbol = element_symbol(site)
        neighbors = structure.get_neighbors(site, cutoff)

        if symbol == "Si":
            count = sum(
                element_symbol(neighbor) == "O"
                for neighbor in neighbors
            )
            si_counts.append(count)

        elif symbol == "O":
            count = sum(
                element_symbol(neighbor) == "Si"
                for neighbor in neighbors
            )
            o_counts.append(count)

        else:
            return {
                "valid": False,
                "reason": f"unexpected element {symbol}",
                "si_counts": si_counts,
                "o_counts": o_counts,
            }

    valid = (
        len(si_counts) > 0
        and len(o_counts) == 2 * len(si_counts)
        and all(count == 4 for count in si_counts)
        and all(count == 2 for count in o_counts)
    )

    if not valid:
        reason = (
            f"Si coordination={sorted(set(si_counts))}; "
            f"O coordination={sorted(set(o_counts))}"
        )
    else:
        reason = "SiO4/OSi2"

    return {
        "valid": valid,
        "reason": reason,
        "si_counts": si_counts,
        "o_counts": o_counts,
    }


def has_exact_sio2_composition(structure):
    """Require ordered, binary, reduced composition SiO2."""
    if not structure.is_ordered:
        return False

    elements = {element.symbol for element in structure.composition.elements}
    if elements != {"Si", "O"}:
        return False

    reduced = structure.composition.reduced_composition.as_dict()
    return reduced == {"Si": 1.0, "O": 2.0}


output_dir = Path("mp_sio2_tetrahedral")
cif_dir = output_dir / "cifs"
cif_dir.mkdir(parents=True, exist_ok=True)

with MPRester(key) as mpr:
    docs = mpr.materials.summary.search(
        formula="SiO2",
        energy_above_hull=(0.0, MAX_EHULL),
        nsites=(3, MAX_SITES),
        deprecated=False,
        fields=[
            "material_id",
            "structure",
            "formula_pretty",
            "symmetry",
            "energy_above_hull",
            "is_stable",
            "nsites",
            "density",
            "theoretical",
        ],
    )

accepted = []
rejected = []

for doc in docs:
    structure = doc.structure
    mpid = str(doc.material_id)

    if not has_exact_sio2_composition(structure):
        rejected.append((mpid, "composition/disorder"))
        continue

    result = analyze_sio2_coordination(structure)

    if not result["valid"]:
        rejected.append((mpid, result["reason"]))
        continue

    cif_path = cif_dir / f"{mpid}.cif"
    structure.to(filename=str(cif_path))

    symmetry = doc.symmetry

    accepted.append(
        {
            "material_id": mpid,
            "formula": doc.formula_pretty,
            "nsites": doc.nsites,
            "spacegroup_number": symmetry.number,
            "spacegroup_symbol": symmetry.symbol,
            "crystal_system": str(symmetry.crystal_system),
            "energy_above_hull": doc.energy_above_hull,
            "is_stable": doc.is_stable,
            "density": doc.density,
            "theoretical": doc.theoretical,
            "cif": str(cif_path),
        }
    )

metadata_path = output_dir / "accepted.csv"
with metadata_path.open("w", newline="", encoding="utf-8") as handle:
    if accepted:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(accepted[0].keys()),
        )
        writer.writeheader()
        writer.writerows(accepted)

rejected_path = output_dir / "rejected.csv"
with rejected_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["material_id", "reason"])
    writer.writerows(rejected)

print(f"API candidates: {len(docs)}")
print(f"Accepted tetrahedral SiO2: {len(accepted)}")
print(f"Rejected: {len(rejected)}")
print(f"CIF directory: {cif_dir}")
print(f"Metadata: {metadata_path}")

import argparse
import os
import random
import numpy as np
import pandas as pd
from pyxtal import pyxtal
from pyxtal.db import database_topology
from pyxtal.lego.builder import builder
from pyxtal.util import new_struc_wo_energy


def make_builder():
    """Creates and configures a pyxtal builder instance."""
    xtal = pyxtal()
    xtal.from_prototype("graphite")
    cif_file = xtal.to_pymatgen()
    bu = builder(["C"], [1], verbose=False)
    bu.set_descriptor_calculator(mykwargs={"rcut": 2.0})
    bu.set_reference_enviroments(cif_file)
    bu.set_criteria(CN={"C": [3]})
    return bu


def make_csv(total_reps, energy, label, discrete, discrete_cell, N_wp, tag):
    """Converts data into csv format for model training."""
    total_reps = np.array(total_reps)
    column_names = ["spg", "a", "b", "c", "alpha", "beta", "gamma"]

    # Determine which columns should remain float based on discretization settings
    float_cols = []
    if not discrete_cell:
        float_cols.extend([1, 2, 3, 4, 5, 6])  # Lattice parameters a-gamma

    for i in range(N_wp):
        column_names.extend([f"wp{i}", f"x{i}", f"y{i}", f"z{i}"])
        if not discrete:
            # Wyckoff positions x, y, z
            float_cols.extend([7 + i * 4 + 1, 7 + i * 4 + 2, 7 + i * 4 + 3])

    if energy:
        column_names.append("energy")
        float_cols.append(len(column_names) - 1)
    if label:
        column_names.append("label")
        # Label column is typically integer, so not added to float_cols

    # Create dictionary for DataFrame, casting types appropriately
    dicts = {}
    for i, col in enumerate(column_names):
        if i in float_cols:
            dicts[col] = total_reps[:, i]  # Keep as float (original type)
        else:
            # Cast non-float columns to int
            dicts[col] = np.array(total_reps[:, i], dtype=int)

    df = pd.DataFrame.from_dict(dicts)

    # Ensure column names match DataFrame columns
    if len(df.columns) != len(column_names):
        N_cols, N_df = len(column_names), len(df.columns)
        raise ValueError(f"Expected {N_cols} columns, but got {N_df}")
    df.columns = column_names # Assign correct column names

    # Determine output filename based on discretization flags
    if discrete:
        filename = (
            f"data/sample/train-{tag}-discell.csv"
            if discrete_cell
            else f"data/sample/train-{tag}-dis.csv"
        )
    else:
        filename = f"data/sample/train-{tag}.csv"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def get_reps_from_xtal(xtal, params):
    """Generates tabular representations for a given crystal structure and its subgroups."""

    # Unpack parameters
    (
        max_dof,
        N_atoms_min,
        N_atoms_max,
        max_E,
        min_spg,
        N_wp,
        max_per_struc,
        energy,
        discrete,
        discrete_cell,
        discrete_res,
        eps,
    ) = params
    bu = make_builder()

    xtal_reps = []

    # Check initial crystal validity
    if not (
        xtal.dof <= max_dof
        and N_atoms_min <= sum(xtal.numIons) <= N_atoms_max
        and xtal.ff_energy < max_E
        and xtal.group.number >= min_spg
        and len(xtal.atom_sites) <= N_wp
    ):
        return xtal_reps # Return empty list if initial xtal is invalid

    current_energy = xtal.energy if energy else None

    # Optimize initial crystal
    xtal_opt, _, _ = bu.optimize_xtal(xtal, add_db=False)
    if xtal_opt is None or not xtal_opt.check_validity(bu.criteria):
        print(f"Initial xtal failed optimization or validity check: {xtal.formula}")
        return xtal_reps # Return empty list if optimization fails

    # Generate representations for the optimized initial crystal
    N_wps = len(xtal_opt.atom_sites)
    # Scale N_max based on Wyckoff positions relative to max allowed
    N_max_initial = int(0.6 * max_per_struc * np.ceil(N_wps / N_wp))
    reps_initial = xtal_opt.get_tabular_representations(
        N_wp=N_wp,
        N_max=N_max_initial,
        discrete=discrete,
        discrete_cell=discrete_cell,
        N_grids=discrete_res,
    )
    if energy and current_energy is not None:
        reps_initial = [np.append(rep, current_energy) for rep in reps_initial]
    xtal_reps.extend(reps_initial)
    # xtal_opt.to_file("0_opt.cif") # Optional: save optimized initial structure

    # --- Generate and process subgroups ---
    max_cell_factor = max([N_atoms_max / sum(xtal_opt.numIons), 1.0])
    trial_xtals_cache = [xtal_opt] # Cache to check for duplicates

    for gtype in ["t", "k"]: # Iterate through subgroup types
        for _ in range(20): # Try generating subgroups multiple times
            if len(xtal_reps) >= max_per_struc:
                return xtal_reps # Stop if max reps reached

            # Attempt to generate subgroup
            xtal_sub = xtal_opt.subgroup_once(
                eps=eps,
                group_type=gtype,
                max_cell=max_cell_factor,
                mut_lat=False,
            )

            # Handle potential failure of subgroup generation
            if xtal_sub is None:
                # Try t-subgroup first if k-subgroup failed initially
                xtal0 = xtal_opt.subgroup_once(group_type='t')
                if xtal0 is not None:
                     xtal_sub = xtal0.subgroup_once(
                        eps=eps,
                        group_type='t', # Stick to t-subgroup if k failed
                        max_cell=max_cell_factor,
                        mut_lat=False)

            if xtal_sub is None:
                continue # Skip if subgroup generation failed

            # --- Validate the generated subgroup ---
            lat = xtal_sub.lattice.get_para(degree=True)
            lengths, angles = lat[:3], lat[3:]

            is_valid_geometry = (
                xtal_sub.get_dof() <= max_dof
                and len(xtal_sub.atom_sites) <= N_wp
                and max(lengths) < 50
                and max(angles) < 150
                and min(angles) > 30
            )

            # Check for novelty using structure comparison
            is_new_structure = new_struc_wo_energy(
                xtal_sub, trial_xtals_cache, 0.025, 0.025, 1.0
            )

            if not (is_valid_geometry and is_new_structure):
                continue # Skip invalid or duplicate structures

            # --- Optimize and process the valid subgroup ---
            xtal_sub_opt, _, _ = bu.optimize_xtal(xtal_sub, add_db=False)

            if xtal_sub_opt is None or not xtal_sub_opt.check_validity(bu.criteria):
                # print(f"Subgroup failed validity ({gtype}): {xtal_sub.formula}")
                continue # Skip if optimization fails

            trial_xtals_cache.append(xtal_sub_opt) # Add optimized structure to cache

            # Generate representations for the optimized subgroup
            N_wps_sub = len(xtal_sub_opt.atom_sites)
            # Scale N_max based on Wyckoff positions
            N_max_sub = int(0.2 * max_per_struc * np.ceil(N_wps_sub / N_wp))
            reps_sub = xtal_sub_opt.get_tabular_representations(
                N_wp=N_wp,
                N_max=N_max_sub,
                discrete=discrete,
                discrete_cell=discrete_cell,
                N_grids=discrete_res,
            )

            # --- Check relaxation consistency (for debugging/analysis) ---
            # This block checks if the structure changes significantly upon
            # re-optimization after being converted from its representation.
            # It might be computationally expensive for large datasets.
            # Consider enabling it only for debugging or specific analysis runs.
            CHECK_RELAXATION = False # Set to True to enable this check
            if CHECK_RELAXATION and discrete: # Check only makes sense for discrete reps
                for rep in reps_sub:
                    xtal_from_rep = pyxtal()
                    try:
                        xtal_from_rep.from_tabular_representation(
                            rep,
                            discrete=discrete,
                            discrete_cell=discrete_cell,
                            N_grids=discrete_res,
                        )
                        # Optimize the structure derived from the representation
                        xtal_reopt, _, _ = bu.optimize_xtal(xtal_from_rep, add_db=False)

                        if xtal_reopt is not None and xtal_reopt.check_validity(bu.criteria):
                            rms_dist = xtal_from_rep.get_rms_dist(xtal_reopt, 0.5, 0.5, 5.0)
                            if rms_dist > 0.4:
                                print(f"Significant change upon relaxation (RMSD: {rms_dist:.3f})")
                                # xtal_from_rep.to_file("debug_raw.cif")
                                # xtal_reopt.to_file("debug_reopt.cif")
                        # else:
                        #     print("Failed re-optimization or validity check from representation.")
                    except Exception as e:
                        print(f"Error processing representation: {e}")


            # Add energy to representations if required
            if energy and current_energy is not None:
                reps_sub = [np.append(rep, current_energy) for rep in reps_sub]

            xtal_reps.extend(reps_sub)

    return xtal_reps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate tabular crystal representations from a database.")
    parser.add_argument("--database", default="data/source/sp2_sacada.db",
        help="Path to the database (.db), default: data/source/sp2_sacada.db")
    parser.add_argument("--max_atoms", type=int, default=500,
        help="Maximum number of atoms in the unit cell (default: 500)")
    parser.add_argument("--min_spg", type=int, default=0,
        help="Minimum space group number to include (default: 0)")
    parser.add_argument("--max_dof", type=int, default=24,
        help="Maximum degrees of freedom allowed (default: 24)")
    parser.add_argument("--max_wp", type=int, default=8,
        help="Maximum number of Wyckoff positions allowed (default: 8)")
    parser.add_argument("--max_energy", type=float, default=0.0,
        help="Maximum relative energy (eV/atom) for structure selection (default: 0.0)")
    parser.add_argument("--max_per_struc", type=int, default=500,
        help="Maximum number of representations per structure (default: 500)")
    parser.add_argument("--label", action="store_true",
        help="Add a label column indicating the source structure index")
    parser.add_argument("--energy", action="store_true",
        help="Include the energy column in the output")
    parser.add_argument("--discrete", type=int, metavar='N_GRIDS',
        help="Use discrete representation for Wyckoff positions with N_GRIDS resolution")
    parser.add_argument("--discrete_cell", action="store_true",
        help="Use discrete representation for cell parameters (requires --discrete)")

    args = parser.parse_args()

    # --- Setup ---
    random.seed(42) # Use a fixed seed for reproducibility
    np.random.seed(42)

    N_wp = args.max_wp
    N_atoms_max = args.max_atoms
    N_atoms_min = 1 # Minimum atoms should be at least 1
    max_E = args.max_energy
    max_per_struc = args.max_per_struc
    max_dof = args.max_dof
    min_spg = args.min_spg
    include_energy_col = args.energy
    include_label_col = args.label
    discrete_cell = args.discrete_cell
    use_discrete_rep = False
    discrete_resolution = None


    # --- Load Data ---
    db = database_topology(args.database)
    # Load energy only if needed for filtering or output
    load_energy_from_db = include_energy_col or max_E > -float('inf') # Check if filtering by energy
    xtals_all = db.get_all_xtals(include_energy=load_energy_from_db)
    print(f"Loaded {len(xtals_all)} structures from {args.database}")

    xtals_filtered = [
        xtal for xtal in xtals_all
        if not load_energy_from_db or xtal.energy <= max_E # Apply energy filter if loaded
    ]
    print(f"Filtered to {len(xtals_filtered)} structures based on initial energy <= {max_E}")

    symdata = []
    engs = []
    for xtal in xtals_filtered:
        wps = [site.wp.get_label() for site in xtal.atom_sites]
        if len(wps) <= N_wp and sum(xtal.numIons) <= N_atoms_max:
            entry = (xtal.group.number, sorted(wps))
            engs.append(xtal.energy)
            if entry not in symdata: symdata.append(entry)
    print(np.min(engs) - np.max(engs), len(engs))
    print(f"Unique space group and Wyckoff position combinations: {len(symdata)}")

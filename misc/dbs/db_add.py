import os
import ase.db as db
from pyxtal import pyxtal
from pyxtal.db import database_topology, mace_opt_single
from glob import glob

# Output merged database file
db = database_topology("lego-sp2.db")
criteria={'CN': {'C': [3]}, 'cutoff': 2.0}

#for cif in glob("Add/*.cif"):
#    print(cif)
#    xtal = pyxtal()
#    xtal.from_seed(cif)
#    xtal, mace_eng, _ = mace_opt_single(0, xtal, 250, 0.05, {})
#    db.add_xtal(xtal)

db.update_row_energy("GULP")
#db.update_row_topology(overwrite=False)

import pandas as pd
from pyxtal.db import database_topology, mace_opt_single
from pyxtal import pyxtal
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os

# Read the CSV file
db = database_topology('lego-sp2.db')
db_ref = database_topology('../LEGO-cryst/data/source/sp2_sacada.db')

ref_data = []
for row in db_ref.db.select():
    if hasattr(row, "topology") and hasattr(row, "mace_energy"):
        ref_data.append(
            (row.topology, row.topology_detail, row.mace_energy))

count = 0
for row in db.db.select():
    if hasattr(row, "topology") and hasattr(row, "mace_energy"):
        for ref in ref_data:
            (top, top_detail, mace_energy) = ref
            if (
                row.topology == top
                and row.topology_detail == top_detail
                and abs(row.mace_energy - mace_energy) < 0.002
            ):
                strs = 'Find {:4d} {:6s}'.format(row.id, row.pearson_symbol)
                strs += ' {:12s} {:10.3f}'.format(row.topology, row.mace_energy)
                strs += f' {mace_energy}'
                print(strs)
                count += 1
                #db.db.update(row.id, remark='Training')
                break
print(count)

import os
from glob import glob
import ase.db as db
from pyxtal import pyxtal

ids = [5, 6, 7, 16, 34, 42, 56, 66, 69, 70, 72, 83, 85, 241, 256, 270, 293, 294,
       351, 360, 800, 854, 904, 1119, 1241]
ids += [31, 38, 39, 61, 62, 150, 163, 252]
#ids += [225, 244, 618, 705, 715, 353, 446, 509, 522, 564, 625, 672, 918, 951, 1100, 1202, 1253, 1379,
#        1474, 1740]
# Output merged database file
db_file = "lego-sp2.db"
merged_db = db.connect(db_file)
for row in merged_db.select():
    #if row.pearson_symbol=='oP12' and row.mace_energy < -9.33: #row.dimension == 3 and 170<row.space_group_number<180 and 15<=row.natoms<=32:
    if row.id in ids:
        name = f"cifs-2/id{row.id}-dim{row.dimension}-{row.topology}-{row.pearson_symbol}-{row.density:.2f}{row.mace_energy:.2f}{row.ff_energy:.2f}.cif"
        with open(name, "w") as f:
            f.write(row.mace_relaxed)
        print(name)

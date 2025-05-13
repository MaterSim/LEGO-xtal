from pyxtal.db import database_topology
from glob import glob
import numpy as np

sp2 = 'data/source/sp2_sacada.db'
db_names = [sp2] + glob('*/final.db')

for db_name in db_names[1:]:
    db = database_topology(db_name)
    # check overlap
    overlaps = db.check_overlap(sp2)
    print(len(overlaps))
    # output creativity
    #ids = [entry[0] for entry in overlaps]
    #db.print_info(ids, cutoff=20)
    # output all structures
    #db.export_structures(folder='out_'+db.db_name[:-3], sort_by='ff_energy', cutoff=100)

# Create a square matrix
matrix = np.zeros([len(db_names), len(db_names)], dtype=int)
for i, db_name in enumerate(db_names):
    db = database_topology(db_name)
    matrix[i, i] = db.db.count()
    for j in range(i+1, len(db_names)):
        db_name2 = db_names[j]
        overlaps = db.check_overlap(db_name2, verbose=False)
        matrix[i, j] = matrix[j, i] = len(overlaps)

# Print the square matrix
for i, db_name in enumerate(db_names):
    if i == 0:
        name = sp2.split("/")[-1]
        labels = f"{name:20s}"
    else:
        labels = "{:20s}".format(db_name)
    for j in range(len(matrix[i])):
        labels += "{:8d}".format(matrix[i, j])
    print(labels)

# Plot mace_energy histogram
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set_context("talk", font_scale=1.1)
f, ax = plt.subplots(figsize=(16, 6))
bins = np.linspace(-9.35, -8, 120)
for db_name in db_names:
    db = database_topology(db_name)
    props = db.get_properties('mace_energy')
    if len(props) > 0:
        label = f"{db_name[-20:]}({len(props)})"
        ax.hist(props, bins, density=False, alpha=0.45, label=label)
ax.set_xlabel('MACE energy (eV/atom)')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig('Mace_energy.pdf')

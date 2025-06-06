import pandas as pd
from pyxtal.db import database_topology, mace_opt_single
from pyxtal import pyxtal
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os

# Read the CSV file
df = pd.read_csv('vasp_energies_r2SCAN.csv')
db = database_topology('../../data/source/lego-sp2.db', ltol=0.3, stol=0.35)

for i in range(len(df)):
    name, _, _, vasp_eng = df.iloc[i]
    if len(name.split('-')) == 1:
        print(f"Skipping {name} as it does not contain a '-'")
        if name == 'graphene':
            E_min = vasp_eng
        continue

    id = int(name.split('-')[0][2:])
    pearson_symbol = name.split('-')[-4]
    found = False
    for row in db.db.select(id=id):
        if row.id == id and row.pearson_symbol == pearson_symbol:
            print(name, row.dimension, row.topology, row.pearson_symbol, vasp_eng, vasp_eng - E_min)
            found = True
            db.db.update(row.id, vasp_energy=vasp_eng-E_min)

            break

    if not found:
        print(f"No match found for {name}")


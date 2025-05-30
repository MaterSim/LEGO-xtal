import pandas as pd
from pyxtal.db import database_topology, mace_opt_single
from pyxtal import pyxtal
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os

# Read the CSV file
df = pd.read_csv('vasp-eng.csv')
db = database_topology('lego-sp2.db', ltol=0.3, stol=0.35)

for i in range(len(df)):
    name, vasp_eng = df.iloc[i]
    vasp_eng /= 1000
    if len(name.split('-')) == 1:
        print(f"Skipping {name} as it does not contain a '-'")
        continue
    symbol = name.split('-')[-3]
    mace_eng = -float(name.split('-')[-1])

    if not os.path.exists(f"Add/{name}.cif"):
        continue

    if vasp_eng > 0.35:
        continue
    print(name, vasp_eng)

    print(name)
    xtal1 = pyxtal()
    xtal1.from_seed(f"Add/{name}.cif", tol=0.01)

    print(xtal1)
    print(xtal1.get_xtal_string(), symbol)
    #xtal1, mace_eng, _ = mace_opt_single(0, xtal1, 200, 0.05, {})

    symbol = xtal1.get_Pearson_Symbol()
    pmg1 = xtal1.to_pymatgen()
    xtal2 = pyxtal()
    for row in db.db.select():
        if hasattr(row, 'vasp_energy'):
            continue
        #print(row.id, row.topology, row.mace_energy, row.pearson_symbol, row.pearson_symbol=='cI228', mace_eng - row.mace_energy)
        if row.pearson_symbol != symbol:
            continue
        #else:
        #    #import sys; sys.exit()
        if  -0.03 < mace_eng - row.mace_energy < 0.05:
            xtal2 = db.get_pyxtal(row.id)
            pmg2 = xtal2.to_pymatgen()
            if xtal1.group.number != xtal2.group.number or len(xtal1.atom_sites) != len(xtal2.atom_sites):
                continue
            if db.matcher.fit(pmg1, pmg2, symmetric=True):
                print(f"Found match for {name} with {row.id} {row.mace_energy} {vasp_eng}")
                db.db.update(row.id, vasp_energy=vasp_eng)
                break
            else:
                print(xtal1)#.get_xtal_string(), mace_eng)
                print(xtal2)#.get_xtal_string(), row.mace_energy)
                #import sys; sys.exit()
    else:
        print(f"No match found for {name}")
        #os.system(f"mv proc/{name}.cif missing/")
        #import sys; sys.exit()


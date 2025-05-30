from pyxtal.db import database_topology

name = 'lego-sp2.db'
db = database_topology(name)
for i in range(1500, 2200):
    xtal = db.get_pyxtal(i, use_relaxed='mace_relaxed')
    if xtal is not None:
        print(i, xtal.get_xtal_string())
    #if xtal.group.number == 224 and sum(xtal.numIons) == 120:
    #    xtal.to_ase().write('2.cif', format='cif')
    #    print(xtal)
    #    import sys; sys.exit()
#db.clean_structures_pmg()

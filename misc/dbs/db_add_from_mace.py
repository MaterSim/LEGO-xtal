from pyxtal.db import database_topology

name = 'clean-sp2.db'
db = database_topology(name)
db.add_strucs_from_db('lego-sp2.db',
                      check=False,
                      freq=10,
                      #use_relaxed='mace_relaxed',
                      sort='mace_energy',
                      criteria={'CN': {'C': [3]}, 'cutoff': 1.95},
                      max_count = 2500,
                      #min_atoms = 0,
                      #max_atoms = 40,
                      same_number = True,
                      )
#db.clean_structures_spg_topology()#dim=3)

from pyxtal.db import database_topology

db = database_topology('data/source/sp2_sacada.db')
db.update_row_topology(overwrite=True)

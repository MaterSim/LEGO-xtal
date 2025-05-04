from pyxtal.db import database_topology
from pyxtal import pyxtal
import argparse
import os
from ast import literal_eval

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Gen Models.")

    # Add arguments
    parser.add_argument('--db', '-d', dest='db', type=str,
                        help='database name')
    parser.add_argument('--output', '-o', dest='output', type=str, default='output',
                        help='output folder')
    parser.add_argument('--code', '-c', dest='code', default='MACE',
                        help='GULP, MACE, VASP')

    args = parser.parse_args()
    code = args.code.lower()
    name = code + '_energy'
    prefix = args.db.split('/')[0]
    db = database_topology(args.db, log_file=prefix + '/db.log')
    dirname = prefix + '/' + args.output
    os.makedirs(dirname, exist_ok=True)

    xtal = pyxtal()
    for row in db.db.select():
        ps, spg, wps = row.pearson_symbol, row.space_group_number, literal_eval(row.wps)
        if hasattr(row, name):
            eng = getattr(row, name)
            den = row.density
            topo = row.topology
            if -9.3 < eng < -8.8 and spg >= 195:
                ps = row.pearson_symbol
                path = f'{dirname}/{prefix}-{ps}-{topo}-E{-eng:.3f}-D{den:.2f}.cif'
                xtal.from_seed(row.toatoms())
                xtal.to_file(path)
                print(f"{row.id:5d} {topo:8s}", xtal.get_xtal_string(dicts={'energy': eng}))

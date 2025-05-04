import os, argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from time import time
from pyxtal.db import database_topology

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Gen Models.")

    # Add arguments
    parser.add_argument('--db', '-d', dest='db', type=str,
                        help='database name')
    parser.add_argument('--ncpu', '-n', dest='ncpu', type=int, default=1,
                        help='N_cpu for parallel computation')
    parser.add_argument('--code', '-c', dest='code', default='MACE',
                        help='GULP, MACE, VASP')
    parser.add_argument('--min', dest='min', type=int, default=1,
                        help='min number of atoms')
    parser.add_argument('--max', dest='max', type=int, default=1000,
                        help='max number of atoms')
    parser.add_argument('--step', '-s', dest='step', type=int, default=200,
                        help='relaxation steps')
    parser.add_argument("--metric", dest='metric', action='store_true',
                        default=False, help="write metric")


    t0 = time()

    args = parser.parse_args()
    folder = args.db.split('/')[0]
    db = database_topology(args.db, log_file=folder+'/'+args.code+'.log')

    db.update_row_energy(args.code,
                         N_atoms=(args.min, args.max),
                         steps=args.step,
                         ncpu=args.ncpu,
                         overwrite=True,
                         use_relaxed=args.code.lower()+'_relaxed')

    print(f"Complete MACE in {args.db} with {(time()-t0)/60:.1f} minutes")

    if args.metric:
        attribute = args.code.lower() + '_energy'
        N_lowE = 0
        N_lowE_cubic = 0
        dirname = args.db.split('/')[-2]
        with open(f'{dirname}/metric.txt', 'a+') as f:
            for row in db.db.select():
                if hasattr(row, attribute):
                    eng = getattr(row, attribute)
                    if -9.4 < eng < -8.8:
                        N_lowE += 1
                        if row.space_group_number >= 195:
                            N_lowE_cubic += 1
            f.write(f'N_lowE_all:      {N_lowE:12d}\n')
            f.write(f'N_lowE_cubic:    {N_lowE_cubic:12d}\n')

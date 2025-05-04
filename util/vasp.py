from pyxtal import pyxtal
from pyxtal.interface.vasp import VASP
import os

if __name__ == "__main__":

    xtal = pyxtal()
    xtal.from_spg_wps_rep(227, ['8a'], [3.6], ['C'])

    # set up the commands
    os.environ['VASP_PP_PATH'] = '/home/x-qiangz/Github/VASP'
    cmd='srun -n $SLURM_NTASKS --mpi=pmi2 /home/x-qiangz/Github/VASP/vasp.5.4.4.pl2/bin/vasp_std > log'

    calc = VASP(xtal, path='tmp', cmd=cmd)
    calc.run()
    print("Energy:", calc.energy)
    print("Forces", calc.forces)

    calc = VASP(xtal, path='tmp', cmd=cmd)
    calc.run(level=4, read_gap=True)
    print("Energy:", calc.energy)
    print("Gap:", calc.gap)

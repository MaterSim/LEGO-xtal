## GULP Installation and test
Download [GULP package](https://gulp.curtin.edu.au/download.html)
```
$ tar -zvxf gulp-6.2-2.tgz
$ cd gulp-6.2/Src 
$ ./mkgulp
```

The compilation may take about 10-20 minutes. And you will find a executable file `gulp` under the `gulp-6.2/Src/` folder.
Now you need to include both path and library to your local environment via either `~/.bashrc` or `~/.zshrc`

```
export PATH=/Users/qzhu8/GitHub/gulp-6.2/Src:$PATH
export GULP_LIB=/Users/qzhu8/GitHub/gulp-6.2/Libraries
```

**To test if GULP is installed correctly, come back to this folder and type**

```
$ gulp < gulp.in
```

You should expect to see the following in the end.

```
  Timing analysis for GULP :

--------------------------------------------------------------------------------
  Task / Subroutine                                          Time (Seconds)
--------------------------------------------------------------------------------
  Calculation of reaxFF energy and derivatives                    0.0749
  Electronegativity equalisation                                  0.0179
  Symmetry generation of equivalent positions                     0.0008
--------------------------------------------------------------------------------
  Total CPU time                                                  0.1086
--------------------------------------------------------------------------------


  Job Finished at 15:00.14  6th May        2024
```



## DFTB+ installation and test

```
$ git clone https://github.com/dftbplus/dftbplus.git
$ pip install cmake
```
Enter the directory and then edit the followings in the file called `config.cmake`

```
option(WITH_OMP "Whether OpenMP thread parallisation should be enabled" FALSE)
option(WITH_SDFTD3 "Whether the s-dftd3 library should be included" TRUE)
```
Then compile and install the code as follows

```
$ FC=gfortran CC=gcc cmake -DCMAKE_INSTALL_PREFIX=$HOME/opt/dftb+ -B _build .
$ cmake --build _build -- -j
$ cmake --install _build
```

**To test if GULP is installed correctly, come back to this folder and type**

```
$ /Users/qzhu8/opt/dftb+/bin/dftb+
```

You should expect to see the following in the end.
```
***  Geometry step: 0

 iSCC Total electronic   Diff electronic      SCC error
>> Charges saved for restart in charges.bin
    1   -0.10199259E+02    0.00000000E+00    0.23923619E-09

Total Energy:                      -10.0526879532 H         -273.5476 eV
Extrapolated to 0K:                -10.0526879542 H         -273.5476 eV
Total Mermin free energy:          -10.0526879552 H         -273.5476 eV
Force related energy:              -10.0526879552 H         -273.5476 eV


--------------------------------------------------------------------------------
DFTB+ running times                          cpu [s]             wall clock [s]
--------------------------------------------------------------------------------
Pre-SCC initialisation                 +       0.03 ( 79.7%)       0.03 ( 80.0%)
--------------------------------------------------------------------------------
Missing                                +       0.01 ( 20.3%)       0.01 ( 20.0%)
Total                                  =       0.04 (100.0%)       0.04 (100.0%)
--------------------------------------------------------------------------------
```

##  MPFR Installation and test (Purdue Anvil only)

To compute the topology, we used the julia package called `CrystalNet.jl`, which requires the use of [MPFR libraray](https://www.mpfr.org/mpfr-current/#download). 
On Purdue-Anvil, the [mpfr](https://www.rcac.purdue.edu/software/mpfr) package is too old. 
In order to check if it works, please run check_topology.py under this folder.

```
$ python check_topology.py
```
if you see that the script is completed successfully, you can stop here. 
Otherwise, **you must manually install the most recent mpfr on Purdue-Anvil.**

```
$ wget https://www.mpfr.org/mpfr-current/mpfr-4.2.1.tar.xz
$ tar -zvxf mpfr-4.2.1.tar.xz 
$ cd mpfr-4.2.1/
$ ./configure --prefix=/home/x-qiangz/libs
$ make
$ make check
$ make install
```

After that, update your environment in `.bashrc`,
```
export LD_LIBRARY_PATH=/home/x-qiangz/libs/lib/:$LD_LIBRARY_PATH
```

Then you can rerun `python check_topology.py` to check if running the script still generates errors. 


## VASP Installation and test

Make sure you have installed the most recent packages
```
pip install vasprun-xml
pip install pyxtal
```

**Note on the VASP 6.4** 
First compile VASP and download the VASP PP file from [here](https://drive.google.com/drive/u/0/folders/15VrDAZ2pkuw4cqeSqCUjX2kj0meSO6W3).
```
# In UNCC cluster
$ module load openmpi/4.1.0-intel
$ module load intel/2020
$ cp arch/makefile.include.intel_ompi_mkl_omp makefile.include
```
Edit the makefile.include
```
FCL        += -mkl
LLIBS      += -L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
```

Second, modify ``cmd`` and ``vasp pseudo potential path`` in ``vasp.py`` 

```
# set up the commands
os.environ['VASP_PP_PATH'] = '/home/x-qiangz/Github/VASP'
cmd='srun -n $SLURM_NTASKS /home/x-qiangz/Github/VASP/vasp.5.4.4.pl2/bin/vasp_std > log'
```


Third, modify the following job script ``myrun_vasp`` if necessary
```
#!/bin/sh -l
#SBATCH --job-name test_vasp
#SBATCH -A dmr180040
#SBATCH -p wholenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --export=ALL
#SBATCH --time=0:30:00
module --force purge # Unload all loaded modules and reset everything to original state.
module load intel  intel-mkl openmpi
python vasp.py > log-vasp
echo "JOB is complete"
```

```
#!/bin/sh -l
#SBATCH --job-name test_vasp
#SBATCH -p Apus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --export=ALL
#SBATCH --time=0:30:00
module load openmpi/4.1.0-intel
module load intel/2020
python vasp.py > log-vasp
echo "JOB is complete"
```

Finally, submit the slurm script and check the ``log-vasp`` file
```
sbatch myrun_vasp
```

```
$ cat log-vasp
 Energy: -73.27065
 Forces [[ 0. -0.  0.]
  [ 0.  0. -0.]
  [-0.  0.  0.]
  [ 0. -0. -0.]
  [-0. -0.  0.]
  [-0. -0.  0.]
  [ 0.  0. -0.]
  [ 0.  0. -0.]]
 Energy: -72.731827
 Gap: 4.073
```

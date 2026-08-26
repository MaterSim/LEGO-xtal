#!/bin/bash
#SBATCH --job-name=juliette-relax # Name of job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1            # number of GPUs per node (max 8 with gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=16      # number of cores per task (1/4 of the 4-GPUs node)
##SBATCH --constraint=mps        # Enable MPS
#SBATCH --hint=nomultithread    # hyperthreading is deactivated
#SBATCH --time=02:00:00         # maximum execution time requested (HH:MM:SS)

#SBATCH --account=yns@v100
##SBATCH -C v100-16g
#SBATCH --partition=gpu_p2
#SBATCH --output=%x-%j.out      # Output file %x is the jobname, %j the jobid
##SBATCH --error=%x-%j.err       # Error file
#SBATCH --qos=qos_gpu-dev       # Uncomment for job requiring 2h
##SBATCH --qos=qos_gpu-t3        # Uncomment for job requiring 20h

source /linkhome/rech/genimm01/uxm25xg/work/python/init.sh
echo "Running on node: $(hostname)"

export pyexec='/linkhome/rech/genimm01/uxm25xg/work/python/LEGO-xtal'
export sourcedb='/linkhome/rech/genimm01/uxm25xg/work/python/LEGO-xtal/data/source'

NCPU=$SLURM_CPUS_PER_TASK
START_TIME=$(date +%s)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun python ${pyexec}/2_relax.py \
  --csv data/sample/train2000-FactorizedVAE-v40-cached-ionic-field-seed42-2000.csv \
  --reference-sio2 alpha_quartz.cif \
  --training-db /linkhome/rech/genimm01/uxm25xg/work/python/LEGO-xtal/data/source/sio2_mp.db \
  --output-dir sio2_relax_final \
  --ncpu ${NCPU} \
  --rcut 3.0 \
  --ff-lib reaxff \
  --end 200

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "Relaxation script completed in $((ELAPSED_TIME / 60)) minutes."


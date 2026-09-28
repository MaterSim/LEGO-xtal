#!/bin/bash
#SBATCH --job-name=lego         # Name of job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:4            # number of GPUs per node (max 8 with gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=16      # number of cores per task (1/4 of the 4-GPUs node)
##SBATCH --constraint=mps        # Enable MPS
#SBATCH --hint=nomultithread    # hyperthreading is deactivated
#SBATCH --time=20:00:00         # maximum execution time requested (HH:MM:SS)

#SBATCH --account=yns@v100
##SBATCH -C v100-16g
#SBATCH --partition=gpu_p2
#SBATCH --output=%x-%j.out      # Output file %x is the jobname, %j the jobid
##SBATCH --error=%x-%j.err       # Error file
##SBATCH --qos=qos_gpu-dev       # Uncomment for job requiring 2h
#SBATCH --qos=qos_gpu-t3        # Uncomment for job requiring 20h


# Print the hostname of the node executing this job
#export OPENBLAS_NUM_THREADS=64
#export OMP_NUM_THREADS=64
#export MKL_NUM_THREADS=64
echo "Running on node: $(hostname)"

source /linkhome/rech/genimm01/uxm25xg/work/python/init.sh

export pyexec='/linkhome/rech/genimm01/uxm25xg/work/python/LEGO-xtal'
export database='/linkhome/rech/genimm01/uxm25xg/work/python/LEGO-xtal/data/source'

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NCPU=$SLURM_CPUS_PER_TASK

START=$(date +%s)

srun python ${pyexec}/0_make_traindata.py \
    --database ./mixed_34_sacada.db \
    --tag mixed34_site_test \
    --rcut 2.4 \
    --max_energy inf \
    --max_atoms 500 \
    --max_wp 8 \
    --max_per_struc 500 \
    --target-coordination \
    --ncpu ${NCPU} --chunksize 1

srun python ${pyexec}/1_train_sample.py \
    --data data/train/mixed34_site_test.csv \
    --model VAE \
    --epochs 250 \
    --nbatch 500 \
    --sample 10000 \
    --seed 42

srun python ${pyexec}/2_relax.py \
    --csv data/sample/mixed34_site_test-VAE-skeleton-dis1-seed42-10000.csv \
    --begin 0 \
    --end 1000 \
    --ncpu ${NCPU} \
    --prototype-cn3 graphite \
    --prototype-cn4 diamond \
    --rcut 2.4 \
    --site-repair-policy map-survivors \
    --skip-overlap \
    --output-dir mixed34_relax_test
    #--skip-topology \
    #--skip-energy \

END=$(date +%s)
ELAPSED_TIME=$((END - START))

echo "LEGO completed in $((ELAPSED_TIME / 60)) minutes."

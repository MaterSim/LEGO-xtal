#!/bin/bash
#SBATCH --job-name=juliette-train # Name of job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1            # number of GPUs per node (max 8 with gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=8       # number of cores per task (1/4 of the 4-GPUs node)
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


# Print the hostname of the node executing this job
#export OPENBLAS_NUM_THREADS=64
#export OMP_NUM_THREADS=64
#export MKL_NUM_THREADS=64
echo "Running on node: $(hostname)"

source /linkhome/rech/genimm01/uxm25xg/work/python/init.sh

export pyexec='/linkhome/rech/genimm01/uxm25xg/work/python/LEGO-xtal'

model=VAE
#model=GAN

#for model in  VAE GAN 
#do
# Check if the CSV file exists, if not create it
NCPU=$((SLURM_CPUS_PER_TASK > 1 ? SLURM_CPUS_PER_TASK - 1 : 1))
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

START=$(date +%s)

srun python ${pyexec}/1_train_sample.py \
  --data train2000.csv \
  --sample 2000 \
  --epochs 250 \
  --geometry-workers ${NCPU}

END=$(date +%s)
ELAPSED_TIME=$((END - START))

echo "Training script completed in $((ELAPSED_TIME / 60)) minutes."

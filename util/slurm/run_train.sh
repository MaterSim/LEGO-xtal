#!/bin/sh -l
#SBATCH --job-name="GAN_VAE"
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Print the hostname of the node executing this job
#export OPENBLAS_NUM_THREADS=64
#export OMP_NUM_THREADS=64
#export MKL_NUM_THREADS=64
echo "Running on node: $(hostname)"
DATAFILE="data/train/${SLURM_JOB_NAME}.csv"
echo $DATAFILE

for model in  VAE #GAN 
do
  # Check if the CSV file exists, if not create it
    START=$(date +%s)
    python 1_train_sample.py --data ${DATAFILE} --epochs 1000 --model ${model} --sample 500000 #--cutoff 100
    END=$(date +%s)
    ELAPSED_TIME=$((END - START))
    echo "Training script completed in $((ELAPSED_TIME / 60)) minutes."
done

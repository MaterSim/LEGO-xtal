#!/bin/sh -l
#SBATCH --job-name="CTGAN_TVAE"
#SBATCH --partition=Apus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --mem=96G
#SBATCH --time=12:00:00

# Print the hostname of the node executing this job
export OPENBLAS_NUM_THREADS=64
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
echo "Running on node: $(hostname)"
DATAFILE="data/train/${SLURM_JOB_NAME}.csv"
echo $DATAFILE

for model in CTGAN TVAE
do
  # Check if the CSV file exists, if not create it
    START=$(date +%s)
    python 1_train_sample.py --data ${DATAFILE} --model ${model} --sample 200000
    END=$(date +%s)
    ELAPSED_TIME=$((END - START))
    echo "Relaxation script completed in $((ELAPSED_TIME / 60)) minutes."
done

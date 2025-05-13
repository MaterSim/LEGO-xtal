#!/bin/sh -l
#SBATCH --job-name="CTGAN_TVAE"
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Print the hostname of the node executing this jobdata
#export OPENBLAS_NUM_THREADS=64
#export OMP_NUM_THREADS=64
#export MKL_NUM_THREADS=64
echo "Running on node: $(hostname)"
DATAFILE="data/train/${SLURM_JOB_NAME}.csv"
echo $DATAFILE

for model in CTGAN TVAE
do
  # Check if the CSV file exists, if not create it
    START=$(date +%s)
    python 1_train_sample.py --data ${DATAFILE} --model ${model} --epochs 1000  --sample 100000 #--cutoff 100
    END=$(date +%s)
    ELAPSED_TIME=$((END - START))
    echo "Training completed in $((ELAPSED_TIME / 60)) minutes."
done

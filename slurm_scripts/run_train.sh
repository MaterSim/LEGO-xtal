#!/bin/sh -l
#SBATCH --job-name="sdv-Hydrus"
#SBATCH --partition=Hydrus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu
#SBATCH --mem=96G
#SBATCH --time=48:00:00

# Print the hostname of the node executing this job
echo "Running on node: $(hostname)"

DATAFILE="data/${SLURM_JOB_NAME}.csv"
echo $DATAFILE

python src/train_sdv.py -d ${DATAFILE} -b 500 -e 500 -m CTGAN -o 100000
python src/train_sdv.py -d ${DATAFILE} -b 500 -e 500 -m TVAE -o 100000

#!/bin/sh -l
#SBATCH --partition=Apus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/dev/null
MODEL=$SLURM_JOB_NAME

# Set RUN_RELAX to 'yes' by default
# Check if a '--no-relax' argument was provided to disable relax.py
RUN_RELAX="yes"
for arg in "$@"
do
  if [ "$arg" = "--no-relax" ]; then
    RUN_RELAX="no"
  fi
done

# Check if the directory does not exist, and create it if needed
if [ ! -d "${MODEL}" ]; then
    mkdir ${MODEL}
fi

# Manually redirect stdout and stderr to log-${MODEL}.txt
exec > ${MODEL}/log-${MODEL}.txt 2>&1

export OMP_NUM_THREADS=1
echo "Running on node: $(hostname)"
NCPU=$SLURM_CPUS_PER_TASK
NCPU1=$((NCPU/2))
NCPU2=$((NCPU/4))

# Conditionally run relax.py if RUN_RELAX is set to "yes"
if [ "$RUN_RELAX" = "yes" ]; then
    echo "Running relax.py with model ${MODEL}"
    python src/relax.py -n ${NCPU} -c data/${MODEL}.csv -e 100000
fi

python row_update.py -n ${NCPU}  -s 250 --min 1   --max 100  -d ${MODEL}/final.db
python row_update.py -n ${NCPU1} -s 100 --min 100 --max 200  -d ${MODEL}/final.db
python row_update.py -n ${NCPU1} -s 100 --min 100 --max 200  -d ${MODEL}/final.db
python row_update.py -n ${NCPU2} -s 50  --min 200 --max 1000 -d ${MODEL}/final.db
python row_update.py -n ${NCPU2} -s 50  --min 200 --max 1000 -d ${MODEL}/final.db
python row_update.py -n ${NCPU2} -s 50  --min 200 --max 1000 -d ${MODEL}/final.db --metric

#sbatch -J CTGAN myrun_relax
#sbatch -J TVAE myrun_relax
#sbatch -J RTF myrun_relax

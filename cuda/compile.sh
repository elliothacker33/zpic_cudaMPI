#!/bin/bash

# =========================================================================== #
# - Course - Parallel Computing (MCA)
# - Authors - Diogo Silva & Tomás Pereira
# =========================================================================== #

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00
#SBATCH --partition normal-a100-40
#SBATCH --gpus=1
#SBATCH --account=f202500010hpcvlabuminhog
#SBATCH --output=tests/slurm_logs/compile_%x_%j.out
#SBATCH --error=tests/slurm_logs/compile_%x_%j.err

ml purge
ml CUDA

mkdir -p tests/slurm_logs

make clean
make Makefile

echo "[DONE] Compilation complete. Executable: zpic"

exit 0


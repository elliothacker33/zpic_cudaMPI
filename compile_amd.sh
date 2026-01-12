#!/bin/bash

# =========================================================================== #
# Compile configuration for AMD EPYC (MPI + OpenMP)                           #
# =========================================================================== #
# - University - University of Minho
# - Course - Parallel Computing (MCA)
# - Authors - Diogo Silva & Tomás Pereira
# =========================================================================== #

#SBATCH -A f202500010hpcvlabuminhox
#SBATCH -p normal-x86
#SBATCH -t 00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=tests/slurm_logs/compile_amd_out.o%j
#SBATCH --error=tests/slurm_logs/compile_amd_err.e%j

# --------- LOAD MODULES -----------------
echo "[STARTING] Loading modules for AMD (MPI + OpenMP)"
modules=(
    "GCC/13.3.0"
    "LLVM/19"
    "OpenMPI/5.0.3-GCC-13.3.0"
)

ml purge

for module in "${modules[@]}"; do
    echo "[LOAD_MODULE] $module"
    ml "$module"
done

mkdir -p tests/slurm_logs

# Force clean old object files (to avoid architecture mismatch)
rm -f src/*.o zpic

# Compile with MPI wrapper (uses clang underneath with OpenMP)
make CC=mpicc CFLAGS="-O3 -ffast-math -march=native -std=c99 -pedantic -Wall -fopenmp" Makefile

echo "[DONE] Compilation complete for AMD (MPI + OpenMP). Executable: zpic"

exit 0

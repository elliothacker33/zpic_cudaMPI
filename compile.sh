#!/bin/bash

# =========================================================================== #
# - Course - Parallel Computing (MCA)
# - Authors - Diogo Silva & Tomás Pereira
# =========================================================================== #

#SBATCH -A f202500010hpcvlabuminhoa
#SBATCH -p normal-arm
#SBATCH -t 00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=tests/slurm_logs/compile_out.o%j
#SBATCH --error=tests/slurm_logs/compile_err.e%j

# --------- LOAD MODULES -----------------
echo "[STARTING] Loading modules"
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

make Makefile CC=mpicc

echo "[DONE] Compilation complete. Executable: zpic"

exit 0

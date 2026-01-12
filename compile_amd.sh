#!/bin/bash

# =========================================================================== #
# Compile configuration for AMD EPYC                                          #
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
#SBATCH --cpus-per-task=1
#SBATCH --output=tests/slurm_logs/compile_amd_out.o%j
#SBATCH --error=tests/slurm_logs/compile_amd_err.e%j

# --------- LOAD MODULES -----------------
echo "[STARTING] Loading modules for AMD"
modules=(
    "GCC/13.3.0"
    "LLVM/19"
)

ml purge

for module in "${modules[@]}"; do
    echo "[LOAD_MODULE] $module"
    ml "$module"
done

mkdir -p tests/slurm_logs

# Force clean old object files (to avoid architecture mismatch)
rm -f src/*.o zpic

make Makefile

echo "[DONE] Compilation complete for AMD. Executable: zpic"

exit 0

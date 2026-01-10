#!/bin/bash
# =========================================================================== #
# ZPIC – Hybrid MPI + OpenMP tests on Fujitsu A64FX (SLURM)                    #
# University of Minho – Parallel Computing (MCA)                              #
# Authors: Diogo Silva & Tomás Pereira                                         #
# =========================================================================== #

#SBATCH -A f202500010hpcvlabuminhoa
#SBATCH -p normal-arm
#SBATCH -t 00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4        
#SBATCH --cpus-per-task=12  
#SBATCH --exclusive
#SBATCH --output=tests/slurm_logs/%x_%j.out
#SBATCH --error=tests/slurm_logs/%x_%j.err

# =========================================================================== #
# MODULES
# =========================================================================== #
ml purge
ml GCC/13.3.0
ml LLVM/19
ml OpenMPI/5.0.3-GCC-13.3.0

# =========================================================================== #
# ARGUMENTS
# =========================================================================== #
TEST_NAME=${1:-run}
RANKS=${2:-4}          # MPI ranks (1–4)
THREADS=${3:-12}       # OpenMP threads per rank (1–12)

# =========================================================================== #
# SANITY CHECKS
# =========================================================================== #
if (( RANKS < 1 || RANKS > 4 )); then
    echo "ERROR: RANKS must be in [1–4] on A64FX"
    exit 1
fi

if (( THREADS < 1 || THREADS > 12 )); then
    echo "ERROR: THREADS must be in [1–12] per CMG"
    exit 1
fi

if [ ! -x "./zpic" ]; then
    echo "ERROR: zpic executable not found"
    exit 1
fi

# =========================================================================== #
# OPENMP ENVIRONMENT
# =========================================================================== #
export OMP_NUM_THREADS=$THREADS
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# =========================================================================== #
# DIRECTORIES
# =========================================================================== #
mkdir -p tests/{slurm_logs,results,perf}

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
TAG="${RANKS}r_${THREADS}t"

# =========================================================================== #
# LAUNCHER BASE (single source of truth)
# =========================================================================== #
SRUN_BASE="srun \
  --ntasks=$RANKS \
  --cpus-per-task=$THREADS \
  --cpu-bind=cores"

# =========================================================================== #
# RUN MODES
# =========================================================================== #
case $TEST_NAME in

  run)
    OUT="tests/results/run_${TAG}_${TIMESTAMP}.txt"
    echo "[RUN] MPI=$RANKS | OMP=$THREADS" | tee "$OUT"
    echo "----------------------------------------" >> "$OUT"

    $SRUN_BASE ./zpic >> "$OUT" 2>&1
    ;;

  perf_stat)
    OUT="tests/perf/perf_stat_${TAG}_${TIMESTAMP}.txt"
    echo "[PERF STAT] MPI=$RANKS | OMP=$THREADS" | tee "$OUT"

    $SRUN_BASE perf stat ./zpic 2>&1 | tee "$OUT"
    ;;

  *)
    echo "Unknown test: $TEST_NAME"
    echo "Valid: run | perf_stat"
    exit 1
    ;;
esac

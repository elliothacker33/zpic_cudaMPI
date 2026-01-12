#!/bin/bash
# =========================================================================== #
# ZPIC – Hybrid MPI + OpenMP tests on AMD EPYC (SLURM)                        #
# University of Minho – Parallel Computing (MCA)                              #
# Authors: Diogo Silva & Tomás Pereira                                         #
# =========================================================================== #

#SBATCH -A f202500010hpcvlabuminhox
#SBATCH -p normal-x86
#SBATCH -t 00:10:00
#SBATCH --nodes=4
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
RANKS=${2:-16}         # MPI ranks (1–64, scales with nodes)
THREADS=${3:-8}        # OpenMP threads per rank (1–16 recommended)

# AMD EPYC 7742: 128 cores per node (2 sockets × 64 cores)
# Recommended: 8 ranks per node × 16 threads = 128 cores
# Or: 16 ranks per node × 8 threads = 128 cores

# =========================================================================== #
# SANITY CHECKS
# =========================================================================== #
if (( RANKS < 1 || RANKS > 64 )); then
    echo "ERROR: RANKS must be in [1–64] (up to 4 nodes × 16 ranks/node)"
    exit 1
fi

if (( THREADS < 1 || THREADS > 64 )); then
    echo "ERROR: THREADS must be in [1–64]"
    exit 1
fi

if [ ! -x "./zpic" ]; then
    echo "ERROR: zpic executable not found. Run: sbatch compile_amd.sh"
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
TAG="amd_${RANKS}r_${THREADS}t"

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
TOTAL_CORES=$((RANKS * THREADS))

case $TEST_NAME in

  run)
    OUT="tests/results/run_${TAG}_${TIMESTAMP}.txt"
    echo "=========================================" | tee "$OUT"
    echo "[RUN] ZPIC Hybrid MPI+OpenMP - AMD EPYC" | tee -a "$OUT"
    echo "=========================================" | tee -a "$OUT"
    echo "MPI Ranks:      $RANKS" | tee -a "$OUT"
    echo "OMP Threads:    $THREADS" | tee -a "$OUT"
    echo "Total Cores:    $TOTAL_CORES" | tee -a "$OUT"
    echo "Nodes:          $SLURM_NNODES" | tee -a "$OUT"
    echo "=========================================" | tee -a "$OUT"
    echo "" >> "$OUT"

    $SRUN_BASE ./zpic 2>&1 | grep -v "Allocating\|Particles per cell" | tee -a "$OUT"
    ;;

  perf_stat)
    OUT="tests/perf/perf_stat_${TAG}_${TIMESTAMP}.txt"
    echo "=========================================" | tee "$OUT"
    echo "[PERF STAT] ZPIC Hybrid MPI+OpenMP - AMD EPYC" | tee -a "$OUT"
    echo "=========================================" | tee -a "$OUT"
    echo "MPI Ranks:      $RANKS" | tee -a "$OUT"
    echo "OMP Threads:    $THREADS" | tee -a "$OUT"
    echo "Total Cores:    $TOTAL_CORES" | tee -a "$OUT"
    echo "Nodes:          $SLURM_NNODES" | tee -a "$OUT"
    echo "=========================================" | tee -a "$OUT"
    echo "" >> "$OUT"

    # perf stat wraps the entire srun command for clean aggregated output
    perf stat -o "${OUT}.perf" -- $SRUN_BASE ./zpic 2>&1 | grep -v "Allocating\|Particles per cell" | tee -a "$OUT"
    
    echo "" >> "$OUT"
    echo "======== PERF STATISTICS ========" >> "$OUT"
    cat "${OUT}.perf" >> "$OUT"
    rm -f "${OUT}.perf"
    ;;

  *)
    echo "Unknown test: $TEST_NAME"
    echo "Valid: run | perf_stat"
    exit 1
    ;;
esac

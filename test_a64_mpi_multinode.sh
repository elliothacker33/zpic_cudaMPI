#!/bin/bash
# =========================================================================== #
# ZPIC – Hybrid MPI + OpenMP tests on Fujitsu A64FX (Multi-Node)              #
# University of Minho – Parallel Computing (MCA)                              #
# Authors: Diogo Silva & Tomás Pereira                                         #
# =========================================================================== #
# Multi-node version - 2 nodes with MPI distributed memory parallelism        #
# =========================================================================== #

#SBATCH -A f202500010hpcvlabuminhoa
#SBATCH -p normal-arm
#SBATCH -t 00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4        
#SBATCH --cpus-per-task=12  
#SBATCH --exclusive
#SBATCH --output=tests/slurm_logs/mpi_multinode_%j.out
#SBATCH --error=tests/slurm_logs/mpi_multinode_%j.err

# =========================================================================== #
# MODULES
# =========================================================================== #
echo "=========================================="
echo "[RUNNING] Executing ZPIC-MPI (Multi-Node)"
echo "=========================================="
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Working directory: $(pwd)"

ml purge
ml GCC/13.3.0
ml LLVM/19
ml OpenMPI/5.0.3-GCC-13.3.0

echo "[INFO] Modules loaded"

# =========================================================================== #
# ARGUMENTS
# =========================================================================== #
TEST_NAME=${1:-compile_run}
NODES=${SLURM_NNODES:-2}
RANKS_PER_NODE=${2:-4}          # MPI ranks per node (1–4 for 4 CMGs)
THREADS=${3:-12}                # OpenMP threads per rank (1–12)

TOTAL_RANKS=$((NODES * RANKS_PER_NODE))

echo ""
echo "=========================================="
echo "[CONFIG] MPI Configuration"
echo "=========================================="
echo "  Nodes: $NODES"
echo "  MPI ranks per node: $RANKS_PER_NODE"
echo "  Total MPI ranks: $TOTAL_RANKS"
echo "  OpenMP threads per rank: $THREADS"
echo "  Total cores used: $((TOTAL_RANKS * THREADS))"
echo "=========================================="
echo ""

# =========================================================================== #
# SANITY CHECKS
# =========================================================================== #
if (( RANKS_PER_NODE < 1 || RANKS_PER_NODE > 4 )); then
    echo "ERROR: RANKS_PER_NODE must be in [1–4] on A64FX (one per CMG)"
    exit 1
fi

if (( THREADS < 1 || THREADS > 12 )); then
    echo "ERROR: THREADS must be in [1–12] per CMG"
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
TAG="${NODES}n_${TOTAL_RANKS}r_${THREADS}t"

# =========================================================================== #
# COMPILE FUNCTION
# =========================================================================== #
compile_mpi() {
    echo "[COMPILING] Building ZPIC with MPI support..."
    
    # Clean previous build
    rm -f zpic_mpi src/*.o 2>/dev/null
    
    # Compile with MPI wrapper
    MPI_CC=mpicc
    MPI_CFLAGS="-O3 -ffast-math -march=native -std=c99 -pedantic -Wall -fopenmp -Isrc -Ilib"
    MPI_LDFLAGS="-lm -fopenmp"
    
    SRC_FILES="src/main.c src/simulation.c src/emf.c src/current.c src/particles.c src/random.c src/timer.c src/zdf.c src/zpic.c"
    
    echo "[CMD] $MPI_CC $MPI_CFLAGS $SRC_FILES $MPI_LDFLAGS -o zpic_mpi"
    $MPI_CC $MPI_CFLAGS $SRC_FILES $MPI_LDFLAGS -o zpic_mpi
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Compilation successful!"
        ls -la zpic_mpi
        return 0
    else
        echo "[ERROR] Compilation failed!"
        return 1
    fi
}

# =========================================================================== #
# LAUNCHER
# =========================================================================== #
SRUN_BASE="srun \
  --nodes=$NODES \
  --ntasks=$TOTAL_RANKS \
  --ntasks-per-node=$RANKS_PER_NODE \
  --cpus-per-task=$THREADS \
  --cpu-bind=cores"

# =========================================================================== #
# RUN MODES
# =========================================================================== #
case $TEST_NAME in

  compile)
    compile_mpi
    ;;

  compile_run)
    compile_mpi
    if [ $? -ne 0 ]; then
        exit 1
    fi
    
    OUT="tests/results/mpi_run_${TAG}_${TIMESTAMP}.txt"
    
    echo "" 
    echo "[INFO] Starting ZPIC execution..."
    echo "-----------------------------------"
    
    {
        echo "=== ZPIC MPI Multi-Node Run ==="
        echo "Nodes: $NODES"
        echo "MPI Ranks: $TOTAL_RANKS"
        echo "OpenMP Threads per rank: $THREADS"
        echo "Date: $TIMESTAMP"
        echo "==============================="
        echo ""
    } > "$OUT"

    $SRUN_BASE ./zpic_mpi >> "$OUT" 2>&1
    
    echo ""
    echo "[INFO] ZPIC execution completed"
    echo "[INFO] End time: $(date)"
    echo "=========================================="
    echo ""
    echo "=== OUTPUT ==="
    cat "$OUT"
    echo ""
    echo "Output saved to: $OUT"
    ;;

  run)
    if [ ! -x "./zpic_mpi" ]; then
        echo "[WARNING] zpic_mpi not found, compiling first..."
        compile_mpi
        if [ $? -ne 0 ]; then
            exit 1
        fi
    fi
    
    OUT="tests/results/mpi_run_${TAG}_${TIMESTAMP}.txt"
    
    echo "[RUN] Nodes=$NODES | MPI=$TOTAL_RANKS | OMP=$THREADS" | tee "$OUT"
    echo "----------------------------------------" >> "$OUT"

    $SRUN_BASE ./zpic_mpi >> "$OUT" 2>&1
    
    echo ""
    cat "$OUT"
    echo ""
    echo "Output saved to: $OUT"
    ;;

  perf_stat)
    if [ ! -x "./zpic_mpi" ]; then
        echo "[WARNING] zpic_mpi not found, compiling first..."
        compile_mpi
        if [ $? -ne 0 ]; then
            exit 1
        fi
    fi
    
    OUT="tests/perf/mpi_perf_stat_${TAG}_${TIMESTAMP}.txt"
    
    echo "[PERF STAT] Nodes=$NODES | MPI=$TOTAL_RANKS | OMP=$THREADS" | tee "$OUT"

    $SRUN_BASE perf stat -e cycles,instructions,cache-references,cache-misses \
        ./zpic_mpi 2>&1 | tee -a "$OUT"
    
    echo ""
    echo "Output saved to: $OUT"
    ;;

  scaling)
    echo "[SCALING TEST] Testing different MPI configurations..."
    
    compile_mpi
    if [ $? -ne 0 ]; then
        exit 1
    fi
    
    OUT="tests/results/mpi_scaling_${TIMESTAMP}.txt"
    
    {
        echo "=== ZPIC MPI Scaling Test ==="
        echo "Date: $TIMESTAMP"
        echo "Nodes: $NODES"
        echo "==============================="
        echo ""
    } > "$OUT"
    
    # Test with different configurations
    for NRANKS in 2 4 8; do
        RPNODE=$((NRANKS / NODES))
        if (( RPNODE < 1 )); then RPNODE=1; fi
        if (( RPNODE > 4 )); then continue; fi
        
        echo "-----------------------------------" >> "$OUT"
        echo "Testing: $NRANKS MPI ranks ($RPNODE per node), $THREADS threads each" >> "$OUT"
        echo "-----------------------------------" >> "$OUT"
        
        srun --nodes=$NODES --ntasks=$NRANKS --ntasks-per-node=$RPNODE \
             --cpus-per-task=$THREADS --cpu-bind=cores \
             ./zpic_mpi >> "$OUT" 2>&1
        
        echo "" >> "$OUT"
    done
    
    echo ""
    cat "$OUT"
    echo ""
    echo "Scaling test output saved to: $OUT"
    ;;

  *)
    echo "Unknown test: $TEST_NAME"
    echo "Valid options: compile | compile_run | run | perf_stat | scaling"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "[DONE] Test completed at $(date)"
echo "=========================================="

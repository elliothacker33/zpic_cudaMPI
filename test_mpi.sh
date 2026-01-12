#!/bin/bash
set -euo pipefail

RANKS=${1:-2}
THREADS=${2:-1}
shift $(( $# > 2 ? 2 : $# ))
EXTRA_ARGS=("$@")

if [ ! -x "./zpic" ]; then
    echo "ERROR: zpic executable not found. Run ./compile.sh first."
    exit 1
fi

export OMP_NUM_THREADS=$THREADS
export OMP_PLACES=cores
export OMP_PROC_BIND=close

mkdir -p tests/results

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
TAG="${RANKS}r_${THREADS}t"
OUT="tests/results/mpi_run_${TAG}_${TIMESTAMP}.txt"

if command -v srun >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
    LAUNCHER=(srun --ntasks="$RANKS" --cpus-per-task="$THREADS" --cpu-bind=cores)
elif command -v mpirun >/dev/null 2>&1; then
    LAUNCHER=(mpirun -np "$RANKS")
elif command -v mpiexec >/dev/null 2>&1; then
    LAUNCHER=(mpiexec -np "$RANKS")
else
    echo "ERROR: mpirun/mpiexec (or srun) not found in PATH"
    exit 1
fi

echo "[RUN] MPI=$RANKS | OMP=$THREADS" | tee "$OUT"
echo "----------------------------------------" >> "$OUT"
"${LAUNCHER[@]}" ./zpic "${EXTRA_ARGS[@]}" >> "$OUT" 2>&1
echo "[DONE] Output: $OUT"

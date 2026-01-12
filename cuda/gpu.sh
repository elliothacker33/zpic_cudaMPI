#!/bin/bash
# =========================================================================== #
# ZPIC – CUDA using 1 A100 GPU                                                #
# University of Minho – Parallel Computing (MCA)                              #
# Authors: Diogo Silva & Tomás Pereira                                         #
# =========================================================================== #

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00
#SBATCH --partition normal-a100-40
#SBATCH --gpus=1
#SBATCH --account=f202500010hpcvlabuminhog
#SBATCH --output=tests/slurm_logs/%x_%j.out
#SBATCH --error=tests/slurm_logs/%x_%j.err

ml purge
ml CUDA

TEST_NAME=${1:-run}

if [ ! -x "./zpic" ]; then
    echo "ERROR: zpic executable not found"
    exit 1
fi

mkdir -p tests/{slurm_logs,results,perf}

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

SRUN_BASE="srun -n1"

case $TEST_NAME in

  run)
    OUT="tests/results/run_${TIMESTAMP}.txt"
    echo "----------------------------------------" >> "$OUT"

    $SRUN_BASE ./zpic >> "$OUT" 2>&1
    ;;

  perf_stat)
    OUT="tests/perf/perf_stat_${TIMESTAMP}.txt"

    $SRUN_BASE perf stat ./zpic 2>&1 | tee "$OUT"
    ;;

  *)
    echo "Unknown test: $TEST_NAME"
    echo "Valid: run | perf_stat"
    exit 1
    ;;
esac

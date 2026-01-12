#!/bin/bash

# =========================================================================== #
# Test configuration for Fujitsu A64FX                                        #
# For the Fujitsu A64FX, we will have a version that uses multi-node and one single node #
# Single node version - also used to test vectorization of SVE instructions   #
# =========================================================================== #
# Test different configurations - ZPIC Project
# =========================================================================== #
# - University - University of Minho
# - Course - Parallel Computing (MCA)
# - Authors - Diogo Silva & Tomás Pereira
# =========================================================================== #

#SBATCH -A f202500010hpcvlabuminhoa
#SBATCH -p normal-arm
#SBATCH -t 00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --output=tests/slurm_logs/test_out.o%j
#SBATCH --error=tests/slurm_logs/test_err.e%j
#SBATCH --exclusive
# #SBATCH --acctg-freq=energy=10

# --------- LOAD MODULES -----------------
echo "[STARTING] Loading modules"
modules=(
    "GCC/13.3.0"
    "LLVM/19"
)

ml purge

for module in "${modules[@]}"; do
    echo "[LOAD_MODULE] $module"
    ml "$module"
done

# --------- PARSE ARGUMENTS -------------
TEST_NAME=$1
CORES=$2

if [ -z "$TEST_NAME" ]; then
    echo "Usage: Insert a right test (run, scorep, perf_stat, perf_record)"
    exit 1
fi

if [ -z "$CORES" ]; then
    CORES=12 # Full saturation of 1 CMG (Core memory group)
fi

echo "ZPIC kernel will be run with $CORES cores"
if [ "$CORES" -gt 48 ] || [ "$CORES" -lt 1 ]; then
    echo "Warning: CORES ($CORES) out of range [1-48]. Setting CORES=12. Full saturation of 1 CMG (Core memory group)"
    CORES=12
fi

# Check if zpic executable exists
if [ ! -f "./zpic" ]; then
    echo "ERROR: zpic executable not found!"
    echo "Please compile first using: sbatch compile.sh"
    exit 1
fi

# --------- RUN TESTS --------------------

mkdir -p tests/slurm_logs
mkdir -p tests/results
mkdir -p tests/perf

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

case $TEST_NAME in
    run)
        echo "Running zpic with $CORES cores"
        RUN_OUTPUT="tests/results/run_${CORES}_threads_${TIMESTAMP}.txt"

        export OMP_NUM_THREADS=$CORES
        export OMP_PLACES=cores
        export OMP_PROC_BIND=close

        echo "=== ZPIC Run ===" > "$RUN_OUTPUT"
        echo "Threads: $CORES" >> "$RUN_OUTPUT"
        echo "Date: $TIMESTAMP" >> "$RUN_OUTPUT"
        echo "===============================" >> "$RUN_OUTPUT"
        echo "" >> "$RUN_OUTPUT"

        srun -c $CORES ./zpic >> "$RUN_OUTPUT" 2>&1

        echo "Output saved to $RUN_OUTPUT"
        ;;

    perf_stat)
        echo "Running perf stat with $CORES cores"
        mkdir -p tests/perf
        PERF_TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
        PERF_DIR="tests/perf/perf_stat_${CORES}_threads_${PERF_TIMESTAMP}.txt"
        export OMP_NUM_THREADS=$CORES
        export OMP_PLACES=cores
        export OMP_PROC_BIND=close
        
        echo "=== ZPIC Perf Stats - ARM A64FX ===" | tee "$PERF_DIR"
        echo "Threads: $CORES" | tee -a "$PERF_DIR"
        echo "Date: $(date)" | tee -a "$PERF_DIR"
        echo "" | tee -a "$PERF_DIR"
        
        # 1. FLOPs (Floating-Point Operations)
        echo "┌─────────────────────────────────────────────────────────────────┐" | tee -a "$PERF_DIR"
        echo "│ 1. FLOPs (Floating-Point Operations)                           │" | tee -a "$PERF_DIR"
        echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$PERF_DIR"
        srun -c $CORES perf stat -e fp_dp_fixed_ops_spec,fp_dp_scale_ops_spec \
            ./zpic 2>&1 | tee -a "$PERF_DIR"
        echo "" | tee -a "$PERF_DIR"
        
        # 2. Memory Traffic
        echo "┌─────────────────────────────────────────────────────────────────┐" | tee -a "$PERF_DIR"
        echo "│ 2. Memory Traffic (Cache Refills & Writebacks)                 │" | tee -a "$PERF_DIR"
        echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$PERF_DIR"
        srun -c $CORES perf stat -e l1d_cache_refill,l2d_cache_refill,l2d_cache_wb \
            ./zpic 2>&1 | tee -a "$PERF_DIR"
        echo "" | tee -a "$PERF_DIR"
        
        # 3. Execution Efficiency
        echo "┌─────────────────────────────────────────────────────────────────┐" | tee -a "$PERF_DIR"
        echo "│ 3. Execution Efficiency (Cycles & Instructions)                │" | tee -a "$PERF_DIR"
        echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$PERF_DIR"
        srun -c $CORES perf stat -e cycles,instructions,stalled-cycles-frontend,stalled-cycles-backend \
            ./zpic 2>&1 | tee -a "$PERF_DIR"
        echo "" | tee -a "$PERF_DIR"
        
        # 4. Complete run for GFLOPS calculation
        echo "┌─────────────────────────────────────────────────────────────────┐" | tee -a "$PERF_DIR"
        echo "│ 4. COMPLETE METRICS (All-in-One for GFLOPS)                    │" | tee -a "$PERF_DIR"
        echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$PERF_DIR"
        PERF_OUTPUT=$(srun -c $CORES perf stat -e task-clock,fp_dp_fixed_ops_spec,fp_dp_scale_ops_spec,cycles,instructions \
            ./zpic 2>&1)
        echo "$PERF_OUTPUT" | tee -a "$PERF_DIR"
        
        # Calculate GFLOPS
        echo "" | tee -a "$PERF_DIR"
        echo "=== GFLOPS Calculation ===" | tee -a "$PERF_DIR"
        
        FP_FIXED=$(echo "$PERF_OUTPUT" | grep "fp_dp_fixed_ops_spec" | awk '{print $1}' | tr -d ',')
        FP_SCALE=$(echo "$PERF_OUTPUT" | grep "fp_dp_scale_ops_spec" | awk '{print $1}' | tr -d ',')
        TIME_MS=$(echo "$PERF_OUTPUT" | grep "task-clock" | awk '{print $1}' | tr -d ',')
        CYCLES=$(echo "$PERF_OUTPUT" | grep -w "cycles" | awk '{print $1}' | tr -d ',')
        INSTR=$(echo "$PERF_OUTPUT" | grep "instructions" | awk '{print $1}' | tr -d ',')
        
        if [ -n "$FP_FIXED" ] && [ -n "$TIME_MS" ]; then
            TOTAL_FLOPS=$((FP_FIXED + FP_SCALE))
            TIME_SEC=$(echo "scale=6; $TIME_MS / 1000" | bc)
            GFLOPS=$(echo "scale=4; $TOTAL_FLOPS / 1000000000" | bc)
            GFLOPS_PER_SEC=$(echo "scale=4; $TOTAL_FLOPS / 1000000000 / $TIME_SEC" | bc)
            
            # CPI calculation
            if [ -n "$INSTR" ] && [ "$INSTR" != "0" ]; then
                CPI=$(echo "scale=2; $CYCLES / $INSTR" | bc)
            else
                CPI="N/A"
            fi
            
            echo "FP Fixed Ops:     $FP_FIXED" | tee -a "$PERF_DIR"
            echo "FP Scale Ops:     $FP_SCALE" | tee -a "$PERF_DIR"
            echo "Total FP Ops:     $TOTAL_FLOPS" | tee -a "$PERF_DIR"
            echo "Execution Time:   $TIME_SEC s" | tee -a "$PERF_DIR"
            echo "Cycles:           $CYCLES" | tee -a "$PERF_DIR"
            echo "Instructions:     $INSTR" | tee -a "$PERF_DIR"
            echo "CPI:              $CPI" | tee -a "$PERF_DIR"
            echo "" | tee -a "$PERF_DIR"
            echo "═══════════════════════════════════════════════════════════════════" | tee -a "$PERF_DIR"
            echo "GFLOPS (total):   $GFLOPS" | tee -a "$PERF_DIR"
            echo "GFLOPS/s:         $GFLOPS_PER_SEC" | tee -a "$PERF_DIR"
            echo "═══════════════════════════════════════════════════════════════════" | tee -a "$PERF_DIR"
        else
            echo "Warning: Could not extract FP operations from perf output" | tee -a "$PERF_DIR"
            echo "Check 'perf list' for available events on this system" | tee -a "$PERF_DIR"
        fi
        
        echo "" | tee -a "$PERF_DIR"
        echo "Perf stats saved to $PERF_DIR"
        ;;

    perf_record)
        echo "Running perf record with $CORES cores"
        mkdir -p tests/perf

        TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
        PERF_DATA="tests/perf/perf_record_${CORES}_threads_${TIMESTAMP}.data"
        PERF_SCRIPT="tests/perf/measurement_${TIMESTAMP}.perf"

        export OMP_NUM_THREADS=$CORES
        export OMP_PLACES=cores
        export OMP_PROC_BIND=close

        echo "Recording performance data..."
        srun -c "$CORES" perf record -g --call-graph dwarf -F 99 -o "$PERF_DATA" -- ./zpic

        echo "Converting to script format..."
        perf script -i "$PERF_DATA" -F +pid > "$PERF_SCRIPT"

        echo "Perf data saved to $PERF_DATA"
        echo "Perf script saved to $PERF_SCRIPT"

        echo ""
        echo "Quick analysis:"
        perf report -i "$PERF_DATA" --stdio --no-children | head -30
        ;;

    *)
        echo "Unknown test name: $TEST_NAME"
        exit 1
        ;;
esac

exit 0


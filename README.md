# ZPIC - Particle in cell Optimizations

The particle in cell simulation is one of the most popular plasma physics simulation.  In this repository we use the infrastructure available at Deucalion to improve this simulation using MPI and CUDA.



## MPI USAGE Multinode

# Compile and run (recommended for first time)
sbatch test_a64_mpi_multinode.sh compile_run

# Just run (if already compiled)
sbatch test_a64_mpi_multinode.sh run

# Run with custom configuration (ranks_per_node, threads)
sbatch test_a64_mpi_multinode.sh run 4 12

# Performance analysis
sbatch test_a64_mpi_multinode.sh perf_stat

# Scaling test (tests 2, 4, 8 MPI ranks)
sbatch test_a64_mpi_multinode.sh scaling
/*
Copyright (C) 2017 Instituto Superior Tecnico

This file is part of the ZPIC Educational code suite

The ZPIC Educational code suite is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

The ZPIC Educational code suite is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with the ZPIC Educational code suite. If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "../lib/zpic.h"
#include "../lib/simulation.h"
#include "../lib/emf.h"
#include "../lib/current.h"
#include "../lib/particles.h"
#include "../lib/timer.h"

// Include Simulation parameters here
#include "input/twostream.c"

int main (int argc, char * argv[]) {

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Declare variables accessible by all ranks
    t_simulation sim;
    int n;
    float t;
    double en_in = 0.0, en_out = 0.0;
    uint64_t t0 = 0, t1 = 0;

    // Only rank 0 initializes the simulation
    if (rank == 0) {
        printf("========================================\n");
        printf("[MPI] Running with %d processes\n", size);
        printf("========================================\n");

        // Initialize simulation
        sim_init(&sim);

        printf("Starting simulation ...\n\n");
        t0 = timer_ticks();
        printf("n = 0, t = 0.0\n");

        // Create temporary buffer for kernel_x
        kernel_tmpbuf_init(sim.current.nx);
    }

    // Broadcast simulation parameters needed by all ranks
    MPI_Bcast(&sim.tmax, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sim.dt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sim.ndump, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sim.n_species, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Main simulation loop - ALL ranks must participate
    for (n = 0, t = 0.0; t <= sim.tmax; n++, t = n * sim.dt) {
        
        // Report before iteration (rank 0 only)
        if (rank == 0) {
            if (report(n, sim.ndump)) sim_report(&sim);
        }

        // ALL ranks call sim_iter (which calls spec_advance with MPI)
        sim_iter(&sim, rank);

        // Report after iteration (only first iteration, rank 0 only)
        if (n == 0 && rank == 0) {
            sim_report_energy_ret(&sim, &en_in);
            sim_report_energy(&sim);
        }
    }

    // Finalization - rank 0 only
    if (rank == 0) {
        kernel_tmpbuf_cleanup();

        printf("n = %i, t = %f\n", n, t);

        t1 = timer_ticks();
        fprintf(stderr, "\nSimulation ended.\n\n");
        sim_report_energy(&sim);
        sim_report_energy_ret(&sim, &en_out);
        printf("Initial energy: %e, Final energy: %e\n", en_in, en_out);
        double ratio = 100 * fabs((en_in - en_out) / en_out);
        printf("\nFinal energy different from Initial Energy. Change in total energy is: %.2f %%\n", ratio);
        if (ratio > 5) {
            printf("ERROR: Large Change\n");
            MPI_Finalize();
            return 1;
        }

        // Simulation times
        sim_timings(&sim, t0, t1);

        // Cleanup data
        sim_delete(&sim);
    }

    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}

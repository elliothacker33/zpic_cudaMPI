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

int main (int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    const int is_root = (mpi_rank == 0);

    // Initialize simulation
    t_simulation sim;
    sim_init(&sim);

    // Run simulation
    int n;
    float t;
    double en_in, en_out;
    
    if (is_root) {
        printf("Starting simulation (%d MPI ranks) ...\n\n", mpi_size);
    }
    uint64_t t0,t1;
    t0 = timer_ticks();
    if (is_root) {
        printf("n = 0, t = 0.0\n");
    }

    // Create temporary buffer for kernel_x
    kernel_tmpbuf_init(sim.current.nx);

    for (n=0,t=0.0; t<=sim.tmax; n++, t=n*sim.dt){
        
        // Report before iteration
        if (is_root && report(n, sim.ndump)) sim_report(&sim);

        sim_iter(&sim);

        // Report after iteration (only first iteration)
        if (is_root && n==0){
            sim_report_energy_ret(&sim, &en_in);
            sim_report_energy (&sim);
        }
    }

    kernel_tmpbuf_cleanup();

    if (is_root) {
        printf("n = %i, t = %f\n",n,t);
    }

    t1 = timer_ticks();
    int exit_code = 0;
    if (is_root) {
        fprintf(stderr, "\nSimulation ended.\n\n");
        sim_report_energy(&sim);
        sim_report_energy_ret(&sim, &en_out);
        printf("Initial energy: %e, Final energy: %e\n", en_in, en_out);
        double ratio=100*fabs((en_in-en_out)/en_out);
        printf("\nFinal energy different from Initial Energy. Change in total energy is: %.2f %% \n",ratio);
        if (ratio>5) {
            printf("ERROR: Large Change\n");
            exit_code = 1;
        }

        // Simulation times
        sim_timings(&sim, t0, t1);
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Cleanup data
    sim_delete(&sim);

    MPI_Finalize();
    
	return exit_code;
}

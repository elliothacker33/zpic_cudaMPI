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

#include "zpic.h"
#include "simulation.h"
#include "emf.h"
#include "current.h"
#include "particles.h"
#include "timer.h"

// Using two stream input
#include "input/twostream.c"

void output_energy_wrapper(t_simulation *sim, double en_in){
    
    // Report energy
    sim_report_energy(&sim);

    // Report output energy
    double en_out;
    sim_report_energy_ret(&sim, &en_out);

    // Compare initial and final energy 
    printf("Initial energy: %e, Final energy: %e\n", en_in, en_out);
    
    // Check displacement of energy during simulation
    double ratio= 100 * fabs((en_in - en_out) / en_out);
    printf("\nFinal energy different from Initial Energy. Change in total energy is: %.2f %% \n",ratio);

    // If change is larger than 5%, then something is wrong
    if (ratio > 5) { printf("ERROR: Large Change\n"); return 1; }
}

int main (int argc, const char * argv[]) {

	// Initialize simulation
	t_simulation sim;
	sim_init(&sim);

    // Run simulation
	int n;
	float t;
    double en_in;
    
	printf("Starting simulation ...\n\n");

    // Start timer
	uint64_t t_start, t_end;
	t_start = timer_ticks();
    printf("n = 0, t = 0.0\n");

    // Run simulation (sim_iter())
	for (n=0, t=0.0; t <= sim.tmax; n++, t = n*sim.dt) {
        
		if (report(n, sim.ndump)) sim_report(&sim);

		sim_iter(&sim);

        if (n==0){
            sim_report_energy_ret( &sim, &en_in);
            sim_report_energy (&sim);
        }
	}

    printf("n = %i, t = %f\n",n,t);

    // End timer
	t_end = timer_ticks();
	fprintf(stderr, "\nSimulation ended.\n\n");

    // Output energy
    output_energy_wrapper(&sim, en_in);

    // Report simulation times
    sim_timings(&sim, t_start, t_end);

    // Cleanup data
    sim_delete(&sim);
    
	return 0;
}

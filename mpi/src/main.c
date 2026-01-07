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

// Std libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ZPIC headers
#include "../lib/zpic.h"
#include "../lib/simulation.h"
#include "../lib/emf.h"
#include "../lib/current.h"
#include "../lib/particles.h"
#include "../lib/timer.h"

// Include Simulation parameters here
#include "input/twostream.c"

// Graph deploys kernels faster than the typical kernel launch
cudaGraph_t graph;
cudaGraphExec_t graph_exec;
bool graph_created = false;

int main (int argc, const char * argv[]) {

    // 1. CPU Initialization
    
    // Initialize simulation
    t_simulation sim;
    sim_init(&sim);

    // Choose CPU or GPU
    sim.arch = SIM_CPU;

    // Main loop variables
    int n;
    float t;
    double en_in, en_out;
    double t0, t1;

    // Timer initialization
    printf("Starting simulation ...\n\n");
    t0 = timer_ticks();
    printf("n = 0, t = 0.0\n");

    if (sim.arch == SIM_CPU){
        // CPU specific context
        printf("Initializing CPU structures ...\n");
        // Temporary buffer for parallelism in current deposition
        kernel_tmpbuf_init(sim.current.nx);
    }
    // Create GPU context
    else if (sim.arch == SIM_GPU){
        // GPU specific context
        // Initialize GPU structures
        printf("Initializing GPU structures ...\n");
        gpu_init(&sim);
        printf("GPU structures initialized\n");
    }
    
    // 2. GPU Handles Simulation
    // Main loop
    for (n=0,t=0.0; t<=sim.tmax; n++, t=n*sim.dt){
        
        // Report before iteration
        //if (report(n, sim.ndump) && sim.arch == SIM_CPU) sim_report(&sim);
        
        // Create graph
        if (!graph_created){
            
            // Capture the graph
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            // Capture the graph
            sim_iter(&sim);

            // Cuda stop the capture
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
            graph_created = true;
        }
        
        /*
        // Report after iteration (only first iteration)        
        if (n==0){
            sim_report_energy_ret(&sim, &en_in);
            sim_report_energy (&sim);
        }
        */
    }

    // GPU Copy back and Cleanup memory
    if (sim.arch == SIM_GPU){
        printf("Cleaning GPU memory ...\n");
        gpu_copy_and_cleanup(&sim);
        printf("GPU cleanup done\n");
    }
    else if (sim.arch == SIM_CPU){
        printf("Cleaning CPU memory ...\n");
        // Cleanup temporary buffer
        kernel_tmpbuf_cleanup();
        printf("CPU cleanup done\n");
    }

    // 3. CPU Cleanup and Reports
    printf("n = %i, t = %f\n",n,t);
    t1 = timer_ticks();
    fprintf(stderr, "\nSimulation ended.\n\n");

    sim_report_energy(&sim);
    sim_report_energy_ret(&sim, &en_out);
    printf("Initial energy: %e, Final energy: %e\n", en_in, en_out);
    double ratio=100*fabs((en_in-en_out)/en_out);
    printf("\nFinal energy different from Initial Energy. Change in total energy is: %.2f %% \n",ratio);
    if (ratio>5) { printf("ERROR: Large Change\n"); return 1;}
    
    // Simulation times
    sim_timings(&sim, t0, t1);
    
    // Cleanup data
    sim_delete(&sim);
    
	return 0;
}

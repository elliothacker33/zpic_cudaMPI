/**
 * @author Diogo Silva, Tomás Pereira
 * @date 2026-01-12
 * @brief This file contains GPU specific functions for the simulation
 * (e.g. GPU initialization, Cuda Kernels, GPU cleanup, etc.)
*/

#ifndef __SIMULATION_GPU__
#define __SIMULATION_GPU__

#include "simulation.h"
#include "particles.h"
#include "emf.h"
#include "current.h"

// GPU simulation structure
/**
 * @brief GPU temporary simulation structure
 * GPU has it's own simulation structure, which is a copy of the CPU one.
*/
typedef struct Simulation_gpu {
	int n_species;			///< Number of particle species
	t_species* species;		///< Particle species
	t_emf emf;				///< EM fields
	t_current current;		///< Electric current density
	float dt;				///< Time step
	float tmax;				///< Final simulation time
	int ndump;				///< Diagnostic frequency
} t_simulation_gpu;

// GPU simulation functions

void gpu_init(t_simulation* sim);

void gpu_cleanup(t_simulation* sim);



 #endif
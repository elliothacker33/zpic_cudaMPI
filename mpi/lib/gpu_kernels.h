/**
 * @file gpu_kernels.h
 * @author Diogo Silva, Tomás Pereira
 * This file contains structures used by GPU kernel on spec_advance and respective kernels
 * The structs don't contain scalar data since they are passed as parameters to the kernels
 */

#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H


#include "particles.h"
#include "emf.h"
#include "current.h"

typedef struct {
    int np;
    int* ix;
    float* x;
    float* ux;
    float* uy;
    float* uz;
    
} t_species_gpu;

typedef struct {

    // Particles
    int n_species;
    t_species_gpu* species;

    // Number of cells
    int nx;
    int gc_current[2]; // Guard cells
    int gc_emf[2]; // Guard cells

    // EM fields
    float* E_chunk_xyz;
    float* E_part_x;
    float* E_part_y;
    float* E_part_z;

    float* B_chunk_xyz;
    float* B_part_x;
    float* B_part_y;
    float* B_part_z;

    // Current fields
    float* J_chunk_xyz;
    float* J_0x;
    float* J_0y;
    float* J_0z;

} gpu_context_t;

// Define functions here 
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocate particles on GPU if needed
 */
void alloc_particles(int free_mem, int species_idx, t_species* spec);

/**
 * @brief Allocate GPU context at host
 */
void alloc_gpu_ctx(t_simulation* sim);

/**
 * @brief Free all GPU resources
 */
void free_gpu_ctx();

/**
 * @brief Update particles on GPU
 */
void update_particles(int species_idx, t_species* spec, bool first_call);

/**
 * @brief Spec advance kernel - Using atomics and a reduction on energy
 */
double spec_advance_gpu(t_species* spec, t_emf* emf, t_current* current, int species_idx);

#ifdef __cplusplus
}
#endif

#endif
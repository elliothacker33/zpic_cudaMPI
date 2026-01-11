/**
* @file spec_advance.cu
* @author Diogo Silva, Tomás Pereira
* @date 2026-01-12
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../lib/gpu_kernels.h"

// Configuration
#define THREADS_PER_BLOCK 256
#define ENERGY_REDUCE_THREADS 512
#define PARTICLE_GROWTH_FACTOR 2.4f 
#define WARP_SIZE 32

// Global GPU state
static gpu_context_t h_gpu_ctx;
static gpu_context_t* d_gpu_ctx = NULL;
static int* h_species_capacity = NULL;

// Energy reduction buffers
static double* d_block_energies = NULL;
static double* d_total_energy = NULL;
static int max_blocks_allocated = 0;

// CUDA streams
static cudaStream_t compute_stream;
static cudaStream_t copy_stream;
static bool streams_initialized = false;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * @brief Warp-level sum reduction
 */
__device__ __forceinline__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

/**
 * @brief Block-level sum reduction (optimized)
 */
__device__ __forceinline__ double block_reduce_sum(double val) {
    __shared__ double warp_sum[32];
    
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) warp_sum[warp] = val;
    
    __syncthreads();
    
    int num_warps = (blockDim.x + 31) >> 5;
    val = (warp == 0 && lane < num_warps) ? warp_sum[lane] : 0.0;
    
    if (warp == 0) val = warp_reduce_sum(val);
    
    return val;
}

/**
 * @brief Optimized particle push kernel with shared memory current buffering
 * 
 * Key optimizations:
 * - Shared memory buffering for current deposition (reduces global atomics)
 * - Coalesced memory access patterns
 * - Register blocking for particle data
 * - Inline energy reduction
 */
__global__ void kernel_spec_advance_push_particles(
    gpu_context_t* d_gpu_ctx, 
    float tem, 
    float dt_dx, 
    float qnx, 
    float q, 
    int species_idx, 
    int np, 
    double* block_energies)
{

    extern __shared__ float shared_current[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    // Load particle buffers (restrict for compiler optimization)
    int* __restrict__ const part_ix = d_gpu_ctx->species[species_idx].ix;
    float* __restrict__ const part_x  = d_gpu_ctx->species[species_idx].x;
    float* __restrict__ const part_ux = d_gpu_ctx->species[species_idx].ux;
    float* __restrict__ const part_uy = d_gpu_ctx->species[species_idx].uy;
    float* __restrict__ const part_uz = d_gpu_ctx->species[species_idx].uz;
    
    // Load EM field buffers
    const float* __restrict__ const E_part_x = d_gpu_ctx->E_part_x;
    const float* __restrict__ const E_part_y = d_gpu_ctx->E_part_y;
    const float* __restrict__ const E_part_z = d_gpu_ctx->E_part_z;
    const float* __restrict__ const B_part_x = d_gpu_ctx->B_part_x;
    const float* __restrict__ const B_part_y = d_gpu_ctx->B_part_y;
    const float* __restrict__ const B_part_z = d_gpu_ctx->B_part_z;
    
    // Load current buffers
    float* __restrict__ const J_0x = d_gpu_ctx->J_0x;
    float* __restrict__ const J_0y = d_gpu_ctx->J_0y;
    float* __restrict__ const J_0z = d_gpu_ctx->J_0z;
    
    const int gc0 = d_gpu_ctx->gc_current[0];
    
    const int shared_size = blockDim.x + 4; 
    float* s_Jx = shared_current;
    float* s_Jy = s_Jx + shared_size;
    float* s_Jz = s_Jy + shared_size;
    
    // Circular buffer to zero shared memory
    for (int i = local_tid; i < shared_size; i += blockDim.x) {
        s_Jx[i] = 0.0f;
        s_Jy[i] = 0.0f;
        s_Jz[i] = 0.0f;
    }

    __syncthreads();
    
    double energy_local = 0.0;
    
    if (tid < np) {
        
        float ux = part_ux[tid];
        float uy = part_uy[tid];
        float uz = part_uz[tid];
        const float x = part_x[tid];
        const int iz = part_ix[tid];
        
        const float w1 = x;
        const int ih = (w1 < 0.5f) ? -1 : 0;
        const int ih_dist = ih + iz;
        const float w1h = w1 + ((w1 < 0.5f) ? 0.5f : -0.5f);
        
        const float Ep_x = E_part_x[ih_dist] * (1.0f - w1h) + E_part_x[ih_dist+1] * w1h;
        const float Ep_y = E_part_y[iz] * (1.0f - w1) + E_part_y[iz+1] * w1;
        const float Ep_z = E_part_z[iz] * (1.0f - w1) + E_part_z[iz+1] * w1;
        
        const float Bp_x = B_part_x[iz] * (1.0f - w1) + B_part_x[iz+1] * w1;
        const float Bp_y = B_part_y[ih_dist] * (1.0f - w1h) + B_part_y[ih_dist+1] * w1h;
        const float Bp_z = B_part_z[ih_dist] * (1.0f - w1h) + B_part_z[ih_dist+1] * w1h;
        
        const float Ep_x_tem = Ep_x * tem;
        const float Ep_y_tem = Ep_y * tem;
        const float Ep_z_tem = Ep_z * tem;
        
        float utx = ux + Ep_x_tem;
        float uty = uy + Ep_y_tem;
        float utz = uz + Ep_z_tem;
        
        const float u2 = utx*utx + uty*uty + utz*utz;
        const float gamma = sqrtf(1.0f + u2);
        energy_local += u2 / (1.0f + gamma);
        
        const float gtem = tem / gamma;
        const float Bp_x_tem = Bp_x * gtem;
        const float Bp_y_tem = Bp_y * gtem;
        const float Bp_z_tem = Bp_z * gtem;
        
        const float otsq = 2.0f / (1.0f + Bp_x_tem*Bp_x_tem + Bp_y_tem*Bp_y_tem + Bp_z_tem*Bp_z_tem);
        
        const float ux_tem = utx + uty*Bp_z_tem - utz*Bp_y_tem;
        const float uy_tem = uty + utz*Bp_x_tem - utx*Bp_z_tem;
        const float uz_tem = utz + utx*Bp_y_tem - uty*Bp_x_tem;
        
        const float Bp_x_otsq = Bp_x_tem * otsq;
        const float Bp_y_otsq = Bp_y_tem * otsq;
        const float Bp_z_otsq = Bp_z_tem * otsq;
        
        utx += uy_tem*Bp_z_otsq - uz_tem*Bp_y_otsq;
        uty += uz_tem*Bp_x_otsq - ux_tem*Bp_z_otsq;
        utz += ux_tem*Bp_y_otsq - uy_tem*Bp_x_otsq;
        
        ux = utx + Ep_x_tem;
        uy = uty + Ep_y_tem;
        uz = utz + Ep_z_tem;
        
        part_ux[tid] = ux;
        part_uy[tid] = uy;
        part_uz[tid] = uz;
        
        const float rg = rsqrtf(1.0f + ux*ux + uy*uy + uz*uz);
        const float dx = dt_dx * rg * ux;
        float x1 = x + dx;
        
        const int di = (x1 >= 1.0f) - (x1 < 0.0f);
        x1 -= (float)di;
        
        part_x[tid] = x1;
        const int new_iz = iz + di;
        part_ix[tid] = new_iz;
        
        const float qvy = q * uy * rg;
        const float qvz = q * uz * rg;
        
        float vp_x0[2], vp_x1[2], vp_dx[2];
        float vp_qvy[2], vp_qvz[2];
        int vp_ix[2];
        int vnp = 1;
        
        vp_x0[0] = x1;
        vp_dx[0] = dx;
        vp_x1[0] = x1;
        vp_qvy[0] = qvy * 0.5f;
        vp_qvz[0] = qvz * 0.5f;
        vp_ix[0] = new_iz;
        
        if (di != 0) {
            int ib = (di == 1);
            float delta = (x + dx - ib) / dx;
            
            vp_x0[1] = 1 - ib;
            vp_x1[1] = x1;
            vp_dx[1] = dx * delta;
            vp_ix[1] = new_iz;
            vp_qvy[1] = vp_qvy[0] * delta;
            vp_qvz[1] = vp_qvz[0] * delta;
            
            vp_x1[0] = ib;
            vp_dx[0] *= (1.0f - delta);
            vp_qvy[0] *= (1.0f - delta);
            vp_qvz[0] *= (1.0f - delta);
            
            vnp++;
        }
        
        #pragma unroll
        for (int k = 0; k < MAX_VNPS; k++) {
            if (k >= vnp) break;
            
            float S0x[2], S1x[2];
            S0x[0] = 1.0f - vp_x0[k];
            S0x[1] = vp_x0[k];
            S1x[0] = 1.0f - vp_x1[k];
            S1x[1] = vp_x1[k];
            
            const int global_idx = vp_ix[k] + gc0;
            const int block_start = blockIdx.x * blockDim.x;
            
            if (global_idx >= block_start - 2 && global_idx < block_start + blockDim.x + 2) {
                int local_idx = global_idx - block_start + 2;    
                atomicAdd(&s_Jx[local_idx], qnx * vp_dx[k]);
                atomicAdd(&s_Jy[local_idx], vp_qvy[k] * (S0x[0] + S1x[0] + (S0x[0] - S1x[0]) * 0.5f));
                atomicAdd(&s_Jy[local_idx + 1], vp_qvy[k] * (S0x[1] + S1x[1] + (S0x[1] - S1x[1]) * 0.5f));
                atomicAdd(&s_Jz[local_idx], vp_qvz[k] * (S0x[0] + S1x[0] + (S0x[0] - S1x[0]) * 0.5f));
                atomicAdd(&s_Jz[local_idx + 1], vp_qvz[k] * (S0x[1] + S1x[1] + (S0x[1] - S1x[1]) * 0.5f));
            } else {
                atomicAdd(&J_0x[global_idx], qnx * vp_dx[k]);
                atomicAdd(&J_0y[global_idx], vp_qvy[k] * (S0x[0] + S1x[0] + (S0x[0] - S1x[0]) * 0.5f));
                atomicAdd(&J_0y[global_idx + 1], vp_qvy[k] * (S0x[1] + S1x[1] + (S0x[1] - S1x[1]) * 0.5f));
                atomicAdd(&J_0z[global_idx], vp_qvz[k] * (S0x[0] + S1x[0] + (S0x[0] - S1x[0]) * 0.5f));
                atomicAdd(&J_0z[global_idx + 1], vp_qvz[k] * (S0x[1] + S1x[1] + (S0x[1] - S1x[1]) * 0.5f));
            }
        }
    }
    
    __syncthreads();
    
    const int block_start = blockIdx.x * blockDim.x + gc0;
    for (int i = local_tid; i < shared_size; i += blockDim.x) {
        if (s_Jx[i] != 0.0f) atomicAdd(&J_0x[block_start - 2 + i], s_Jx[i]);
        if (s_Jy[i] != 0.0f) atomicAdd(&J_0y[block_start - 2 + i], s_Jy[i]);
        if (s_Jz[i] != 0.0f) atomicAdd(&J_0z[block_start - 2 + i], s_Jz[i]);
    }
    
    // Reduce energy
    double energy_block = block_reduce_sum(energy_local);
    if (threadIdx.x == 0) block_energies[blockIdx.x] = energy_block;
}

/**
 * @brief Optimized multi-level energy reduction
 */
__global__ void kernel_energy_reduce(double* block_energies, int n_blocks, double* total_energy) {
    i
    int tid = threadIdx.x;
    
    double sum = 0.0;
    
    for (int i = tid; i < n_blocks; i += blockDim.x) {
        sum += block_energies[i];
    }
    
    sum = block_reduce_sum(sum);
    
    if (tid == 0) *total_energy = sum;
}

/**
 * @brief Allocate particles
 * @param free_mem True if the memory should be freed
 * @param species_idx Index of the species to update
 * @param spec Pointer to the species data
 * In this function we use a growth factor to avoid frequent reallocations since communication
 * trough PCI-E is expensive.
 */
void alloc_particles(int free_mem, int species_idx, t_species* spec) {
    
    t_species_gpu h_spec_temp;
    t_species_gpu* d_spec_ptr = h_gpu_ctx.species + species_idx;
    
    CUDA_CHECK(cudaMemcpy(&h_spec_temp, d_spec_ptr, sizeof(t_species_gpu), cudaMemcpyDeviceToHost));
    
    if (free_mem) {
        if (h_spec_temp.ix) cudaFree(h_spec_temp.ix);
        if (h_spec_temp.x)  cudaFree(h_spec_temp.x);
        if (h_spec_temp.ux) cudaFree(h_spec_temp.ux);
        if (h_spec_temp.uy) cudaFree(h_spec_temp.uy);
        if (h_spec_temp.uz) cudaFree(h_spec_temp.uz);
    }
    
    // Pre-Allocate for the next iterations
    int alloc_size = (int)(spec->np * PARTICLE_GROWTH_FACTOR);
    
    CUDA_CHECK(cudaMalloc((void**)&h_spec_temp.ix, alloc_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&h_spec_temp.x,  alloc_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&h_spec_temp.ux, alloc_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&h_spec_temp.uy, alloc_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&h_spec_temp.uz, alloc_size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(h_spec_temp.ix, spec->part.ix, spec->np * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_spec_temp.x,  spec->part.x,  spec->np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_spec_temp.ux, spec->part.ux, spec->np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_spec_temp.uy, spec->part.uy, spec->np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_spec_temp.uz, spec->part.uz, spec->np * sizeof(float), cudaMemcpyHostToDevice));
    
    h_spec_temp.np = spec->np;
    h_spec_temp.np_max = alloc_size;
    
    CUDA_CHECK(cudaMemcpy(d_spec_ptr, &h_spec_temp, sizeof(t_species_gpu), cudaMemcpyHostToDevice));
    
    // Change capacity of this species for realloc
    if (h_species_capacity != NULL) {
        h_species_capacity[species_idx] = alloc_size;
    }
    
    // Reallocate needed blocks for energy reduction since number of particles changed
    int needed_blocks = (spec->np + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (needed_blocks > max_blocks_allocated) {
        if (d_block_energies) cudaFree(d_block_energies);
        CUDA_CHECK(cudaMalloc((void**)&d_block_energies, needed_blocks * sizeof(double)));
        max_blocks_allocated = needed_blocks;
    }
}

/**
 * @brief Initialize GPU context (called once at startup)
 * @param sim Pointer to the simulation data
 */
void alloc_gpu_ctx(t_simulation* sim) {
    
    // Create CUDA streams
    if (!streams_initialized) {
        CUDA_CHECK(cudaStreamCreate(&compute_stream));
        CUDA_CHECK(cudaStreamCreate(&copy_stream));
        streams_initialized = true;
    }
    
    // Create device pointer
    if (d_gpu_ctx == NULL) {
        CUDA_CHECK(cudaMalloc((void**)&d_gpu_ctx, sizeof(gpu_context_t)));
    }
    
    // Create capacity buffer    
    if (h_species_capacity == NULL) {
        h_species_capacity = (int*)calloc(sim->n_species, sizeof(int));
    }
    
    // Host context (scalars)
    h_gpu_ctx.num_species = sim->n_species;
    h_gpu_ctx.gc_current[0] = sim->current.gc[0];
    h_gpu_ctx.gc_current[1] = sim->current.gc[1];
    h_gpu_ctx.gc_emf[0] = sim->emf.gc[0];
    h_gpu_ctx.gc_emf[1] = sim->emf.gc[1];
    h_gpu_ctx.nx = sim->emf.nx;
    
    // Device energy for reduction
    if (d_total_energy == NULL) {
        CUDA_CHECK(cudaMalloc((void**)&d_total_energy, sizeof(double)));
    }
    
    // Pre-allocate species array
    CUDA_CHECK(cudaMalloc((void**)&h_gpu_ctx.species, sim->n_species * sizeof(t_species_gpu)));
    t_species_gpu* temp_species = (t_species_gpu*)calloc(sim->n_species, sizeof(t_species_gpu));
    CUDA_CHECK(cudaMemcpy(h_gpu_ctx.species, temp_species, sim->n_species * sizeof(t_species_gpu), cudaMemcpyHostToDevice));
    free(temp_species);
    
    // Allocate EM field buffers 
    // This buffers are only allocated once per simulation since the parameters are 
    // constant for all species
    int size_emf = sim->emf.gc[0] + sim->emf.nx + sim->emf.gc[1];
    CUDA_CHECK(cudaMalloc((void**)&h_gpu_ctx.E_chunk_xyz, 3 * size_emf * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&h_gpu_ctx.B_chunk_xyz, 3 * size_emf * sizeof(float)));
    
    h_gpu_ctx.E_part_x = h_gpu_ctx.E_chunk_xyz + sim->emf.gc[0];
    h_gpu_ctx.E_part_y = h_gpu_ctx.E_chunk_xyz + sim->emf.gc[0] + size_emf;
    h_gpu_ctx.E_part_z = h_gpu_ctx.E_chunk_xyz + sim->emf.gc[0] + 2 * size_emf;
    
    h_gpu_ctx.B_part_x = h_gpu_ctx.B_chunk_xyz + sim->emf.gc[0];
    h_gpu_ctx.B_part_y = h_gpu_ctx.B_chunk_xyz + sim->emf.gc[0] + size_emf;
    h_gpu_ctx.B_part_z = h_gpu_ctx.B_chunk_xyz + sim->emf.gc[0] + 2 * size_emf;
    
    // Allocate current buffers 
    int size_current = sim->current.gc[0] + sim->current.nx + sim->current.gc[1];
    CUDA_CHECK(cudaMalloc((void**)&h_gpu_ctx.J_chunk_xyz, 3 * size_current * sizeof(float)));
    
    h_gpu_ctx.J_0x = h_gpu_ctx.J_chunk_xyz + sim->current.gc[0];
    h_gpu_ctx.J_0y = h_gpu_ctx.J_chunk_xyz + sim->current.gc[0] + size_current;
    h_gpu_ctx.J_0z = h_gpu_ctx.J_chunk_xyz + sim->current.gc[0] + 2 * size_current;
    
    // Transfer context to device 
    CUDA_CHECK(cudaMemcpy(d_gpu_ctx, &h_gpu_ctx, sizeof(gpu_context_t), cudaMemcpyHostToDevice));
}

/**
 * @brief Free all GPU resources
 */
void free_gpu_ctx() {
    
    // Destroy CUDA streams
    if (streams_initialized) {
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(copy_stream);
        streams_initialized = false;
    }
    
    // Free device energy reduction buffer
    if (d_block_energies) {
        cudaFree(d_block_energies);
        d_block_energies = NULL;
    }
    if (d_total_energy) {
        cudaFree(d_total_energy);
        d_total_energy = NULL;
    }

    // Free chunk buffers
    if (h_gpu_ctx.E_chunk_xyz) cudaFree(h_gpu_ctx.E_chunk_xyz);
    if (h_gpu_ctx.B_chunk_xyz) cudaFree(h_gpu_ctx.B_chunk_xyz);
    if (h_gpu_ctx.J_chunk_xyz) cudaFree(h_gpu_ctx.J_chunk_xyz);
    
    // Free species data
    if (h_gpu_ctx.species) {
        for (int i = 0; i < h_gpu_ctx.num_species; i++) {
            t_species_gpu h_spec;
            cudaMemcpy(&h_spec, h_gpu_ctx.species + i, sizeof(t_species_gpu), cudaMemcpyDeviceToHost);
            if (h_spec.ix) cudaFree(h_spec.ix);
            if (h_spec.x)  cudaFree(h_spec.x);
            if (h_spec.ux) cudaFree(h_spec.ux);
            if (h_spec.uy) cudaFree(h_spec.uy);
            if (h_spec.uz) cudaFree(h_spec.uz);
        }
        cudaFree(h_gpu_ctx.species);
    }
    
    // Free device pointer
    if (d_gpu_ctx) {
        cudaFree(d_gpu_ctx);
        d_gpu_ctx = NULL;
    }

    // Free capacity buffer
    if (h_species_capacity) {
        free(h_species_capacity);
        h_species_capacity = NULL;
    }
    
    // Free first call flags
    if (first_call_flags) {
        free(first_call_flags);
        first_call_flags = NULL;
    }

    max_blocks_allocated = 0;
}

/**
 * @brief Update particles on GPU (minimal transfers)
 * @param species_idx Index of the species to update
 * @param spec Pointer to the species data
 * @param first_call True if this is the first call to the function
 */
static inline void update_particles(int species_idx, t_species* spec, bool first_call) {
    
    t_species_gpu h_spec_temp;
    CUDA_CHECK(cudaMemcpy(&h_spec_temp, h_gpu_ctx.species + species_idx, sizeof(t_species_gpu), cudaMemcpyDeviceToHost));
    
    // Full copy on first call or when the moving window was updated
    if (first_call || spec->moving_window || spec->bc_type == PART_BC_OPEN) {
        
        // Full copy needed
        CUDA_CHECK(cudaMemcpyAsync(h_spec_temp.ix, spec->part.ix, spec->np * sizeof(int), cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(h_spec_temp.x,  spec->part.x,  spec->np * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(h_spec_temp.ux, spec->part.ux, spec->np * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(h_spec_temp.uy, spec->part.uy, spec->np * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(h_spec_temp.uz, spec->part.uz, spec->np * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
    } else {
        
        // Only copy new positions on grid cell
        CUDA_CHECK(cudaMemcpyAsync(h_spec_temp.ix, spec->part.ix, spec->np * sizeof(int), cudaMemcpyHostToDevice, copy_stream));
    }
    
    // Update particle data
    h_spec_temp.np = spec->np;
    CUDA_CHECK(cudaMemcpy(h_gpu_ctx.species + species_idx, &h_spec_temp, sizeof(t_species_gpu), cudaMemcpyHostToDevice));
}

/**
 * @brief Advanced particls on GPU A100
 * @param spec Pointer to the species data
 * @param emf Pointer to the electric field data
 * @param current Pointer to the current data
 * @param species_idx Index of the species to update
 */
double spec_advance_gpu(t_species* spec, t_emf* emf, t_current* current, int species_idx) {
    
    /// 1 - CPU to GPU transfer
    /// Optimized to do the least of transfers possible - Only necessary data per iteration
    /// Open boundaries are also handled

    const int np = spec->np;
    if (np == 0) return 0.0;
    
    static int* first_call_flags = NULL;
    if (first_call_flags == NULL) {
        first_call_flags = (int*)calloc(h_gpu_ctx.num_species, sizeof(int));
    }
    
    bool first_call = (first_call_flags[species_idx] == 0);
    
    // Handle particle allocation with growth factor
    if (h_species_capacity[species_idx] < np) {
        alloc_particles(h_species_capacity[species_idx] > 0, species_idx, spec);
        first_call = true;
    }
    
    // Minimal particle updates
    update_particles(species_idx, spec, first_call);
    
    // Update EM fields (async copy overlapped with previous kernel)
    int size_emf = emf->gc[0] + emf->nx + emf->gc[1];
    CUDA_CHECK(cudaMemcpyAsync(h_gpu_ctx.E_chunk_xyz, emf->E_buf.chunk_xyz, 3 * size_emf * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_gpu_ctx.B_chunk_xyz, emf->B_buf.chunk_xyz, 3 * size_emf * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
    
    int size_current = current->gc[0] + current->nx + current->gc[1];
    CUDA_CHECK(cudaMemsetAsync(h_gpu_ctx.J_chunk_xyz, 0, 3 * size_current * sizeof(float), copy_stream));
    
    // Wait for copies to complete before kernel launch
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));
    
    // Kernel parameters
    const float tem   = 0.5f * (spec->dt / spec->m_q);
    const float dt_dx = spec->dt / spec->dx;
    const float qnx   = (spec->q * spec->dx) / spec->dt;
    
    const int numBlocks = (np + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int shared_mem_size = 3 * (THREADS_PER_BLOCK + 4) * sizeof(float);
    
    // 2. Kernel launch
    // Spec advance (Using atomics could be faster with sort and shared)
    // Since it's specified to not use sort we keep only atomics approach.
    kernel_spec_advance_push_particles<<<numBlocks, THREADS_PER_BLOCK, shared_mem_size, compute_stream>>>(d_gpu_ctx, tem, dt_dx, qnx, spec->q, species_idx, np, d_block_energies);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    
    // Kernel Energy reduction on blocks of different SMs
    kernel_energy_reduce<<<1, ENERGY_REDUCE_THREADS, 0, compute_stream>>>(d_block_energies, numBlocks, d_total_energy);
    CUDA_CHECK(cudaGetLastError());

    
    // 3. COPY GPU-CPU
    // Copyback results
    t_species_gpu h_spec_ptr_copy;
    CUDA_CHECK(cudaMemcpyAsync(&h_spec_ptr_copy, h_gpu_ctx.species + species_idx, sizeof(t_species_gpu), cudaMemcpyDeviceToHost, copy_stream));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));
    
    CUDA_CHECK(cudaMemcpyAsync(spec->part.ix, h_spec_ptr_copy.ix, np * sizeof(int), 
                               cudaMemcpyDeviceToHost, copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(spec->part.x,  h_spec_ptr_copy.x,  np * sizeof(float), 
                               cudaMemcpyDeviceToHost, copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(spec->part.ux, h_spec_ptr_copy.ux, np * sizeof(float), 
                               cudaMemcpyDeviceToHost, copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(spec->part.uy, h_spec_ptr_copy.uy, np * sizeof(float), 
                               cudaMemcpyDeviceToHost, copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(spec->part.uz, h_spec_ptr_copy.uz, np * sizeof(float), 
                               cudaMemcpyDeviceToHost, copy_stream));
    
    CUDA_CHECK(cudaMemcpyAsync(current->J_buf.chunk_xyz, h_gpu_ctx.J_chunk_xyz, 
                               3 * size_current * sizeof(float), cudaMemcpyDeviceToHost, copy_stream));
    
    // Get energy
    double h_energy_val;
    CUDA_CHECK(cudaMemcpyAsync(&h_energy_val, d_total_energy, sizeof(double), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));
    
    first_call_flags[species_idx] = 1;
    
    return h_energy_val;
}
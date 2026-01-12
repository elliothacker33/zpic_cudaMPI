/**
 * @file gpu_kernels.cu
 * @author Ricardo Fonseca, Diogo Silva, Tomás Pereira
 * @brief GPU kernels
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
#define WARP_SIZE 32

// Global GPU state
static gpu_context_t h_gpu_ctx;
static gpu_context_t* d_gpu_ctx = NULL;
static int* first_call_flags = NULL;

// Energy reduction buffers
static double* d_block_energies = NULL;
static double* d_total_energy = NULL;

// CUDA streams
static cudaStream_t compute_stream;
static cudaStream_t copy_stream;
static int streams_initialized = 0;

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
 * @brief Warp-level reduction
 */
__device__ __forceinline__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

/**
 * @brief Block-level reduction
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
 * @brief Particle push kernel and current deposition
 */
__global__ void kernel_spec_advance_push_particles(
    gpu_context_t* d_gpu_ctx, 
    float tem, 
    float dt_dx, 
    float qnx, 
    float q, 
    int species_idx, 
    int np, 
    double* d_block_energies)
{
	 int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Only can do shared if maybe the particles are sorted. If not its possible but the cost is stlii high
    // due to atomics will be mostly unique to global and not accumulate on shared
    // Depends a lot on simulation

    int* __restrict__ const part_ix = d_gpu_ctx->species[species_idx].ix;
    float* __restrict__ const part_x  = d_gpu_ctx->species[species_idx].x;
    float* __restrict__ const part_ux = d_gpu_ctx->species[species_idx].ux;
    float* __restrict__ const part_uy = d_gpu_ctx->species[species_idx].uy;
    float* __restrict__ const part_uz = d_gpu_ctx->species[species_idx].uz;

    const float* __restrict__ const E_part_x = d_gpu_ctx->E_part_x;
    const float* __restrict__ const E_part_y = d_gpu_ctx->E_part_y;
    const float* __restrict__ const E_part_z = d_gpu_ctx->E_part_z;
    const float* __restrict__ const B_part_x = d_gpu_ctx->B_part_x;
    const float* __restrict__ const B_part_y = d_gpu_ctx->B_part_y;
    const float* __restrict__ const B_part_z = d_gpu_ctx->B_part_z;

    float* __restrict__ const J_0x = d_gpu_ctx->J_0x;
    float* __restrict__ const J_0y = d_gpu_ctx->J_0y;
    float* __restrict__ const J_0z = d_gpu_ctx->J_0z;

    double energy_local = 0.0;

    if (tid < np) {

        // Particle push

        float ux = part_ux[tid];
        float uy = part_uy[tid];
        float uz = part_uz[tid];
        float x = part_x[tid];
        int iz = part_ix[tid];

        float w1 = x;
        int ih = (w1 < 0.5f) ? -1 : 0;
        float w1h = w1 + ((w1 < 0.5f) ? 0.5f : -0.5f);
        int ih_dist = ih + iz;

        float Ep_x = E_part_x[ih_dist] * (1.0f - w1h) + E_part_x[ih_dist+1] * w1h;
        float Ep_y = E_part_y[iz] * (1.0f - w1)  + E_part_y[iz+1] * w1;
        float Ep_z = E_part_z[iz] * (1.0f - w1)  + E_part_z[iz+1] * w1;

        float Bp_x = B_part_x[iz] * (1.0f - w1)  + B_part_x[iz+1] * w1;
        float Bp_y = B_part_y[ih_dist] * (1.0f - w1h) + B_part_y[ih_dist+1] * w1h;
        float Bp_z = B_part_z[ih_dist] * (1.0f - w1h) + B_part_z[ih_dist+1] * w1h;

        Ep_x *= tem;
        Ep_y *= tem;
        Ep_z *= tem;

        float utx = ux + Ep_x;
        float uty = uy + Ep_y;
        float utz = uz + Ep_z;

        float u2 = utx*utx + uty*uty + utz*utz;
        float gamma = sqrtf(1.0f + u2);
        energy_local += u2 / (1.0f + gamma);

        float gtem = tem / gamma;
        Bp_x *= gtem;
        Bp_y *= gtem;
        Bp_z *= gtem;

        float otsq = 2.0f / (1.0f + Bp_x*Bp_x + Bp_y*Bp_y + Bp_z*Bp_z);

        float ux_temp = utx + uty*Bp_z - utz*Bp_y;
        float uy_temp = uty + utz*Bp_x - utx*Bp_z;
        float uz_temp = utz + utx*Bp_y - uty*Bp_x;

        Bp_x *= otsq;
        Bp_y *= otsq;
        Bp_z *= otsq;

        utx += uy_temp*Bp_z - uz_temp*Bp_y;
        uty += uz_temp*Bp_x - ux_temp*Bp_z;
        utz += ux_temp*Bp_y - uy_temp*Bp_x;

        ux = utx + Ep_x;
        uy = uty + Ep_y;
        uz = utz + Ep_z;

        part_ux[tid] = ux;
        part_uy[tid] = uy;
        part_uz[tid] = uz;

        float rg = rsqrtf(1.0f + ux*ux + uy*uy + uz*uz);
        float dx = dt_dx * rg * ux;
        float x1 = x + dx;
        int di = (x1 >= 1.0f) - (x1 < 0.0f);
        x1 -= (float)di;

        part_x[tid] = x1;
        part_ix[tid] = iz + di;

        float qvy = q * uy * rg;
        float qvz = q * uz * rg;

        float vp_x0[2], vp_x1[2], vp_dx[2], vp_qvy[2], vp_qvz[2];
        int vp_ix[2];
        int vnp = 1;

        vp_x0[0] = x;
        vp_dx[0] = dx;
        vp_x1[0] = x + dx;
        vp_qvy[0] = qvy * 0.5f;
        vp_qvz[0] = qvz * 0.5f;
        vp_ix[0] = iz;

        if (di != 0) {
            int ib = (di == 1);
            float delta = (x + dx - ib) / dx;

            vp_x0[1] = 1 - ib;
            vp_x1[1] = (x + dx) - di;
            vp_dx[1] = dx * delta;
            vp_ix[1] = iz + di;
            vp_qvy[1] = vp_qvy[0] * delta;
            vp_qvz[1] = vp_qvz[0] * delta;

            vp_x1[0] = ib;
            vp_dx[0] *= (1.0f - delta);
            vp_qvy[0] *= (1.0f - delta);
            vp_qvz[0] *= (1.0f - delta);

            vnp = 2;
        }

        // Deposit current
        for (int k = 0; k < vnp; k++) {
            float S0x[2], S1x[2];
            S0x[0] = 1.0f - vp_x0[k];
            S0x[1] = vp_x0[k];
            S1x[0] = 1.0f - vp_x1[k];
            S1x[1] = vp_x1[k];

            int idx = vp_ix[k];
            const int block_start = blockIdx.x * blockDim.x;

            atomicAdd(&J_0x[idx], qnx * vp_dx[k]);

            float weight0 = (S0x[0] + S1x[0] + (S0x[0] - S1x[0]) * 0.5f);
            float weight1 = (S0x[1] + S1x[1] + (S0x[1] - S1x[1]) * 0.5f);

            atomicAdd(&J_0y[idx],     vp_qvy[k] * weight0);
            atomicAdd(&J_0y[idx + 1], vp_qvy[k] * weight1);
            atomicAdd(&J_0z[idx],     vp_qvz[k] * weight0);
            atomicAdd(&J_0z[idx + 1], vp_qvz[k] * weight1);
        }
    }


    // Energy reduction
    double energy_block = block_reduce_sum(energy_local);
    if (threadIdx.x == 0) {
        d_block_energies[blockIdx.x] = energy_block;
    }
}

/**
 * @brief Energy reduction kernel
 * Energy reduction trough Warp-level reduction
 * Single-block reduction
 */
__global__ void kernel_energy_reduce(double* block_energies, int n_blocks, double* total_energy) {
    
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
 */
void alloc_particles(int species_idx, t_species* spec) {
    
    t_species_gpu h_spec_temp;
    t_species_gpu* d_spec_ptr = h_gpu_ctx.species + species_idx;
    
    CUDA_CHECK(cudaMemcpy(&h_spec_temp, d_spec_ptr, sizeof(t_species_gpu), cudaMemcpyDeviceToHost));
    
    // Only for realloc future implementations
    // We did a version with realloc but it's not needed for this configuration so we removed it
    if (h_spec_temp.ix) cudaFree(h_spec_temp.ix);
    if (h_spec_temp.x)  cudaFree(h_spec_temp.x);
    if (h_spec_temp.ux) cudaFree(h_spec_temp.ux);
    if (h_spec_temp.uy) cudaFree(h_spec_temp.uy);
    if (h_spec_temp.uz) cudaFree(h_spec_temp.uz);
      
    CUDA_CHECK(cudaMalloc((void**)&h_spec_temp.ix, spec->np * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&h_spec_temp.x,  spec->np * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&h_spec_temp.ux, spec->np * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&h_spec_temp.uy, spec->np * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&h_spec_temp.uz, spec->np * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(h_spec_temp.ix, spec->part.ix, spec->np * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_spec_temp.x,  spec->part.x,  spec->np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_spec_temp.ux, spec->part.ux, spec->np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_spec_temp.uy, spec->part.uy, spec->np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_spec_temp.uz, spec->part.uz, spec->np * sizeof(float), cudaMemcpyHostToDevice));
    
    h_spec_temp.np = spec->np;
    
    CUDA_CHECK(cudaMemcpy(d_spec_ptr, &h_spec_temp, sizeof(t_species_gpu), cudaMemcpyHostToDevice));
}

/**
 * @brief Initialize GPU context
 */
void alloc_gpu_ctx(t_simulation* sim) {
    
    if (!streams_initialized) {
        CUDA_CHECK(cudaStreamCreate(&compute_stream));
        CUDA_CHECK(cudaStreamCreate(&copy_stream));
        streams_initialized = 1;
    }
    
    if (d_gpu_ctx == NULL) {
        CUDA_CHECK(cudaMalloc((void**)&d_gpu_ctx, sizeof(gpu_context_t)));
    }
     
    h_gpu_ctx.n_species = sim->n_species;
    h_gpu_ctx.gc_current[0] = sim->current.gc[0];
    h_gpu_ctx.gc_current[1] = sim->current.gc[1];
    h_gpu_ctx.gc_emf[0] = sim->emf.gc[0];
    h_gpu_ctx.gc_emf[1] = sim->emf.gc[1];
    h_gpu_ctx.nx = sim->emf.nx;
    
    if (d_total_energy == NULL) {
        CUDA_CHECK(cudaMalloc((void**)&d_total_energy, sizeof(double)));
    }
    
    if (d_block_energies == NULL) {
        
        int max_particles = 0;
        for (int i = 0; i < sim->n_species; i++) {
            if (sim->species[i].np > max_particles) max_particles = sim->species[i].np;
        }

        int max_blocks = (max_particles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        CUDA_CHECK(cudaMalloc((void**)&d_block_energies, max_blocks * sizeof(double)));

    }
    
    CUDA_CHECK(cudaMalloc((void**)&h_gpu_ctx.species, sim->n_species * sizeof(t_species_gpu)));
    t_species_gpu* temp_species = (t_species_gpu*)calloc(sim->n_species, sizeof(t_species_gpu));
    CUDA_CHECK(cudaMemcpy(h_gpu_ctx.species, temp_species, sim->n_species * sizeof(t_species_gpu), cudaMemcpyHostToDevice));
    free(temp_species);
    
    int size_emf = sim->emf.gc[0] + sim->emf.nx + sim->emf.gc[1];
    
    // Allocate full buffers
    float *E_buf_x, *E_buf_y, *E_buf_z;
    float *B_buf_x, *B_buf_y, *B_buf_z;
    
    CUDA_CHECK(cudaMalloc((void**)&E_buf_x, size_emf * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&E_buf_y, size_emf * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&E_buf_z, size_emf * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&B_buf_x, size_emf * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&B_buf_y, size_emf * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&B_buf_z, size_emf * sizeof(float)));
    
    h_gpu_ctx.E_part_x = E_buf_x + sim->emf.gc[0];
    h_gpu_ctx.E_part_y = E_buf_y + sim->emf.gc[0];
    h_gpu_ctx.E_part_z = E_buf_z + sim->emf.gc[0];
    h_gpu_ctx.B_part_x = B_buf_x + sim->emf.gc[0];
    h_gpu_ctx.B_part_y = B_buf_y + sim->emf.gc[0];
    h_gpu_ctx.B_part_z = B_buf_z + sim->emf.gc[0];
    
    h_gpu_ctx.E_buf_x_base = E_buf_x;
    h_gpu_ctx.E_buf_y_base = E_buf_y;
    h_gpu_ctx.E_buf_z_base = E_buf_z;
    h_gpu_ctx.B_buf_x_base = B_buf_x;
    h_gpu_ctx.B_buf_y_base = B_buf_y;
    h_gpu_ctx.B_buf_z_base = B_buf_z;
    
    int size_current = sim->current.gc[0] + sim->current.nx + sim->current.gc[1];
    
    // Allocate current buffers
    float *J_buf_x, *J_buf_y, *J_buf_z;
    
    CUDA_CHECK(cudaMalloc((void**)&J_buf_x, size_current * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&J_buf_y, size_current * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&J_buf_z, size_current * sizeof(float)));
    
    h_gpu_ctx.J_0x = J_buf_x + sim->current.gc[0];
    h_gpu_ctx.J_0y = J_buf_y + sim->current.gc[0];
    h_gpu_ctx.J_0z = J_buf_z + sim->current.gc[0];
    
    h_gpu_ctx.J_buf_x_base = J_buf_x;
    h_gpu_ctx.J_buf_y_base = J_buf_y;
    h_gpu_ctx.J_buf_z_base = J_buf_z;
        
    CUDA_CHECK(cudaMemcpy(d_gpu_ctx, &h_gpu_ctx, sizeof(gpu_context_t), cudaMemcpyHostToDevice));
}

/**
 * @brief Free GPU resources
 */
void free_gpu_ctx() {
    if (streams_initialized) {
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(copy_stream);
        streams_initialized = 0;
    }
    
    if (d_block_energies) {
        cudaFree(d_block_energies);
        d_block_energies = NULL;
    }
    if (d_total_energy) {
        cudaFree(d_total_energy);
        d_total_energy = NULL;
    }

    if (h_gpu_ctx.E_buf_x_base) cudaFree(h_gpu_ctx.E_buf_x_base);
    if (h_gpu_ctx.E_buf_y_base) cudaFree(h_gpu_ctx.E_buf_y_base);
    if (h_gpu_ctx.E_buf_z_base) cudaFree(h_gpu_ctx.E_buf_z_base);
    if (h_gpu_ctx.B_buf_x_base) cudaFree(h_gpu_ctx.B_buf_x_base);
    if (h_gpu_ctx.B_buf_y_base) cudaFree(h_gpu_ctx.B_buf_y_base);
    if (h_gpu_ctx.B_buf_z_base) cudaFree(h_gpu_ctx.B_buf_z_base);
    if (h_gpu_ctx.J_buf_x_base) cudaFree(h_gpu_ctx.J_buf_x_base);
    if (h_gpu_ctx.J_buf_y_base) cudaFree(h_gpu_ctx.J_buf_y_base);
    if (h_gpu_ctx.J_buf_z_base) cudaFree(h_gpu_ctx.J_buf_z_base);
    
    if (h_gpu_ctx.species) {
        for (int i = 0; i < h_gpu_ctx.n_species; i++) {
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
    
    if (d_gpu_ctx) {
        cudaFree(d_gpu_ctx);
        d_gpu_ctx = NULL;
    }
    
    if (first_call_flags) {
        free(first_call_flags);
        first_call_flags = NULL;
    }
}

/**
 * @brief This method updates particle position per species per iteration
 * Since the conditions are periodic and no realloc only need to update 
 * particles current cell index
 */
void update_particles(int species_idx, t_species* spec) {
    t_species_gpu h_spec_temp;
    CUDA_CHECK(cudaMemcpy(&h_spec_temp, h_gpu_ctx.species + species_idx, sizeof(t_species_gpu), cudaMemcpyDeviceToHost));
    
    // Only copy positions
    CUDA_CHECK(cudaMemcpyAsync(h_spec_temp.ix, spec->part.ix, spec->np * sizeof(int), cudaMemcpyHostToDevice, copy_stream));
    
    h_spec_temp.np = spec->np;
    CUDA_CHECK(cudaMemcpy(h_gpu_ctx.species + species_idx, &h_spec_temp, sizeof(t_species_gpu), cudaMemcpyHostToDevice));
}

/**
 * @brief This method implements spec_advance logic to run in gpu
 * @param spec Particle specie
 * @param emf  Electric field
 * @param current Current density
 * @param species_idx Index of the species to be executed
 */
double spec_advance_gpu(t_species* spec, t_emf* emf, t_current* current, int species_idx) {
    
    const int np = spec->np;
    if (np == 0) return 0.0;
    
    // First call logic (Allocate particles)
    // This logic is better do to realloc later for future implementations
    if (first_call_flags == NULL) {
        first_call_flags = (int*)calloc(h_gpu_ctx.n_species, sizeof(int));
    }
    
    int first_call = (first_call_flags[species_idx] == 0);
    
    // Allocate or update particles
    if (first_call) {
        alloc_particles(species_idx, spec);
    } else {
        update_particles(species_idx, spec);
    }
    
    int size_emf = emf->gc[0] + emf->nx + emf->gc[1];
    int size_current = current->gc[0] + current->nx + current->gc[1];

    // Copy Emf fields and memset current only at first specie to be faster
    if (species_idx == 0) {
        
        // E/B fields
        CUDA_CHECK(cudaMemcpyAsync(h_gpu_ctx.E_buf_x_base, emf->E_buf.x, size_emf * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(h_gpu_ctx.E_buf_y_base, emf->E_buf.y, 
                                 size_emf * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(h_gpu_ctx.E_buf_z_base, emf->E_buf.z, 
                                   size_emf * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
        
        CUDA_CHECK(cudaMemcpyAsync(h_gpu_ctx.B_buf_x_base, emf->B_buf.x, 
                                   size_emf * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(h_gpu_ctx.B_buf_y_base, emf->B_buf.y, 
                                   size_emf * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(h_gpu_ctx.B_buf_z_base, emf->B_buf.z, 
                                   size_emf * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
        
        // Memset                          
        CUDA_CHECK(cudaMemsetAsync(h_gpu_ctx.J_buf_x_base, 0, size_current * sizeof(float), copy_stream));
        CUDA_CHECK(cudaMemsetAsync(h_gpu_ctx.J_buf_y_base, 0, size_current * sizeof(float), copy_stream));
        CUDA_CHECK(cudaMemsetAsync(h_gpu_ctx.J_buf_z_base, 0, size_current * sizeof(float), copy_stream));
	
    }

    // Zero energy before kernel
    CUDA_CHECK(cudaMemsetAsync(d_total_energy, 0, sizeof(double), copy_stream));

    // Wait for all copies to complete
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));
    
    // Kernel parameters
    const float tem   = 0.5f * (spec->dt / spec->m_q);
    const float dt_dx = spec->dt / spec->dx;
    const float qnx   = (spec->q * spec->dx) / spec->dt;
    
    const int numBlocks = (np + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int shared_mem = 3 * (THREADS_PER_BLOCK + 4); // 3 - x, y, z per block + guards
    
    /// 2. Kernel execution
    ///    - First kernel : particle push + current deposition
    ///    - Second kernel : energy reduction

    // Launch particle push kernel
    kernel_spec_advance_push_particles<<<numBlocks, THREADS_PER_BLOCK, 0, compute_stream>>>(
        d_gpu_ctx, tem, dt_dx, qnx, spec->q, species_idx, np, d_block_energies);
    CUDA_CHECK(cudaGetLastError());

    
    // Launch energy reduction
    kernel_energy_reduce<<<1, ENERGY_REDUCE_THREADS, 0, compute_stream>>>(
d_block_energies, numBlocks, d_total_energy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    
    // Copy particle results back
    t_species_gpu h_spec_ptr_copy;
    CUDA_CHECK(cudaMemcpy(&h_spec_ptr_copy, h_gpu_ctx.species + species_idx, sizeof(t_species_gpu), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(spec->part.ix, h_spec_ptr_copy.ix, np * sizeof(int), cudaMemcpyDeviceToHost, copy_stream));
    
    //! Do these if want report trough iterations
    //CUDA_CHECK(cudaMemcpyAsync(spec->part.x,  h_spec_ptr_copy.x,  np * sizeof(float), cudaMemcpyDeviceToHost, copy_stream));
    //CUDA_CHECK(cudaMemcpyAsync(spec->part.ux, h_spec_ptr_copy.ux, np * sizeof(float), cudaMemcpyDeviceToHost, copy_stream));
    //CUDA_CHECK(cudaMemcpyAsync(spec->part.uy, h_spec_ptr_copy.uy, np * sizeof(float), cudaMemcpyDeviceToHost, copy_stream));
    //CUDA_CHECK(cudaMemcpyAsync(spec->part.uz, h_spec_ptr_copy.uz, np * sizeof(float), cudaMemcpyDeviceToHost, copy_stream));
    
    double h_energy_val;
    CUDA_CHECK(cudaMemcpy(&h_energy_val, d_total_energy, sizeof(double), 
                          cudaMemcpyDeviceToHost));
    
    if (species_idx == h_gpu_ctx.n_species - 1) {
        
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        
        CUDA_CHECK(cudaMemcpyAsync(current->J_buf.x, h_gpu_ctx.J_buf_x_base, size_current * sizeof(float), cudaMemcpyDeviceToHost, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(current->J_buf.y, h_gpu_ctx.J_buf_y_base, size_current * sizeof(float), cudaMemcpyDeviceToHost, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(current->J_buf.z, h_gpu_ctx.J_buf_z_base, size_current * sizeof(float), cudaMemcpyDeviceToHost, copy_stream));
    }
    
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));
    
    first_call_flags[species_idx] = 1;
    
    return h_energy_val;
}


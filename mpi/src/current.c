/**
 * @file current.c
 * @author Ricardo Fonseca, Diogo Silva, Tomás Pereira
 * @brief Electric current density and kernel binomial filtering
 * @date 2026-01-12
*/

// Std libraries
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

// ZPIC headers
#include "../lib/zdf.h"
#include "../lib/current.h"

// Guard cells 
#define GC_TOP 2
#define GC_BOTTOM 1

/* ---------------------------------------------------------------------------
 #  ############## TEMPORARY BUFFER USED FOR KERNEL_X ###############
 ------------------------------------------------------------------------------
                1) - Init the temporary buffer
                2) - Free the temporary buffer
                3) - Get the temporary buffer pointer
*/

// Temporary buffer to help on kernel vectorization
// If we dind't use this buffer, we would have to call alloc_float3Buffer for every kernel_x call
static float3Buffer tmp;
static int is_allocated = 0;

void kernel_tmpbuf_init(int nx){
    if (!is_allocated){
        alloc_float3Buffer(&tmp, nx);
        is_allocated= 1;
    }
}


void kernel_tmpbuf_cleanup() {
    if (is_allocated) {
        free_float3Buffer(&tmp);
        is_allocated = 0;
    }
}

float3Buffer* kernel_tmpbuf_get(t_current* current) {
    if (!is_allocated){
        kernel_tmpbuf_init(current->chunk_size);
        is_allocated = 1;
    }   
    return &tmp;
}


/**
 * @brief Allocates chunks of memory for other MPI ranks except rank 0 wich manages the whole buffer
 * @param current Current density object
 * @param chunk Chunk of memory to be allocated
 * @param nx Number of grid cells
 * @param gc0 Number of guard cells on the lower boundary
 * @param gc1 Number of guard cells on the upper boundary
 */
void mpi_distributed_alloc_float3Buffer(t_current* current, float3Buffer* chunk, int nx, int gc0, int gc1) {

    // Get MPI ranks and communicator size
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    
    size_t total_size = gc0 + nx + gc1;
    size_t chunk_size = total_size / comm_size;
    size_t chunk_rem = total_size % comm_size; 


    // Guard cells need to fit rank 0
    if (rank == 0 && chunk_size <= gc0){
        printf("ERROR: Not enough guard cells to fit rank 0\n");
        exit(EXIT_FAILURE);
    }

    // Handle last MPI rank (Edge case)
    if (rank == comm_size - 1){ 
        chunk_size += chunk_rem;
        if (chunk_size <= gc1){
            printf("ERROR: Not enough guard cells to fit rank %i\n", rank);
            exit(EXIT_FAILURE);
        }
    }

    alloc_float3Buffer(chunk, chunk_size);
    current->chunk_size = chunk_size;

    if (rank == 0){
        current->J_0x = &current->J_buf.x[gc0];
        current->J_0y = &current->J_buf.y[gc0];
        current->J_0z = &current->J_buf.z[gc0];
    }

    current->J_0x = &current->J_buf.x[0];
    current->J_0y = &current->J_buf.y[0];
    current->J_0z = &current->J_buf.z[0];

    current->nx = nx;
}


/**
 * @brief Initializes Electric current density object
 * 
 * @param current   Electric current density
 * @param nx        Number of grid cells
 * @param box       Physical box size
 * @param dt        Simulation time step
 */
void current_new(t_current *current, int nx, float box, float dt){
    
    // Number of guard cells for linear interpolation
    int gc[2] = {GC_BOTTOM, GC_TOP}; 
    
    // MPI distributed allocation
    mpi_distributed_alloc_float3Buffer(&current->J_buf, nx, gc[0], gc[1]);
    
    // Set cell sizes and box limits
    current -> box = box;
    current -> dx  = box / nx;
    current -> gc[0] = gc[0];
    current -> gc[1] = gc[1];

    // Clear smoothing options
    current -> smooth = (t_smooth) {
        .xtype = NONE,
        .xlevel = 0
    };

    // Initialize time information
    current -> iter = 0;
    current -> dt = dt;

    // Default to periodic boundaries
    current -> bc_type = CURRENT_BC_PERIODIC;

    // Zero initial current
    // This is only relevant for diagnostics, current is always zeroed before deposition
    current_zero(current);
    
}

/**
 * @brief Frees dynamic memory from electric current density
 * 
 * @param current   Electric current density
 */
void current_delete(t_current *current){
    
    // Free memory allocated on J_buf and set J0 pointers to NULL
    free_float3Buffer(&current->J_buf);

    // Avoid memory leaks
    current->J_0x = NULL;
    current->J_0y = NULL;
    current->J_0z = NULL;
    
}

/**
 * @brief Sets all electric current density values to zero
 * 
 * @param current   Electric current density
 */
void current_zero(t_current *current){
    // Mem setting J_buf with zero
    mem_set_float3Buffer(&current->J_buf, current->chunk_size, 0);
}

/**
 * @brief Updates guard cell values
 * 
 * When using periodic boundaries the electric current that was added to
 * the upper guard cells will be added to the corresponding lower grid
 * cells, and the values then copied to the upper grid cells
 * 
 * @param current Electric current density
 */
void current_update_gc(t_current *current){

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (current -> bc_type == CURRENT_BC_PERIODIC && (rank == 0 || rank == comm_size - 1)) {
        
        float* restrict const J_0x = current -> J_0x;
        float* restrict const J_0y = current -> J_0y;
        float* restrict const J_0z = current -> J_0z;

        size_t transfer_size = current->gc[0] + current->gc[1];            
        float tmp_x[transfer_size];
        float tmp_y[transfer_size];
        float tmp_z[transfer_size];

        if (rank == 0){

            // Receive from final rank values
            MPI_Recv(tmp, transfer_size, MPI_FLOAT, comm_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(tmp, transfer_size, MPI_FLOAT, comm_size - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(tmp, transfer_size, MPI_FLOAT, comm_size - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // lower - add the values from upper boundary (both gc and inside box)
            for (int i = 0; i < transfer_size; i++){
                J_0x[i - current->gc[0]] += tmp_x[i];
                J_0y[i - current->gc[0]] += tmp_y[i];
                J_0z[i - current->gc[0]] += tmp_z[i];
            }

            // Send new values to final rank
            MPI_Send(&J_0x[-current->gc[0]], transfer_size, MPI_FLOAT, comm_size - 1, 0, MPI_COMM_WORLD);
            MPI_Send(&J_0y[-current->gc[0]], transfer_size, MPI_FLOAT, comm_size - 1, 1, MPI_COMM_WORLD);
            MPI_Send(&J_0z[-current->gc[0]], transfer_size, MPI_FLOAT, comm_size - 1, 2, MPI_COMM_WORLD);
        }
        else{
            
            const int idx_send = current -> chunk_size - transfer_size - 1;

            MPI_Send(&J_0x[idx_send], transfer_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&J_0y[idx_send], transfer_size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&J_0z[idx_send], transfer_size, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);

            MPI_Recv(tmp_x, transfer_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(tmp_y, transfer_size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(tmp_z, transfer_size, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // upper - just copy the values from the lower boundary     
            for (int i = 0; i < transfer_size; i++){
                J_0x[idx_send + i] = tmp_x[i];
                J_0y[idx_send + i] = tmp_y[i];
                J_0z[idx_send + i] = tmp_z[i];
            }
        }
    }
}

/**
 * @brief Advances electric current density 1 time step
 * 
 * The routine will:
 * 1. Update the guard cells
 * 2. Apply digitial filtering (if configured)
 * 3. Advance iteration number
 * 
 * @param current Electric current density
 */
void current_update(t_current *current){
    
    // Boundary conditions / guard cells
    current_update_gc(current);

    // Smoothing
    current_smooth(current);

    // Advance iteration number
    current -> iter++;
    
}

void current_report(const t_current *current, const int jc){
    
    if (jc < 0 || jc > 2){
        fprintf(stderr, "(*error*) Invalid current component (jc) selected, returning\n");
        return;
    }

    // Pack the information
    float buf[current->nx];
    const int nx = current->nx;

    switch (jc) {
        case 0: {
            const float* restrict const fx = current->J_0x;
            for (int i = 0; i < nx; i++){
                buf[i] = fx[i];
            }
            break;
        }
        case 1: {
            const float* restrict const fy = current->J_0y;
            for (int i = 0; i < nx; i++){
                buf[i] = fy[i];
            }
            break;
        }
        case 2: {
            const float* restrict const fz = current->J_0z;
            for (int i = 0; i < nx; i++){
                buf[i] = fz[i];
            }
            break;
        }
    }

    char vfname[16];
    char vflabel[16];

    snprintf(vfname, 3, "J%1u", jc);
    char comp[] = {'x','y','z'};
    snprintf(vflabel,4,"J_%c",comp[jc]);

    t_zdf_grid_axis axis[1];
    axis[0] = (t_zdf_grid_axis) {
        .min = 0.0,
        .max = current->box,
        .name = "x",
        .label = "x",
        .units = "c/\\omega_p"
    };

    t_zdf_grid_info info = {
        .ndims = 1,
        .name = vfname,
        .label = vflabel,
        .units = "e \\omega_p^2 / c",
        .axis = axis
    };

    info.count[0] = current->nx;

    t_zdf_iteration iter = {
        .name = "ITERATION",
        .n = current->iter,
        .t = current -> iter * current -> dt,
        .time_units = "1/\\omega_p"
    };

    zdf_save_grid((void *) buf, zdf_float32, &info, &iter, "CURRENT");
}

/**
 * @brief Gets the value of the compensator kernel for an n pass binomial kernel
 * 
 * This kernel eliminates the $k^2$ dependency of the transfer function
 * near $k = 0$. The resulting kernel will be in the form [a,b,a], with
 * the values of a and b being determined by this function. The result
 * is normalized.
 * 
 * @param n Number of binomial passes
 * @param sa a value of the compensator kernel
 * @param sb b value of the compensator kernel
 */
void get_smooth_comp(int n, float* sa, float* sb) {
    float a,b,total;
    a = -1;
    b = (4.0 + 2.0*n)/n;
    total = 2*a + b;

    *sa = a / total;
    *sb = b / total;
}

/**
 * @brief Applies a 3 point kernel convolution along x
 * 
 * The kernel has the form [a,b,a]. The routine accounts for periodic
 * boundaries.
 * 
 * @param current 
 * @param sa kernel a value
 * @param sb kernel b value
 */
void kernel_x(t_current* const current, const float sa, const float sb){

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Temporary buffers
    float* restrict const J0_x = current -> J_0x;
    float* restrict const J0_y = current -> J_0y;
    float* restrict const J0_z = current -> J_0z;

    // Get temporary buffer with chunk size
    float3Buffer* tmp = kernel_tmpbuf_get();
    float* restrict const tmp_x = tmp->x;
    float* restrict const tmp_y = tmp->y;
    float* restrict const tmp_z = tmp->z;
    
    // Stencil operation (Vectorized)
    // Convolution (Using HALO cells)
    for(int i = 0; i < current -> ; i++){
        tmp_x[i] = sa * J0_x[i-1] + sb * J0_x[i] + sa * J0_x[i+1];
        tmp_y[i] = sa * J0_y[i-1] + sb * J0_y[i] + sa * J0_y[i+1];
        tmp_z[i] = sa * J0_z[i-1] + sb * J0_z[i] + sa * J0_z[i+1];
    }

    // Copy back
    for (int i = 0; i < current -> nx_chunk; i++){
        J0_x[i] = tmp_x[i];
        J0_y[i] = tmp_y[i];
        J0_z[i] = tmp_z[i];
    }

    // Update x boundaries for periodic boundaries
    if (current -> bc_type == CURRENT_BC_PERIODIC){
        
        int gc0 = -current->gc[0];
        int gc1 = current->gc[1];

        for(int i = -current->gc[0]; i < 0; i++){
            J0_x[i] = J0_x[current->nx + i];
            J0_y[i] = J0_y[current->nx + i];
            J0_z[i] = J0_z[current->nx + i];
        }
        
        for (int i = 0; i < gc1; i++){
            J0_x[current->nx + i] = J0_x[i];
            J0_y[current->nx + i] = J0_y[i];
            J0_z[current->nx + i] = J0_z[i];
        }
    }
}

/**
 * @brief Applies digital filtering to the current density
 * 
 * Filtering is applied through a sequence of 3 point kernel convolutions.
 * The routine will apply a binomial kernel ([1,2,1]) n times, followed by
 * an optional compensator kernel.
 * 
 * Filtering parameters are set by the `current -> smooth` variable.
 * 
 * @param current Electric current density
 */
void current_smooth(t_current* const current){

    // filter kernel [sa, sb, sa]
    float sa, sb;

    // x-direction filtering
    if (current -> smooth.xtype != NONE) {
        
        // binomial filter
        sa = 0.25; sb = 0.5;
        for(int i = 0; i < current -> smooth.xlevel; i++){
            kernel_x(current, 0.25, 0.5);
        }

        // Compensator
        if (current -> smooth.xtype == COMPENSATED){
            get_smooth_comp(current -> smooth.xlevel, &sa, &sb);
            kernel_x(current, sa, sb);
        }
    }

}


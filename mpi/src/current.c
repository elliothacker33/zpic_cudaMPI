/**
 * @file current.c
 * @author Diogo Silva, Ricardo Fonseca, Tomás Pereira
 * @brief Eletric current density
 * @date 2026-01-12
 */

// Std libraries
#include <stdlib.h>
#include <assert.h>
#include <string.h>

// Libraries
#include "../lib/zdf.h"
#include "../lib/zpic.h"
#include "../lib/current.h"

#define GC_TOP 1
#define GC_BOTTOM 2

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

float3Buffer* kernel_tmpbuf_get() {
    return &tmp;
}

void current_smooth(t_current* const current);

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
    int gc[2] = {GC_TOP, GC_BOTTOM}; 
    
    // Allocate global array
    int size;
    size = gc[0] + nx + gc[1];
    
    // Allocate buffer
    alloc_float3Buffer(&current->J_buf, size);

    // Store number of grid cells and guard cells
    current->nx = nx;
    current->gc[0] = gc[0];
    current->gc[1] = gc[1];
    
    // Make J point to cell [0]
    current->J_0x = current->J_buf.x + gc[0];
    current->J_0y = current->J_buf.y + gc[0];
    current->J_0z = current->J_buf.z + gc[0];
    
    // Set cell sizes and box limits
    current -> box = box;
    current -> dx  = box / nx;

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
    free_float3Buffer(&current->J_buf);
}

/**
 * @brief Sets all electric current density values to zero
 * 
 * @param current Electric current density
 */
void current_zero(t_current *current){
    
    // Zero current density
    int size = current->gc[0] + current->nx + current->gc[1];
    mem_set_float3Buffer(&current->J_buf, size, 0.0f);
    
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
    
    if (current -> bc_type == CURRENT_BC_PERIODIC) {
        
        // Get pointers to the first element of each coordinate
        float* restrict const J_0x = current -> J_0x;
        float* restrict const J_0y = current -> J_0y;
        float* restrict const J_0z = current -> J_0z;

        // Get number of grid cells
        const int nx = current -> nx;

        // lower - add the values from upper boundary ( both gc and inside box )
        for (int i=-current->gc[0]; i<current->gc[1]; i++) {
            J_0x[i] += J_0x[nx + i];
            J_0y[i] += J_0y[nx + i];
            J_0z[i] += J_0z[nx + i];
        }
        
        // upper - just copy the values from the lower boundary 
        for (int i=-current->gc[0]; i<current->gc[1]; i++) {
            J_0x[nx + i] = J_0x[i];
            J_0y[nx + i] = J_0y[i];
            J_0z[nx + i] = J_0z[i];
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

/**
 * @brief Saves electric current density diagnostic information to disk
 * 
 * Saves the selected current density component to disk in directory
 * "CURRENT". Guard cell values are discarded.
 * 
 * @param current Electric current object
 * @param jc Current component to save, must be one of {0,1,2}
 */
void current_report(const t_current *current, const int jc){
	
    if (jc < 0 || jc > 2) {
		fprintf(stderr, "(*error*) Invalid current component (jc) selected, returning\n");
		return;
	}

    // Pack the information
    float buf[current->nx];
    float* f = current->J_0x;

    switch (jc) {
        case 0: {
            float* restrict const f_0x = current->J_0x;
            for (int i = 0; i < current->nx; i++) {
                buf[i] = f_0x[i];
            }
            break;
        }
        case 1: {
            float* restrict const f_0y = current->J_0y;
            for (int i = 0; i < current->nx; i++) {
                buf[i] = f_0y[i];
            }
            break;
        }
        case 2: {
            float* restrict const f_0z = current->J_0z;
            for ( int i = 0; i < current->nx; i++ ) {
                buf[i] = f_0z[i];
            }
            break;
        }
    }

	char vfname[16];	// Dataset name
	char vflabel[16];	// Dataset label (for plots)

    snprintf(vfname, 3, "J%1u", jc);
	char comp[] = {'x','y','z'};
    snprintf(vflabel, 4, "J_%c", comp[jc]);

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
    
    float* restrict const J0_x = current -> J_0x;
    float* restrict const J0_y = current -> J_0y;
    float* restrict const J0_z = current -> J_0z;

    float3Buffer* tmp = kernel_tmpbuf_get();
    float* restrict const tmp_x = tmp->x;
    float* restrict const tmp_y = tmp->y;
    float* restrict const tmp_z = tmp->z;

    // Stencil operation (Vectorized)
    // Convolution
    for(int i = 0; i < current -> nx; i++){
        tmp_x[i] = sa * J0_x[i-1] + sb * J0_x[i] + sa * J0_x[i+1];
        tmp_y[i] = sa * J0_y[i-1] + sb * J0_y[i] + sa * J0_y[i+1];
        tmp_z[i] = sa * J0_z[i-1] + sb * J0_z[i] + sa * J0_z[i+1];
    }

    // Copy back
    for (int i = 0; i < current -> nx; i++){
        J0_x[i] = tmp_x[i];
        J0_y[i] = tmp_y[i];
        J0_z[i] = tmp_z[i];
    }

    // Update x boundaries for periodic boundaries
    if (current -> bc_type == CURRENT_BC_PERIODIC){
        
        int gc0 = -current->gc[0];
        int gc1 = current->gc[1];

        for(int i = gc0; i<0; i++){
            J0_x[i] = J0_x[current->nx + i];
            J0_y[i] = J0_y[current->nx + i];
            J0_z[i] = J0_z[current->nx + i];
        }
        
        for (int i=0; i<gc1; i++){
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
        
        // Binomial filter
        sa = 0.25; sb = 0.5;
        for(int i = 0; i < current -> smooth.xlevel; i++) {
            kernel_x(current, 0.25, 0.5);
        }

        // Compensator
        if (current -> smooth.xtype == COMPENSATED) {
            get_smooth_comp(current -> smooth.xlevel, &sa, &sb);
            kernel_x(current, sa, sb);
        }
    }

}

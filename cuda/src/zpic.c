/**
 * @file zpic.c
 * @brief ZPIC utilities (Float3Buffer, Float3)
 * @author Diogo Silva, Tomás Pereira
*/

#define _POSIX_C_SOURCE 200112L

// Std libraries
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ZPIC headers
#include "../lib/zpic.h"

// Use Adequate cache line size for different architectures:
// A64FX: 256 bytes
// A100: 64 bytes
// EPYC 7742: 64 bytes
#define CACHE_LINE_SIZE 256

/**
 * @brief Allocates a buffer in SoA (structure of arrays) format
 * @param buffer Pointer to the buffer
 * @param size Number of elements in the buffer
 * The SoA format allows for better vectorization and cache efficiency
 */
void alloc_float3Buffer(float3Buffer* buffer, int size) {

    // Total size for the chunk (x,y,z)
    int total_chunk_size = size * 3 * sizeof(float);

    // Allocate memory for the buffer
    if (posix_memalign((void**)&buffer->chunk_xyz, CACHE_LINE_SIZE, total_chunk_size)) {
        fprintf(stderr, "Error allocating memory for float3Buffer\n");
        exit(EXIT_FAILURE);
    }

    // Assign pointers to each coordinate
    buffer->x = buffer->chunk_xyz;
    buffer->y = buffer->chunk_xyz + size;
    buffer->z = buffer->chunk_xyz + size * 2;
}

/**
 * @brief Set all the values on a float3Buffer to a given value
 * @param buffer Pointer to the buffer
 * @param size Number of elements in the buffer
 * @param value Value to set the buffer to
 * We use this to set values to zero usually 
*/
void mem_set_float3Buffer(float3Buffer* buffer, int size, float value) {
   
   if (buffer->chunk_xyz == NULL) {
       fprintf(stderr, "Buffer is not allocated\n");
       exit(EXIT_FAILURE);
   }

   memset(buffer->chunk_xyz, value, size * 3 * sizeof(float));
}

/**
 * @brief Free allocated memory for a float3Buffer
 * @param buffer Pointer to the buffer
 * This frees only the chunk_xyz memory, and sets the pointers to NULL
*/
void free_float3Buffer(float3Buffer* buffer) {
    
    if (buffer->chunk_xyz == NULL) {
        fprintf(stderr, "Buffer is not allocated\n");
        exit(EXIT_FAILURE);
    }

    // Free memory of chunk xyz and set pointers to NULL
    free(buffer->chunk_xyz);
    buffer->chunk_xyz = NULL;
    buffer->x = NULL;
    buffer->y = NULL;
    buffer->z = NULL;
}

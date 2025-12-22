/**
 * @file zpic.c
 * @brief ZPIC utilities (Float3Buffer, Float3)
 * @author Diogo Silva, Tomás Pereira
*/


#define _POSIX_C_SOURCE 200112L
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../lib/zpic.h"

#define CACHE_LINE_SIZE 256

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

void mem_set_float3Buffer(float3Buffer* buffer, int size, float value) {
   
   // Set all the memory chunk to zeros (x,y,z components) 
   if (buffer->chunk_xyz == NULL) {
       fprintf(stderr, "Buffer is not allocated\n");
       exit(EXIT_FAILURE);
   }

   memset(buffer->chunk_xyz, 0, size * 3 * sizeof(float));
}

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

/*
 *  zpic.h
 *  zpic
 *
 *  Created by Ricardo Fonseca on 12/8/10.
 *  Copyright 2010 Centro de Física dos Plasmas. All rights reserved.
 *
 */

#ifndef __ZPIC__
#define __ZPIC__


/**
 * @struct Float3Buffer holds a chunnk of memory used for x y and z coordinates
 * @brief Buffer of 3 float arrays
 * float* x -> x[i],  x coordinates
 * float* y -> y[i],  y coordinates
 * float* z -> z[i],  z coordinates
 */
typedef struct Float3Buffer {
    float* chunk_xyz;
	float* x;	///< x vector component
	float* y;	///< y vector component
	float* z;	///< z vector component
} float3Buffer;

/**
 * @brief Three component vector
 * 
 */
typedef struct Float3Cpu {
	float x;	///< x vector component
	float y;	///< y vector component
	float z;	///< z vector component
} float3Cpu;


// -- Utilities to create Float3Buffers --

/**
 * @brief Allocates a float3Buffer (float*x, float*y, float*z)
 * @param buffer Pointer to the buffer
 * @param size Number of elements in the buffer
*/
void alloc_float3Buffer(float3Buffer* buffer, int size);

/**
 * @brief Mem setting the buffer to a specific value
 * @param buffer Pointer to the buffer
 * @param size Number of elements in the buffer
 * @param value Value to initialize the buffer
 */
void mem_set_float3Buffer(float3Buffer* buffer, int size, float value);

/**
 * @brief Free memory allocated of a float3Buffer
 * @param buffer Pointer to the buffer
 */
void free_float3Buffer(float3Buffer* buffer);

/* ANSI C does not define math constants */

#ifndef M_PI
#define M_PI        3.14159265358979323846264338327950288   ///< pi
#endif

#ifndef M_PI_2
#define M_PI_2      1.57079632679489661923132169163975144   ///< pi/2
#endif

#ifndef M_PI_4
#define M_PI_4      0.785398163397448309615660845819875721  ///< pi/4
#endif


#endif

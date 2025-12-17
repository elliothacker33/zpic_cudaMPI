/*
 *  current.h
 *  zpic
 *
 *  Created by Ricardo Fonseca on 12/8/10.
 *  Copyright 2010 Centro de Física dos Plasmas. All rights reserved.
 *
 */

#ifndef __CURRENT__
#define __CURRENT__

#include "zpic.h"

/**
 * @brief Types of digital filtering
 * Our current implementation uses no filtering.
*/
enum smooth_type { 
	NONE,		// No filtering 
	BINOMIAL,	// Binomial filtering
	COMPENSATED	// Compensated binomial filtering
};

/**
 * @brief Types of boundary conditions
*/
enum current_boundary{ 
	CURRENT_BC_NONE,		// No boundary conditions
	CURRENT_BC_PERIODIC		// Periodic boundary conditions
};

/**
 * @brief Digital filtering parameters
 * 
 * Stores digital filtering parameters
 */
typedef struct Smooth {
	enum smooth_type xtype;	///< Type of digital filtering
	int xlevel;				///< Number of filter passes
} t_smooth;


/**
 * @brief Current density object
 * This object stores the current density, guard cells and boundary conditions.
 */
typedef struct Current {
	
	//float3vc_buffer J; // Grid cell [0] (Not counting guard cells)
	float3* J;
	
	//float3vc_buffer J_buf; // Current density buffer (including guard cells)
	float3* J_buf;
	
	int nx;	// Number of grid cells

	int gc[2]; // Guard cells (L/R)
	
	float box; // Box dimensions
	
	float dx; // Grid cell size (length of a cell edge)

	t_smooth smooth; // Digital filtering parameters

	float dt; // Time step

	int iter; // Current iteration number

	enum current_boundary bc_type;	// Type of boundary condition
	
} t_current;

void current_smooth(t_current* const current);

/**
 * @brief Initializes Electric current density object
 * 
 * @param current 	Electric current density
 * @param nx 		Number of cells
 * @param box 		Physical box size
 * @param dt 		Simulation time step
  */
void current_new( t_current *current, int nx, float box, float dt );

/**
 * @brief Frees dynamic memory from electric current density
 * 
 * @param current Electric current density object
 */
void current_delete( t_current *current );

/**
 * @brief Sets all electric current density values to zero
 * 
 * @param current Electric current density object
 */
void current_zero( t_current *current );

/**
 * @brief Advances electric current density 1 time step
 * 
 * @param current Electric current density object
 */
void current_update( t_current *current );

/**
 * @brief Saves electric current density diagnostic information to disk
 * 
 * @param current Electric current density object
 * @param jc Current component to save, must be one of {0,1,2}
 */
void current_report( const t_current *current, const int jc );

#endif

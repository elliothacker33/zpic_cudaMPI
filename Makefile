# ================================================ #
# Makefile for zpic - OpenMP + MPI + CUDA          #
# ================================================ #
# Authors: Diogo Silva (pg6444) & Tomás Pereira (pg59810)
# University of Minho
# Master in Advanced Computing (MCA)
# ================================================ #

SHELL := /bin/bash

# OpenMP 
OMP_NUM_THREADS?=64  # Number of cores of one node of AMD EPYC 7742
OMP_PLACES?=cores    # Bind each thread to a core
OMP_PROC_BIND?=close # Bind each thread with close affinity

MAKEFLAGS += --no-print-directory

# Compiler
CC = mpicc

# GCC flags
CFLAGS = -Ofast -march=native -fopenmp -std=c99 -pedantic -Wall
LDFLAGS = -lm -fopenmp


.PHONY: all clean run docs Makefile

SRC_DIR = src
LIB_DIR = lib
SOURCE = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(SOURCE:.c=.o)

TARGET = zpic

# =========================================================================== #
#                                  BUILD
# =========================================================================== #

all: $(TARGET)

$(TARGET): $(OBJ)
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) \
	OMP_PLACES=$(OMP_PLACES) \
	OMP_PROC_BIND=$(OMP_PROC_BIND) \
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) -c $(CFLAGS) -I$(LIB_DIR) $< -o $@

# =========================================================================== #
#                                  RUN
# =========================================================================== #

# Endpoint to run the code
run: all
	@echo "=========================================="
	@echo "Running ZPIC on gpu partition (EPYC 7742 + A100)"
	@echo "Compiler: $(CC)"
	@echo "Threads: $(OMP_NUM_THREADS)"
	@echo "=========================================="
	@echo "Loading modules..."
	@mkdir -p output
	@TIMESTAMP=$$(date +"%Y-%m-%d_%H-%M-%S"); \
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) \
	OMP_PLACES=$(OMP_PLACES) \
	OMP_PROC_BIND=$(OMP_PROC_BIND) \
	./$(TARGET) 2>&1 | tee output/zpic_run_$${TIMESTAMP}.out


# Endpoint to compile the code
Makefile: all


# =========================================================================== #
#                                DOCS
# =========================================================================== #

DOCSBASE = docs
DOCS = $(DOCSBASE)/html/index.html

docs: $(DOCS)

$(DOCS): $(SOURCE)
	@doxygen ./Doxyfile


# =========================================================================== #
#                                CLEAN
# =========================================================================== #

clean:
	rm -f $(TARGET) $(OBJ)
	rm -rf $(DOCSBASE)



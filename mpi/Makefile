# ============================================================= #
#               Second delivery (WA2) Makefile                  #
#             Authors: Diogo Silva and Tomás Pereira            #
#                      Date:  12/1/2026                         #
# ============================================================= #
# To use this Makefile:
#  make Makefile - Entrypoint used to compile files in Makefile # 
#  make run - Entrypoint used to run the binaries               #
# ============================================================= #

# ============================================================= #
#                 Makefile Settings                             #
# ============================================================= #
#SHELL := /bin/bash

OMP_NUM_THREADS ?= 1 # 128 on EPYC 7742 and 48 on A64FXi
OMP_PLACES = cores
OMP_PROC_BIND = close

MAKEFLAGS += --no-print-directory

# Our makefile can run with both GCC or CLANG on both X86 and ARM architectures
DISTRIBUTE ?= N

# Only use OpenMP
CC ?= clang
CFLAGS ?= -O3 -ffast-math -march=native -std=c99 -pedantic -Wall -fopenmp

# Use MPI tests
#MPI_CC = mpicc
#MPI_CFLAGS = -cc=$(CC)

LDFLAGS = -lm -fopenmp

PHONY: all clean run docs Makefile run

SRC_DIR = src
LIB_DIR = lib
SOURCE = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(SOURCE:.c=.o)
TARGET = zpic

# =========================================================================== #
#                                  BUILD
# =========================================================================== #

all: $(TARGET)
	@echo "====================================="
	@echo "Compiling ZPIC using compiler: $(CC) "

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

# =========================================================================== #
#                                SHORTCUTS
# =========================================================================== #
Makefile: all # Compilation

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



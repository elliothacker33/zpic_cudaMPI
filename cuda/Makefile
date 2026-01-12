# ---------------------------------------------------- #
# Makefile for CUDA implementation of ZPIC             #
# ---------------------------------------------------- #

BIN = zpic
CC = nvcc
LINKER = nvcc

SRC_DIR = src
LIB_DIR = lib
BUILD_DIR = build

CFLAGS = -O3 -g -std=c++11 -Wno-deprecated-gpu-targets
NCCFLAGS = -arch=sm_80 -O3 -Wno-deprecated-gpu-targets --use_fast_math -lineinfo -std=c++11 -Xcompiler -Wall,-Wextra
LDFLAGS = -lm
INC = -I$(LIB_DIR)

# ONLY ALTERED THIS LINE: Exclude the input directory from C_SOURCES
C_SOURCES := $(shell find $(SRC_DIR) -type f -name '*.c' ! -path "$(SRC_DIR)/input/*")
CU_SOURCES := $(shell find $(SRC_DIR) -type f -name '*.cu')
HEADERS := $(shell find $(LIB_DIR) -type f -name '*.h')

C_OBJS := $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(C_SOURCES))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SOURCES))

.PHONY: all clean run Makefile

all: $(BIN)

$(BIN): $(C_OBJS) $(CU_OBJS)
	$(LINKER) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.c $(HEADERS)
	@mkdir -p $(dir $@)
	$(CC) -c $< $(CFLAGS) $(INC) -o $@

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cu $(HEADERS)
	@mkdir -p $(dir $@)
	$(CC) -c $< $(NCCFLAGS) $(INC) -o $@

Makefile: all

run: all
	./$(BIN)

clean:
	rm -rf $(BUILD_DIR) $(BIN)

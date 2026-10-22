# Build the Fortran Crank-Nicolson solver.
#   make            optimised build  -> build/cn_propagate
#   make debug      bounds checking, NaN-initialised reals, FP traps, -Werror -> build/debug/cn_propagate
#   make run        run with the example configuration
#   make clean

FC      ?= gfortran
BUILD   := build
SRC     := src/tdse_cn.f90 src/cn_propagate.f90
TARGET  := $(BUILD)/cn_propagate

FFLAGS_COMMON := -std=f2008 -Wall -Wextra -Wimplicit-interface
FFLAGS_RELEASE := -O2 -march=native
FFLAGS_DEBUG := -O0 -g -fcheck=all -finit-real=snan -ffpe-trap=invalid,zero,overflow -fbacktrace -Werror

# GNU make sets FC=f77 by default; fall back to gfortran in that case.
ifeq ($(FC),f77)
FC := gfortran
endif

.PHONY: all debug run clean

all: FFLAGS := $(FFLAGS_COMMON) $(FFLAGS_RELEASE)
all: $(TARGET)

debug: $(SRC)
	mkdir -p $(BUILD)/debug
	$(FC) $(FFLAGS_COMMON) $(FFLAGS_DEBUG) -J$(BUILD)/debug $(SRC) -o $(BUILD)/debug/cn_propagate

$(TARGET): $(SRC)
	mkdir -p $(BUILD)
	$(FC) $(FFLAGS) -J$(BUILD) $(SRC) -o $@

run: all
	./$(TARGET) examples/soft_coulomb.nml

clean:
	rm -rf $(BUILD)

# Makefile for cute_layout_playground.cpp
# Usage:
#   make [all] CUTLASS_DIR=/path/to/cutlass
# Optional:
#   make COMPILER=nvcc
#   make COMPILER=g++
#   make run
#   make clean

# ---- config ----
APP          := layout_demo
SRC          := cute_layout.cpp

# Set this to your CUTLASS root (must contain include/cute/*.hpp)
CUTLASS_DIR  ?= ../4th/cutlass

# Choose compiler: "auto" (default), "nvcc", or "g++"
COMPILER     ?= auto
CXX          ?= g++
NVCC         ?= nvcc

# Pick your GPU arch (H100/Hopper)
GPU_ARCH ?= sm_90a

# nvcc flags
NVCCFLAGS += -arch=$(GPU_ARCH)

# (optional) embed PTX for forward-compat JIT on newer drivers
NVCCFLAGS += -gencode arch=compute_90a,code=compute_90a

# (optional) quiet some host warnings
CXXFLAGS  += -Wno-psabi -Wno-unused-parameter -Wno-sign-compare
NVCCFLAGS += -Xcompiler -Wno-psabi -Xcompiler -Wno-unused-parameter -Xcompiler -Wno-sign-compare


# Common flags
INCS         := -I$(CUTLASS_DIR)/include
CXXFLAGS     ?= -std=c++17 -O3 -Wall -Wextra -fPIC
NVCCFLAGS    ?= -std=c++17 -O3 -lineinfo -Xcompiler -fPIC

# ---- autodetect nvcc if COMPILER=auto ----
ifeq ($(COMPILER),auto)
  HAVE_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes || echo no)
  ifeq ($(HAVE_NVCC),yes)
    COMPILER := nvcc
  else
    COMPILER := g++
  endif
endif

# ---- build rules ----
.PHONY: all clean run help
all: $(APP)

$(APP): $(SRC)
ifeq ($(COMPILER),nvcc)
	@echo ">> Building with NVCC ($(NVCC))"
	$(NVCC) $(NVCCFLAGS) $(INCS) $< -o $@
else ifeq ($(COMPILER),g++)
	@echo ">> Building with G++ ($(CXX))"
	$(CXX) $(CXXFLAGS) $(INCS) $< -o $@
else
	$(error Unknown COMPILER='$(COMPILER)'; use 'nvcc', 'g++', or leave as 'auto')
endif

run: $(APP)
	./$(APP)

clean:
	$(RM) $(APP)

help:
	@echo "Targets: all (default), run, clean"
	@echo "Vars:"
	@echo "  CUTLASS_DIR=/path/to/cutlass   (must contain include/cute)"
	@echo "  COMPILER=auto|nvcc|g++         (default: auto)"
	@echo "  CXXFLAGS='...'                 (for g++)"
	@echo "  NVCCFLAGS='...'                (for nvcc)"



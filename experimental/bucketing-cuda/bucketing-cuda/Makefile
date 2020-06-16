CPP=g++
STD=-std=c++14
CFLAGS=-Wall -O3 $(STD)
NVCCFLAGS=-ccbin $(CPP) $(STD) -arch=sm_52
INCLUDE=/usr/local/cuda/include ./headers ../../bpplib/include
LDFLAGS=
LIBS=
LIBDIRS=/usr/local/cuda/lib64
HEADERS=$(shell find . -name '*.hpp')
CU_HEADERS=$(shell find . -name '*.cuh')
TARGET=bucketing-cuda


.PHONY: all clear clean purge

all: $(TARGET)



# Building Targets

$(TARGET): $(TARGET).cpp $(HEADERS) kernels.obj
	@echo Compiling and linking executable "$@" ...
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) -lcudart kernels.obj $< -o $@


kernels.obj: kernels/kernels.cu
	@echo Compiling kernels ...
	@nvcc $(NVCCFLAGS) $(addprefix -I,$(INCLUDE)) --compile -cudart static $< -o $@



# Cleaning Stuff

clear:
	@echo Removing object files ...
	-@rm -f *.obj

clean: clear

purge: clear
	@echo Removing executable ...
	-@rm -f $(TARGET)

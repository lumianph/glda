#GPU architecture specification
GPU_ARCH_FLAG   = arch=compute_20,code=sm_20


#C++ compiler configuration
CXX             = g++
CXXFLAGS        = -O3


#CUDA compiler configuration
NVCC_HOME       = /usr/local/cuda-5.0
NVCC            = nvcc
CUDA_INC        = -I$(NVCC_HOME)/include
CUDA_LIB        = -L$(NVCC_HOME)/lib64 -lcuda -lcudart
CUDA_FLAGS      = -O3 -m64 -gencode $(GPU_ARCH_FLAG)



#Project configuration
INCLUDE		= $(CUDA_INC)
LIB		= $(CUDA_LIB)



OBJS=           strtokenizer.o dataset.o utils.o model.o lda.o CUDASampling.o sample_kernel.o
MAIN=           lda

all: $(OBJS)
	$(CXX) -o $(MAIN) $(OBJS) ${LIB} ${CXXFLAGS}
	
	

strtokenizer.o: ./src/strtokenizer.h ./src/strtokenizer.cpp
	$(CXX) -c -o strtokenizer.o ./src/strtokenizer.cpp $(CXXFLAGS) $(INCLUDE)

dataset.o: ./src/dataset.h ./src/dataset.cpp
	$(CXX) -c -o dataset.o ./src/dataset.cpp $(CXXFLAGS) $(INCLUDE)

utils.o: ./src/utils.h ./src/utils.cpp
	$(CXX) -c -o utils.o ./src/utils.cpp $(CXXFLAGS) $(INCLUDE)

model.o: ./src/model.h ./src/model.cpp
	$(CXX) -c -o model.o ./src/model.cpp $(CXXFLAGS) $(INCLUDE)
	
CUDASampling.o: ./src/CUDASampling.h ./src/CUDASampling.cpp
	$(CXX) -c -o CUDASampling.o ./src/CUDASampling.cpp $(CXXFLAGS) $(INCLUDE)

sample_kernel.o: ./src/sample_kernel.h ./src/sample_kernel.cu
	$(NVCC) -c ./src/sample_kernel.cu $(INCLUDE) $(CUDA_FLAGS)


lda.o: ./src/lda.cpp
	$(CXX) -c -o lda.o ./src/lda.cpp $(CXXFLAGS) $(INCLUDE)


clean:
	rm $(OBJS) 
	rm $(MAIN)

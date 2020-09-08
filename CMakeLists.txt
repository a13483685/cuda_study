cmake_minimum_required(VERSION 3.12)
project(cuda_test)


#set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD  11)
#set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)
set(CMAK_NVCC_FLAGS "CMAKE_NVCC_FLAGS -std=c++11")
find_package(CUDA QUIET REQUIRED)
set(CUDA_DIR /usr/local/cuda-10.2/lib64)
set(CUDA_LIB ${CUDA_DIR}/})
#set(CUDA_NVCC_FLAGS  "-D_FORCE_INLINES -Xcompiler -fPIC -gencode arch=compute_${GPU_ARCHS},code=sm_${GPU_ARCHS} -gencode arch=compute_${GPU_ARCHS},code=compute_${GPU_ARCHS}")

include_directories(/usr/local/cuda-10.2/include)
link_libraries()
cuda_add_executable(squaresum main.cu)

#add_executable(test1 main.c)
//
// Created by xiezheng on 2020/9/8.
//
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <sys/time.h>
#include <math.h>
#define ROWS 1024
#define COLS 1024
//extern "C"
//{
//
//}
using namespace std;

__global__ void Plus(float A[], float B[], float C[],int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    C[i] = A[i] + B[i];

}

namespace test_cpu //在cpu计算
{
    void add_cpu_demo()
    {


        float *A,*B,*C;
        int n = 1024*1024;
        int size = n * sizeof(float);
        A = (float*)malloc(size);
        B = (float*)malloc(size);
        C = (float*)malloc(size);

        for (int i = 0; i < n; ++i) {
            A[i] = 90.0;
            B[i] = 10.0;
        }

        for (int j = 0; j < n; ++j) {
            C[j] = A[j] + B[j];
        }


        float max_error = 0.0;
        for (int k = 0; k < n; ++k) {
            max_error += fabs(100.0 - C[k]);
        }
        std::cout << "max_error is " << max_error << std::endl;

        delete A;
        delete B;
        delete C;

    }
}

namespace test_gpu
{
    void add_gpu_demo()
    {
        float *A, *B, *C, *Ad,*Bd,*Cd;
        int n = 1024*1024;
        int size = n* sizeof(int);
        A = (float*)malloc(n* sizeof(float));
        B = (float*)malloc(n* sizeof(float));
        C = (float*)malloc(n* sizeof(float));


        for (int i = 0; i < n; ++i) {
            A[i] = 90.0;
            B[i] = 10.0;
        }


        cudaMalloc((void**)&Ad,size);
        cudaMalloc((void**)&Bd,size);
        cudaMalloc((void**)&Cd,size);

        cudaMemcpy(Ad,A,size,cudaMemcpyHostToDevice);
        cudaMemcpy(Bd,B,size,cudaMemcpyHostToDevice);
        cudaMemcpy(Cd,C,size,cudaMemcpyHostToDevice);

        dim3 dimBlock(512);
        dim3 dimGrid(n/512);

        Plus<<<dimGrid,dimBlock>>>(Ad,Bd,Cd,n);

        cudaMemcpy(C,Cd,size,cudaMemcpyHostToDevice);

        // 校验误差
        float max_error = 0.0;
        for(int i=0;i<n;i++)
        {
            max_error += fabs(100.0 - C[i]);
        }
        cout << "max error is " << max_error << endl;

        cudaFree(Ad);
        cudaFree(Bd);
        cudaFree(Cd);

        delete A;
        delete B;
        delete C;


    }
}


int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout<<"devices count = "<<deviceCount<<std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp,i);
        std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
        std::cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
        std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
        std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
        std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << std::endl;
        std::cout << "======================================================" << std::endl;

    }

    struct timeval start,end;
    gettimeofday(&start,NULL);

//    test_cpu::add_cpu_demo();
    test_gpu::add_gpu_demo();
    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    std::cout << "total time is " << timeuse/1000 << "ms" <<std::endl;


    return 0;

}
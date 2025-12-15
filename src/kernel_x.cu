#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void convolution(const float sa, const float sb, float* J_x, float* J_y, float* J_z, float* J_out_x, float* J_out_y, float* J_out_z, const int nx){

    // Get thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx - 1 || i <= 0) return;

    // Convolution
    J_out_x[i] = sa * J_x[i-1] + sb * J_x[i] + sa * J_x[i+1];
    J_out_y[i] = sa * J_y[i-1] + sb * J_y[i] + sa * J_y[i+1];
    J_out_z[i] = sa * J_z[i-1] + sb * J_z[i] + sa * J_z[i+1];

}


extern "C" void launchKernelX_with_tmp(const float sa, const float sb, t_current* current)
{
    const int nx = current->nx;
    float* J_out_x = nullptr;
    float* J_out_y = nullptr;
    float* J_out_z = nullptr;
    

    // Allocate temporary buffers
    const size_t bytes = nx * sizeof(float);
    CUDA_CHECK(cudaMalloc(&J_out_x, bytes));
    CUDA_CHECK(cudaMalloc(&J_out_y, bytes));
    CUDA_CHECK(cudaMalloc(&J_out_z, bytes));
    
    CUDA_CHECK(cudaMemset(J_out_x, 0, bytes));
    CUDA_CHECK(cudaMemset(J_out_y, 0, bytes));
    CUDA_CHECK(cudaMemset(J_out_z, 0, bytes));

    
    const int threads = 256;
    const int blocks = (nx + threads - 1) / threads;
    
    // Run kernel
    convolution<<<blocks, threads>>>(sa, sb, current->J.x, current->J.y, current->J.z, J_out_x, J_out_y, J_out_z, nx);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back to original J buffer
    CUDA_CHECK(cudaMemcpy(current->J.x, J_out_x, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(current->J.y, J_out_y, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(current->J.z, J_out_z, bytes, cudaMemcpyDeviceToDevice));
    
    // Free temporary buffers
    CUDA_CHECK(cudaFree(J_out_x));
    CUDA_CHECK(cudaFree(J_out_y));
    CUDA_CHECK(cudaFree(J_out_z));
}


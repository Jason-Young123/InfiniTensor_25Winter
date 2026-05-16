#include <iostream>
#include <vector>
#include <cuda_fp16.h>



template <typename T>
__global__ void GEMM_naive(size_t M, size_t N, size_t K, float alpha, float beta, const T *A, const T *B, T *C){
    //A * B = C, (MxK) @ (KxN) = (MxN)
    //对应C的row和column，只考虑一个block内 
    int row = blockIdx.y * blockDim.y + threadIdx.x;
    int col = blockIdx.x * blockDim.x + threadIdx.y;

    if(row < M && col < N){
        T sum = T(0);
        for(size_t k = 0; k < K; k++){
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}


template <typename T>
__global__ void GEMM_coalescing(size_t M, size_t N, size_t K, float alpha, float beta, const T *A, const T *B, T *C){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        T sum = T(0);
        for(size_t k = 0; k < K; k++){
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }

}
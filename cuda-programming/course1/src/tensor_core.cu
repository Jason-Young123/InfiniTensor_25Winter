#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <mma.h>


#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err = (call);                                                   \
        if(err != cudaSuccess){                                                     \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)                  \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
            std::exit(EXIT_FAILURE);                                                \
        }                                                                           \
    } while(0)


// 最小Tensor Core示例: 每个block计算一个16x16的C tile。
// blockDim = 16x16, 每个线程分别加载一个A_tile元素和一个B_tile元素。
// WMMA/MMA本身是warp-level指令, 因此这里只让block中的第一个warp调用mma_sync。
__global__ void tensor_core_matmul(float* C, const float* A, const float* B, int M, int K, int N){
    using namespace nvcuda;

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidy * blockDim.x + tidx;

    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 16;
    const int MMA_K = 8;

    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;

    __shared__ float A_tile[TILE_M][TILE_K];
    __shared__ float B_tile[TILE_K][TILE_N];

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> C_frag;

    if(tid < 32){
        wmma::fill_fragment(C_frag, 0.0f);
    }

    for(int k0 = 0; k0 < K; k0 += TILE_K){
        int a_gm_row = block_row + tidy;
        int a_gm_col = k0 + tidx;
        if(a_gm_row < M && a_gm_col < K){
            A_tile[tidy][tidx] = wmma::__float_to_tf32(A[a_gm_row * K + a_gm_col]);
        }
        else{
            A_tile[tidy][tidx] = 0.0f;
        }

        int b_gm_row = k0 + tidy;
        int b_gm_col = block_col + tidx;
        if(b_gm_row < K && b_gm_col < N){
            B_tile[tidy][tidx] = wmma::__float_to_tf32(B[b_gm_row * N + b_gm_col]);
        }
        else{
            B_tile[tidy][tidx] = 0.0f;
        }

        __syncthreads();

        if(tid < 32){
            #pragma unroll
            for(int kk = 0; kk < TILE_K; kk += MMA_K){
                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> A_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> B_frag;

                wmma::load_matrix_sync(A_frag, &A_tile[0][kk], TILE_K);
                wmma::load_matrix_sync(B_frag, &B_tile[kk][0], TILE_N);
                wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
            }
        }

        __syncthreads();
    }

    if(tid < 32 && block_row + TILE_M <= M && block_col + TILE_N <= N){
        wmma::store_matrix_sync(&C[block_row * N + block_col], C_frag, N, wmma::mem_row_major);
    }
}


int main(){
    int M = 1024, K = 1024, N = 1024;

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];

    std::fill(h_A, h_A + M * K, 1.0f);
    std::fill(h_B, h_B + K * N, 1.0f);
    std::fill(h_C, h_C + M * N, 0.0f);

    size_t totalByte_A = sizeof(float) * M * K;
    size_t totalByte_B = sizeof(float) * K * N;
    size_t totalByte_C = sizeof(float) * M * N;

    //step1: device malloc
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, totalByte_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, totalByte_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, totalByte_C));

    //step2: copy H2D
    CUDA_CHECK(cudaMemcpy(d_A, h_A, totalByte_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, totalByte_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, totalByte_C));

    //step3: launch kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim((N + 15) / 16, (M + 15) / 16);
    tensor_core_matmul<<<grid_dim, block_dim>>>(d_C, d_A, d_B, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //step4: copy D2H
    CUDA_CHECK(cudaMemcpy(h_C, d_C, totalByte_C, cudaMemcpyDeviceToHost));

    std::cout << h_C[0] << " - " << h_C[1] << " - " << h_C[N] << " - " << h_C[N + 1] << std::endl;

    //step5: clean-up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

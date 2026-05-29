/*### 4. 层归一化 (Layer Normalization / RMSNorm)

题目：实现大模型中极其常用的 LayerNorm（或 RMSNorm）算子之前向传播。

面试官附加限制

- 针对形状为 `[Batch, Hidden_dim]` 的输入，在 `Hidden_dim` 上求均值和方差。
- 同样要求使用 Shared Memory 或 Warp 原语来优化均值和方差的规约过程。
*/


#include <iostream>




template <typename T>
__device__ T warp_reduce_max_down(T val){
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1){
        T val_other = __shfl_down_sync(0xffffffff, val, offset);
        val = ::max(val_other, val);
    }
    return val;
}

template <typename T>
__device__ T warp_reduce_sum_down(T val){
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1){
        T val_other = __shfl_down_sync(0xffffffff, val, offset);
        val += val_other;
    }
    return val;
}


//计算一行的均值和方差; d_in为该行的起始地址; 1-pass方案
template <typename T>
__device__ void get_mean_var_1pass(T* d_in, T* mean, T* var, size_t cols){
    int tidx = threadIdx.x;
    int stride = blockDim.x;

    __shared__ T smem_sum[32];
    __shared__ T smem_square_sum[32];

    T sum = T(0);
    T square_sum = T(0);
    for(int i = tidx; i < cols; i += stride){
        sum += d_in[i];
        square_sum += d_in[i] * d_in[i];
    }

    T sum_warp = warp_reduce_sum_down(sum);
    T square_sum_warp = warp_reduce_sum_down(square_sum);

    if(tidx % 32 == 0){
        smem_sum[tidx/32] = sum_warp;
        smem_square_sum[tidx/32] = square_sum_warp;
    }
    __syncthreads();

    T sum_block = T(0);
    T square_sum_block = T(0);
    if(tidx < 32){
        sum_warp = (tidx < (blockDim.x + 31)/32) ? smem_sum[tidx] : T(0);
        square_sum_warp = (tidx < (blockDim.x + 31)/32) ? smem_square_sum[tidx] : T(0);
        sum_block = warp_reduce_sum_down(sum_warp);
        square_sum_block = warp_reduce_sum_down(square_sum_warp);
        if(tidx == 0){
            T mean_tmp = sum_block / cols;
            *mean = mean_tmp;
            *var = square_sum_block / cols - mean_tmp * mean_tmp;
        }
    }
}





//计算一行的均值和方差; d_in为该行的起始地址; 2-pass方案
template <typename T>
__device__ void get_mean_var_2pass(T* d_in, T* mean, T* var, size_t cols){
    int tidx = threadIdx.x;
    int stride = blockDim.x;

    __shared__ T smem[32];
    __shared__ T _mean, _var;

    //1-pass, 求mean
    T sum = T(0);
    for(int i = tidx; i < cols; i += stride){
        sum += d_in[i];
    }

    T sum_warp = warp_reduce_sum_down(sum);

    if(tidx % 32 == 0){
        smem[tidx/32] = sum_warp;
    }
    __syncthreads();

    if(tidx < 32){
        sum_warp = (tidx < (blockDim.x + 31)/32) ? smem[tidx] : T(0);
        T sum_block = warp_reduce_sum_down(sum_warp);
        if(tidx == 0){
            _mean = sum_block / cols;
        }
    }
    __syncthreads();//等待_mean共享内存写入


    //2-pass,求var
    sum = T(0);
    for(int i = tidx; i < cols; i += stride){
        sum += (d_in[i] - _mean) * (d_in[i] - _mean);
    }

    sum_warp = warp_reduce_sum_down(sum);
    if(tidx % 32 == 0){
        smem[tidx/32] = sum_warp;
    }
    __syncthreads();

    if(tidx < 32){
        sum_warp = (tidx < (blockDim.x + 31)/32) ? smem[tidx] : T(0);
        T sum_block = warp_reduce_sum_down(sum_warp);
        if(tidx == 0){
            _var = sum_block / cols;
            *mean = _mean;
            *var = _var;
        }
    }
    __syncthreads();
}






template <typename T>
__global__ void layerNorm_2d(T* d_in, T* d_out, size_t cols){
    int tidx = threadIdx.x;
    int stride = blockDim.x;
    int row = blockIdx.x;

    const T epsilon = T(1e-5);
    __shared__ T mean, var;

    get_mean_var_2pass(d_in + row * cols, &mean, &var, cols);
    __syncthreads();

    for(int i = tidx; i < cols; i += stride){
        d_out[row * cols + i] = (d_in[row * cols + i] - mean) / ::sqrt(var + epsilon);
    }
}








int main(){
    int M = 128, N = 4096;//128 x 4096的输入矩阵, 对每行的4096个元素进行softmax
    float* h_in = new float[M * N];
    float* h_out = new float[M * N];
    for(int i = 0; i < M * N; ++i){
        h_in[i] = i;
    }

    float* d_in, *d_out;
    cudaMalloc((void**)&d_in, sizeof(float) * M * N);
    cudaMalloc((void**)&d_out, sizeof(float) * M * N);

    cudaMemcpy(d_in, h_in, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    dim3 block_dim(512);//每行由1个block中的512个线程处理
    dim3 grid_dim(M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    layerNorm_2d<float><<<grid_dim, block_dim>>>(d_in, d_out, N);
    cudaDeviceSynchronize(); // 必须等待预热完成
    
    cudaEventRecord(start);
    layerNorm_2d<float><<<grid_dim, block_dim>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start, stop);
    std::cout << "Naive 2D Softmax runtime: " << milliseconds1 << " ms" << std::endl;


    cudaMemcpy(h_out, d_out, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    //check result
    std::cout << h_in[0] << " - " << h_in[1] << " - " << h_in[2] << " - " << h_in[3] << std::endl;
    std::cout << h_out[0] << " - " << h_out[1] << " - " << h_out[2] << " - " << h_out[3] << std::endl;


    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;

    return 0;
}



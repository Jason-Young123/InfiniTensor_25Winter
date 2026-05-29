#include <iostream>


//2d softmax, 每一行交由一个block完成归约, 总block数 = 行数
#define MYINFINITY 1e20


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

template <typename T>
__device__ void warp_online_softmax(T& local_m, T& local_d){
    //printf("local_m = %f, local_d = %f\n", float(local_m), float(local_d));
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1){
        T other_m = __shfl_down_sync(0xffffffff, local_m, offset);
        T other_d = __shfl_down_sync(0xffffffff, local_d, offset);
        //printf("id = %d, other_m = %f, other_d = %f\n", threadIdx.x, float(other_m), float(other_d));
        
        // 根据 Online Softmax 公式更新
        T new_m = ::max(local_m, other_m);
        T new_d = local_d * ::exp(local_m - new_m) + other_d * ::exp(other_m - new_m);
        //printf("id = %d, new_m = %f, new_d = %f\n", threadIdx.x, float(new_m), float(new_d));

        local_m = new_m;
        local_d = new_d;
    }
    //printf("out: local_m = %f, local_d = %f\n", float(local_m), float(local_d));
}


//d_in为该行第一个元素的起始地址; n = #cols;
//d_m_block为该行归约后的max值, d_d_block为该行归约后的expsum值
template <typename T>
__device__ void get_m_d(T* d_in, T* d_m_block, T* d_d_block, size_t cols){
    int tidx = threadIdx.x;
    int stride = blockDim.x;

    __shared__ T smem_m[32];//最大只需32
    __shared__ T smem_d[32];

    T m_old = T(-MYINFINITY);
    T m_new = T(-MYINFINITY);
    T d_old = T(0);
    T d_new = T(0);
    for(int i = tidx; i < cols; i += stride){
        //printf("d_in[%d] = %f\n", i, float(d_in[i]));
        m_new = ::max(m_old, d_in[i]);
        d_new = d_old * ::exp(m_old - m_new) + ::exp(d_in[i] - m_new);
        m_old = m_new; d_old = d_new;//更新
    }
    //printf("m_new = %f, d_new = %f\n", float(m_new), float(d_new));

    warp_online_softmax(m_new, d_new);//完成warp内归约
    T& m_warp = m_new; T& d_warp = d_new;
    //printf("m_warp = %f, d_warp = %f\n", float(m_warp), float(d_warp));
    if(tidx % 32 == 0){
        smem_m[tidx/32] = m_warp;
        smem_d[tidx/32] = d_warp;
    }
    __syncthreads();
    //printf("okok\n");

    if(tidx < 32){
        m_warp = (tidx < (blockDim.x + 31)/32) ? smem_m[tidx] : T(-MYINFINITY);
        d_warp = (tidx < (blockDim.x + 31)/32) ? smem_d[tidx] : T(0);
        warp_online_softmax(m_warp, d_warp);
        T& m_block = m_warp; T& d_block = d_warp;
        if(tidx == 0){
            *d_m_block = m_block; *d_d_block = d_block;
        }
    }

}






//d_in为完整的矩阵地址
template <typename T>
__global__ void softmax_2d(T* d_in, T* d_out, size_t cols){
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;

    __shared__ T m_block, d_block;//每个线程都有这两个变量,但根据softmax_1d函数,只有tidx == 0的线程会写这两个变量; 
    //用shared从而每个线程在softmax_1d函数之后都可以访问这两个变量

    int row = bidx;
    get_m_d(d_in + row * cols, &m_block, &d_block, cols);//每个block各自计算出本行的m_block和d_block
    __syncthreads();//!!!必须等待shared mem被彻底写入才允许后续执行,避免之后线程读到垃圾值

    for(int i = tidx; i < cols; i += blockDim.x){
        d_out[row * cols + i] = ::exp(d_in[row * cols + i] - m_block) / d_block;
    }

}







int main(){
    int M = 1, N = 4096;//128 x 4096的输入矩阵, 对每行的4096个元素进行softmax

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

    softmax_2d<float><<<grid_dim, block_dim>>>(d_in, d_out, N);
    cudaDeviceSynchronize(); // 必须等待预热完成
    
    cudaEventRecord(start);
    softmax_2d<float><<<grid_dim, block_dim>>>(d_in, d_out, N);
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







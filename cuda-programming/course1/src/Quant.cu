/*
对LLM中的权重矩阵进行Grouped Quant:
    输入矩阵 W: d x d(d为hidden_dim) fp16
    量化时,应该严格按照llm标准, 对于纵向每列中一个group的权重进行量化, 并输出量化系数S
    输出: (1)量化后权重矩阵 Wq: d x d int8
          (2)量化系数矩阵 S: d/group_size x d fp16
*/



/*
注意,由于是对一列内的group进行量化/反量化,因此在通常实践中(比如当前版本),会让一个warp/block处理一列中
若干行数据, 从而面临严重的uncoalesced访存; 工业级处理方案:
1. 直接将矩阵存储为访存高效的形式, 比如这里用列主序而非行主序, 这样一个warp就会处理一行中连续若干列数据, 解决了非合并访存问题;
2. 如果输入必须是 row-major W[row][col]，那更好的量化 kernel 不会让一个 warp 只处理同一列。会改成一个 block 处理一个小 tile：
     rows: group_size
     cols: 例如 32 或 64 列
     线程布局类似：
     threadIdx.x -> col
     threadIdx.y -> row
     这样加载时，一个 warp 访问同一行的连续列：
     W[row * d + col + lane]
     这就是合并访存。然后在 shared memory 或寄存器里对每一列做 absmax reduce，得到多个 scale。

     这种方式的思路是：
     旧写法：一个 block 处理 1 列 x 1 group，访存 stride = d
     工程写法：一个 block 处理 多列 x 1 group，按行连续加载，再列内归约
     这比“先转置再量化”少一个显式转置步骤，但 kernel 会复杂一些。
*/





#include <iostream>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>


//向下洗牌归约, 求warp内最大值
template <typename T>
__device__ T warp_reduce_max_down(T val){
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1){
        T val_other = __shfl_down_sync(0xffffffff, val, offset);
        val = ::max(val_other, val);
    }
    return val;
}


//d_W为完整矩阵地址; 每个block负责某一列中的一个group, 求该group内的absmax
__device__ void get_absmax(half* d_W, float* d_absmax_block, size_t d, size_t grpSize){
    int tidx = threadIdx.x;
    int stride = blockDim.x;
    int grp = blockIdx.x;
    int col = blockIdx.y;

    __shared__ float smem[32];//最大只需32

    size_t row_start = grp * grpSize;
    size_t row_end = (row_start + grpSize < d) ? row_start + grpSize : d;

    float absmax = 0.0f;
    for(size_t row = row_start + tidx; row < row_end; row += stride){
        float val = __half2float(d_W[row * d + col]);
        absmax = ::max(absmax, ::fabsf(val));
    }

    float absmax_warp = warp_reduce_max_down(absmax);//完成warp内归约
    if(tidx % 32 == 0){
        smem[tidx/32] = absmax_warp;
    }
    __syncthreads();

    if(tidx < 32){
        absmax_warp = (tidx < (blockDim.x + 31)/32) ? smem[tidx] : 0.0f;
        absmax_warp = warp_reduce_max_down(absmax_warp);
        if(tidx == 0){
            *d_absmax_block = absmax_warp;
        }
    }
}




//为简单起见不用模板,指定输入矩阵为fp16,量化后矩阵为int8
__global__ void Quant(half* d_W, int8_t* d_Wq, half* d_S, size_t d, size_t grpSize){
    int tidx = threadIdx.x;
    int stride = blockDim.x;
    int grp = blockIdx.x;
    int col = blockIdx.y;

    __shared__ float absmax_block, scale_block;//每个block对应一个量化scale

    size_t row_start = grp * grpSize;
    size_t row_end = (row_start + grpSize < d) ? row_start + grpSize : d;

    get_absmax(d_W, &absmax_block, d, grpSize);//求出该列该group内权重绝对值的最大值
    __syncthreads();//!!!必须等待shared mem被彻底写入才允许后续执行,避免之后线程读到垃圾值

    if(tidx == 0){
        scale_block = absmax_block / 127.0f;
        d_S[grp * d + col] = __float2half(scale_block);
    }
    __syncthreads();//等待scale_block写入

    for(size_t row = row_start + tidx; row < row_end; row += stride){//量化公式: Q(W) = round(W/S), 注意还要clipped一下
        float val = __half2float(d_W[row * d + col]);
        int q = (scale_block == 0.0f) ? 0 : __float2int_rn(val / scale_block);
        q = (q > 127) ? 127 : q;
        q = (q < -127) ? -127 : q;
        d_Wq[row * d + col] = int8_t(q);
    }

}




__global__ void DeQuant(int8_t* d_Wq, half* d_S, half* d_W, size_t d, size_t grpSize){
    int tidx = threadIdx.x;
    int stride = blockDim.x;
    int grp = blockIdx.x;
    int col = blockIdx.y;

    size_t row_start = grp * grpSize;
    size_t row_end = (row_start + grpSize < d) ? row_start + grpSize : d;

    float scale_block = __half2float(d_S[grp * d + col]);
    for(size_t row = row_start + tidx; row < row_end; row += stride){//反量化公式: W' = Q(W) * S
        int8_t q = d_Wq[row * d + col];
        d_W[row * d + col] = __float2half(float(q) * scale_block);
    }
}






int main(){
    const size_t d = 2048;
    const size_t grpSize = 128;
    const size_t tmpDim = (d + grpSize - 1)/grpSize;

    half* h_W = new half[d * d];//量化前权重矩阵d x d
    int8_t* h_Wq = new int8_t[d * d];//量化后同尺寸权重矩阵d x d
    half* h_S = new half[tmpDim * d];
    half* h_W_dequant = new half[d * d];

    for(size_t i = 0; i < d * d; ++i){
        float val = float(int(i % 251) - 125) * 0.01f;
        h_W[i] = __float2half(val);
    }

    half* d_W, *d_S, *d_W_dequant;
    int8_t* d_Wq;

    cudaMalloc((void**)&d_W, sizeof(half) * d * d);
    cudaMalloc((void**)&d_Wq, sizeof(int8_t) * d * d);
    cudaMalloc((void**)&d_S, sizeof(half) * tmpDim * d);
    cudaMalloc((void**)&d_W_dequant, sizeof(half) * d * d);

    cudaMemcpy(d_W, h_W, sizeof(half) * d * d, cudaMemcpyHostToDevice);

    dim3 block_dim(128);//每列的每个group由1个block中的128个线程处理
    dim3 grid_dim(tmpDim, d);//grid.x为group数量, grid.y为列数

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Quant<<<grid_dim, block_dim>>>(d_W, d_Wq, d_S, d, grpSize);
    DeQuant<<<grid_dim, block_dim>>>(d_Wq, d_S, d_W_dequant, d, grpSize);
    cudaDeviceSynchronize(); // 必须等待预热完成
    
    cudaEventRecord(start);
    Quant<<<grid_dim, block_dim>>>(d_W, d_Wq, d_S, d, grpSize);
    DeQuant<<<grid_dim, block_dim>>>(d_Wq, d_S, d_W_dequant, d, grpSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start, stop);
    std::cout << "Grouped Quant + DeQuant runtime: " << milliseconds1 << " ms" << std::endl;

    cudaMemcpy(h_Wq, d_Wq, sizeof(int8_t) * d * d, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_S, d_S, sizeof(half) * tmpDim * d, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W_dequant, d_W_dequant, sizeof(half) * d * d, cudaMemcpyDeviceToHost);

    //check result
    float max_abs_err = 0.0f;
    float max_allowed_err = 0.0f;
    int bad_cnt = 0;
    for(size_t row = 0; row < d; ++row){
        size_t grp = row / grpSize;
        for(size_t col = 0; col < d; ++col){
            size_t idx = row * d + col;
            float val = __half2float(h_W[idx]);
            float val_dequant = __half2float(h_W_dequant[idx]);
            float scale = __half2float(h_S[grp * d + col]);
            float err = std::fabs(val - val_dequant);
            float allowed_err = scale * 0.6f + 1e-3f;
            max_abs_err = (max_abs_err > err) ? max_abs_err : err;
            max_allowed_err = (max_allowed_err > allowed_err) ? max_allowed_err : allowed_err;
            if(err > allowed_err){
                ++bad_cnt;
            }
        }
    }

    std::cout << __half2float(h_W[0]) << " - " << __half2float(h_W[1]) << " - "
              << __half2float(h_W[2]) << " - " << __half2float(h_W[3]) << std::endl;
    std::cout << int(h_Wq[0]) << " - " << int(h_Wq[1]) << " - "
              << int(h_Wq[2]) << " - " << int(h_Wq[3]) << std::endl;
    std::cout << __half2float(h_W_dequant[0]) << " - " << __half2float(h_W_dequant[1]) << " - "
              << __half2float(h_W_dequant[2]) << " - " << __half2float(h_W_dequant[3]) << std::endl;
    std::cout << "max_abs_err = " << max_abs_err
              << ", max_allowed_err = " << max_allowed_err
              << ", bad_cnt = " << bad_cnt << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_W); cudaFree(d_Wq); cudaFree(d_S); cudaFree(d_W_dequant);
    delete[] h_W; delete[] h_Wq; delete[] h_S; delete[] h_W_dequant;

    return 0;
}

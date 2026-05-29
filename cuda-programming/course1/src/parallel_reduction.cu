/*
2. 并行归约求和/求最大值 (Parallel Reduction)
题目：给定一个长度为 N 的一维浮点数组，使用 CUDA 编写一个核函数求其所有元素的和（或最大值）。

面试官附加限制：

- 初始版本可以使用 Shared Memory 交错寻址。
- 进阶要求：如何消除 Warp Divergence（分支发散）？如何解决 Shared Memory 的 Bank Conflict（存储体冲突）？
- 终极要求：请使用 Warp-level 原语（如 `__shfl_down_sync`）实现极致优化版本的归约。
*/


#include <stdio.h>

__device__ __forceinline__ float atomicMax(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        float assumed_f = __int_as_float(assumed);
        float max_f = ::fmaxf(assumed_f, val);
        old = atomicCAS(address_as_int, assumed, __float_as_int(max_f));
    } while (assumed != old);
    return __int_as_float(old);
}



/*
几种__shfl_*函数的效果:
    __shfl_down_sync: 向下shuffle, 即lane_i中最终存储 lane_i ~ lane_31的归约结果; 常用于归约(如求max/sum),只取lane0作为最终归约结果
    __shfl_up_sync: 向上shuffle, 即lane_i中最终存储 lane_0 ~ lane_i的归约结果; 常用于求前缀sum/max,此时每个lane的结果都有用
    __shfl_sync: 获取指定lane编号的数据,比如__shfl_sync(0xffffffff, val, 15)可以让每个lane都获取lane15的数值
*/
template <typename T> 
__device__ T warp_reduce_down(T val){
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1){
        T val_other = __shfl_down_sync(0xffffffff, val, offset);//获取要被归约的另一个元素
        val = ::max(val_other, val);
    }
    return val;
}


template <typename T>
__device__ T warp_reduce_up(T val){
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1){
        T val_other = __shfl_up_sync(0xffffffff, val, offset);
        val = ::max(val_other, val);
    }
    return val;
}


//由于涉及到atomic 归约,最好采用grid-strided loop, 而不是根据元素数量决定block总数, 否则当元素总量过多时会导致最后一步atomic归约效率下降
//以归约求最大值为例
template <typename T>
__global__ void parallel_reduction(T* d_in, T* d_out, size_t n){//n代表总元素数目
    int tidx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    T __shared__ tmp[32];//最大分配32即可,因为一级归约后,每个block内元素总数至多为1024/32 = 32

    //注意d_out的初始化不能放在核函数内,而是应该放在main函数中
    T reduced_l0 = T(-INFINITY);
    for(int i = idx; i < n; i += stride){//grid-strided
        reduced_l0 = ::max(reduced_l0, d_in[i]);
    }

    T reduced_l1 = warp_reduce_down(reduced_l0);//一级归约, warp内归约
    if(tidx % 32 == 0){//每个warp内lane0负责把结果写入shared mem
        tmp[tidx/32] = reduced_l1;
    }
    __syncthreads();

    if(tidx < 32){//二级归约,每个block内只需前32个线程工作
        T reduced_l2 = (tidx < (blockDim.x + 31)/32) ? tmp[tidx] : T(-INFINITY);
        reduced_l2 = warp_reduce_down(reduced_l2);
        if(tidx == 0){
            atomicMax(d_out, reduced_l2);
        }
    }
}



int main(){
    //step1: 资源分配
    size_t TOTAL_NUM = 1000000;
    float *h_in = new float[TOTAL_NUM];//待归约数据
    float h_out = -INFINITY;//归约后结果

    float* d_in, *d_out;
    cudaMalloc((void**)&d_in, sizeof(float) * TOTAL_NUM);
    cudaMalloc((void**)&d_out, sizeof(float));

    //step2: copy H2D
    cudaMemcpy(d_in, h_in, sizeof(float) * TOTAL_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, &h_out, sizeof(float), cudaMemcpyHostToDevice);

    //step3: launch kernel
    dim3 block_dim(256);
    dim3 grid_dim(32);

    parallel_reduction<float><<<grid_dim, block_dim>>>(d_in, d_out, TOTAL_NUM);

    //step4: copy D2H
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    //step5: check result

    //step6: cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;

    return 0;
}
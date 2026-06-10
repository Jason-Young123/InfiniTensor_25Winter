/*### 5. 并行前缀和 (Prefix Sum / Scan)

题目： 对一个一维数组实现并行前缀和（Inclusive or Exclusive Scan）。

面试官附加限制：

- 要求实现具有工作效能（Work-efficient）的算法（如 Blelloch Scan），即总加法次数应当与 CPU 串行版本保持在同一数量级（$O(N)$）。
- 需要清晰说明 Up-sweep（归约阶段）和 Down-sweep（分发阶段）的过程。
*/

#include <iostream>


//warp内求inclusive prefix_sum
__device__ float warp_prefix_sum(float val){
    int lane = threadIdx.x & 31;
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1){
        float tmp_val = __shfl_up_sync(0xffffffff, val, offset);//必须每个线程都如同掩码所示那样参与归约
        if(lane >= offset){//然后根据lane选择性地进行累加
            val += tmp_val;
        }
    }
    return val;
}




//简单起见直接用float, 并且blockDim.x固定为1024
//pass1目的: 每个block负责1024个元素的局部prefix_sum;
//输出: 512个元素各自的前缀和 + 整个block的sum
//本质上, prefix_sum_pass1就是对任意数量的元素，以blockDim.x(1024)的粒度输出局部块的prefix_sum
__global__ void prefix_sum_blockwise(float* d_blockwise_pfxsum, float* d_blockwise_sum, const float* d_in, const int N){
    int tidx = threadIdx.x;
    int warp_id = tidx / warpSize;
    int lane_id = tidx % warpSize;
    int bidx = blockIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float warpwise_pfxsum[1024];//按照每个block最多处理1024个线程分配smem
    __shared__ float warpwise_sum[32];//每个block最多包含32个warp

    //step1: 对每个warp单独求prefix_sum
    float data = (idx < N) ? d_in[idx] : 0.0;
    data = warp_prefix_sum(data);//每个warp内求prefix_sum
    warpwise_pfxsum[tidx] = data;//存入warp_pfxsum


    //step2: 把每个warp的sum取出来放入warpsize_sum,再求一轮prefix_sum
    if(lane_id == warpSize - 1){//取每个warp的最后一个lane, 将结果存入warp_sum
        warpwise_sum[warp_id] = data;
    }
    __syncthreads();

    if(warp_id == 0){//block内对所有warp_sum做prefix_sum
        warpwise_sum[lane_id] = warp_prefix_sum(warpwise_sum[lane_id]);
    }
    __syncthreads();

    //step3: 将第二轮求得的prefix_sum加回到第一轮结果上
    warpwise_pfxsum[tidx] += ((warp_id > 0) ? warpwise_sum[warp_id - 1] : 0.0);

    //epilogue: store data; 最终得到blockwise的prefix_sum以及blockwise的sum
    if(idx < N){
        d_blockwise_pfxsum[idx] = warpwise_pfxsum[tidx];
    }
    if(d_blockwise_sum != NULL){
        d_blockwise_sum[bidx] = warpwise_sum[31];
    }
}




//pass2, 这里的N为block总数
__global__ void prefix_sum_pass2(float* d_prefix_sum, const float* d_blockwise_pfxsum, const float* d_blockwise_increment, const int N){
    int tidx = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < N){
        float increment = (blockIdx.x > 0) ? d_blockwise_increment[idx/1024 - 1] : 0.0;
        d_prefix_sum[idx] = d_blockwise_pfxsum[idx] + increment;
    }
}






int main(){
    const int N = 100000;//以10w数据为例
    float* h_in = new float[N];
    float* h_out = new float[N];
    for(int i = 0; i < N; ++i){
        h_in[i] = float(i) / N;
    }

    dim3 block_dim(1024);
    dim3 grid_dim((N + 1023)/1024);


    float* d_in, *d_blockwise_pfxsum, *d_blockwise_sum, *d_out;
    cudaMalloc((void**)&d_in, sizeof(float) * N);
    cudaMalloc((void**)&d_blockwise_pfxsum, sizeof(float) * N);
    cudaMalloc((void**)&d_blockwise_sum, sizeof(float) * grid_dim.x);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    cudaMemcpy(d_in, h_in, sizeof(float) * N, cudaMemcpyHostToDevice);


    //pass1
    prefix_sum_blockwise<<<grid_dim, block_dim>>>(d_blockwise_pfxsum, d_blockwise_sum, d_in, N);
    //cudaMemcpy(h_out, d_blockwise_pfxsum, sizeof(float) * N, cudaMemcpyDeviceToHost);
    //std::cout << h_out[0] << "--" << h_out[N - 2] << "--" << h_out[N - 3] << std::endl;


    //pass2
    prefix_sum_blockwise<<<1, block_dim>>>(d_blockwise_sum, NULL, d_blockwise_sum, grid_dim.x);

    //pass3
    prefix_sum_pass2<<<grid_dim, block_dim>>>(d_out, d_blockwise_pfxsum, d_blockwise_sum, N);
    

    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    std::cout << h_out[1] << "--" << h_out[N - 1] << "--" << h_out[N - 3] << std::endl;


    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_blockwise_pfxsum);
    cudaFree(d_blockwise_sum);
    delete[] h_in;
    delete[] h_out;

    return 0;
}















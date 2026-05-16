#include <iostream>
#include <vector>
#include <cuda_fp16.h>
#include <cub/cub.cuh>//启用cuda unbound作为reference

//1. warp内归约 -> 利用register间传递数据,效率最高,通常用__shfl_down_sync实现
//2. block内(warp间)归约 -> 利用shared memory实现,效率较低,且发射时需要<<<grid_dim, block_dim, smem_size>>>指定smem空间
//3. block间归约 -> 要首先将每个block内数据转移到同一个block内,需经过global memory,效率更低






//简单原子累加,本质为线程串行,效率极低
template <typename T>
__global__ void reduce_sum_atomic(T *output, const T *input, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    //grid-strided
    for(size_t i = idx; i < n; i += stride){
        atomicAdd(output, input[i]);
    }
}


//每一个__global__ 核函数都考虑一个block内的情形
//block粒度伪并行
template <typename T>
__global__ void reduce_sum_blockwise(T *output, const T *input, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    if(threadIdx.x == 0){//只有每个block的第一个thread在运行,其他线程闲置
        T block_sum = 0;
        for(size_t i = 0; i < blockDim.x; i++){//第一个线程模拟全部线程
            for(size_t j = idx + i; j < n; j += stride){
                block_sum += input[j];
            }
        }
        atomicAdd(output, block_sum);
    }
}


//warp粒度伪并行,要求必须满足blockDim(每个block中线程数)为32的整数倍
template <typename T>
__global__ void reduce_sum_warpwise(T *output, const T *input, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    if(threadIdx.x % 32 == 0){//
        T warp_sum = 0;
        for(size_t i = 0; i < 32; i++){
            for(int j = idx + i; j < n; j += stride){
                warp_sum += input[j];
            }
        }
        atomicAdd(output, warp_sum);
    }
}


//自定义粒度伪并行,要求必须满足blockDim(每个block中线程数)为grain的整数倍
template <typename T>
__global__ void reduce_sum_grained(T *output, const T *input, size_t n, size_t grain){
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    if(tid % grain == 0){//
        T warp_sum = 0;
        for(size_t i = 0; i < grain; i++){
            for(int j = idx + i; j < n; j += stride){
                warp_sum += input[j];
            }
        }
        atomicAdd(output, warp_sum);
    }
}




//树状归约真并行,利用shared mem,这里固定对block内全部线程进行规约
//方式1,对半树状归约,即thread[0] += thread[s/2], thread[1] += thread[s/2+1],...,where s = blockDim.x
//注意：加载时有s个线程参与；第一轮for循环只有s/2个线程参与;此后每轮for循环参与线程数均减半
template <typename T>
__global__ void reduce_sum_smem_tree1(T *output, const T *input, size_t n){
    extern __shared__ T smem[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    //为一个block内的所有线程建立共享内存smem,长度等于blockDim
    //smem[tid] = (idx < n) ? input[idx] : 0;
    T sum = 0;
    for(size_t i = idx; i < n; i += stride){
        sum += input[i];
    }
    smem[tid] = sum;//初始化smem
    __syncthreads();

    for(size_t s = blockDim.x/2; s > 0; s >>= 1){
        if(tid < s){
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        atomicAdd(output, smem[0]);//每个block中的smem[0]进行原子累加
    }
}

//树状归约真并行
//方式2,首尾树状归约,即thread[0] += thread[s-1], thread[1] += thread[s-2],...,where s = blockDim.x
template <typename T>
__global__ void reduce_sum_smem_tree2(T *output, const T *input, size_t n){
    extern __shared__ T smem[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    //为一个block内的所有线程建立共享内存smem,长度等于blockDim
    //smem[tid] = (idx < n) ? input[idx] : 0;
    T sum = 0;
    for(size_t i = idx; i < n; i += stride){
        sum += input[i];
    }
    smem[tid] = sum;//初始化smem
    __syncthreads();

    for(size_t s = blockDim.x/2; s > 0; s >>= 1){
        if(tid < s){
            smem[tid] += smem[2*s - tid - 1];
        }
        __syncthreads();
    }

    if(tid == 0){
        atomicAdd(output, smem[0]);//每个block中的smem[0]进行原子累加
    }
}


//树状归约真并行
//方式3,相邻归约,即thread[0] += thread[1], thread[2] += thread[3],...
template <typename T>
__global__ void reduce_sum_smem_tree3(T *output, const T *input, size_t n){
    extern __shared__ T smem[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    //为一个block内的所有线程建立共享内存smem,长度等于blockDim
    //smem[tid] = (idx < n) ? input[idx] : 0;
    T sum = 0;
    for(size_t i = idx; i < n; i += stride){
        sum += input[i];
    }
    smem[tid] = sum;//初始化smem
    __syncthreads();

    for(size_t s = 1; s < blockDim.x; s <<= 1){
        size_t j = 2 * s * tid;
        if(j < blockDim.x){
            smem[j] += smem[j + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        atomicAdd(output, smem[0]);//每个block中的smem[0]进行原子累加
    }
}


//线程shfl真并行,要求必须满足blockDim(每个block中线程数)为grain的整数倍,且warp(32)为grain的整数倍,
//因为shuffle只工作在一个warp内
template <typename T>
__global__ void reduce_sum_shfl1(T *output, const T *input, size_t n, size_t grain){
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    T sum = 0;
    for(size_t i = idx; i < n; i+= stride){
        sum += input[i];
    }

    for(size_t offset = grain/2; offset > 0; offset >>= 1){
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if(tid % grain == 0){//原子操作次数 = warp数量 = block数量 * 每个block内warp数量
        atomicAdd(output, sum);
    }

}





//warp内数据可以通过寄存器互相访问,但warp间数据只能通过更为低效的smem进行
//warp内(32个线程)并行归约,只能32,多于32时多余线程未归约,小于32时需要将未使用的线程补0
template <typename T>
__device__ T warp_reduce(T val){
#pragma unroll//短循环自动展开,省去分支预测,提升效率
    for(int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__global__ void reduce_sum_shfl2(T *output, const T *input, size_t n){
    extern __shared__ T smem[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    T sum = 0;
    for(size_t i = idx; i < n; i += stride){
        sum += input[i];
    }
    //一级归约,每个warp内所有线程归约
    T warp_sum = warp_reduce(sum);
    //准备二级规约,将每个warp内lane[0]的值拷贝到smem中
    if(tid % 32 == 0){
        smem[tid / 32] = warp_sum;
    }
    __syncthreads();

    //默认一级归约后每个block内的线程不超过32(本质为一个block内warp不超过32,但这是一定的,因为maxThreadsPerBlock就是1024)
    if(tid < 32){
        //准备二级规约,多余线程补0;用每个block内warp[0]处理smem中数据
        T block_sum = (tid < (blockDim.x + 31)/32) ? smem[tid] : T(0);
        //二级归约,每个block内所有lane[0]线程归约
        block_sum = warp_reduce(block_sum);
        if(tid == 0){//原子操作次数 = block数量
            atomicAdd(output, block_sum);
        }
    }
}






//2-pass法之first pass,依次进行warp内、block内线程归约,最终得到intermediate数组,长度
//由原本元素总数降为启用的block总数
template <typename T>
__global__ void reduce_sum_first_pass(T * intermediate, const T *input, size_t n){
    extern __shared__ T smem[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    //grid-strided
    T sum = 0;
    for(size_t i = idx; i < n; i += stride){
        sum += input[i];
    }

    //一级归约,每个block内warp内归约
    T warp_sum = warp_reduce(sum);

    if(tid % 32 == 0){//一级归约后结果存储在smem中
        smem[tid/32] = warp_sum;
    }
    __syncthreads();

    //二级归约,每个block内归约
    if(tid < 32){
        T block_sum = (tid < (blockDim.x + 31)/32) ? smem[tid] : T(0);
        block_sum = warp_reduce(block_sum);
        //注意此处的不同,二级归约结果不会直接进行atomicAdd,而是作为中间值暂存起来
        if(tid == 0){
            intermediate[blockIdx.x] = block_sum;
        }
    }
}


//2-pass之second pass,用一个block处理first pass得到的intermediate数组
template <typename T>
__global__ void reduce_sum_second_pass(T *output, const T *intermediate, size_t n){
    extern __shared__ T smem[];
    size_t tid = threadIdx.x;
    size_t idx = tid;//由于只启用一个block,tid = idx
    size_t stride = blockDim.x;

    T sum = 0;
    for(size_t i = idx; i < n; i += stride){
        sum += intermediate[i];
    }
    
    //一级归约(实际为三级归约),该特定block内每个warp进行规约
    T warp_sum = warp_reduce(sum);

    if(tid % 32 == 0){
        smem[tid / 32] = warp_sum;
    }
    __syncthreads();

    //二级归约(实际为四级归约),该特定block内归约
    if(tid < 32){
        T block_sum = (tid < (blockDim.x + 31)/32) ? smem[tid] : T(0);
        block_sum = warp_reduce(block_sum);
        if(tid == 0){
            *output += block_sum;//得到最终结果;由于此时只剩一个block,因此无需atomicAdd
        }
    }
}




template <typename T>
void reduce_sum_2pass(T *d_result, const T *d_input, size_t n, const dim3 &grid, const dim3 &block){
    T *intermediate;
    cudaMalloc(&intermediate, grid.x * sizeof(T));//second-pass所需空间,等于block数目

    //first pass
    //smem_size1 = 每个block内warp数量
    size_t smem_size1 = ((block.x + 31)/32) * sizeof(T);
    reduce_sum_first_pass<<<grid, block, smem_size1>>>(intermediate, d_input, n);

    //second pass
    dim3 grid2(1);//只用一个block
    dim3 block2(min(grid.x, block.x));//有可能不会启用该block内全部的线程
    size_t smem_size2 = ((block2.x + 31)/32) * sizeof(T);
    reduce_sum_second_pass<<<grid2, block2, smem_size2>>>(d_result, intermediate, grid.x);

    cudaFree(intermediate);
}





/*int main(){
    
    const size_t SIZE = 1 << 23;//1M
    size_t size_bytes = SIZE * sizeof(float);


    //step1: 初始化 host(cpu) 端数据; 预留 device(gpu) 数据端空间
    std::vector<float> host_a(SIZE, 1);
    float *host_b = new float;

    float *device_a, *device_b;//注意device端仅支持裸指针
    cudaMalloc(&device_a, size_bytes);
    cudaMalloc(&device_b, sizeof(float));

     //step2: copy from host to device; 注意需传输host_a的数据内容,而非vector对象host_a
    cudaMemcpy(device_a, host_a.data(), size_bytes, cudaMemcpyHostToDevice);

     //step3: gpu calculation via kernel function
    //dim3 grid_dim(4096);
    //dim3 block_dim(64);
    dim3 block_dim(256);  // 尝试不同大小：128, 256, 512
    //dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);
    dim3 grid_dim(512);
    size_t smem_size = block_dim.x * sizeof(float);

    //reduce_sum_atomic<float><<<grid_dim, grid_dim>>>(device_b, device_a, SIZE);
    
    //reduce_sum_blockwise<float><<<grid_dim, grid_dim>>>(device_b, device_a, SIZE);
    
    //reduce_sum_warpwise<float><<<grid_dim, block_dim>>>(device_b, device_a, SIZE);
    //reduce_sum_grained<float><<<grid_dim, block_dim>>>(device_b, device_a, SIZE, 256);

    reduce_sum_smem_tree1<float><<<grid_dim, block_dim, smem_size>>>(device_b, device_a, SIZE);
    reduce_sum_smem_tree2<float><<<grid_dim, block_dim, smem_size>>>(device_b, device_a, SIZE);
    reduce_sum_smem_tree3<float><<<grid_dim, block_dim, smem_size>>>(device_b, device_a, SIZE);
    //reduce_sum_shfl<float><<<grid_dim, block_dim>>>(device_b, device_a, SIZE, 32);
    //reduce_sum_shfl<float><<<grid_dim, block_dim>>>(device_b, device_a, SIZE, 8);
    //reduce_sum_shfl<float><<<grid_dim, block_dim>>>(device_b, device_a, SIZE, 32);
    //reduce_sum_shfl2<float><<<grid_dim, block_dim, smem_size>>>(device_b, device_a, SIZE);

    //reduce_sum_2pass(device_b, device_a, SIZE, grid_dim, block_dim);

    //step4: copy from device to host
    cudaMemcpy(host_b, device_b, sizeof(float), cudaMemcpyDeviceToHost);

    //step5: free cuda memory
    if(device_a)
        cudaFree(device_a);
    if(device_b)
        cudaFree(device_b);


    std::cout << "result: " << *host_b << std::endl;

    return 0;
}*/







int main() {
    // 数据规模：约 1.34 亿个 float (约 512 MB)
    size_t N = 1 << 27; 
    size_t bytes = N * sizeof(float);

    std::cout << "Data Size: " << N << " elements (" << bytes / (1024.0 * 1024.0) << " MB)\n";

    // Host 内存分配与初始化
    std::vector<float> h_input(N, 1.0f); // 全填1，正确结果应该是 N
    float h_custom_out = 0.0f;
    float h_cub_out = 0.0f;

    // Device 内存分配
    float *d_input, *d_custom_out, *d_cub_out;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_custom_out, sizeof(float));
    cudaMalloc(&d_cub_out, sizeof(float));

    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // CUDA Event 计时器
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_runs = 100;
    float milliseconds = 0;

    // ==========================================
    // 测试 A: 你的自定义算子
    // ==========================================
    int blockSize = 256;
    // 获取 SM 数量并计算最优 GridSize
    int num_SMs;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0);
    int gridSize = std::min((int)((N + blockSize - 1) / blockSize), num_SMs * 32);
    size_t smemSize = ((blockSize + 31) / 32) * sizeof(float);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        cudaMemset(d_custom_out, 0, sizeof(float)); // 切记清零！
        reduce_sum_shfl2<<<gridSize, blockSize, smemSize>>>(d_custom_out, d_input, N);
    }
    cudaDeviceSynchronize();

    // 计时循环
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; ++i) {
        cudaMemset(d_custom_out, 0, sizeof(float));
        reduce_sum_shfl2<<<gridSize, blockSize, smemSize>>>(d_custom_out, d_input, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float custom_avg_time = milliseconds / num_runs;
    
    cudaMemcpy(&h_custom_out, d_custom_out, sizeof(float), cudaMemcpyDeviceToHost);


    // ==========================================
    // 测试 B: CUB 官方库
    // ==========================================
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // 第一步：获取临时内存大小
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_cub_out, N);
    
    // 分配临时内存
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_cub_out, N);
    }
    cudaDeviceSynchronize();

    // 计时循环
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; ++i) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_cub_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float cub_avg_time = milliseconds / num_runs;

    cudaMemcpy(&h_cub_out, d_cub_out, sizeof(float), cudaMemcpyDeviceToHost);


    // ==========================================
    // 输出结果对比
    // ==========================================
    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Expected : " << (float)N << std::endl;
    std::cout << "Custom   : " << h_custom_out << " | Time: " << custom_avg_time << " ms" << std::endl;
    std::cout << "CUB      : " << h_cub_out << " | Time: " << cub_avg_time << " ms" << std::endl;

    // 计算有效内存带宽 (Effective Memory Bandwidth)
    // 规约操作主要受限于显存读取速度，公式：带宽 = 数据量 / 时间
    float custom_bw = (bytes / 1e9) / (custom_avg_time / 1000.0f);
    float cub_bw = (bytes / 1e9) / (cub_avg_time / 1000.0f);

    std::cout << "\n--- Bandwidth ---" << std::endl;
    std::cout << "Custom BW: " << custom_bw << " GB/s" << std::endl;
    std::cout << "CUB BW   : " << cub_bw << " GB/s" << std::endl;

    // 释放内存
    cudaFree(d_input);
    cudaFree(d_custom_out);
    cudaFree(d_cub_out);
    cudaFree(d_temp_storage);

    return 0;
}



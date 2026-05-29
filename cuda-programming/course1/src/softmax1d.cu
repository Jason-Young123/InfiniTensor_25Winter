/*
3. Softmax 算子 (Softmax Operator)

题目：针对形状为 `[Batch, Seq_len]` 的二维张量，在 `Seq_len` 维度上实现 Softmax 操作。

面试官附加限制：

- 为了数值稳定性，必须先减去每一行的最大值（Safe Softmax）。
- 思考：如何在一个 Block 内高效完成“找最大值 -> 求指数和 -> 归一化”这三次不同的规约和遍历，同时尽量减少与 Global Memory 的反复交互？
*/

//1d的最佳实现: naive 3-pass

//注意, atomic操作必须针对GM内的变量, 不能针对局部变量


#include <iostream>


//原生atomicMax不支持float
__device__ __forceinline__ float atomicMax(float* address, float val) {
    // 1. 将 float 指针强转为 int 指针，因为 atomicCAS 支持 int
    int* address_as_int = (int*)address;
    
    // 2. 读取当前内存里的旧值
    int old = *address_as_int;
    int assumed;

    // 3. 自旋锁循环
    do {
        assumed = old;
        
        // 将整型假定值转回 float 进行比大小
        float assumed_f = __int_as_float(assumed);
        float max_f = ::fmaxf(assumed_f, val); 
        
        // 将算出的最大值再转回 int
        int max_i = __float_as_int(max_f);

        // atomicCAS 会去尝试更新：
        // 如果 address_as_int 里的值还是 assumed，就更新为 max_i，并返回 assumed
        // 如果这期间被别人改了，它会返回新的旧值，此时 assumed != old，循环继续
        old = atomicCAS(address_as_int, assumed, max_i);
        
    } while (assumed != old);

    // 返回原本在那里的旧值（转回 float）
    return __int_as_float(old);
}




__device__ __forceinline__ void atomicSM(float2* address, float m_block, float d_block) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old_ull = *address_as_ull;
    unsigned long long assumed_ull, new_ull;

    do {
        assumed_ull = old_ull;
        
        float old_m = __uint_as_float((unsigned int)(assumed_ull & 0xFFFFFFFF));
        float old_d = __uint_as_float((unsigned int)(assumed_ull >> 32));
        
        float new_m = ::max(old_m, m_block);
        float new_d = old_d * ::exp(old_m - new_m) + d_block * ::exp(m_block - new_m);
        
        unsigned int new_m_bits = __float_as_uint(new_m);
        unsigned int new_d_bits = __float_as_uint(new_d);
        new_ull = ((unsigned long long)new_d_bits << 32) | new_m_bits;
        
        old_ull = atomicCAS(address_as_ull, assumed_ull, new_ull);
        
    } while (assumed_ull != old_ull);
}


//向下洗牌归约
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



/*template <typename T>
__device__ void warp_online_softmax(T val, T* m, T* d){
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>=1){
        T val_other = __shfl_down_sync(0xffffffff, val, offset);
        *m = ::max(val_other, val);
        *d = val * ::exp(val - *m) + val_other * ::exp(val_other - *m);
    }
}*/

template <typename T>
__device__ void warp_online_softmax(T& local_m, T& local_d){
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1){
        T other_m = __shfl_down_sync(0xffffffff, local_m, offset);
        T other_d = __shfl_down_sync(0xffffffff, local_d, offset);
        
        // 根据 Online Softmax 公式更新
        T new_m = ::max(local_m, other_m);
        T new_d = local_d * ::exp(local_m - new_m) + other_d * ::exp(other_m - new_m);
        
        local_m = new_m;
        local_d = new_d;
    }
}



//m_global为全局最大值, d_global为全局分母(即减去全局最大值后的expsum)
template <typename T>
__global__ void softmax_1d_stage1_online(T* d_in, float2* d_max_expsum, size_t n){
    int tidx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    __shared__ T smem_m[32];//按最大32分配, 1024/32 = 32
    __shared__ T smem_d[32];

    T m_old = T(-INFINITY);
    T m_new = T(-INFINITY);
    T d_old = T(0);
    T d_new = T(0);
    

    //grid-strided loop
    for(int i = idx; i < n; i += stride){
        m_new = ::max(m_old, d_in[i]);
        d_new = d_old * ::exp(m_old - m_new) + ::exp(d_in[i] - m_new);
        //d_new = d_old + ::exp(d_in[i] - m_new);
        m_old = m_new; d_old = d_new;//更新
    }

    //一级归约, warp内归约
    warp_online_softmax(m_new, d_new);
    T& m_warp = m_new; T& d_warp = d_new;//得到warp内的m和d

    if(tidx % 32 == 0){
        smem_m[tidx/32] = m_warp;
        smem_d[tidx/32] = d_warp;
    }
    __syncthreads();

    if(tidx < 32){
        m_warp = (tidx < (blockDim.x + 31)/32) ? smem_m[tidx] : T(-INFINITY);
        d_warp = (tidx < (blockDim.x + 31)/32) ? smem_d[tidx] : T(0);
        warp_online_softmax(m_warp, d_warp);
        T& m_block = m_warp; T& d_block = d_warp;//得到block内的m和d
        
        if(tidx == 0){
            //原子操作
            atomicSM(d_max_expsum, m_block, d_block);
        }
    }
}






//传统softmax, pass 1, 仅求全局最大值
template <typename T>
__global__ void softmax_1d_stage1_naive(T* d_in, T* d_out, size_t n){
    int tidx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    T __shared__ tmp[32];//最大分配32即可,因为一级归约后,每个block内元素总数至多为1024/32 = 32

    T reduced_l0 = T(-INFINITY);
    for(int i = idx; i < n; i += stride){//grid-strided
        reduced_l0 = ::max(reduced_l0, d_in[i]);
    }

    T reduced_l1 = warp_reduce_max_down(reduced_l0);//一级归约, warp内归约
    if(tidx % 32 == 0){//每个warp内lane0负责把结果写入shared mem
        tmp[tidx/32] = reduced_l1;
    }
    __syncthreads();

    if(tidx < 32){//二级归约,每个block内只需前32个线程工作
        T reduced_l2 = (tidx < (blockDim.x + 31)/32) ? tmp[tidx] : T(-INFINITY);
        reduced_l2 = warp_reduce_max_down(reduced_l2);
        if(tidx == 0){
            atomicMax(d_out, reduced_l2);
        }
    }
}


//pass-2, 仅求sigma(exp(x - max_global))
template <typename T>
__global__ void softmax_1d_stage2_naive(T* d_in, T* d_max, T* d_out, size_t n){
    int tidx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    T __shared__ tmp[32];//最大分配32即可,因为一级归约后,每个block内元素总数至多为1024/32 = 32

    T reduced_l0 = T(0);
    for(int i = idx; i < n; i += stride){//grid-strided
        reduced_l0 += ::exp(d_in[i] - *d_max);
    }

    T reduced_l1 = warp_reduce_sum_down(reduced_l0);//一级归约, warp内归约
    if(tidx % 32 == 0){//每个warp内lane0负责把结果写入shared mem
        tmp[tidx/32] = reduced_l1;
    }
    __syncthreads();

    if(tidx < 32){//二级归约,每个block内只需前32个线程工作
        T reduced_l2 = (tidx < (blockDim.x + 31)/32) ? tmp[tidx] : T(0);
        reduced_l2 = warp_reduce_sum_down(reduced_l2);
        if(tidx == 0){
            atomicAdd(d_out, reduced_l2);
        }
    }
}


//3-pass, 计算最终结果并写回
template <typename T>
__global__ void softmax_1d_stage3_naive(T* d_in, T* d_max, T* d_expsum, T* d_out, size_t n){
    int tidx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = idx; i < n; i += stride){
        d_out[i] = ::exp(d_in[i] - *d_max) / *d_expsum;
    }
}

















int main(){
    size_t TOTAL_NUM = 100000;
    float* h_in = new float[TOTAL_NUM];
    float* h_out = new float[TOTAL_NUM];
    float h_max = float(-INFINITY);
    float h_expsum = float(0);
    float2 h_max_expsum = make_float2(h_max, h_expsum);
    
    float* d_in;
    float* d_max;
    float* d_expsum; 
    float* d_out;
    float2* d_max_expsum;

    cudaMalloc((void**)&d_in, sizeof(float) * TOTAL_NUM);
    cudaMalloc((void**)&d_max, sizeof(float));
    cudaMalloc((void**)&d_expsum, sizeof(float));
    cudaMalloc((void**)&d_out, sizeof(float) * TOTAL_NUM);
    cudaMalloc((void**)&d_max_expsum, sizeof(float2));

    cudaMemcpy(d_in, h_in, sizeof(float) * TOTAL_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &h_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expsum, &h_expsum, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_expsum, &h_max_expsum, sizeof(float2), cudaMemcpyHostToDevice);


    dim3 block_dim(512);
    dim3 grid_dim(108);//限制大小,用grid-strided loop避免大量atomic操作

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    /*计时1*/
    //warm-up
    softmax_1d_stage1_naive<float><<<grid_dim, block_dim>>>(d_in, d_max, TOTAL_NUM);
    softmax_1d_stage2_naive<float><<<grid_dim, block_dim>>>(d_in, d_max, d_expsum, TOTAL_NUM);
    softmax_1d_stage3_naive<float><<<grid_dim, block_dim>>>(d_in, d_max, d_expsum, d_out, TOTAL_NUM);
    cudaDeviceSynchronize(); // 必须等待预热完成

    cudaEventRecord(start);
    softmax_1d_stage1_naive<float><<<grid_dim, block_dim>>>(d_in, d_max, TOTAL_NUM);
    softmax_1d_stage2_naive<float><<<grid_dim, block_dim>>>(d_in, d_max, d_expsum, TOTAL_NUM);
    softmax_1d_stage3_naive<float><<<grid_dim, block_dim>>>(d_in, d_max, d_expsum, d_out, TOTAL_NUM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start, stop);
    std::cout << "Naive 3-Pass Softmax 执行时间: " << milliseconds1 << " ms" << std::endl;
    

    

    /*计时2*/
    //warm-up
    cudaMemcpy(d_max, &h_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expsum, &h_expsum, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_expsum, &h_max_expsum, sizeof(float2), cudaMemcpyHostToDevice);
    softmax_1d_stage1_online<float><<<grid_dim, block_dim>>>(d_in, d_max_expsum, TOTAL_NUM);
    softmax_1d_stage3_naive<float><<<grid_dim, block_dim>>>(d_in, (float*)d_max_expsum, (float*)d_max_expsum + 1, d_out, TOTAL_NUM);//注意host端不允许解引用device端的指针,比如访问ptr -> x等
    cudaDeviceSynchronize(); // 必须等待预热完成

    cudaEventRecord(start);
    softmax_1d_stage1_online<float><<<grid_dim, block_dim>>>(d_in, d_max_expsum, TOTAL_NUM);
    softmax_1d_stage3_naive<float><<<grid_dim, block_dim>>>(d_in, (float*)d_max_expsum, (float*)d_max_expsum + 1, d_out, TOTAL_NUM);//注意host端不允许解引用device端的指针,比如访问ptr -> x等
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start, stop);
    std::cout << "Online 2-Pass Softmax 执行时间: " << milliseconds2 << " ms" << std::endl;


    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);



    cudaMemcpy(h_out, d_out, sizeof(float) * TOTAL_NUM, cudaMemcpyDeviceToHost);

    //check result

    cudaFree(d_in); cudaFree(d_max); cudaFree(d_expsum); cudaFree(d_out);
    delete[] h_in; delete[] h_out;

    return 0;
}
#include <iostream>
#include <vector>
#include <cuda_fp16.h>


//纯__device__函数无法在cpu上运行，可以被核函数或其他__device__函数调用
template <typename T>
__device__ T add(const T &a, const T &b){
    if constexpr(std::is_same_v<T, float>){
        return a + b;
    }
    else if constexpr(std::is_same_v<T, float1>){
        return make_float1(a.x + b.x);
    }
    else if constexpr(std::is_same_v<T, float2>){
        return make_float2(a.x + b.x, a.y + b.y);
    }
    else if constexpr(std::is_same_v<T, float4>){
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
    else if constexpr(std::is_same_v<T, half>){//半精度
        return __hadd(a, b);
    }
    else if constexpr(std::is_same_v<T, half2>){//并行半精度
        return __hadd2(a, b);
    }
    else{
        ;        
    }
}

//采用Grid_Strided的更高效版本
//__global__可以调用__device__
template<typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n, size_t step){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(size_t i = idx; i < n; i += step){
        //c[i] = a[i] + b[i];
        c[i] = add(a[i], b[i]);
    }
}

template<typename T>
void add_cuda(T *c, const T *a, const T* b, size_t n, const dim3 &grid, const dim3 &block){
    size_t step = grid.x * block.x;
    add_kernel<T><<<grid, block>>>(c, a, b, n, step);
}




//T: float/float1/float2/float4
template <typename T>
void test_case_SIMD(const size_t SIZE, dim3 grid_dim, dim3 block_dim){
    size_t size_bytes = SIZE * sizeof(float);
    
    //step1: 初始化 host(cpu) 端数据; 预留 device(gpu) 数据端空间
    std::vector<float> host_a(SIZE, 1);
    std::vector<float> host_b(SIZE, 2);
    std::vector<float> host_c(SIZE, 3);
    
    T *device_a, *device_b, *device_c;//注意device端仅支持裸指针
    cudaMalloc(&device_a, size_bytes);
    cudaMalloc(&device_b, size_bytes);
    cudaMalloc(&device_c, size_bytes);

    //step2: copy from host to device; 注意需传输host_a的数据内容,而非vector对象host_a
    cudaMemcpy(device_a, (T*)(host_a.data()), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, (T*)(host_b.data()), size_bytes, cudaMemcpyHostToDevice);

    //step3: gpu calculation via kernel function
    add_cuda<T>(device_c, device_a, device_b, size_bytes/sizeof(T), grid_dim, block_dim);

    //step4: copy from device to host
    cudaMemcpy((T*)(host_c.data()), device_c, size_bytes, cudaMemcpyDeviceToHost);

    //step5: free cuda memory
    if(device_a)
        cudaFree(device_a);
    if(device_b)
        cudaFree(device_b);
    if(device_c)
        cudaFree(device_c);

}




//T: half/half2
template <typename T>
void test_case_half(const size_t SIZE, dim3 grid_dim, dim3 block_dim){
    size_t size_bytes = SIZE * sizeof(half);
    
    //step1: 初始化 host(cpu) 端数据; 预留 device(gpu) 数据端空间
    std::vector<half> host_a(SIZE, __float2half(1.0f));
    std::vector<half> host_b(SIZE, __float2half(2.0f));
    std::vector<half> host_c(SIZE, __float2half(3.0f));
    
    T *device_a, *device_b, *device_c;//注意device端仅支持裸指针
    cudaMalloc(&device_a, size_bytes);
    cudaMalloc(&device_b, size_bytes);
    cudaMalloc(&device_c, size_bytes);

    //step2: copy from host to device; 注意需传输host_a的数据内容,而非vector对象host_a
    cudaMemcpy(device_a, (T*)(host_a.data()), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, (T*)(host_b.data()), size_bytes, cudaMemcpyHostToDevice);

    //step3: gpu calculation via kernel function
    add_cuda<T>(device_c, device_a, device_b, size_bytes/sizeof(T), grid_dim, block_dim);

    //step4: copy from device to host
    cudaMemcpy((T*)(host_c.data()), device_c, size_bytes, cudaMemcpyDeviceToHost);

    //step5: free cuda memory
    if(device_a)
        cudaFree(device_a);
    if(device_b)
        cudaFree(device_b);
    if(device_c)
        cudaFree(device_c);

}



int main(){
    const size_t SIZE = 1 << 24;//1M
    dim3 grid_dim(1024);
    dim3 block_dim(128);

    test_case_SIMD<float>(SIZE, grid_dim, block_dim);

    test_case_SIMD<float2>(SIZE, grid_dim, block_dim);

    test_case_SIMD<float4>(SIZE, grid_dim, block_dim);

    test_case_half<half>(SIZE, grid_dim, block_dim);

    test_case_half<half2>(SIZE, grid_dim, block_dim);

    return 0;
}

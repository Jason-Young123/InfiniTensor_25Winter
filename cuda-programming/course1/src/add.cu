#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

//error indicator
#define CUDA_CHECK(call) {\
    cudaError_t err = call;\
    if(err != cudaSuccess){\
        std::cerr << "CUDA error @ "<< __FILE__ << ":" << __LINE__\
        << "-" << cudaGetErrorString(err) << "\n";\
        exit(1);\
    }\
}




//传统cpu计算函数,传inference
void add_cpu(std::vector<float> &c, const std::vector<float> &a,
            const std::vector<float> &b) {
    for(size_t i = 0; i < a.size(); i++){
        c[i] = a[i] + b[i];
    }
}




//更低效
//每个线程处理一个元素,内无循环,外部多次启动核函数
//核函数
template<typename T>
__global__ void add_kernel1(T *c, const T *a, const T *b, int n, size_t step){
    int idx = blockIdx.x * blockDim.x + threadIdx.x + step;
    if(idx < n)
        c[idx] = a[idx] + b[idx];
}

//n: 待计算元素总数
//grid: #block
//block: #thread per block
//  step: 开启的线程总数
template<typename T>
void add_cuda1(T *c, const T *a, const T* b, size_t n, const dim3 &grid, const dim3 &block){
    size_t step = grid.x * block.x;
    for(size_t i = 0; i < n; i += step){//反复launch 核函数,低效
        add_kernel1<T><<<grid, block>>>(c, a, b, n, i);
    }
}


//更高效
//每个线程处理若干元素,每个元素间隔step(即线程总数),内有循环
template<typename T>
__global__ void add_kernel2(T *c, const T *a, const T *b, size_t n, size_t step){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(size_t i = idx; i < n; i += step){
        c[i] = a[i] + b[i];
    }
}

//n: 待计算元素总数
//grid: #block
//block: #thread per block
//  step: 开启的线程总数
template<typename T>
void add_cuda2(T *c, const T *a, const T* b, size_t n, const dim3 &grid, const dim3 &block){
    size_t step = grid.x * block.x;
    add_kernel2<T><<<grid, block>>>(c, a, b, n, step);
}




int main(){

    //print critical device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // 0 表示第一个 GPU 设备
    printf("----------device info-----------\n");
    printf("maxGridSize([x, y, z]):   [%d  %d  %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("maxThreadsDim([x, y, z]): [%d  %d  %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    //printf("maxT")
    //std::cout << "maxGridSize: " << prop.maxGridSize << endl;

    const size_t SIZE = 1 << 20;//1M
    size_t size_bytes = SIZE * sizeof(float);


    //step1: 初始化 host(cpu) 端数据; 预留 device(gpu) 数据端空间
    std::vector<float> host_a(SIZE, 1);
    std::vector<float> host_b(SIZE, 2);
    std::vector<float> host_c(SIZE, 3);

    float *device_a, *device_b, *device_c;//注意device端仅支持裸指针
    CUDA_CHECK(cudaMalloc(&device_a, size_bytes));
    CUDA_CHECK(cudaMalloc(&device_b, size_bytes));
    CUDA_CHECK(cudaMalloc(&device_c, size_bytes));


    //step2: copy from host to device; 注意需传输host_a的数据内容,而非vector对象host_a
    CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_c, host_c.data(), size_bytes, cudaMemcpyHostToDevice));

    //step3: gpu calculation via kernel function
    //dim3 block_dim(256);
    //dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);// = ceil(SIZE/block_dim.x)
    dim3 grid_dim(256);
    dim3 block_dim(256);
    printf("------------usage---------\n");
    std::cout << "grid.x (blocks per grid): " << grid_dim.x << std::endl;
    std::cout << "block.x (threads per block): " << block_dim.x << std::endl;
    //grid_dim: 所需block数目; block_dim: 每个block中线程数
    //add_kernel1<float><<<grid_dim, block_dim>>>(device_c, device_a, device_b, SIZE);

    //比较不同方法：
    // auto t0 = std::chrono::high_resolution_clock::now();

    // add_cuda1<float>(device_c, device_a, device_b, SIZE, grid_dim, block_dim);
    // cudaDeviceSynchronize();
    // auto t1 = std::chrono::high_resolution_clock::now();

    add_cuda2<float>(device_c, device_a, device_b, SIZE, grid_dim, block_dim);
    // cudaDeviceSynchronize();
    // auto t2 = std::chrono::high_resolution_clock::now();

    // add_cpu(host_c, host_a, host_b);
    // auto t3 = std::chrono::high_resolution_clock::now();


    // auto time1 = std::chrono::duration<float, std::milli>(t1 - t0).count();
    // auto time2 = std::chrono::duration<float, std::milli>(t2 - t1).count();
    // auto time3 = std::chrono::duration<float, std::milli>(t3 - t2).count();
    // printf("-------------perf analysis-----------\n");
    // std::cout << "Relaunch-Kernel time: " << time1 << " ms" << std::endl;
    // std::cout << "Grid-Stride time: " << time2 << " ms" << std::endl;
    // std::cout << "Cpu time: " << time3 << " ms" << std::endl; 

    //step4: copy from device to host
    CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, size_bytes, cudaMemcpyDeviceToHost));

    //step5: free memory
    if(device_a)
        CUDA_CHECK(cudaFree(device_a));
    if(device_b)
        CUDA_CHECK(cudaFree(device_b));
    if(device_c)
        CUDA_CHECK(cudaFree(device_c)); 

    return 0;
}
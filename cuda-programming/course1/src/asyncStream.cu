/*
### 10. 异步流与内存流水线 (Asynchronous Streams Pipeline)

题目：有一个巨大的数组，需要对其做简单的向量加法; (假设)由于显存装不下，需要分块处理。

面试官附加限制：

- 必须使用多个 `cudaStream_t` 配合 `cudaMemcpyAsync`，实现 Host-To-Device 数据传输、Kernel 执行、Device-To-Host 数据传回三者的重叠（Overlap）。

*/



#include <iostream>
#include <cstdlib>


//error indicator
#define CUDA_CHECK(call) {\
    cudaError_t err = call;\
    if(err != cudaSuccess){\
        std::cerr << "CUDA error @ "<< __FILE__ << ":" << __LINE__\
        << "-" << cudaGetErrorString(err) << "\n";\
        exit(1);\
    }\
}



//简单起见直接用float
__global__ void vectorAdd(const float* d_va, const float* d_vb, float* d_vout, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for(size_t i = idx; i < n; i += stride){
        d_vout[i] = d_va[i] + d_vb[i];
    }

}







int main(){
    const size_t N = 1ULL << 24;
    const int num_streams = 4;
    const size_t chunk_size = N / num_streams;
    const size_t bytes = N * sizeof(float);
    const size_t chunk_bytes = chunk_size * sizeof(float);

    cudaStream_t streams[num_streams] = {};
    float *h_va, *h_vb, *h_vout_multi, *h_vout_single;
    float *d_va, *d_vb, *d_vout;

    // cudaMemcpyAsync要真正异步，host端内存需要是page-locked memory。
    CUDA_CHECK(cudaMallocHost((void**)&h_va, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_vb, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_vout_multi, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_vout_single, bytes));

    for(size_t i = 0; i < N; ++i){
        h_va[i] = 1.0f;
        h_vb[i] = 2.0f;
        h_vout_multi[i] = 0.0f;
        h_vout_single[i] = 0.0f;
    }

    CUDA_CHECK(cudaMalloc((void**)&d_va, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_vb, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_vout, bytes));

    dim3 block_dim(256);
    dim3 chunk_grid_dim((chunk_size + block_dim.x - 1) / block_dim.x);
    dim3 full_grid_dim((N + block_dim.x - 1) / block_dim.x);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for(int s = 0; s < num_streams; ++s){//分批次创建stream
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    //step1: 4个stream分块执行H2D -> kernel -> D2H
    CUDA_CHECK(cudaEventRecord(start));
    for(int s = 0; s < num_streams; ++s){//4个stream循环执行copyH2D, kernel计算, copyD2H
        size_t offset = s * chunk_size;

        CUDA_CHECK(cudaMemcpyAsync(d_va + offset, h_va + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_vb + offset, h_vb + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[s]));
        vectorAdd<<<chunk_grid_dim, block_dim, 0, streams[s]>>>(d_va + offset, d_vb + offset, d_vout + offset, chunk_size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_vout_multi + offset, d_vout + offset, chunk_bytes, cudaMemcpyDeviceToHost, streams[s]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float multi_stream_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&multi_stream_ms, start, stop));

    //step2: 默认stream执行完整数组的H2D -> kernel -> D2H
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpyAsync(d_va, h_va, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(d_vb, h_vb, bytes, cudaMemcpyHostToDevice));
    vectorAdd<<<full_grid_dim, block_dim>>>(d_va, d_vb, d_vout, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_vout_single, d_vout, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float single_stream_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&single_stream_ms, start, stop));

    bool check_pass = true;
    for(size_t i = 0; i < N; ++i){
        if(h_vout_multi[i] != 3.0f || h_vout_single[i] != 3.0f){
            check_pass = false;
            std::cerr << "check failed at idx " << i
                      << ", multi: " << h_vout_multi[i]
                      << ", single: " << h_vout_single[i]
                      << ", expected 3" << std::endl;
            break;
        }
    }

    std::cout << "N: " << N << std::endl;
    std::cout << "Multi-stream time: " << multi_stream_ms << " ms" << std::endl;
    std::cout << "Single-stream time: " << single_stream_ms << " ms" << std::endl;
    std::cout << "Check: " << (check_pass ? "PASS" : "FAIL") << std::endl;

    for(int s = 0; s < num_streams; ++s){//stream销毁
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_va));
    CUDA_CHECK(cudaFree(d_vb));
    CUDA_CHECK(cudaFree(d_vout));
    CUDA_CHECK(cudaFreeHost(h_va));
    CUDA_CHECK(cudaFreeHost(h_vb));
    CUDA_CHECK(cudaFreeHost(h_vout_multi));
    CUDA_CHECK(cudaFreeHost(h_vout_single));
    CUDA_CHECK(cudaDeviceReset());

    return check_pass ? 0 : 1;
}

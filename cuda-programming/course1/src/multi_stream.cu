/*
    仅用于测试多流行为; 一定要搞清楚各种stream配置方式 + 内存分配方式对于多任务效率的影响
*/





//待测试kernel1: vector add, grided stride
__global__ void vectorAdd(const float* d_va, const float* d_vb, float* d_vout, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = idx; i < n; i += stride){
        d_vout[i] = d_va[i] + d_vb[i];
    }
}


//待测试kernel2: 
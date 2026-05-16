#include <iostream>


template <typename T>
__global__ void transpose1(T* dst, const T* src, const int M, const int N){//M行N列
    //block内相对线程编号
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    //grid层面绝对线程编号(转置前)
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    //转置后的绝对线程编号
    int idx_x_new = blockIdx.y * blockDim.x + threadIdx.x;
    int idx_y_new = blockIdx.x * blockDim.y + threadIdx.y;

    //声明sharedMem,固定大小为32 x 32
    __shared__ T smem[32][32];
    
    //step1: load from GM to SM, 注意边界; 此时访问GM完全合并, 访问SM也不存在bank conflict
    if(idx_x < N && idx_y < M){
        smem[tidy][tidx] = src[idx_y * N + idx_x];
    }

    __syncthreads();

    //step2: store from SM to GM, 注意边界; 此时访问GM依然完全合并, 但访问SM存在严重bank conflict
    if(idx_x_new < M && idx_y_new < N){
        dst[idx_y_new * M + idx_x_new] = smem[tidx][tidy];
    }
}







int main(){
    //矩阵尺寸
    int M = 1000;//行
    int N = 1000;//列

    float *h_src = new float[M * N];
    float *h_dst = new float[N * M];
    size_t totalByte = M * N * sizeof(float);

    //初始化输入和输出矩阵
    for (int i = 0; i < M * N; ++i) {
        h_src[i] = static_cast<float>(i);
    }
    memset(h_dst, 0., totalByte);

    //step1: GPU端资源分配
    float *d_dst, *d_src;
    cudaMalloc((void **)&d_src, totalByte);
    cudaMalloc((void **)&d_dst, totalByte);

    //step2: copy H2D
    cudaMemcpy(d_src, h_src, totalByte, cudaMemcpyHostToDevice);

    //step3: launch kernel
    dim3 block_dim(32, 32);
    dim3 grid_dim((N + 31)/32, (M + 31)/32);
    transpose1<float><<<grid_dim, block_dim>>>(d_dst, d_src, M, N);
    cudaDeviceSynchronize(); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err) << std::endl;
    }

    //step4: copy D2H
    cudaMemcpy(h_dst, d_dst, totalByte, cudaMemcpyDeviceToHost);

    //check result
    bool is_correct = true;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            if (std::abs(h_dst[c * M + r] - h_src[r * N + c]) > 1e-5) {
                is_correct = false;
                std::cout << "Error at (" << r << ", " << c << ")\n";
                std::cout << h_dst[c * M + r] << std::endl;
                std::cout << h_src[r * N + c] << std::endl;
                break;
            }
        }
        if (!is_correct) break; // 发现错误退出外层循环
    }

    if (is_correct) {
        std::cout << "Success: Matrix transpose is correct!" << std::endl;
    } else {
        std::cout << "Failed: Matrix transpose results do not match." << std::endl;
    }


    //step5: cleanup
    cudaFree(d_dst);
    cudaFree(d_src);
    delete[] h_src;
    delete[] h_dst;

    return 0;
}




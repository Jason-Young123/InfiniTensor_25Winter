#include <iostream>
#include <cublas_v2.h>



//blockDim.x = blockDim.y = 32
template <typename T>
__global__ void transpose1(T* dst, const T* src, const int M, const int N, const dim3 sharedDim){//M行N列; smem尺寸动态决定为sharedDim.x列sharedDim.y行
    //block内相对线程编号
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    //grid层面绝对线程编号(转置前)
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    //转置后的绝对线程编号
    int idx_x_new = blockIdx.y * blockDim.x + threadIdx.x;
    int idx_y_new = blockIdx.x * blockDim.y + threadIdx.y;

    //声明sharedMem
    extern __shared__ T smem[];
    
    //step1: load from GM to SM, 注意边界; 此时访问GM完全合并, 访问SM也不存在bank conflict
    if(idx_x < N && idx_y < M){
        //smem[tidy][tidx] = src[idx_y][idx_x];
        smem[tidy * sharedDim.x + tidx] = src[idx_y * N + idx_x];
    }

    __syncthreads();//必须确保block内每个thread都已经加载完毕才能进行下一步写入操作

    //step2: store from SM to GM, 注意边界; 此时访问GM依然完全合并, 但访问SM存在严重bank conflict
    if(idx_x_new < M && idx_y_new < N){
        //dst[idx_y_new][idx_x_new] = smem[tidx][tidy];
        dst[idx_y_new * M + idx_x_new] = smem[tidx * sharedDim.x + tidy];
    }
}


//blockDim.x = 32, blockDim.y = 8
template <typename T>
__global__ void transpose2(T* dst, const T* src, const int M, const int N, const dim3 sharedDim){
    //block内相对线程编号
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    
    //grid层面绝对线程编号(转置前)
    int TILE_DIM = blockDim.x;//32 x 32数据块
    int idx_x = blockIdx.x * TILE_DIM + threadIdx.x;
    int idx_y = blockIdx.y * TILE_DIM + threadIdx.y;

    //转置后的绝对线程编号
    int idx_x_new = blockIdx.y * TILE_DIM + threadIdx.x;
    int idx_y_new = blockIdx.x * TILE_DIM + threadIdx.y;

    //声明sharedMem
    extern __shared__ T smem[];

    //step1: load from GM to SM, 注意边界; 此时访问GM完全合并, 访问SM也不存在bank conflict; 但是现在每个线程需要负责加载4个数据
    for(int i = 0; i < TILE_DIM / blockDim.y; ++i) {
        int current_idx_y = idx_y + i * blockDim.y; 

        if (idx_x < N && current_idx_y < M) {
            smem[(tidy + i * blockDim.y) * sharedDim.x + tidx] = src[current_idx_y * N + idx_x];
        }
    }

    __syncthreads();//必须同步

    //step2: store from SM to GM
    for(int i = 0; i < TILE_DIM / blockDim.y; ++i) {
        int current_idx_y_new = idx_y_new + i * blockDim.y;

        if (idx_x_new < M && current_idx_y_new < N) {
            dst[current_idx_y_new * M + idx_x_new] = smem[tidx * sharedDim.x + (tidy + i * blockDim.y)];
        }
    }
}



//利用cublas的cublasSsgeam函数进行转置;这里的S就代表单精度,因而不支持模板调用
//cublasSgeam 的数学公式是：C = alpha * op(A) + beta * op(B); 注意, cublas中所有矩阵都是按列主序进行存储, 而非cpp的行主序存储
void transpose_cublas(float* dst, const float* src, const int M, const int N){//M行N列
    cublasHandle_t handle;
    cublasCreate(&handle);

    //设置标量参数
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    //设定 Leading Dimension (主维度)
    //在 C++ 的行优先中，lda 实际上就是矩阵的“物理列数（宽度）”
    int lda = N; // 源矩阵宽度
    int ldb = M; // B 不重要，随便填个合法的
    int ldc = M; // 目标矩阵宽度

    cublasStatus_t status = cublasSgeam(
        handle, 
        CUBLAS_OP_T, // 对 A 转置
        CUBLAS_OP_N, // B 不做操作
        M,           // 输出矩阵在 cuBLAS 视角下的行数
        N,           // 输出矩阵在 cuBLAS 视角下的列数
        &alpha, 
        src, lda, 
        &beta, 
        src, ldb,  // B 用不到，传 d_src 占位防止空指针报错
        dst, ldc
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS failed!" << std::endl;
    }

    cublasDestroy(handle);
}




int main(){
    //矩阵尺寸
    int M = 1000;//行
    int N = 2000;//列

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
    dim3 block_dim1(32, 32);
    dim3 block_dim2(32, 8);
    dim3 grid_dim1((N + 31)/32, (M + 31)/32);
    dim3 grid_dim2((N + 31)/32, (M + 31)/32);//由于trnaspose2中每个block实际负责32x32的数据块,因此grid_dim依然是/32
    dim3 shared_dim(33, 32);//列padding
    size_t sharedByte = shared_dim.x * shared_dim.y * sizeof(float);


    transpose1<float><<<grid_dim1, block_dim1, sharedByte>>>(d_dst, d_src, M, N, shared_dim);//32x32 blockDim, sm有列padding
    cudaDeviceSynchronize(); 
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err1) << std::endl;
    }
    
    transpose2<float><<<grid_dim2, block_dim2, sharedByte>>>(d_dst, d_src, M, N, shared_dim);//32x8 blockDim, sm有列padding
    cudaDeviceSynchronize(); 
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err2) << std::endl;
    }

    transpose_cublas(d_dst, d_src, M, N);
    cudaDeviceSynchronize();
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err3) << std::endl;
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




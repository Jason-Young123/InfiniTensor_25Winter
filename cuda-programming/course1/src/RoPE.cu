/*
cuda实现RoPE算子:
    输入: Q/K
    输出: 调制相位(即对每一行进行旋转编码)之后的Q/K矩阵

*/






#include <iostream>
#include <cmath>

#define BASE 10000.0f//θi = base ^ (-2i/d)

//为矩阵(N x d)添加旋转编码, 矩阵实际调用时可以为Q/K
//d_M: 输入矩阵, 尺寸为N x d, 且假定其中token的绝对位置就是0 - N-1(模拟prefill阶段)
//d_Mrope: 调制完毕之后的输出矩阵, 其中每一行都通过hadmard积进行了旋转编码
__global__ void apply_rope(float* d_M, float* d_Mrope, size_t d){
    int tidx = threadIdx.x;
    int stride = blockDim.x;
    int row = blockIdx.x;//token的绝对位置

    size_t pair_num = d / 2;

    for(size_t i = tidx; i < pair_num; i += stride){
        size_t col = i * 2;
        float x0 = d_M[row * d + col];
        float x1 = d_M[row * d + col + 1];

        //原论文中每一对维度(x_2i, x_2i+1)乘以一个2x2旋转矩阵
        //也等价于: x * cos(mθ) + rotate_half(x) * sin(mθ), 即hadamard积形式
        float theta = ::powf(BASE, -2.0f * float(i) / float(d));
        float angle = float(row) * theta;
        float sin_val, cos_val;
        __sincosf(angle, &sin_val, &cos_val);

        d_Mrope[row * d + col] = x0 * cos_val - x1 * sin_val;
        d_Mrope[row * d + col + 1] = x0 * sin_val + x1 * cos_val;
    }

    //RoPE一般要求d为偶数; 这里为了兼容奇数d,最后一维直接拷贝
    if((d % 2 == 1) && tidx == 0){
        d_Mrope[row * d + d - 1] = d_M[row * d + d - 1];
    }
}









int main(){
    int M = 128, N = 4096;//128 x 4096的输入矩阵, 对每行token添加RoPE

    float* h_in = new float[M * N];
    float* h_out = new float[M * N];
    float* h_ref = new float[M * N];
    for(int i = 0; i < M * N; ++i){
        h_in[i] = float((i % 97) - 48) * 0.01f;
    }

    float* d_in, *d_out;
    cudaMalloc((void**)&d_in, sizeof(float) * M * N);
    cudaMalloc((void**)&d_out, sizeof(float) * M * N);

    cudaMemcpy(d_in, h_in, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    dim3 block_dim(512);//每行由1个block中的512个线程处理
    dim3 grid_dim(M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    apply_rope<<<grid_dim, block_dim>>>(d_in, d_out, N);
    cudaDeviceSynchronize(); // 必须等待预热完成
    
    cudaEventRecord(start);
    apply_rope<<<grid_dim, block_dim>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start, stop);
    std::cout << "RoPE runtime: " << milliseconds1 << " ms" << std::endl;

    cudaMemcpy(h_out, d_out, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    //check result
    for(int row = 0; row < M; ++row){
        for(int i = 0; i < N / 2; ++i){
            int col = i * 2;
            float x0 = h_in[row * N + col];
            float x1 = h_in[row * N + col + 1];
            float theta = std::pow(BASE, -2.0f * float(i) / float(N));
            float angle = float(row) * theta;
            float sin_val = std::sin(angle);
            float cos_val = std::cos(angle);
            h_ref[row * N + col] = x0 * cos_val - x1 * sin_val;
            h_ref[row * N + col + 1] = x0 * sin_val + x1 * cos_val;
        }
    }

    float max_abs_err = 0.0f;
    int bad_cnt = 0;
    for(int i = 0; i < M * N; ++i){
        float err = std::fabs(h_out[i] - h_ref[i]);
        max_abs_err = (max_abs_err > err) ? max_abs_err : err;
        if(err > 1e-4f){
            ++bad_cnt;
        }
    }

    std::cout << h_in[0] << " - " << h_in[1] << " - " << h_in[2] << " - " << h_in[3] << std::endl;
    std::cout << h_out[0] << " - " << h_out[1] << " - " << h_out[2] << " - " << h_out[3] << std::endl;
    std::cout << h_out[N] << " - " << h_out[N + 1] << " - " << h_out[N + 2] << " - " << h_out[N + 3] << std::endl;
    std::cout << "max_abs_err = " << max_abs_err << ", bad_cnt = " << bad_cnt << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out; delete[] h_ref;

    return 0;
}

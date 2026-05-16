#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define RUNTIME_ERR_TYPE cudaError_t
#define RUNTIME_SUCCESS_CODE cudaSuccess
#define RUNTIME_GET_ERROR_STR cudaGetErrorString

#define RUNTIME_CHECK(call)                                                    \
  do {                                                                         \
    RUNTIME_ERR_TYPE err = call;                                               \
    if (err != RUNTIME_SUCCESS_CODE) {                                         \
      std::cerr << "Runtime error at " << __FILE__ << ":" << __LINE__ << " - " \
                << RUNTIME_GET_ERROR_STR(err) << "\n";                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)




template <typename T>
__device__ T myexp(T x) {
    /*if constexpr (std::is_same<T, __half>::value) {
        float fx = __half2float(x);
        float result = expf(fx);
        return __float2half(result);
    } else {
        float fx = static_cast<float>(x);
        return static_cast<T>(expf(fx));
    }*/
    return T(0.5);
}

__device__ float myexp1(float x){
  return float(exp(double(x)));
}

template <typename T>
__device__ T warp_reduce_max(T val){
#pragma unroll//短循环自动展开,省去分支预测,提升效率
    for(int offset = 16; offset > 0; offset >>= 1){
        T tmp = __shfl_down_sync(0xffffffff, val, offset);
        val = (val > tmp) ? val : tmp;
    }
    return val;
}


//完成32x32矩阵求rowMax功能
template <typename T>
__device__ void rowMax(const T* src, T* dst){
  int tid_x = threadIdx.x;//横向,32列
  int tid_y = threadIdx.y;//纵向,32行

  T val = src[tid_y * 32 + tid_x];

  val = warp_reduce_max(val);//按行归约,因为同一warp的线程共享相同的threadIdx.y（行索引）

  if(tid_x == 0){
    dst[tid_y] = val;
  }
}

//完成32x32矩阵求rowSoftmax功能
template <typename T>
__device__ void rowSoftmax(T* src, T* m, T* l){
  int tid_x = threadIdx.x;//横向,32列
  int tid_y = threadIdx.y;//纵向,32行

  //定义临时m和l向量
  __shared__ T m_tmp[32];
  __shared__ T m_new[32];
  __shared__ T l_tmp[32];
  __shared__ T l_new[32];

  //step1: m = rowMax(src), row-wise
  T val = src[tid_y * 32 + tid_x];
  val = warp_reduce_max(val);//按行归约,因为同一warp的线程共享相同的threadIdx.y（行索引）
  if(tid_x == 0){
    m_tmp[tid_y] = val;
  }
  __syncthreads();

  //step2: P = exp(src - m), point-wise
  src[tid_y * 32 + tid_x] = myexp(src[tid_y * 32 + tid_x] - m_tmp[tid_y]);
  __syncthreads();

  //step3: l = rowSum(P), row-wise
  T val1 = src[tid_y * 32 + tid_x];
  val1 = warp_reduce_sum(val1);//按行归约
  if(tid_x == 0){
    l_tmp[tid_y] = val1;
  }
  __syncthreads();

  //step4: 对入口参数m进行更新
  m_new[tid_y] = (m_tmp[tid_y] > m[tid_y]) ? m_tmp[tid_y] : m[tid_y];
  __syncthreads();

  l_new[tid_y] = myexp<T>(m[tid_y] - m_new[tid_y]) * l[tid_y] + myexp<T>(m_tmp[tid_y] - m_new[tid_y]) * l_tmp[tid_y];
  __syncthreads();
}



//对应flash attention v2原文算法, block采用三维布局
template <typename T>
__global__ void kernel_flashAttention(int batch_size, int target_seq_len, int src_seq_len, int q_heads, int kv_heads, int head_dim, bool is_causal, const T* Q, const T* K, const T* V, T* O){
  int tid_x = threadIdx.x;//横向,blockDim.x列
  int tid_y = threadIdx.y;//纵向,blockDim.y行
  int bid_x = blockIdx.x;//x方向,总数 = #q_heads
  int bid_y = blockIdx.y;//y方向,总数 = #batch
  int bid_z = blockIdx.z;//z方向,总数 = Tr
  const int p = q_heads / kv_heads;//计算比例系数
  const int Br = 24;//Q纵向每块大小, 默认为32 (RTX 5090)
  const int Bc = 24;//K/V纵向分块大小, 默认为32
  const int Tc = (src_seq_len + Bc - 1) / Bc;//对应原始论文中K/V纵向分块数Tc,其中Bc = 32

  //预计算常量
  const int QO_index = ((((bid_y * target_seq_len) + (Br * bid_z + tid_y)) * q_heads) + bid_x) * head_dim;
  const float scale_factor = float(1.0) / sqrtf(head_dim);
  //const float scale_factor = __fdividef(1.0, head_dim);

  //定义一系列临时变量
  __shared__ float SP[Br][Bc];//复用S和P
  __shared__ float m_prev[Br], m_new[Br];
  __shared__ float l_prev[Br], l_new[Br];

    int bound_tid_y = min(Br, target_seq_len - Br * bid_z);
    //step0: reset l to 0 and m to -INFINITY
    if(tid_x == 0){
      m_prev[tid_y] = float(-8192);
      l_prev[tid_y] = float(0);
    }
    __syncthreads();

    for(int j = 0; j < Tc; ++j){//对于每个K/V分块
      if(is_causal && bid_z < j){//early exit, 直接跳过
        __syncthreads();
        continue;
      }

      SP[tid_y][tid_x] = float(-8192);
      __syncthreads();
      int bound_tid_x = min(Bc, src_seq_len - Bc * j);
      bool is_compute = (!is_causal) || (bid_z > j) || (bid_z == j && tid_y >= tid_x);
    
      //step1: S = Q @ K.T, point-wise
      /*if(tid_y < bound_tid_y && tid_x < bound_tid_x){//用于边缘不完整块
        float val0 = float(0);//临时sum  
        if(is_compute){
          #pragma unroll
          for(int k = 0; k < head_dim; ++k){
          //S[tid_y][tid_x] += Q[bid_y][Br * bid_z + tid_y][bid_x][k] * K[bid_y][Bc * j + tid_x][bid_x / p][k];
            val0 += float(Q[QO_index + k]) *\
                    float(K[((((bid_y * src_seq_len) + (Bc * j + tid_x)) * kv_heads) + bid_x / p) * head_dim + k]);
          }
          SP[tid_y][tid_x] = val0 * scale_factor;//缩放因子
        }
      }
      __syncthreads();

      //step2: m_new = max(m_prev, rowMax(S))
      //float val1 = SP[tid_y][tid_x];
      //val1 = warp_reduce_max(val1);
      if(tid_x == 0 && tid_y < bound_tid_y){
        float val1 = SP[tid_y][0];//手动实现非并行求行最大值
        #pragma unroll
        for(int h = 1; h < Bc; ++h){
          val1 = (val1 < SP[tid_y][h]) ? SP[tid_y][h] : val1;
        }
        m_new[tid_y] = (val1 > m_prev[tid_y]) ? val1 : m_prev[tid_y];
      }
      __syncthreads();*/

      //step3: P = exp(S - m_new), point-wise
      /*if(tid_y < bound_tid_y && tid_x < bound_tid_x){
        if(is_compute){
          SP[tid_y][tid_x] = myexp<float>(SP[tid_y][tid_x] - m_new[tid_y]);
        }
        else{
          SP[tid_y][tid_x] = float(0);
        }
      }
      else{
        SP[tid_y][tid_x] = float(0);
      }
      __syncthreads();

      //step4: l_new = exp(m_prev - m_new) * l_prev + rowSum(P)
      //float val2 = SP[tid_y][tid_x];
      //val2 = warp_reduce_sum(val2);
      float exp_result = myexp<float>(m_prev[tid_y] - m_new[tid_y]);
      if(tid_x == 0 && tid_y < bound_tid_y){
        float val2 = 0;//手动实现非并行求rowSum
        #pragma unroll
        for(int h = 0; h < Bc; ++h){
          val2 += SP[tid_y][h];
        }
        l_new[tid_y] = exp_result * l_prev[tid_y] + val2;
      }
      __syncthreads();

      //step5: O = 1/(exp(m_prev - m_new)) * O + P @ V
      if(tid_x < bound_tid_x && tid_y < bound_tid_y){//32路并行计算Oi的每一行
        for(int u = tid_x; u < head_dim; u += blockDim.x){
          float val3 = float(0);
          #pragma unroll(4)
          for(int w = 0; w < Bc; ++w){//val2 += P[tid_y][w] * V[bid_y][Bc * j + w][bid_x / p][u];
            val3 += SP[tid_y][w] * float(V[((((bid_y * src_seq_len) + (Bc * j + w)) * kv_heads) + (bid_x / p)) * head_dim + u]);
          }
          //O[bid_y][Br * bid_z + tid_y][bid_x][u]
          O[QO_index + u] = T(float(O[QO_index + u]) * exp_result + val3);
        }
      }
      __syncthreads();
      
      //step6: m_prev <- m_new; l_prev <- l_new
      if (tid_x == 0 && tid_y < bound_tid_y) {//向量更新只使用第1列线程
        m_prev[tid_y] = m_new[tid_y];
        l_prev[tid_y] = l_new[tid_y];
      }
      __syncthreads();*/

    }

    //post process, O = O/l_prev
    if(tid_x == 0 && tid_y < bound_tid_y){//32路并行计算Oi的每一行
      #pragma unroll
      for(int u = 0; u < head_dim; ++u){
        O[QO_index + u] =\
        T(float(O[QO_index + u]) / l_prev[tid_y]);
      }
    }
    __syncthreads();
}


template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  //step0: basic check

  //step1: 初始化,预留device端空间
  const size_t size_bytes_q = h_q.size() * sizeof(T);
  const size_t size_bytes_k = h_k.size() * sizeof(T);
  const size_t size_bytes_v = h_v.size() * sizeof(T);
  const size_t size_bytes_o = h_o.size() * sizeof(T);
  //const size_t size_bytes_lm = target_seq_len * query_heads * batch_size * sizeof(T);
  T *d_q, *d_k, *d_v, *d_o;//device端只支持裸指针
  RUNTIME_CHECK(cudaMalloc(&d_q, size_bytes_q));
  RUNTIME_CHECK(cudaMalloc(&d_k, size_bytes_k));
  RUNTIME_CHECK(cudaMalloc(&d_v, size_bytes_v));
  RUNTIME_CHECK(cudaMalloc(&d_o, size_bytes_o));
  //RUNTIME_CHECK(cudaMalloc(&d_l, size_bytes_lm));//l向量,长度 = target_seq_len
  //RUNTIME_CHECK(cudaMalloc(&d_m, size_bytes_lm));//m向量,长度 = target_seq_len

  //step2: 拷贝数据from host to device
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), size_bytes_q, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), size_bytes_k, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), size_bytes_v, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_o, 0, size_bytes_o));//d_o初始化为全0

  //step3: device端计算
  int Br = 32, Bc = 32;
  int grid_dim_z = (target_seq_len + Br - 1) / Br;
  dim3 block_dim(Br, Bc);
  dim3 grid_dim(query_heads, batch_size, grid_dim_z);
  //size_t smem_size = Br * Bc * sizeof(T) + 4 * Br * sizeof(T);//SP, m_prev, m_new, l_prev, l_new

  kernel_flashAttention<T><<<grid_dim, block_dim>>>(batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal, d_q, d_k, d_v, d_o);
  //注意核函数返回类型只能为void

  //step4: 拷贝数据from device to host(not needed)
  cudaMemcpy(h_o.data(), d_o, size_bytes_o, cudaMemcpyDeviceToHost);

  //step5: free memory
  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));

  return;
}


// 生成-1到1之间的随机数
std::vector<float> generate_random_data(size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

// 验证结果（简单验证，检查是否有NaN/Inf）
void validate_results(const std::vector<float>& output) {
    bool has_nan = false;
    bool has_inf = false;
    bool has_zero = false;
    float sum = 0.0f;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    for (const auto& val : output) {
        if (std::isnan(val)) has_nan = true;
        if (std::isinf(val)) has_inf = true;
        if (val == 0.0f) has_zero = true;
        sum += val;
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    
    std::cout << "Validation Results:" << std::endl;
    std::cout << "  Has NaN: " << (has_nan ? "YES" : "NO") << std::endl;
    std::cout << "  Has Inf: " << (has_inf ? "YES" : "NO") << std::endl;
    std::cout << "  All zeros: " << (has_zero && output[0] == 0.0f ? "YES" : "NO") << std::endl;
    std::cout << "  Min value: " << min_val << std::endl;
    std::cout << "  Max value: " << max_val << std::endl;
    std::cout << "  Average value: " << sum / output.size() << std::endl;
}

// 打印张量形状信息
void print_tensor_info(const char* name, int batch, int seq_len, int heads, int head_dim) {
    size_t total_elements = batch * seq_len * heads * head_dim;
    size_t total_bytes = total_elements * sizeof(float);
    
    std::cout << name << " Tensor Info:" << std::endl;
    std::cout << "  Shape: [" << batch << ", " << seq_len << ", " << heads << ", " << head_dim << "]" << std::endl;
    std::cout << "  Total elements: " << total_elements << std::endl;
    std::cout << "  Memory size: " << total_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
}

int main() {
    // 设置CUDA设备
    int device_id = 0;
    RUNTIME_CHECK(cudaSetDevice(device_id));
    
    cudaDeviceProp prop;
    RUNTIME_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    std::cout << "==========================================" << std::endl;
    std::cout << "Running Flash Attention Test on: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "==========================================" << std::endl << std::endl;
    
    // 设置参数
    const int batch_size = 2;
    const int target_seq_len = 256;
    const int src_seq_len = 256;
    const int query_heads = 16;
    const int kv_heads = 16;
    const int head_dim = 32;
    const bool is_causal = true;
    
    // 计算张量大小
    const size_t q_elements = batch_size * target_seq_len * query_heads * head_dim;
    const size_t k_elements = batch_size * src_seq_len * kv_heads * head_dim;
    const size_t v_elements = batch_size * src_seq_len * kv_heads * head_dim;
    const size_t o_elements = batch_size * target_seq_len * query_heads * head_dim;
    
    // 打印信息
    std::cout << "Test Configuration:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  target_seq_len: " << target_seq_len << std::endl;
    std::cout << "  src_seq_len: " << src_seq_len << std::endl;
    std::cout << "  query_heads: " << query_heads << std::endl;
    std::cout << "  kv_heads: " << kv_heads << std::endl;
    std::cout << "  head_dim: " << head_dim << std::endl;
    std::cout << "  is_causal: " << (is_causal ? "true" : "false") << std::endl;
    std::cout << std::endl;
    
    print_tensor_info("Q", batch_size, target_seq_len, query_heads, head_dim);
    print_tensor_info("K", batch_size, src_seq_len, kv_heads, head_dim);
    print_tensor_info("V", batch_size, src_seq_len, kv_heads, head_dim);
    print_tensor_info("O", batch_size, target_seq_len, query_heads, head_dim);
    
    std::cout << std::endl << "Generating random data..." << std::endl;
    
    // 生成随机数据
    auto h_q = generate_random_data(q_elements);
    auto h_k = generate_random_data(k_elements);
    auto h_v = generate_random_data(v_elements);
    std::vector<float> h_o(o_elements, 0.0f);  // 输出初始化为0
    
    std::cout << "Data generation complete." << std::endl << std::endl;
    
    // 预热运行（消除初始化开销）
    std::cout << "Warm-up run..." << std::endl;
    try {
        flashAttention<float>(h_q, h_k, h_v, h_o, 
                              batch_size, target_seq_len, src_seq_len,
                              query_heads, kv_heads, head_dim, is_causal);
    } catch (const std::exception& e) {
        std::cerr << "Error during warm-up: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after warm-up: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }
    
    // 清空输出
    std::fill(h_o.begin(), h_o.end(), 0.0f);
    
    // 正式运行并计时
    std::cout << "Running Flash Attention..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        flashAttention<float>(h_q, h_k, h_v, h_o, 
                              batch_size, target_seq_len, src_seq_len,
                              query_heads, kv_heads, head_dim, is_causal);
    } catch (const std::exception& e) {
        std::cerr << "Error during Flash Attention: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // 检查CUDA错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after Flash Attention: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << std::endl << "Flash Attention completed!" << std::endl;
    std::cout << "Execution time: " << duration.count() / 1000.0 << " ms" << std::endl;
    
    // 验证结果
    std::cout << std::endl;
    validate_results(h_o);
    
    // 可选：保存部分结果用于调试
    const int sample_size = 10;
    std::cout << std::endl << "Sample output values (first " << sample_size << "):" << std::endl;
    for (int i = 0; i < sample_size && i < h_o.size(); ++i) {
        std::cout << "  h_o[" << i << "] = " << std::scientific << std::setprecision(6) << h_o[i] << std::endl;
    }
    
    // 计算FLOPs（近似值）
    // Flash Attention的FLOPs大约是 2 * batch * heads * seq_len^2 * head_dim
    double flops = 2.0 * batch_size * query_heads * 
                   static_cast<double>(target_seq_len) * src_seq_len * head_dim;
    double gflops = flops / 1e9;
    double gflops_per_sec = gflops / (duration.count() / 1e6);
    
    std::cout << std::endl << "Performance Metrics:" << std::endl;
    std::cout << "  Approximate FLOPs: " << flops << std::endl;
    std::cout << "  GFLOPs: " << gflops << std::endl;
    std::cout << "  GFLOP/s: " << gflops_per_sec << std::endl;
    
    // 检查设备内存是否正确释放
    size_t free_mem_before, total_mem_before;
    size_t free_mem_after, total_mem_after;
    
    RUNTIME_CHECK(cudaMemGetInfo(&free_mem_before, &total_mem_before));
    
    // 可选：运行多次取平均值
    const int num_runs = 10;
    if (num_runs > 1) {
        std::cout << std::endl << "Running " << num_runs << " iterations for average timing..." << std::endl;
        
        double total_time = 0.0;
        for (int i = 0; i < num_runs; ++i) {
            // 清空输出
            std::fill(h_o.begin(), h_o.end(), 0.0f);
            
            auto run_start = std::chrono::high_resolution_clock::now();
            flashAttention<float>(h_q, h_k, h_v, h_o, 
                                  batch_size, target_seq_len, src_seq_len,
                                  query_heads, kv_heads, head_dim, is_causal);
            auto run_end = std::chrono::high_resolution_clock::now();
            
            auto run_duration = std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_start);
            total_time += run_duration.count();
            
            // 检查错误
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA Error in iteration " << i << ": " << cudaGetErrorString(err) << std::endl;
                break;
            }
        }
        
        double avg_time_ms = total_time / num_runs / 1000.0;
        std::cout << "Average execution time over " << num_runs << " runs: " << avg_time_ms << " ms" << std::endl;
    }
    
    RUNTIME_CHECK(cudaMemGetInfo(&free_mem_after, &total_mem_after));
    
    std::cout << std::endl << "Memory Usage:" << std::endl;
    std::cout << "  Free memory before: " << free_mem_before / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  Free memory after: " << free_mem_after / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  Memory leak: " << (free_mem_before - free_mem_after) / 1024.0 / 1024.0 << " MB" << std::endl;
    
    std::cout << std::endl << "Test completed successfully!" << std::endl;
    
    return EXIT_SUCCESS;
}




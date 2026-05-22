#include <iostream>
#include <vector>


//对应flash attention v2原文算法, block采用三维布局
template <typename T>
__global__ void kernel_flashAttention(int batch_size, int target_seq_len, int src_seq_len, int q_heads, int kv_heads, int head_dim, bool is_causal, const T* Q, const T* K, const T* V, T* O){
  int tid_x = threadIdx.x;//横向,blockDim.x列
  int tid_y = threadIdx.y;//纵向,blockDim.y行
  int bid_x = blockIdx.x;//x方向,总数 = #q_heads
  int bid_y = blockIdx.y;//y方向,总数 = Tr
  int bid_z = blockIdx.z;//z方向,总数 = #batch
  const int p = q_heads / kv_heads;//计算比例系数
  const int Br = blockDim.y;//Q纵向每块大小, 默认为32 (RTX 5090)
  const int Bc = blockDim.x;//K/V纵向分块大小, 默认为32
  const int Tc = (src_seq_len + Bc - 1) / Bc;//对应原始论文中K/V纵向分块数Tc,其中Bc = 32

  //预计算常量
  const double scale_factor = 1.0 / sqrt(double(head_dim));//保留精度,采用double

  //定义一系列临时变量
  /*__shared__ double SP[Br][Bc];//复用S和P
  __shared__ double m_prev[Br], m_new[Br];
  __shared__ double l_prev[Br], l_new[Br];

  __shared__ float Q_sm[Br][64];
  __shared__ float K_T_sm[64][Bc];//transpose for K
  __shared__ float V_sm[Bc][64];
  __shared__ float O_sm[Br][64];*/

  extern __shared__ char shared_mem[];
  char* ptr = shared_mem;  
  //计算中间变量,包括S, P(复用为SP), m_prev, m_new, l_prev, l_new; 为保留精度, SP采用double
  double* SP = reinterpret_cast<double*>(ptr);    // double SP[Br][Bc]
  ptr += Br * Bc * sizeof(double);
  float* m_prev = reinterpret_cast<float*>(ptr);  // float m_prev[Br]
  ptr += Br * sizeof(float);
  float* m_new = reinterpret_cast<float*>(ptr);   // float m_new[Br] 
  ptr += Br * sizeof(float);
  float* l_prev = reinterpret_cast<float*>(ptr);  // float l_prev[Br]
  ptr += Br * sizeof(float);
  float* l_new = reinterpret_cast<float*>(ptr);   // float l_new[Br] 
  ptr += Br * sizeof(float);  

  //原始数据QKV和计算结果O; 全采用float
  float* Q_sm = reinterpret_cast<float*>(ptr);    // float Q_sm[Br][head_dim] 
  ptr += Br * head_dim * sizeof(float);  
  float* K_T_sm = reinterpret_cast<float*>(ptr);  // float K_T_sm[head_dim][Bc]
  ptr += head_dim * Bc * sizeof(float);
  float* V_sm = reinterpret_cast<float*>(ptr);    // float V_sm[Br][head_dim] 
  ptr += Bc * head_dim * sizeof(float);  
  float* O_sm = reinterpret_cast<float*>(ptr);    // float O_sm[Br][head_dim]

  //定义访问宏
  /*#define   SP_AT(y, x)       SP[y][x]
  #define   Q_sm_AT(y, x)     Q_sm[y][x]
  #define   K_T_sm_AT(y, x)   K_T_sm[y][x]
  #define   V_sm_AT(y, x)     V_sm[y][x]
  #define   O_sm_AT(y, x)     O_sm[y][x]*/

  #define   SP_AT(y, x)       SP[y * Bc + x]
  #define   Q_sm_AT(y, x)     Q_sm[y * head_dim + x]
  #define   K_T_sm_AT(y, x)   K_T_sm[y * Bc + x]
  #define   V_sm_AT(y, x)     V_sm[y * head_dim + x]
  #define   O_sm_AT(y, x)     O_sm[y * head_dim + x]


  /****************************preparation**************************/
  int bound_tid_y = ::min(Br, target_seq_len - Br * bid_y);

  //preparation-1: load Qi from GM to SM, and reset Oi to 0
  //Q[bid_z][Br * bid_y + tid_y][bid_x][*]
  for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
    O_sm_AT(tid_y, idx) = 0.0;
    Q_sm_AT(tid_y, idx) = 0.0;
    if(tid_y < bound_tid_y){
      Q_sm_AT(tid_y, idx) = float(Q[((((bid_z * target_seq_len) + (Br * bid_y + tid_y)) * q_heads) + bid_x) * head_dim + idx]);
    }
  }
  __syncthreads();

  //preparation-2: reset m_prev to -INFINITY and l_prev to 0
  if(tid_x == 0){
    m_prev[tid_y] = -8192.0;
    l_prev[tid_y] = 0.0;
  }
  __syncthreads();
  /****************************end-of-preparation*************************/


  /****************************main-loop**************************/
  #pragma unroll 4
  for(int j = 0; j < Tc; ++j){//对于每个K/V分块
    bool skip = (is_causal && bid_y < j);
    if(skip){//early exit, 直接跳过
    __syncthreads();
      continue;
    }

    SP_AT(tid_y, tid_x) = -8192.0;
    __syncthreads();
    int bound_tid_x = ::min(Bc, src_seq_len - Bc * j);
    bool is_compute = true;//optimization: 分支处理,加速branch-resolving
    if (is_causal) {
      if (bid_y < j) {
        is_compute = false;  // 早期退出情况
      } else if (bid_y == j) {
        is_compute = (tid_y >= tid_x);  // 对角线以上
      }
    }

    //step-1: load Ki, Vi from GM to SM, reset Oi to 0
    //K[bid_z][Bc * j + tid_y][bid_x / p][*], V[bid_z][Bc * j + tid_y][bid_x / p][*]
    #pragma unroll
    for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
      K_T_sm_AT(idx, tid_y) = 0.0;
      V_sm_AT(tid_y, idx) = 0.0;
      if(tid_y < bound_tid_x){//注意这里是bound_tid_x
        K_T_sm_AT(idx, tid_y) = float(K[((((bid_z * src_seq_len) + (Bc * j + tid_y)) * kv_heads) + (bid_x / p)) * head_dim + idx]);
        V_sm_AT(tid_y, idx) = float(V[((((bid_z * src_seq_len) + (Bc * j + tid_y)) * kv_heads) + (bid_x / p)) * head_dim + idx]);
      }
    }
    __syncthreads();

    //step-2: S = Q @ K.T, point-wise
    if(tid_y < bound_tid_y && tid_x < bound_tid_x){//用于边缘不完整块
      float val0 = 0.0;//临时sum
      if(is_compute){
        #pragma unroll
        for(int k = 0; k < head_dim; ++k){
          val0 += Q_sm_AT(tid_y, k) * K_T_sm_AT(k, tid_x);
        }
        SP_AT(tid_y, tid_x) = double(val0) * scale_factor;//必须用double,对精度影响最大的计算步骤
      }
    }
    __syncthreads();

    //step-3: m_new = max(m_prev, rowMax(S))
    float val1 = float(SP_AT(tid_y, tid_x));
    val1 = warp_reduce_max(val1);
    if(tid_x == 0 && tid_y < bound_tid_y){
      /*double val1 = SP_AT(tid_y, 0);//手动实现非并行求行最大值
      for(int h = 1; h < Bc; ++h){
        val1 = (val1 < SP_AT(tid_y, h)) ? SP_AT(tid_y, h) : val1;
      }*/
      m_new[tid_y] = (val1 > m_prev[tid_y]) ? val1 : m_prev[tid_y];
    }
    __syncthreads();

    //step-4: P = exp(S - m_new), point-wise
    if(tid_y < bound_tid_y && tid_x < bound_tid_x){
      if(is_compute){
        SP_AT(tid_y, tid_x) = myexp<double>(SP_AT(tid_y, tid_x) - double(m_new[tid_y]));
      }
      else{
        SP_AT(tid_y, tid_x) = 0.0;
      }
    }
    else{
      SP_AT(tid_y, tid_x) = 0.0;
    }
    
    __syncthreads();

    //step-5: l_new = exp(m_prev - m_new) * l_prev + rowSum(P)
    float val2 = float(SP_AT(tid_y, tid_x));
    val2 = warp_reduce_sum(val2);
    float exp_result = myexp<float>(m_prev[tid_y] - m_new[tid_y]);
    //float exp_result = expf(m_prev[tid_y] - m_new[tid_y]);
    if(tid_x == 0 && tid_y < bound_tid_y){
      /*double val2 = 0.0;//手动实现非并行求rowSum
      for(int h = 0; h < Bc; ++h){
        val2 += SP_AT(tid_y, h);
      }*/
      l_new[tid_y] = exp_result * l_prev[tid_y] + val2;
    }
    __syncthreads();

    //step-6: O = 1/(exp(m_prev - m_new)) * O + P @ V
    if(tid_x < bound_tid_x && tid_y < bound_tid_y){//32路并行计算Oi的每一行
      for(int u = tid_x; u < head_dim; u += blockDim.x){
        float val3 = 0.0;
        #pragma unroll
        for(int w = 0; w < Bc; ++w){//val3 += P[tid_y][w] * V[bid_z][Bc * j + w][bid_x / p][u];
          val3 += float(SP_AT(tid_y, w)) * V_sm_AT(w, u);
        }
        O_sm_AT(tid_y, u) = O_sm_AT(tid_y, u) * exp_result + val3;
      }
    }
    __syncthreads();
      
    //step-7: m_prev <- m_new; l_prev <- l_new
    if (tid_x == 0 && tid_y < bound_tid_y) {//向量更新只使用第1列线程
      m_prev[tid_y] = m_new[tid_y];
      l_prev[tid_y] = l_new[tid_y];
    }
    __syncthreads();

  }
  /****************************end-of-main-loop**************************/

  /*****************************post-process****************************/
  //O(GM) = O/l_prev, aka O_sm /= l_prev and write Oi from SM to GM
  //O[bid_z][Br * bid_y + tid_y][bid_x][*]
  #pragma unroll
  for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
    if(tid_y < bound_tid_y){
      O[((((bid_z * target_seq_len) + (Br * bid_y + tid_y)) * q_heads) + bid_x) * head_dim + idx] = T(O_sm_AT(tid_y, idx) / float(l_prev[tid_y]));
    }
  }
  __syncthreads();
  /*****************************end-of-post-process****************************/

  //取消访问宏定义
  #undef   SP_AT
  #undef   Q_sm_AT
  #undef   K_T_sm_AT
  #undef   V_sm_AT
  #undef   O_sm_AT

}



template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  //step0: basic check

  //step1: 初始化,预留device端空间
  //cudaStream_t stream2;
  cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, -1);


  const size_t size_bytes_q = h_q.size() * sizeof(T);
  const size_t size_bytes_k = h_k.size() * sizeof(T);
  const size_t size_bytes_v = h_v.size() * sizeof(T);
  const size_t size_bytes_o = h_o.size() * sizeof(T);
  const size_t total_bytes = size_bytes_q + size_bytes_k + size_bytes_v + size_bytes_o;

  //device端只支持裸指针
  T* d_all = nullptr;
  RUNTIME_CHECK(cudaMallocAsync(&d_all, total_bytes, stream2));
  // 切片为Q/K/V/O
  T *d_q = d_all;
  T *d_k = d_q + h_q.size();
  T *d_v = d_k + h_k.size();
  T *d_o = d_v + h_v.size();

  //step2: 拷贝数据from host to device
  RUNTIME_CHECK(cudaMemcpyAsync(d_q, h_q.data(), size_bytes_q, cudaMemcpyHostToDevice, stream2));
  RUNTIME_CHECK(cudaMemcpyAsync(d_k, h_k.data(), size_bytes_k, cudaMemcpyHostToDevice, stream2));
  RUNTIME_CHECK(cudaMemcpyAsync(d_v, h_v.data(), size_bytes_v, cudaMemcpyHostToDevice, stream2));
  RUNTIME_CHECK(cudaMemsetAsync(d_o, 0, size_bytes_o, stream2));//d_o初始化为全0

  //step3: device端计算
  int Br = 32, Bc = 32;
  int grid_dim_y = (target_seq_len + Br - 1) / Br;
  dim3 block_dim(Br, Bc);
  dim3 grid_dim(query_heads, grid_dim_y, batch_size);
  size_t smem_size = (Br * Bc) * sizeof(double) + (Br * 4) * sizeof(float) + (Br * head_dim * 2 + Bc * head_dim * 2) * sizeof(float);

  kernel_flashAttention<T><<<grid_dim, block_dim, smem_size, stream2>>>(batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal, d_q, d_k, d_v, d_o);//注意核函数返回类型只能为void
  RUNTIME_CHECK(cudaStreamSynchronize(stream2));//important

  //step4: 拷贝数据from device to host
  RUNTIME_CHECK(cudaMemcpyAsync(h_o.data(), d_o, size_bytes_o, cudaMemcpyDeviceToHost, stream2));
  RUNTIME_CHECK(cudaStreamSynchronize(stream2));//important

  //step5: free memory
  RUNTIME_CHECK(cudaFreeAsync(d_all, stream2));

  //std::cout << "h_o[0] is: " << float(h_o[0]) << std::endl;
  return;
}

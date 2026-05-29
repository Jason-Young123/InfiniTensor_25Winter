/*### 5. 并行前缀和 (Prefix Sum / Scan)

题目： 对一个一维数组实现并行前缀和（Inclusive or Exclusive Scan）。

面试官附加限制：

- 要求实现具有工作效能（Work-efficient）的算法（如 Blelloch Scan），即总加法次数应当与 CPU 串行版本保持在同一数量级（$O(N)$）。
- 需要清晰说明 Up-sweep（归约阶段）和 Down-sweep（分发阶段）的过程。
*/

#include <iostream>



template <typename T>
__device__ T warp_prefix_sum(T val){
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_up_sync(0xffffffff, val, offset);
    }
    return val;
}







template <typename T>
__global__ void prefix_sum(T* d_in, size_t n){
    int tidx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //归约阶段,每次循环之后, 数据量变为原来的1/32
    for(int i = n; i > 0; i >>= 5){

    }


}




























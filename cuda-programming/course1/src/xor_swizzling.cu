//用最小化测试用例探究xor swizzling, padding和naive的矩阵转置




//一切从简,就假定矩阵尺寸为32x32,数据类型float;就启用一个block
__global__ void transpose_naive(float* d_out, const float* d_in){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    
    __shared__ float smem[32][32];

    smem[tidy][tidx] = d_in[tidy * 32 + tidx];

    __syncthreads();

    d_out[tidy * 32 + tidx] = smem[tidx][tidy];

}


//引入padding
__global__ void transpose_padding(float* d_out, const float* d_in){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    
    __shared__ float smem[32][33];//多一列

    smem[tidy][tidx] = d_in[tidy * 32 + tidx];

    __syncthreads();

    d_out[tidy * 32 + tidx] = smem[tidx][tidy];
}


//引入xor swizzling
__global__ void transpose_swizzling(float* d_out, const float* d_in){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    __shared__ float smem[32][32];//不会多占用存储空间

    int swizzled_col = tidx ^ tidy;
    smem[tidy][swizzled_col] = d_in[tidy * 32 + tidx];

    __syncthreads();

    int read_col = tidy ^ tidx;
    d_out[tidy * 32 + tidx] = smem[tidx][read_col];
}

/*
可以把 smem[row][col] 分成两个坐标：
  逻辑坐标: smem[row][col]
  物理坐标: smem[row][col ^ row]
  也就是说，swizzling 不改变“这个元素逻辑上属于哪一行哪一列”，只改变它在 shared memory 这一行里的实际列位置。

  写入时：
  // 逻辑上要写 smem[tidy][tidx]
  int swizzled_col = tidx ^ tidy;
  smem[tidy][swizzled_col] = d_in[tidy * 32 + tidx];

  所以逻辑元素：
  smem[tidy][tidx]
  实际存到：
  smem[tidy][tidx ^ tidy]

  读取 transpose 时，原来要读的是：
  smem[tidx][tidy]

  这里的逻辑坐标变成：
  row = tidx
  col = tidy
  按照同一个 swizzle 规则，物理列应该是：
  physical_col = col ^ row
               = tidy ^ tidx
  所以读取时：
  int read_col = tidy ^ tidx;
  d_out[tidy * 32 + tidx] = smem[tidx][read_col];
*/




/*
异或能消除 conflict 的核心原因是：
  shared memory bank 由地址的低 5 bit 决定
  对 float 来说，一个 bank 正好 4B，所以：
  bank = word_offset % 32
  如果是二维 row-major shared memory：
  float smem[ROWS][COLS];
  那么：
  word_offset = row * COLS + col
  bank = (row * COLS + col) % 32

  以 32x32 transpose 为例
  原始读转置时：
  smem[tidx][tidy]
  一个 warp 内通常 tidy 固定，tidx = lane，所以：
  row = lane
  col = fixed
  bank = (lane * 32 + fixed) % 32
       = fixed
  32 个线程访问 32 个不同地址，但全部落到同一个 bank，典型 32-way conflict。

  引入 swizzling：
  physical_col = logical_col ^ row
  读取逻辑上的：
  smem[tidx][tidy]
  实际读：
  smem[tidx][tidy ^ tidx]
  此时：
  row = lane
  physical_col = fixed ^ lane
  bank = (lane * 32 + (fixed ^ lane)) % 32
       = fixed ^ lane
  lane = 0..31 时，fixed ^ lane 也是 0..31 的一个排列，所以 32 个线程被打散到 32 个 bank。
  这就是 XOR 的作用：把 row/lane 里的变化信息混到低 5 bit 里。因为 bank 正是看低 5 bit，所以 bank 被打散了。

  XOR 还有两个好处：
  1. x ^ mask 是双射，不会两个逻辑列映射到同一个物理列。
  2. XOR 自反，(x ^ mask) ^ mask = x，读写用同一套规则很好还原。

  对应到 matmul5 的 A_tile
  你的 A_tile[64][8]：
  word_offset = row * 8 + k
  bank = (row * 8 + k) % 32
  只看 k=0, m=0：
  lane 0-3   -> row = 0
  lane 4-7   -> row = 4
  lane 8-11  -> row = 8
  ...
  lane 28-31 -> row = 28

  原始 bank：
  row = 4g
  bank = (4g * 8 + 0) % 32
       = 32g % 32
       = 0
  所以 8 个不同地址全部打到 bank 0。每 4 个 lane 内部是 broadcast，不是问题；问题是 8 组不同地址同 bank。

  这时可以用：
  physical_k = logical_k ^ ((row >> 2) & 7);
  因为 row >> 2 正好把：
  row 0,4,8,...,28
  变成：
  0,1,2,...,7
  于是：
  bank = (row * 8 + physical_k) % 32
       = (4g * 8 + (k ^ g)) % 32
       = k ^ g

  当 k=0 时：
  group 0 -> bank 0
  group 1 -> bank 1
  group 2 -> bank 2
  ...
  group 7 -> bank 7
  这就把 8 个不同地址打散到了 8 个 bank。注意，这里不需要 32 个 lane 全部分到不同 bank，因为实际上只有 8 个不同 word；每组 4 个 lane 读同一个 word 是 broadcast，
  保留即可。

*/



/*
以后怎么找映射公式
  按这个流程来：
  1. 写出 bank 公式：
  bank = word_offset % 32
  比如：
  bank = (row * stride + col) % 32

  2. 写出一个 warp 内每个 lane 的访问坐标：
  row = f(lane)
  col = g(lane)

  3. 代入 bank 公式，看哪些 lane 是：
  不同地址，同一个 bank
  相同地址是 broadcast，不算 conflict。

  4. 找出“区分这些冲突地址”的变量。
  比如 matmul5 A_tile 里，冲突地址由：
  row = 0,4,8,...,28
  区分，所以冲突组编号是：
  group = row >> 2;

  5. 把这个 group 编号 XOR 到会影响 bank 低位的维度里。
  对于 row-major 数组，通常是列维度：
  physical_col = logical_col ^ group;
  或者你的 A_tile 是 K 维：
  physical_k = logical_k ^ group;

  6. 重新计算 bank，确认是否变成排列或至少冲突减少。
  判断标准不是“公式看起来高级”，而是重新算：
  new_bank = (row * stride + swizzled_col) % 32
  看同一条 warp 指令里的不同地址是否分散。
  如果 swizzle 的维度大小是 8，那最多只能把冲突打散到 8 个位置；如果有 16 或 32 个不同地址都冲突，而你只 swizzle 3 bit，无法彻底消除。那就要改更大的维度、
  padding、tile layout，或者换访问映射。
*/




/*
关键不是“异或有魔法”，而是 swizzling 需要满足两个条件：
  1. 对固定的 row/key，logical_col -> physical_col 必须是一一映射，否则 shared memory 里会覆盖数据。
  2. 对同一个 warp 的冲突访问，physical_col 要把 bank 低位打散。
  对 float smem[32][32]：
  bank = (row * 32 + physical_col) % 32
       = physical_col
  所以只要让 physical_col 随着 row/lane 变化，就能打散 bank。

  xor 做的是：
  physical_col = logical_col ^ row
  读转置时，logical_col 固定，row = lane：
  bank = fixed_col ^ lane
  lane = 0..31 时，fixed_col ^ lane 仍然是 0..31 的一个排列，所以 32 个线程落到 32 个 bank。

  为什么 AND/OR 不行
  比如：
  physical_col = logical_col | row
  对某一 bit 来说，如果 row 这一 bit 是 1：
  0 | 1 = 1
  1 | 1 = 1
  两个不同 logical col 被映射到同一个 physical col，数据会重叠。
  再比如：
  physical_col = logical_col & row
  如果 row 这一 bit 是 0：
  0 & 0 = 0
  1 & 0 = 0
  同样不是一一映射。
  NAND/NOR 只是 AND/OR 取反，本质上仍然会把两个输入压成一个输出，也不行。

  以后找公式时，就按这个流程：
  1. 写出 bank = (row * stride + col) % 32
  2. 写出 warp 内 row/col 如何随 lane 变化
  3. 找到造成 conflict 的 key，例如 row、row>>2、lane_m
  4. 把 key XOR 到影响 bank 低位的那个坐标上
  5. 重新代入 bank 公式，确认 bank 变成排列或至少更分散
  如果冲突模式不是 32-way，也一样：先找“哪些不同地址落到同一 bank”，再把能区分这些地址的 key 混到低 5 bit 里。
*/




int main(){
    const int N = 32;
    float* h_in = new float[N * N];
    for(int i = 0; i < 32 * 32; ++i){
        h_in[i] = i;
    }
    float* h_out = new float[N * N];

    //step1: device malloc
    float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in, sizeof(float) * N * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N * N);

    //step2: copy H2D
    cudaMemcpy(d_in, h_in, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    //step3: launch kernel
    dim3 grid_dim(1, 1);
    dim3 block_dim(32, 32);
    transpose_naive<<<grid_dim, block_dim>>>(d_out, d_in);

    transpose_padding<<<grid_dim, block_dim>>>(d_out, d_in);

    transpose_swizzling<<<grid_dim, block_dim>>>(d_out, d_in);

    //step4: copy D2H
    cudaMemcpy(h_out, d_out, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

    //step5: check & cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
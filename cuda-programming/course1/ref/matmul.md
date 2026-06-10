# CUDA Matmul 优化复盘

本文档整理截至目前在 `src/matmul.cu` 中尝试过的矩阵乘法优化方法、当前性能结果、后续优化方向，以及面试中最可能被追问的问题和回答要点。

## 1. 当前任务背景

目标计算：

```text
C = A @ B
A: M x K
B: K x N
C: M x N
```

当前 benchmark 规模：

```text
M = 1024
K = 1024
N = 1024
数据类型: float
存储布局: C/C++ row-major
```

当前主要对照：

- 自写 CUDA kernel: `matmul3_db`
- 库函数基线: `cublasSgemm`
- Profiling 工具: Nsight Compute, `analysis/matmul.ncu-rep`
- 编译目标: `sm_120`

## 2. 当前最佳结果

基于当前 NCU report：

| Kernel | Duration | 估算吞吐 | 相对 cuBLAS |
| --- | ---: | ---: | ---: |
| cuBLAS / CUTLASS SGEMM | `238.91 us` | `8.99 TFLOPS` | `100%` |
| `matmul3` | `375.68 us` | `5.72 TFLOPS` | `63.6%` |
| `matmul3_db` | `335.78 us` | `6.40 TFLOPS` | `71.1%` |

计算方式：

```text
FLOPs = 2 * M * N * K
      = 2 * 1024^3
      = 2.147 GFLOPs
```

因此当前最佳自写版本 `matmul3_db` 相比 `matmul3`：

```text
时间从 375.68 us 降到 335.78 us
耗时降低约 10.6%
吞吐提升约 11.9%
```

注意：这里的比例来自当前 NCU report。普通运行计时和 NCU replay 模式下的数值可能不同，面试时建议表述为“在当前 NCU profiling 条件下达到 cuBLAS 的约 71%”。

## 3. 已尝试的优化路径

### 3.1 `matmul1`: naive 版本

核心做法：

- 一个 thread 计算一个 `C[row][col]`。
- 每个 thread 直接从 global memory 读取一整行 A 和一整列 B。
- 不使用 shared memory。

主要问题：

- 同一个 A/B 元素会被多个 thread 重复从 global memory 加载。
- global memory 访问复用差，算术强度低。
- 对大矩阵而言，很容易被 memory latency 和 bandwidth 限制。

面试讲法：

> naive kernel 的瓶颈不是计算能力，而是 global memory 访问。每个输出元素都重新读取 K 个 A 和 K 个 B，线程之间没有显式复用，所以大量带宽被浪费。

### 3.2 `matmul2`: shared memory tiling

核心做法：

- 将 C 按 block tile 切分。
- 每个 block 协同加载一块 `A_tile` 和 `B_tile` 到 shared memory。
- block 内 thread 复用 shared memory 中的 tile 数据。
- 使用 `__syncthreads()` 保证 tile 加载完成后再计算。

当前形态：

```text
TILE_M = 32
TILE_N = 32
TILE_K = 32
每个 thread 仍然只计算一个 C 元素
```

优化收益：

- 大幅减少 global memory 重复读取。
- 将部分 global memory 访问转化为 shared memory 访问。
- 提升数据复用。

新的瓶颈：

- 每个 thread 只算一个输出，寄存器复用少。
- shared memory 到 register 的读取压力上升。
- arithmetic intensity 仍然有限。

面试讲法：

> shared tiling 的核心是把 global memory 的跨线程复用显式搬到 shared memory。这样一个 A/B tile 被整个 block 多个 thread 重复使用，而不是每个 thread 都从 global memory 重新读。

### 3.3 `matmul3`: thread tiling / register blocking

核心做法：

- 一个 thread 不再只计算一个 C 元素，而是计算一个 `4 x 4` 的小块。
- 每个 thread 使用多个 accumulator 保存在 register 中。
- 每次从 shared memory 读入 `A_reg[4]` 和 `B_reg[4]`。
- 使用寄存器内的 outer-product 计算 16 个输出。

当前形态：

```text
blockDim = (32, 8)
每个 block 有 256 threads

TILE_M = 8
TILE_N = 32
TILE_THREAD_M = 4
TILE_THREAD_N = 4
TILE_K = 16

实际 C block tile:
M 方向 = 8 * 4 = 32
N 方向 = 32 * 4 = 128
即每个 block 计算 32 x 128 的 C 子块
```

shared memory：

```text
A_tile: 32 x 16
B_tile: 16 x 128
总 shared memory = (32*16 + 16*128) * 4B = 10 KB
```

优化收益：

- 提高每个 thread 的计算量。
- 提高 A/B 从 shared memory 读入 register 后的复用。
- 每个 thread 的 16 个 accumulator 保存在 register 中，减少中间结果写回。

代价：

- register 使用上升。
- 写回 C 时，一个 thread 要写 4 行，每行 4 个元素。
- shared memory 读访问模式更复杂，可能产生 bank conflict 或 L1TEX/shared pipeline 压力。

面试讲法：

> `matmul3` 的关键是 register blocking。一个 thread 计算 `4x4` 输出块后，每次加载 4 个 A 和 4 个 B，可以做 16 次 FMA。这样提升了 shared memory 到 register 后的数据复用。

### 3.4 cooperative loading

当前实现了两个协同加载版本：

```text
cooperative_ldst
cooperative_ldst_vector
```

`cooperative_ldst`：

- block 内所有 thread 共同搬运一个二维 tile。
- 每个 thread 根据 `flat_tid` 负责若干元素。
- 支持边界判断和 zero padding。

`cooperative_ldst_vector`：

- 默认 float 类型。
- 每个 thread 搬运 `float4`。
- 要求整体 tile 的 `COL` 是 4 的倍数。
- 边界完整时使用 16B 向量化 load/store。
- 边界不完整时退化为 scalar path。

对应的 SASS 现象：

```text
global load: LDG.E.128
shared store: STS.128
shared load: LDS.128
global store: STG.E.128
```

优化收益：

- 减少 load/store 指令数。
- 提升 global memory coalescing。
- 写回 C 时用 `float4` 合并写，减少 store 指令数量。

面试讲法：

> 向量化搬运不是改变计算逻辑，而是让每条 memory instruction 搬更多连续数据。对 float 来说 `float4` 是 16B，通常更容易生成 128-bit load/store。

### 3.5 NCU 驱动的问题定位

在 `matmul3` 中曾观察到：

```text
Long Scoreboard / L1TEX dependency stall
```

本质原因：

- global memory load 之后，数据要先进入 register。
- 随后再由 `STS` 写入 shared memory。
- `STS` 依赖前面的 `LDG` 结果，因此 SASS 中高 stall sampling 可能显示在 `STS` 行。

之前汇编中典型模式：

```text
LDG.E    Rxx, [global]
STS      [shared], Rxx
```

含义：

- stall 显示在 `STS` 不代表 shared store 本身一定慢。
- 它可能是在等待前面的 global load 数据返回。

面试讲法：

> NCU 报 L1TEX scoreboard 时，我不会只看被标红的那一行，而会追溯它依赖的 producer instruction。这里 `STS` 被标红，是因为它要等待前面的 `LDG` 返回数据。

### 3.6 `matmul3_db`: cp.async + double buffering

这是当前最佳版本。

核心做法：

- 在 `matmul3` 基础上保留原本的 `4x4` thread tile。
- shared memory 从单 buffer 改成双 buffer：

```cpp
__shared__ T A_tile[2][32][16];
__shared__ T B_tile[2][16][128];
```

- 使用 `cp.async` 将 global memory 直接异步搬到 shared memory。
- 计算当前 stage 时，预取下一 stage。
- 通过 `cp.async.commit_group` 和 `cp.async.wait_group 0` 管理异步 copy 的完成。

关键收益：

- 避免 global memory 到 shared memory 搬运必须经过显式 register 中转。
- 将 global-to-shared copy 与当前 tile 的计算重叠。
- 降低 load/use dependency 对 scheduler 的影响。

当前 NCU 对比：

| 指标 | `matmul3` | `matmul3_db` |
| --- | ---: | ---: |
| Duration | `375.68 us` | `335.78 us` |
| Compute Throughput | `48.92%` | `52.72%` |
| SM Busy | `61.66%` | `68.38%` |
| Executed IPC Active | `2.47` | `2.74` |
| No Eligible | `39.24%` | `31.58%` |
| Eligible Warps / Scheduler | `1.54` | `2.28` |
| Registers / Thread | `48` | `58` |
| Static Shared / Block | `10.24 KB` | `20.48 KB` |
| Theoretical Occupancy | `83.33%` | `66.67%` |
| Achieved Occupancy | `50.36%` | `45.75%` |

结论：

- `matmul3_db` 虽然 occupancy 更低，但性能更好。
- 说明当前瓶颈不是单纯 occupancy，而是 memory latency hiding 和 instruction scheduling。
- double buffering 增加了 register/shared memory 使用，但换来了更好的 pipeline overlap。

面试讲法：

> `matmul3_db` 证明了 occupancy 不是唯一目标。虽然 double buffering 让 shared memory 和 register 都增加，理论 occupancy 下降，但它减少了等待 global memory 的时间，scheduler 中 eligible warp 反而更多，所以整体更快。

### 3.7 `cp.async.commit_group` 和 `cp.async.wait_group`

`cp.async.commit_group`：

- 将此前发出的若干 `cp.async` 指令提交为一个 group。
- 表示这一批异步 copy 已经发完，可以由硬件异步推进。

`cp.async.wait_group 0`：

- 等待所有已提交的 cp.async group 完成。
- `wait_group 0` 是最保守的等待方式。
- 等待后还需要 `__syncthreads()`，保证 block 内所有 thread 都看到 shared memory 中的数据。

当前使用方式：

```text
预取第一个 tile
commit_group
wait_group 0
__syncthreads()

for each K tile:
    预取 next tile
    commit_group
    计算 current tile
    wait_group 0
    __syncthreads()
```

面试讲法：

> `cp.async` 只保证当前 thread 发起的异步 copy 最终完成，`wait_group` 负责等待 copy 完成，而 `__syncthreads()` 负责 block 内线程同步。二者不能互相替代。

### 3.8 `cp.async.ca` vs `cp.async.cg`

当前实现使用：

```ptx
cp.async.ca.shared.global
```

含义：

- `.ca` 表示 cache at all levels，数据可能进入 L1 和 L2。

曾尝试直接替换为：

```ptx
cp.async.cg.shared.global
```

遇到问题：

```text
Argument 2 of instruction 'cp.async': unexpected value '4', expected to be 16
```

原因：

- `cp.async.cg` 只支持 16B copy 粒度。
- 当前边界 fallback 里存在 4B copy。
- 因此不能机械地把 `.ca` 全部替换成 `.cg`。

正确思路：

- 完整 `float4` 路径可以测试 `.cg`。
- 4B 边界 fallback 继续保留 `.ca`。
- 或者重写边界路径，用 16B copy + `src-size` zero-fill。

面试讲法：

> `.cg` 不是 `.ca` 的等价替换。`cp.async.cg` 要求 copy size 是 16B，而我的通用边界路径中有 4B fallback，所以直接替换会被 ptxas 拒绝。

### 3.9 warp tiling: `matmul4` / `matmul5` / `matmul7`

后续还尝试过更明确的层级划分：

```text
Block Tile -> Warp Tile -> Thread Tile
```

`matmul4`：

- 引入 warp tile 概念。
- 每个 warp 负责一个更规整的 C 子块。
- 尝试让 warp 内 thread 的计算和写回更规则。

`matmul5`：

- 使用 `64 x 64` block tile。
- 每个 warp 负责 `32 x 16` 子块。
- 每个 thread 负责 `4 x 4` 输出。
- 写回阶段直接使用 4 次 `float4`。

`matmul7`：

- 在 `matmul5` 基础上给 `A_tile` 的 K 维增加 padding：

```text
A_tile: TILE_M x (TILE_K + 1)
```

- 目的是打散 shared memory bank 映射，减少 A_tile 读阶段的 bank conflict。

重要认识：

- padding 只对特定 shared memory 访问模式有效。
- 对 WMMA 来说，随意 padding 成 9 列通常不可行，因为 `wmma::load_matrix_sync` 对 leading dimension 和对齐有要求。
- 如果要 padding WMMA tile，通常应 pad 到满足 alignment/ldm 约束的值，例如 16，而不是 9。

面试讲法：

> warp tiling 的目的，是让一个 warp 的线程协作计算一个规则输出子块，从而改善 shared memory 访问和 global writeback。padding 是处理 shared bank conflict 的常见技巧，但必须结合实际访问模式分析。

### 3.10 Tensor Core / WMMA / MMA: `matmul8`

还探索了 TF32 Tensor Core 方向。

核心概念：

- WMMA 是 CUDA C++ 提供的 warp-level matrix API。
- MMA 是更底层的 PTX/SASS matrix multiply-accumulate 指令。
- Tensor Core 负责执行这些矩阵乘加指令。

`matmul8` 的设计：

- 基于 `matmul7` 的 block/warp tile 思路。
- 每个 warp 仍然对应一个 `32 x 16` 的输出子块。
- 由于 WMMA TF32 常用形状是 `m16n16k8`，所以一个 warp 的 `32 x 16` 需要两个 accumulator：

```text
C_frag0: 16 x 16
C_frag1: 16 x 16
```

- FP32 输入先用 `nvcuda::wmma::__float_to_tf32` 转为 TF32 精度。
- MMA 阶段使用 TF32 multiply + FP32 accumulate。

TF32 认知：

- TF32 保留 FP32 的 8-bit exponent。
- mantissa 精度低于 FP32，大约 10-bit 显式精度量级。
- Tensor Core 上 TF32 比严格 FP32 FFMA 快很多。
- 代价是乘法输入精度低于 FP32。

面试讲法：

> Tensor Core 不通常拿来做严格 FP32 multiply，因为硬件面积、功耗和吞吐之间有取舍。NVIDIA 给 FP32 输入提供 TF32 Tensor Core 路线，本质上是保留 FP32 动态范围，但降低乘法 mantissa 精度，用 FP32 accumulate 换吞吐。

## 4. 当前 NCU 关键结论

### 4.1 当前不是 DRAM 带宽瓶颈

`matmul3_db` 当前 NCU：

```text
DRAM Throughput: 5.52%
L2 Hit Rate: 92.57%
L1/TEX Throughput: 77.01%
Memory Throughput: 59.37%
Compute Throughput: 52.72%
```

解读：

- DRAM 使用率很低，说明不是显存带宽打满。
- L2 hit rate 很高，说明很多 global load 能从 L2 命中。
- L1/TEX throughput 很高，说明压力集中在 L1TEX/shared memory 相关路径。
- 后续优化应更多关注 shared memory layout、cp.async 搬运效率、barrier 和计算 pipeline，而不是单纯优化 DRAM bandwidth。

### 4.2 active cycles 分布不均

NCU warning：

```text
One or more SMs / L1 slices / SMSPs have a much higher number of active cycles
Maximum instance value is about 21.5% above average
Minimum instance value is about 4.5% below average
```

当前 launch：

```text
grid_dim3 = (8, 32)
总 block 数 = 256
SM 数 = 82
matmul3_db 最多约 4 blocks / SM
```

256 个 block 分给 82 个 SM，大致是：

```text
部分 SM 拿到 4 个 block
大部分 SM 拿到 3 个 block
```

因此 active cycles 最大值高于平均值是可以解释的。它不一定表示某个 SM 异常慢，而可能是小矩阵下 block 数不够多，尾波分布不均。

后续方向：

- 对更大矩阵测试，block 数自然增加。
- 对 1024 这种规模，可以考虑更小 block tile 增加 grid block 数。
- 或尝试 split-K 增加并行块数，但需要额外 reduction。

面试讲法：

> 这个 warning 我会先看 grid size 和 waves per SM。当前只有 256 个 block，而 GPU 有 82 个 SM，所以 block 分布天然不够均匀，有些 SM 做 4 个 block，有些做 3 个 block，active cycles 差异是合理的。

### 4.3 `LDGSTS.E.128` 的 shared wavefront excessive

NCU 指出 SASS L81 和 L227：

```text
LDGSTS.E.128
50% shared wavefronts are excessive
```

含义：

- `LDGSTS.E.128` 是 `cp.async` 编译后的 global-to-shared 搬运指令。
- warning 不是发生在 FFMA 计算本身，而是在异步搬运写 shared 的阶段。
- 说明当前 cp.async 的 shared 写入 wavefront 还不是理想状态。

可能原因：

- 通用边界逻辑导致 predication 和路径复杂。
- shared memory 目标地址模式不完全理想。
- 每个 warp 发起的 16B shared 写入在 bank / wavefront 层面仍存在额外拆分。

后续方向：

- 为整除尺寸写 no-boundary fast path，只保留 16B `cp.async`。
- 分别 profile A tile 和 B tile 的 cp.async 路径。
- 尝试调整 shared layout 或 swizzle。
- 只在完整 16B 路径测试 `cp.async.cg`。

面试讲法：

> 这说明我虽然用了 `cp.async` 消除了 LDG 到 STS 的 register 中转，但 global-to-shared 搬运本身的 shared 写入 wavefront 还没有完全理想。下一步应该针对 `LDGSTS.E.128` 的地址模式和边界分支做专门优化。

### 4.4 barrier stall

NCU 指出 K-loop 末尾附近存在 barrier stall。

对应代码语义：

```text
cp.async.wait_group 0
__syncthreads()
切换 double buffer stage
```

原因：

- 每轮 K tile 计算完后，必须保证下一块 tile 已经搬到 shared。
- `wait_group 0` 是保守等待，会等待所有已提交 cp.async group 完成。
- `__syncthreads()` 会等待 block 内所有线程到达。

后续方向：

- 增大 `TILE_K`，减少 K-loop 轮数和 barrier 次数，但会增加 shared memory。
- 使用 triple buffering 和 `wait_group 1` 做更深 pipeline，但会进一步增加 shared memory 和复杂度。
- 消除边界 fast path 后，减少每轮进入 barrier 前的非计算指令。

面试讲法：

> double buffering 已经把加载和计算重叠了一部分，但每个 K tile 末尾仍然需要等待下一 stage 完成并同步整个 block。barrier stall 说明 pipeline 还有空间，但继续优化要在 shared memory、寄存器和代码复杂度之间权衡。

## 5. 后续还能怎么优化

### 5.1 给 `matmul3_db` 做 no-boundary fast path

当前 kernel 保留了通用边界处理：

- load A/B 有边界判断。
- writeback C 有边界判断。
- cp.async 有 16B path 和 4B fallback。

对于当前 `1024 x 1024`：

```text
M, N, K 都能被当前 tile 整除
大部分边界判断都是无效开销
```

可优化：

- 专门写整除尺寸版本。
- A/B load 只发 16B `cp.async`。
- C writeback 直接 `float4`。
- 移除 `gm_row < M`、`gm_col < N` 等判断。

预期收益：

- 减少 branch instruction。
- 减少 predicate。
- 降低寄存器和地址计算压力。
- 也可能改善 `LDGSTS.E.128` 的 shared wavefront warning。

### 5.2 调整 `cp.async.ca/cg`

由于当前 L1 hit rate 很低：

```text
L1/TEX Hit Rate: 0.03%
```

可以测试：

- 完整 16B copy 使用 `cp.async.cg`
- 4B fallback 继续使用 `cp.async.ca`

预期：

- `.cg` 可能减少 L1 cache 污染。
- 但不保证一定更快，需要实测。

注意：

- `cp.async.cg` 不支持 4B copy size。
- 不能直接全局替换 `.ca`。

### 5.3 优化 shared memory layout

当前 `matmul3_db` 计算阶段：

```cpp
A_reg[thread] = A_tile[curr_stage][tidy * 4 + thread][k];
B_reg[thread] = B_tile[curr_stage][k][tidx * 4 + thread];
```

潜在问题：

- B 的访问在 warp 内呈现 `4 * tidx + thread` 的 stride 模式。
- 这种模式容易给 shared memory bank 带来压力。

后续方案：

- 对 `B_tile` 做 swizzle。
- 改变 B 在 shared memory 中的存储布局。
- 改变 warp/thread tile 映射。
- 将 `matmul5/matmul7` 中更规整的 warp tiling 思路和 `matmul3_db` 的 cp.async 结合。

### 5.4 扫 `TILE_K`

当前：

```text
TILE_K = 16
```

可测试：

```text
TILE_K = 8
TILE_K = 16
TILE_K = 32
```

权衡：

- `TILE_K` 增大：K-loop 轮数减少，barrier 次数减少，单次 tile 计算更多。
- `TILE_K` 增大：shared memory 增加，可能降低 occupancy。
- 对 double buffering，`TILE_K=32` 会让 shared memory 大约变成 40 KB/block，可能只能放更少 block。

### 5.5 增加 grid block 数或 split-K

当前 1024 规模下只有 256 个 block，SM 分布不够均匀。

可选方向：

- 减小 block tile，增加 block 数。
- split-K，把 K 维切成多个 block 并行计算，再做 reduction。

代价：

- block tile 变小可能降低数据复用。
- split-K 需要额外 reduction 和更多 global memory 写读。

### 5.6 Tensor Core 路线

如果允许 TF32 精度：

- 使用 WMMA/MMA/Tensor Core 是更有上限的方向。
- 当前 `matmul8` 已经实现了基本思路。
- 后续可以改用更底层 MMA 指令，做更精细的 fragment layout、shared swizzle 和 pipeline。

如果必须严格 FP32：

- 继续走 CUDA core FFMA 路线。
- 上限会低于 TF32 Tensor Core。

## 6. 面试高频问题与回答

### Q1: naive matmul 的主要瓶颈是什么？

回答要点：

> 主要瓶颈是 global memory 访问。每个 thread 计算一个 C 元素，需要读 K 个 A 和 K 个 B。不同 thread 之间会重复读取大量相同数据，但 naive 版本没有 shared memory 或 cache blocking 来显式复用这些数据，所以算术强度低，容易 memory-bound。

### Q2: shared memory tiling 为什么能加速？

回答要点：

> 因为矩阵乘法有天然数据复用。一个 A tile 会被多个输出列使用，一个 B tile 会被多个输出行使用。把 A/B tile 先协同搬到 shared memory 后，block 内线程可以复用这些数据，减少 global memory load 次数。

### Q3: 为什么 `matmul3` 要让一个 thread 计算 `4x4`？

回答要点：

> 这是 register blocking。每个 thread 一次读入 4 个 A 和 4 个 B，可以在寄存器里做 16 次 FMA。这样提高了从 shared memory 读到 register 后的数据复用，也提高了每个 thread 的计算密度。

### Q4: `matmul3` 的 block tile 是多少？

回答要点：

> blockDim 是 `(32, 8)`，每个 thread 计算 `4x4`，所以一个 block 计算 `32 x 128` 的 C 子块。K 方向每次处理 `TILE_K=16`。

### Q5: 为什么要用 `float4` 向量化加载和写回？

回答要点：

> `float4` 是 16B，可以让编译器生成 128-bit load/store，比如 `LDG.E.128`、`STS.128`、`STG.E.128`。这样能减少 memory instruction 数量，并且更容易实现 coalesced memory access。

### Q6: shared memory tiling 里为什么需要 `__syncthreads()`？

回答要点：

> 因为 A/B tile 是整个 block 协同加载的。计算前必须保证所有 thread 都完成对应 shared memory 写入，否则有些 thread 可能读到未完成的数据。计算结束后也要同步，避免下一轮 K tile 覆盖 shared memory 时，其他 thread 还在读旧 tile。

### Q7: NCU 中 L1TEX 是什么？

回答要点：

> L1TEX 是 NVIDIA GPU 上负责 L1 cache、texture、surface 和 shared memory 相关访问的片上子系统。很多 global load、shared load/store、texture load 的压力都会体现在 L1TEX 相关指标上。

### Q8: Long Scoreboard stall 是什么意思？

回答要点：

> Scoreboard 用来跟踪指令依赖。Long Scoreboard 通常表示 warp 等待一个较长延迟的 memory operation 返回。比如 `STS` 要写 shared，但它的数据来自前面的 `LDG`，如果 `LDG` 还没返回，`STS` 就会被 stall。

### Q9: 为什么 `matmul3_db` 要用 `cp.async`？

回答要点：

> 普通 global-to-shared 搬运通常是 `LDG` 到 register，再 `STS` 到 shared。`cp.async` 可以让 global memory 直接异步拷贝到 shared memory，减少显式 register 中转，并且可以和当前 tile 的计算重叠，从而隐藏部分 memory latency。

### Q10: `commit_group` 和 `wait_group` 分别做什么？

回答要点：

> `commit_group` 表示把之前发出的 cp.async 指令提交为一个异步 copy group。`wait_group 0` 表示等待所有已提交 group 完成。等待完成后还需要 `__syncthreads()`，保证 block 内所有线程都同步到同一阶段。

### Q11: 为什么用了 double buffering 后 occupancy 下降但性能上升？

回答要点：

> double buffering 需要两份 shared memory，也增加一些寄存器，所以 occupancy 下降。但它把下一块 tile 的加载和当前 tile 的计算重叠起来，减少 scheduler 没有 eligible warp 的时间。当前 NCU 中 `No Eligible` 从 `39.24%` 降到 `31.58%`，所以整体性能更好。

### Q12: occupancy 是不是越高越好？

回答要点：

> 不是。occupancy 只是可驻留 warp 数量，真正影响性能的是是否有足够 eligible warp 发射指令。如果 kernel 受 latency 影响，适当提高 occupancy 有帮助；但如果寄存器增加换来更高 ILP 或更好数据复用，occupancy 降低也可能更快。

### Q13: ptxas 里的 registers 和 smem 分别是什么粒度？

回答要点：

> `registers per thread` 是每个 thread 使用的寄存器数量。`smem` 通常是每个 block/CTA 的 shared memory 使用量。occupancy 分析时会用每个 thread 的 register、每个 block 的 shared memory、block size 等共同决定一个 SM 上最多能驻留多少 block。

### Q14: CTA 是什么？

回答要点：

> CTA 是 Cooperative Thread Array，在 CUDA 编程模型里基本对应一个 thread block。一个 CTA 内的 thread 可以通过 shared memory 协作，并用 `__syncthreads()` 同步。

### Q15: shared memory bank conflict 是什么？

回答要点：

> shared memory 被分成多个 bank。一个 warp 同时访问 shared memory 时，如果多个 lane 访问落到同一个 bank 的不同地址，就会产生 bank conflict，访问会被拆成多个 wavefront 或 transaction。padding 或 swizzle 可以改变地址到 bank 的映射，减少冲突。

### Q16: 为什么 `A_tile` padding 有时有效？

回答要点：

> 如果 shared memory 的二维数组行宽刚好导致不同 lane 访问映射到相同 bank，给行尾加 padding 可以改变下一行的起始 bank，从而打散访问模式。但 padding 是否有效取决于具体的 warp 访问模式。

### Q17: 为什么 WMMA 不能随便把 `A_tile` 从 8 列 pad 到 9 列？

回答要点：

> WMMA 的 `load_matrix_sync` 对 leading dimension 和内存对齐有要求。TF32 的 `m16n16k8` 不是普通 scalar load，硬件按矩阵 fragment 的格式加载。把物理 stride 改成 9 可能不满足对齐或 ldm 约束。要 padding 通常要 pad 到满足约束的值，比如 16。

### Q18: TF32 是什么？

回答要点：

> TF32 是 NVIDIA Tensor Core 支持的一种计算格式。它保留 FP32 的 8-bit exponent，所以动态范围接近 FP32，但 mantissa 精度比 FP32 低。Tensor Core 用 TF32 做 multiply，用 FP32 accumulate，因此速度比严格 FP32 FFMA 高，但结果精度会略低。

### Q19: 为什么 Tensor Core 不直接做原生 FP32 矩阵乘？

回答要点：

> 严格 FP32 multiply 的硬件成本、功耗和面积都更高。Tensor Core 的设计目标是高吞吐矩阵乘，因此更常支持 FP16/BF16/TF32 等格式。TF32 是 FP32 输入场景下的折中：保留 FP32 动态范围，降低乘法精度来换吞吐。

### Q20: 为什么一个 warp 做 `32 x K @ K x 16` 需要两个 accumulator？

回答要点：

> 因为常用 TF32 WMMA tile 是 `16 x 16 x 8`。一个 warp 的输出是 `32 x 16`，M 方向是 32，需要拆成两个 `16 x 16` fragment，所以要两个 accumulator：一个负责上半 `16 x 16`，一个负责下半 `16 x 16`。

### Q21: 为什么 cuBLAS 里要把 A/B 顺序换一下？

回答要点：

> cuBLAS 默认按 column-major 理解矩阵，而 C++ 代码里数据是 row-major。row-major 的 `C = A @ B` 可以等价看成 column-major 视角下的 `C^T = B^T @ A^T`，所以调用 cuBLAS 时传参顺序需要调整。

### Q22: 当前自写 kernel 和 cuBLAS 差距主要在哪里？

回答要点：

> 当前最佳 `matmul3_db` 在 NCU 下约为 cuBLAS 的 71%。差距主要来自几个方面：shared memory layout 还不够理想，`LDGSTS.E.128` 有 excessive shared wavefront，K-loop 还有 barrier stall，边界判断和通用路径带来额外指令，另外 cuBLAS/CUTLASS 在 tile shape、pipeline、instruction scheduling 上更成熟。

### Q23: 看到 active cycles 在 SM 之间不均，怎么判断是不是问题？

回答要点：

> 我会先看 grid block 数和 SM 数。当前只有 256 个 block，GPU 有 82 个 SM，分配下来有些 SM 做 4 个 block，有些做 3 个 block，所以 active cycles 有 20% 左右差异是合理的。这更像小矩阵下并行粒度不足，而不是某条指令在某些 SM 上异常。

### Q24: 如果继续优化，你优先做什么？

回答要点：

> 我会先做 `matmul3_db` 的 no-boundary fast path，把当前 1024 整除场景下无用的边界判断去掉，只保留 16B `cp.async` 和 `float4` writeback。然后测试 16B 路径的 `cp.async.cg`，再针对 B_tile 的 shared memory layout 做 swizzle，最后扫 `TILE_K` 和更深 pipeline。

### Q25: 怎么保证优化后的正确性？

回答要点：

> 每次优化后都应该和 CPU reference 或 cuBLAS 结果对比。严格 FP32 FFMA 路线可以用较小误差阈值；如果使用 TF32 Tensor Core，误差阈值要放宽，因为乘法输入精度降低了。性能测试则应单独 profile 一个 kernel，避免多个 kernel 写同一个输出导致测量混淆。

## 7. 面试时的简短总结版本

可以按下面这段讲：

> 我从 naive SGEMM 开始，先用 shared memory tiling 减少 global memory 重复访问，然后做 thread tiling，让一个 thread 计算 `4x4` 输出块，把中间结果放在 register 中，提高 shared-to-register 后的数据复用。之后我用 `float4` 做 global/shared 搬运和 C 写回，观察 SASS 中生成了 128-bit load/store。通过 NCU 发现 `matmul3` 主要不是 DRAM 带宽瓶颈，而是 L1TEX/shared 路径和 load-use dependency，于是在 `matmul3` 基础上加入 `cp.async` 和 double buffering，计算当前 K tile 时预取下一 tile。当前 `matmul3_db` 在 NCU report 下从 `375.68 us` 优化到 `335.78 us`，约达到 cuBLAS 的 `71%`。后续我会优先做 no-boundary fast path、优化 cp.async 的 shared wavefront、调整 B_tile shared layout，并继续探索 TF32 Tensor Core / MMA 路线。


# CUDA Copy 与缓存路径梳理

本文按对话顺序整理：`cp.async`、同步加载、`__ldg`、read-only cache，以及 constant/L1/L2/texture/L1TEX/shared 的物理位置。

## 1. `cp.async` 是什么

`cp.async` 是 kernel 内由 GPU 线程发起的异步拷贝指令，方向是：

```text
global memory -> shared memory
```

它不是 host memory 到 device memory 的拷贝。不要和 `cudaMemcpyAsync` 混淆：

```text
cudaMemcpyAsync: host/device/device 之间的 API 级异步拷贝，通常由 copy engine/DMA 执行
cp.async: kernel 内 PTX/SASS 指令，由 SM 中线程发起，搬 global 到 shared
```

典型 PTX：

```ptx
cp.async.ca.shared.global [smem_addr], [gmem_addr], 16;
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16;
```

一次 `cp.async` 通常搬 4/8/16 bytes。发起后数据不一定立即可用，需要配合：

```ptx
cp.async.commit_group;
cp.async.wait_group 0;
```

或者 CUDA C++ 的 `cuda::pipeline` / `memcpy_async` 等接口。

## 2. `cp.async` 的硬件路径

普通同步 GM 到 SMEM 搬运：

```cpp
float4 v = *gmem_ptr;
*smem_ptr = v;
```

数据路径是：

```text
global -> L2 -> L1/L1TEX 或绕过 L1 -> register -> shared
```

`cp.async` 的核心路径是：

```text
global -> L2 -> 可选 L1/L1TEX 路径 -> shared
```

关键区别：

- `cp.async` 的 payload 数据不经过线程可见的通用寄存器。
- 地址计算、谓词、字节数、循环变量仍然需要寄存器。
- `cp.async` 是 global memory 读请求，所以会经过或受 L2 支撑。
- 是否使用 L1 由 cache operator 决定。

`.ca` 和 `.cg` 的区别：

```text
cp.async.ca.shared.global:
    cache at all levels，允许使用/填充 L1 + L2

cp.async.cg.shared.global:
    cache global，倾向绕过 L1，只在 L2 层级缓存
```

## 3. `cp.async` 相比同步加载的价值

主要收益：

1. 隐藏 global memory 延迟  
   可以一边计算当前 tile，一边异步预取下一个 tile 到 shared。

2. 减少寄存器压力  
   普通 `LDG -> register -> STS` 需要临时寄存器保存 payload；`cp.async` 不需要 payload 临时寄存器。

3. 减少显式搬运指令  
   同步路径通常是 global load + shared store；`cp.async` 是专门的 global-to-shared copy。

4. 可避免 L1 污染  
   对 matmul/attention/stencil 这类 tiled kernel，global tile 只搬进 shared 一次，后续复用发生在 shared。此时把 tile 再缓存进 L1 往往是重复缓存，`cp.async.cg` 可以减少 L1 污染。

5. 适合 double buffering / multi-stage pipeline  
   如果发起 `cp.async` 后马上 `wait`，收益会很小。它真正的价值在于和计算重叠。

## 4. 为什么 `cp.async` 特别强调绕过 L1

同步普通 load 也能通过 cache modifier 控制 L1，例如：

```ptx
ld.global.ca   // cache all levels
ld.global.cg   // cache global，倾向 L2-only
```

CUDA C++ 也有：

```cpp
__ldca(&x); // cache all
__ldcg(&x); // cache global，偏 L2-only
```

但 `cp.async` 更常被强调绕过 L1，是因为它的典型用途就是：

```text
global tile -> shared tile -> block 内反复复用
```

此时 shared memory 已经是显式管理的缓存。如果同一份 tile 又填进 L1，会：

- 占用 L1/L1TEX 容量。
- 污染其他真正需要 L1 的数据。
- 和 shared 形成重复缓存。

所以 tiled GEMM/FlashAttention 里常倾向使用 `.cg`，让 staging 流量走 L2 而不污染 L1。

## 5. `src/matmul.cu` 中同步加载为什么不用 `__ldg`

`cooperative_ldst` 里类似：

```cpp
dst[i] = src[gm_row * max_cols + gm_col];
```

用于 matmul 时含义是：

```text
global A/B tile -> shared A_tile/B_tile
```

同步路径是：

```text
global -> register -> shared
```

如果改成：

```cpp
dst[i] = __ldg(&src[gm_row * max_cols + gm_col]);
```

路径仍然是：

```text
global -> read-only/L1TEX cache -> register -> shared
```

它不能做到 `global -> shared`，也不能避免 payload 经过寄存器，更不能自动异步化。

所以对这个 helper 来说，主要优化方向不是 `__ldg`，而是：

- global load coalescing
- `float4` 向量化加载
- shared memory layout
- bank conflict
- `cp.async`
- double buffering / multi-stage buffering
- `.cg` 避免 staging 流量污染 L1

`cooperative_ldst_vector` 里的：

```cpp
float4 value = *reinterpret_cast<const float4*>(&src[idx]);
*reinterpret_cast<float4*>(&dst[smem_offset]) = value;
```

一般会生成类似：

```ptx
ld.global.v4.f32
st.shared.v4.f32
```

这比标量 `__ldg` 更贴近 GM -> SMEM staging 的目标。

## 6. `__ldg` 到底是什么

`__ldg` 是 read-only data cache load intrinsic。

典型代码：

```cpp
float c = __ldg(&coeff[k]);
```

含义不是“把 `coeff` 放进只读内存”，而是：

```text
从 global memory 地址 coeff+k 读取；
声明这次 load 是只读数据访问；
走 read-only/L1TEX cache path；
结果放入寄存器 c。
```

如果 `coeff` 来自：

```cpp
float* d_coeff;
cudaMalloc(&d_coeff, size);
```

那么 `coeff` 的真实数据仍在：

```text
HBM/GDDR global memory
```

`__ldg` 只是改变 load 的 cache 路径：

```text
当前 SM 的 read-only/L1TEX cache
    -> L2 cache
    -> HBM/GDDR
    -> register
```

## 7. `__ldg` 适合什么时候用

适合：

- 数据在 kernel 生命周期内只读。
- 多个线程或多次访问会读到相同/邻近地址。
- 不适合或不值得手动搬到 shared。
- 想利用 SM-local read-only/L1TEX cache。

例子：

```cpp
float v = __ldg(&lookup_table[index[i]]);
```

适合只读查表、不规则 gather、大只读数组、权重表等。

另一个例子：

```cpp
float c0 = __ldg(&coeff[0]);
float c1 = __ldg(&coeff[1]);
float c2 = __ldg(&coeff[2]);
```

如果 `coeff` 很小且 warp 内线程读同一地址，constant memory 可能更合适；如果是较大的只读表，`__ldg` 更自然。

不适合：

- 数据在同一个 kernel 中可能被写。
- 数据只是搬到 shared 后复用。
- 每个数据只读一次且没有局部性。
- 普通 `const __restrict__` 已经让编译器选择了合适只读路径。

注意 `const float*` 不等于一定只读无别名。`const` 只表示不能通过这个指针写；如果另一个指针 alias 到同一地址，仍可能写。`const __restrict__` 能给编译器更强的信息。

## 8. 为什么不都用 `cp.async` 搬到 shared 当只读缓存

shared memory 和 read-only cache 解决的是不同问题。

`cp.async + shared` 适合：

```text
访问范围可预测
block 内大量复用
tile 能放进 shared
同步/搬运成本能被复用摊薄
```

例如 matmul：

```text
A/B tile -> shared -> block 内多次使用
```

但很多只读访问不适合 shared：

- shared 是 block 私有的，多个 block 会重复搬同一份大表。
- shared 容量有限，大表/embedding/稀疏数据放不下。
- 不规则访问很难提前 staging。
- 搬到 shared 需要同步或 pipeline wait。
- shared layout 不好会有 bank conflict。
- 如果数据只用一次，`global -> shared -> register` 反而比 `global/read-only cache -> register` 更重。

因此：

```text
block 内 tile 复用强：shared / cp.async
只读大表、不规则 gather：__ldg / read-only cache
小常量且 warp uniform：constant memory
图像/2D 空间局部性：texture memory
```

## 9. 各类 cache 和 memory 的物理位置

最重要的分类：

```text
global / constant / texture memory:
    CUDA 内存空间，真实数据通常在 HBM/GDDR。

L1 / L1TEX / texture cache / read-only cache / constant cache / L2:
    cache，是 GPU die 上的临时副本。

shared memory:
    GPU die 上每个 SM 内的显式可编程 SRAM，不是自动 cache。
```

物理位置表：

| 名称 | 物理位置 | 共享范围 | 说明 |
| --- | --- | --- | --- |
| global memory | HBM/GDDR memory die | 全 GPU | `cudaMalloc` 数据主体 |
| local memory | HBM/GDDR memory die | 每线程逻辑私有 | spill/大局部数组等，物理仍在 device memory |
| constant memory | HBM/GDDR memory die | 全 GPU 只读 | `__constant__` 数据主体 |
| texture memory | HBM/GDDR memory die | 全 GPU | texture object 绑定的数据主体通常仍是 device memory |
| L2 cache | GPU die | 所有 SM 共享 | global/texture/constant 等访问的共享下一级 cache |
| L1 cache | GPU die，每个 SM 内 | 当前 SM 私有 | 普通 global/local load 的近端 cache，受架构和 cache policy 影响 |
| L1TEX cache | GPU die，每个 SM 内 | 当前 SM 私有 | 现代架构常见的 unified L1/Texture 结构 |
| texture cache | GPU die，每个 SM 内 | 当前 SM 私有 | 现代架构通常由 L1TEX 承担 |
| read-only cache | GPU die，每个 SM 内 | 当前 SM 私有 | `__ldg` 使用的只读路径，现代架构通常映射到 L1TEX |
| constant cache | GPU die，每个 SM 内 | 当前 SM 私有 | constant memory 专用缓存路径，适合 warp 内广播 |
| shared memory | GPU die，每个 SM 内 | block 内共享 | 显式管理的 SRAM；thread block cluster 可访问 distributed shared memory |
| register file | GPU die，每个 SM 内 | 每线程逻辑私有 | load 结果和计算中间值所在 |

## 10. constant cache、texture cache、read-only cache、L1TEX 的关系

这些名字在不同架构上的物理实现会变化，不应理解成永远独立的 SRAM。

实用理解：

```text
constant cache:
    每个 SM 私有，服务 constant memory。
    适合小只读数据和 warp 内同地址广播。

texture cache:
    每个 SM 私有，服务 texture/surface 访问。
    现代架构通常并入 L1TEX。

read-only cache:
    每个 SM 私有，服务 __ldg/read-only global load。
    现代架构通常映射到 L1TEX。

L1TEX:
    每个 SM 私有的统一 L1/Texture 结构，承载普通 L1、texture、read-only 等近端访问路径的一部分。

L2:
    GPU die 上全 SM 共享。

HBM/GDDR:
    global/constant/texture 数据主体所在。
```

constant memory 和 constant cache 不是一回事：

```text
constant memory: CUDA 内存空间，数据主体通常在 HBM/GDDR
constant cache: 每个 SM 上缓存 constant memory 的片上 cache
```

texture memory 和 texture cache 也不是一回事：

```text
texture memory: texture 访问的数据/对象抽象，主体通常在 device memory
texture cache: 每个 SM 上服务 texture fetch 的 cache path
```

shared memory 和 L1 也不是一回事。它们在一些架构上可能共享同一片物理片上 SRAM 的容量预算，但语义不同：

```text
L1 cache: 硬件自动管理
shared memory: 程序员显式管理
```

## 11. 面试争议点：同步加载也能重叠吗

不能简单说“同步加载完全不能重叠”。更准确的说法是：

```text
普通同步 load 对依赖它结果的指令是阻塞的；
但如果后续指令不依赖该 load 的目标寄存器，GPU 可能继续执行这些独立指令；
warp scheduler 也可以切到其他 warp 来隐藏 memory latency。
```

所以在 `cp.async` 出现之前，高性能 GEMM 也能做流水线。常见方式是：

```text
当前 tile 在 shared 中计算
提前发起下一 tile 的 global load
下一 tile 先进入 register
当前计算过程中隐藏部分 global load 延迟
之后 register -> shared
切换 shared buffer
```

这通常叫：

```text
register prefetch
software pipelining
shared-memory double buffering
```

数据路径仍然是：

```text
global -> register -> shared
```

所以面试中如果对方说“同步加载也能重叠”，合理回应不是否定，而是区分：

```text
同步 load 可以通过 register prefetch/software pipelining 隐藏一部分延迟；
但它不是硬件语义上的 global -> shared async copy。
cp.async 的优势不是第一次允许 overlap，而是更适合表达和实现 GM -> SMEM pipeline。
```

## 12. `cp.async` 从什么架构开始

硬件 `cp.async` 是 Ampere 开始引入的，要求：

```text
Compute Capability >= 8.0
典型目标：sm_80+
```

Ampere 之前没有硬件 global-to-shared async copy 指令。之前 GEMM 的 pipeline overlap 主要依赖：

```text
LDG 提前发到 register
中间插入不依赖该 register 的计算
之后 STS 写 shared
```

也就是 register prefetch / software pipelining。

注意：CUDA 高层 API 可能在不同架构上都有形式，但真正由硬件加速的 `global -> shared` async copy 是 Ampere/SM80 及之后的能力。

## 13. `cp.async` 相比 register prefetch 的优势

### 13.1 不占 payload 寄存器

register prefetch：

```text
global -> register -> shared
```

下一 tile 的 payload 会占用寄存器。

GEMM 本来就有很多寄存器压力：

```text
accumulators
A/B fragment
地址变量
循环变量
predicate
prefetch buffer
```

多加 prefetch register 可能导致：

```text
registers/thread 上升
occupancy 下降
spill 到 local memory
```

`cp.async`：

```text
global -> shared
```

payload 不进入通用寄存器，通常是最重要的优势。

### 13.2 直接表达 global-to-shared pipeline

register prefetch 要拆成：

```text
LDG
计算当前 tile
STS
```

`cp.async` 直接表达：

```text
copy global to shared asynchronously
commit_group
compute current tile
wait_group
consume shared
```

pipeline 语义更清楚，也更贴合 shared-memory tiled kernel。

### 13.3 显式等待语义

register prefetch 的等待是隐式的：

```text
当后续指令使用目标寄存器时，如果数据没回来，warp stall。
```

`cp.async` 有显式 group 管理：

```ptx
cp.async.commit_group;
cp.async.wait_group N;
```

这更适合控制 2-stage、3-stage、4-stage pipeline。

### 13.4 可减少 L1 污染

register prefetch 的普通 `LDG` 如果走 L1，可能把 streaming tile 数据放进 L1。

`cp.async.cg.shared.global` 可以倾向绕过 L1，只使用 L2：

```text
global -> L2 -> shared
```

这符合“tile 搬到 shared 后复用”的访问模式。

### 13.5 少掉显式 shared store 阶段

register prefetch 最终还要：

```text
register -> shared
```

这需要 `STS` 指令，占用调度资源和寄存器读端口。

`cp.async` 直接完成：

```text
global -> shared
```

## 14. 为什么 register prefetch 不利于做深 multi-stage

不是不能做，而是 stage 越深，成本越高。

假设每个线程每个 stage 预取 `float4`。2-stage 可能需要：

```cpp
float4 next_a;
float4 next_b;
```

4-stage 可能需要多个未来 tile 的 payload 寄存器：

```cpp
float4 prefetch_a0, prefetch_a1, prefetch_a2;
float4 prefetch_b0, prefetch_b1, prefetch_b2;
```

问题：

```text
每多一个 stage，就增加一组 payload register；
GEMM accumulator 本来已经很占寄存器；
register pressure 上升会降低 occupancy 或造成 spill；
多个 stage 的寄存器生命周期、依赖关系、写 shared 时机更难调度；
等待语义是隐式的，不如 wait_group 清楚。
```

`cp.async` 的 multi-stage：

```text
global -> shared stage 0
global -> shared stage 1
global -> shared stage 2
...
```

stage 主要消耗 shared memory，而不是 payload 寄存器，并且可以用：

```ptx
cp.async.commit_group;
cp.async.wait_group N;
```

管理 pipeline 状态。因此它更适合系统地做 3-stage/4-stage buffering。

## 15. 面试合理回答模板

如果被问“同步加载也能实现访存和计算重叠，`cp.async` 到底强在哪里”，可以这样答：

> 您说同步加载也能重叠是对的。普通 global load 发出后，如果后续指令不依赖目标寄存器，编译器和 warp scheduler 可以通过 software pipelining、register prefetch 或切换 warp 来隐藏一部分 latency。Ampere 之前的高性能 GEMM 也会这么做。  
> 但这种方式的数据路径仍然是 `global -> register -> shared`，下一 tile 的 payload 要占用额外寄存器，最后还要显式 `st.shared`，等待也是通过寄存器依赖隐式发生的。  
> `cp.async` 是 Ampere/SM80 开始提供的硬件 global-to-shared 异步 copy。它直接把数据从 global 搬到 shared，payload 不经过通用寄存器，可以用 `commit_group/wait_group` 显式管理多 stage pipeline，还可以用 `.cg` 绕过 L1，减少 staging 流量对 L1 的污染。  
> 所以 `cp.async` 的优势不是“同步 load 完全不能 overlap”，而是它用更低的寄存器压力、更清晰的等待语义和更直接的 GM->SMEM 数据路径，更适合 shared-memory tiled GEMM/attention 这类 kernel。

一句话版本：

```text
Ampere 之前也能靠 register prefetch 做 overlap；
Ampere 之后 cp.async 提供了真正的 global -> shared async pipeline，
减少 payload register、减少显式 STS、支持 commit/wait group，并可绕过 L1。
```

## 16. 最短总结

`cp.async`：

```text
global -> shared
payload 不经过通用寄存器
可用 .cg 绕过 L1
适合 tile staging + pipeline
```

普通同步搬运：

```text
global -> register -> shared
```

`__ldg`：

```text
global -> read-only/L1TEX cache -> register
不是 shared copy
不是 constant memory
适合只读大表/查表/gather
```

物理位置：

```text
数据主体：HBM/GDDR
L2：GPU die，全 SM 共享
L1/L1TEX/read-only/texture/constant cache：GPU die，每 SM 私有
shared memory：GPU die，每 SM 内，显式管理
register：GPU die，每 SM 内，每线程逻辑私有
```

## 17. double buffering / multi-stage buffering 与 `wait_group`

### 17.1 简单 double buffering

double buffering 的核心不是“有两个数组”本身，而是：

```text
compute current tile
prefetch next tile
```

两者在时间上重叠。典型结构：

```cpp
// prologue
load tile 0 -> smem[0];
commit;

for tile i:
    if i + 1 exists:
        load tile i + 1 -> smem[(i + 1) % 2];
        commit;

    wait current tile ready;
    __syncthreads();

    compute smem[i % 2];

    __syncthreads(); // 保护 shared buffer 复用
```

其中第一处 `__syncthreads()` 用来保证整个 block 搬完当前 tile；第二处用来保证没有线程还在读当前 stage，下一轮才能覆盖这个 shared buffer。

如果把 `cp.async` 换成同步 `LDG -> register -> STS`，即使 shared memory 仍然开两个 buffer，通常也不能得到同样的 GMEM->SMEM 异步流水效果。此时加载延迟已经在进入计算前支付掉了。

### 17.2 multi-stage buffering

multi-stage buffering 通常指多级 shared-memory buffering，而不是把一次 tile matmul 的计算拆成多个计算阶段。

`S` stage 的标准形态：

```text
smem[0], smem[1], ..., smem[S - 1]
```

4-stage 示例：

```cpp
// prologue: 先发 S - 1 个 tile
load tile 0 -> smem[0]; commit;
load tile 1 -> smem[1]; commit;
load tile 2 -> smem[2]; commit;

for tile i:
    prefetch = i + S - 1;

    if prefetch exists:
        load prefetch -> smem[prefetch % S];
        commit;
        wait_group S - 1; // steady state
    else:
        wait_group future_count; // tail drain: S-2, ..., 1, 0

    __syncthreads();
    compute smem[i % S];
    __syncthreads();
```

这里 stage 数表示预取距离：

```text
2-stage: tile i+1 最多提前 1 个 compute tile 发出
4-stage: tile i+3 最多提前 3 个 compute tile 发出
```

### 17.3 multi-stage 什么时候有收益

设：

```text
L = 一个 K tile 的 global -> shared 延迟
C = 一个 K tile 的计算时间
S = buffering stage 数
```

double buffering 最多隐藏约 `1 * C` 的延迟：

```text
stall_2stage = max(0, L - C)
```

`S` stage buffering 最多隐藏约 `(S - 1) * C` 的延迟：

```text
stall_Sstage = max(0, L - (S - 1) * C)
```

因此 multi-stage 有收益的典型窗口是：

```text
L > C                    // double buffering 藏不住
L <= (S - 1) * C          // 更深预取距离能藏住
memory bandwidth 还有余量
更多 shared memory 不显著降低 occupancy
```

没有收益或可能变慢的情况：

```text
L <= C:
    double buffering 已经足够隐藏访存延迟。

compute-bound:
    关键路径是计算，更多预取不会减少 FMA 时间。

bandwidth-bound:
    总搬运字节数没有减少，multi-stage 不能突破带宽上限。

shared memory 占用过高:
    stage 更多会增加 smem/block，可能降低 active blocks/SM。

copy pipeline / shared memory 通路压力变大:
    更多 pending cp.async 不是免费资源。
```

所以 multi-stage buffering 提升的是 latency hiding capability，不提升 arithmetic intensity，也不减少 global memory 总访问量。

### 17.4 `cp.async.commit_group` / `wait_group` 语义

`commit_group`：

```text
把当前线程此前发出的、尚未 commit 的 cp.async 指令组成一个 group。
group 的逻辑顺序由 commit 的程序顺序决定。
```

常见写法是一整个 K tile 一个 group：

```cpp
issue all cp.async for tile i;
cp.async.commit_group;
```

`wait_group N`：

```text
不是等待第 N 个 group。
而是等待到“最多只剩最近 N 个 committed groups 还 pending”。
```

例子：

```text
已提交 group0, group1, group2, group3

wait_group 0: 等 group0,1,2,3 全部完成
wait_group 1: 等 group0,1,2 完成，允许 group3 继续 pending
wait_group 2: 等 group0,1 完成，允许 group2,3 继续 pending
wait_group 3: 等 group0 完成，允许 group1,2,3 继续 pending
```

因此 4-stage steady state 中，如果 wait 前已有：

```text
current tile + 3 个 future tiles
```

就可以用：

```ptx
cp.async.wait_group 3;
```

它等待 current tile 对应的更老 group 完成，同时保留后 3 个 future groups 在路上。

### 17.5 `wait_group` 的常见坑

1. `N` 必须是 PTX immediate。

不能传运行时变量：

```cpp
int n = 3;
asm volatile("cp.async.wait_group %0;\n" :: "r"(n)); // 错
```

应使用常量、模板参数，或 tail 阶段用 `switch`：

```cpp
switch (future_count) {
    case 3: asm volatile("cp.async.wait_group 3;\n"); break;
    case 2: asm volatile("cp.async.wait_group 2;\n"); break;
    case 1: asm volatile("cp.async.wait_group 1;\n"); break;
    case 0: asm volatile("cp.async.wait_group 0;\n"); break;
}
```

2. `wait_group` 是 per-thread 语义。

它只保证当前线程自己发出的 cp.async group 达到等待条件。协同加载的 GEMM tile 会读其他线程搬入 shared memory 的数据，所以通常需要：

```cpp
cp.async.wait_group N;
__syncthreads();
compute shared tile;
```

3. 实际完成顺序可以乱序，但逻辑等待不会错位。

硬件不需要保证 `A0, A1, A2` 物理上按顺序返回。`wait_group N` 按 `commit_group` 建立的逻辑 group 队列等待“更老的 groups”，不会因为 `A2` 先完成就把它当成 `A0` 消费。

4. 同一个 group 内没有可靠的 cp.async 相互排序。

不要让同一个 group 内多个 cp.async 写同一个 shared memory 地址。一个 tile 的不同元素可以放在同一个 group，但地址必须不冲突。

5. tail drain 不能一直用 steady-state 的 `wait_group S-1`。

最后没有新的 future tile 可发时，应逐步 drain：

```text
4-stage tail: wait_group 2 -> wait_group 1 -> wait_group 0
```

如果最后一轮仍然 `wait_group 3`，可能什么都不等，随后读 shared memory 就有风险。

6. `__syncthreads()` 不能替代 `wait_group`。

`__syncthreads()` 只同步 block 内线程到达某个点，不保证 cp.async 数据已经写入 shared memory。正确顺序通常是：

```cpp
cp.async.wait_group N;
__syncthreads();
```

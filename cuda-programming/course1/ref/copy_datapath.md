# CUDA Copy Datapath 与缓存层级梳理

本文档整理 CUDA 中常见数据搬运路径和缓存/存储空间的关系，重点覆盖：

- `cp.async`
- 普通同步 global load/store
- `__ldg`
- constant cache
- L1 cache
- L2 cache
- texture cache / Tex cache
- read-only cache
- L1TEX cache
- shared memory

核心目标是厘清三个问题：

1. 数据的真实存储位置在哪里。
2. 一次 load/copy 经过哪些片上结构。
3. 什么时候应该用普通 load、`__ldg`、shared memory 或 `cp.async`。

## 1. 最重要的区分：内存空间 vs cache vs shared memory

CUDA 里很多名字容易混在一起，但它们不是同一类概念。

### 1.1 内存空间

内存空间是编程模型里的地址空间或数据放置方式。

典型例子：

- global memory
- local memory
- constant memory
- texture memory

这些空间的真实数据通常位于设备显存，即 HBM/GDDR memory die 上。它们的数据可以被 GPU die 上的 cache 临时缓存。

### 1.2 cache

cache 是 GPU die 上的片上 SRAM，用来缓存显存中的数据副本。

典型例子：

- L1 cache
- L2 cache
- constant cache
- texture cache
- read-only cache
- L1TEX cache

cache 不是数据的所有者。`cudaMalloc` 得到的数据主体仍在 global memory/HBM/GDDR 中，cache 里只是临时副本。

### 1.3 shared memory

shared memory 不是普通意义上的 cache，而是每个 SM 上显式可编程的片上 SRAM。

程序员需要显式声明：

```cpp
__shared__ float tile[1024];
```

并显式把数据搬进去：

```cpp
tile[threadIdx.x] = global_ptr[idx];
__syncthreads();
```

或者用 `cp.async` 搬进去。

shared memory 的内容由程序控制，不是硬件自动缓存 global memory 的结果。

## 2. 物理位置总表

| 名称 | 真实物理位置 | 共享范围 | 是否程序员显式管理 | 说明 |
| --- | --- | --- | --- | --- |
| global memory | HBM/GDDR memory die | 全 GPU，可由 host 通过 API 访问 | 是，分配/释放显式；cache 自动 | `cudaMalloc` 的主体数据在这里 |
| local memory | HBM/GDDR memory die | 每个线程逻辑私有 | 编译器生成 | 寄存器 spill、大数组等可能落到 local memory；物理上仍是 device memory |
| constant memory | HBM/GDDR memory die | 全 GPU 只读 | 是，通过 `__constant__` 或 symbol copy | 真实数据在 device memory，访问时走 constant cache |
| texture memory | HBM/GDDR memory die | 全 GPU 只读或按 texture/surface 语义访问 | 是，通过 texture object/surface 等绑定 | 真实数据通常仍来自 device memory，访问时走 texture path/cache |
| L2 cache | GPU die | 所有 SM 共享 | 否 | 全芯片级 cache，global/local/texture/constant 等访问通常会经过或由它支撑 |
| L1 cache | GPU die，每个 SM 内 | 当前 SM 私有 | 否 | 普通 load 的近端 cache；具体行为受架构和 cache operator 影响 |
| L1TEX cache | GPU die，每个 SM 内 | 当前 SM 私有 | 否 | 现代架构中统一的 L1/Texture 路径，承担 L1 data cache、texture cache、read-only path 等功能的一部分 |
| texture cache | GPU die，每个 SM 内 | 当前 SM 私有 | 否 | 传统上服务 texture fetch；现代架构通常并入 L1TEX/unified L1/Texture cache |
| read-only cache | GPU die，每个 SM 内 | 当前 SM 私有 | 否 | `__ldg`/read-only load 使用的缓存路径；现代架构通常映射到 L1TEX |
| constant cache | GPU die，每个 SM 内 | 当前 SM 私有 | 否 | constant memory 的专用缓存路径，适合 warp 内同地址广播 |
| shared memory | GPU die，每个 SM 内 | block 内线程共享；thread block cluster 可访问其他 block 的 distributed shared memory | 是 | 显式可编程 SRAM，不是自动 cache |
| register file | GPU die，每个 SM 内 | 每个线程逻辑私有 | 编译器分配 | 普通 load 的结果最终进入寄存器 |

## 3. 一张总图

粗略数据通路如下：

```text
HBM/GDDR memory die
    |
    |  global / local / constant / texture 数据主体
    v
GPU die
    |
    +-- L2 cache                         全 GPU 共享
    |
    +-- SM 0
    |     +-- L1 / L1TEX / read-only / texture cache
    |     +-- constant cache
    |     +-- shared memory
    |     +-- register file
    |
    +-- SM 1
    |     +-- L1 / L1TEX / read-only / texture cache
    |     +-- constant cache
    |     +-- shared memory
    |     +-- register file
    |
    +-- ...
```

因此：

```text
global/constant/texture 数据主体：通常在 HBM/GDDR
L2：GPU die 上，全 SM 共享
L1/L1TEX/read-only/texture/constant cache：GPU die 上，每个 SM 私有
shared memory：GPU die 上，每个 SM 私有，由程序显式管理
register：GPU die 上，每个 SM 内，每个线程逻辑私有
```

## 4. 普通同步 global load/store 的路径

典型代码：

```cpp
float x = A[i];
```

逻辑含义：

```text
从 global memory 地址 A+i 读取一个 float，结果放到当前线程寄存器 x。
```

典型 cache 查找路径：

```text
register 需要 A[i]
    -> 当前 SM 的 L1/L1TEX，是否使用取决于架构和 cache policy
    -> L2 cache
    -> HBM/GDDR global memory
    -> 数据返回 register
```

从程序员视角可以理解为：

```text
A[i] 的真实数据在显存；
如果当前 SM 的 L1/L1TEX 命中，就直接从片上 cache 得到；
否则查 L2；
L2 miss 再访问 HBM/GDDR；
最终结果写入寄存器。
```

如果随后写 shared memory：

```cpp
__shared__ float tile[1024];
float x = A[i];
tile[threadIdx.x] = x;
```

cache 查找和数据返回路径变成：

```text
register 需要 A[i]
    -> L1/L1TEX 或绕过 L1
    -> L2
    -> HBM/GDDR
    -> register
    -> shared memory
```

也就是普通同步搬运 global 到 shared 时，payload 数据通常会经过通用寄存器。

在 `src/matmul.cu` 的标量 helper 中：

```cpp
dst[i] = src[gm_row * max_cols + gm_col];
```

当 `direction == false`，`dst` 是 shared memory，`src` 是 global memory 时，这就是同步 GM 到 SMEM 搬运：

```text
global -> register -> shared
```

## 5. cache operator：`.ca`、`.cg` 和相关 builtin

PTX load/copy 可以带 cache operator。它们通常是性能 hint，不改变程序语义。

常见含义：

| 写法 | 大意 | L1 行为 | L2 行为 |
| --- | --- | --- | --- |
| `.ca` | cache at all levels | 允许使用/填充 L1 | 使用 L2 |
| `.cg` | cache global | 通常不在 L1 缓存 | 使用 L2 |
| `.cs` | streaming/evict-first 类 hint | 倾向减少 cache 污染 | 使用 L2 |
| `.lu` | last use hint | 倾向尽快淘汰 | 使用 L2 |
| `.cv` | volatile-like cache behavior for global load | 语义更特殊 | 使用 L2/内存系统 |

CUDA C++ 里也有 cache hint load intrinsic，例如：

```cpp
float x = __ldcg(&A[i]); // cache global，偏 L2-only
float y = __ldca(&A[i]); // cache all，允许 L1 + L2
```

注意：

- `__ldg` 不是 `.cg`。
- `__ldg` 表达 read-only cache path。
- 如果目标是“不污染 L1”，更接近的是 `.cg` / `__ldcg` / `cp.async.cg`。

## 6. `cp.async` 的含义

`cp.async` 指 kernel 内部由 GPU 线程发起的异步拷贝指令，方向是：

```text
global memory -> shared memory
```

它不是 host memory 到 device memory 的异步拷贝。

不要混淆：

```text
cudaMemcpyAsync:
    host <-> device 或 device <-> device
    由 runtime/driver/copy engine/DMA 等执行

cp.async:
    kernel 内指令
    由 SM 中线程发起
    global -> shared
```

PTX 形式：

```ptx
cp.async.ca.shared.global [smem_addr], [gmem_addr], 16;
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16;
```

含义：

```text
从 global 地址 gmem_addr 异步复制 4/8/16 bytes 到 shared 地址 smem_addr。
```

## 7. `cp.async` 的底层数据路径

对比普通同步搬运：

```cpp
float4 v = *reinterpret_cast<const float4*>(gmem);
*reinterpret_cast<float4*>(smem) = v;
```

同步路径：

```text
L1/L1TEX 或绕过 L1 -> L2 -> HBM/GDDR -> register -> shared
```

`cp.async` 路径：

```text
可选 L1/L1TEX 路径 -> L2 -> HBM/GDDR -> shared
```

关键差异：

```text
payload 数据不落到线程可见的通用寄存器。
```

地址计算仍然需要寄存器。例如 shared 地址、global 地址、谓词、循环变量都在寄存器里。但被复制的数据本身不需要先进入通用寄存器再 `st.shared`。

NVIDIA Ampere Tuning Guide 对此的总结是：

- 异步 global-to-shared copy 可以和计算重叠。
- 可以避免额外寄存器用于 memory copy。
- 可以绕过 L1 cache。

## 8. `cp.async` 是否经过 L2、L1、寄存器

### 8.1 是否经过 L2

是。`cp.async` 的源地址是 global state space，因此它属于 global memory 读请求。global memory 的片上共享 cache 是 L2。

PTX 还支持 `.L2::64B`、`.L2::128B`、`.L2::256B` 这类 prefetch size hint，说明 `cp.async` 可以显式给 L2 预取 hint。

### 8.2 是否经过 L1

取决于 cache operator：

```ptx
cp.async.ca.shared.global ...
```

`.ca` 表示 cache at all levels，允许使用/填充包括 L1 在内的缓存层级。

```ptx
cp.async.cg.shared.global ...
```

`.cg` 表示 cache global，PTX 文档说明它只在 global level cache，即 L2，缓存数据，不在 L1 缓存。

所以：

```text
cp.async.ca：可以经过/填充 L1/L1TEX
cp.async.cg：倾向绕过 L1，只使用 L2
```

这些 cache operator 是性能 hint，不应当写出依赖它们改变可见语义的程序。

### 8.3 是否经过寄存器

payload 数据不经过线程可见的通用寄存器。

普通同步 GM 到 SMEM：

```text
global -> register -> shared
```

`cp.async`：

```text
global -> shared
```

但下列内容仍然使用寄存器：

- global address
- shared address
- byte count
- predicate
- loop index
- boundary check

不要把“payload 不经过寄存器”误解成“整条指令完全不使用寄存器”。

## 9. `cp.async` 的同步模型

`cp.async` 是非阻塞发起，不代表数据立刻可用。

典型写法：

```ptx
cp.async.ca.shared.global [smem0], [gmem0], 16;
cp.async.ca.shared.global [smem1], [gmem1], 16;
cp.async.commit_group;
cp.async.wait_group 0;
```

语义：

```text
cp.async 发起异步拷贝；
commit_group 把之前发起的拷贝提交为一组；
wait_group 等待指定数量的组完成；
之后才能安全读取目标 shared memory。
```

在 CUDA C++ 中也可以使用 `cuda::pipeline` / `memcpy_async` 这类更高层接口表达同一类逻辑。

常见错误：

```text
发起 cp.async 后马上读 shared，但没有 wait；
只用了 __syncthreads()，却没有正确等待 cp.async group；
循环双缓冲中覆盖了还没被消费或还没写完的 shared stage。
```

`__syncthreads()` 是线程同步，不等价于所有未完成的 `cp.async` 自动完成。需要用 `cp.async.wait_group` 或对应的 pipeline wait。

## 10. `cp.async` 相比同步加载的好处

### 10.1 可以隐藏 global memory 延迟

同步搬运：

```text
load tile
wait load 完成
store shared
__syncthreads()
compute
```

异步双缓冲：

```text
预取 tile 0
wait tile 0
compute tile 0，同时 cp.async 预取 tile 1
compute tile 1，同时 cp.async 预取 tile 2
...
```

核心收益是让 global -> shared 搬运和当前 tile 的计算重叠。

### 10.2 减少寄存器压力

同步搬运需要临时寄存器保存 payload：

```cpp
float4 value = *gmem_ptr;
*smem_ptr = value;
```

`value` 占用寄存器。

`cp.async` 不需要 payload 临时寄存器，因此可能降低 registers/thread，提高 occupancy，或者减少 spill。

### 10.3 减少显式指令数量

同步路径通常是：

```text
LDG
STS
```

`cp.async` 逻辑上是一条 global-to-shared copy 指令。不同架构上的最终 SASS 表达可能不同，但优化目标是硬件加速这一搬运路径。

### 10.4 可以减少 L1 污染

对于 tiling matmul/attention/stencil，global 数据通常只搬进 shared 一次，后续复用发生在 shared。

如果这些 staging 数据同时填入 L1，会出现重复缓存：

```text
同一份 tile 既在 shared memory 中，又占据 L1/L1TEX cache line。
```

使用 `cp.async.cg` 可以倾向只走 L2，避免把 streaming tile copy 流量塞进 L1。

### 10.5 更适合显式流水线

`cp.async` 的优势通常需要配合：

- double buffering
- multi-stage buffering
- `commit_group`
- `wait_group`
- 合理的 shared layout
- 足够的 compute 覆盖 load latency

如果发起 `cp.async` 后马上 `wait_group 0`，没有重叠计算，收益会明显下降。

## 11. 为什么 `cp.async` 特别强调绕过 L1

普通 global load 也可以通过 cache modifier 或 builtin 控制是否使用 L1，例如 `ld.global.cg` / `__ldcg`。

但 `cp.async` 的典型场景更需要强调 L1 bypass：

```text
global -> shared
shared 作为显式管理的 block-local cache
后续计算从 shared 复用
```

此时 L1 往往不是主要复用载体。

如果 tile copy 填充 L1，可能带来两个问题：

1. 同一份数据在 shared 和 L1 中重复占用片上资源。
2. streaming tile 流量污染 L1，挤掉其他真正需要 L1 的数据。

因此对 GEMM、FlashAttention、convolution、stencil 等 tiled kernel：

```text
cp.async.cg.shared.global
```

常常比 `.ca` 更符合意图，尤其当 global tile 只被搬一次、复用完全发生在 shared 中时。

## 12. `__ldg` 的含义

`__ldg` 是 read-only data cache load intrinsic。

典型代码：

```cpp
float c = __ldg(&coeff[k]);
```

语义：

```text
从 global memory 地址 coeff+k 读取一个 float；
这次读取被声明为只读数据 load；
允许走 read-only cache path；
结果放入寄存器 c。
```

重要：

```text
__ldg 不改变 coeff 的存储位置。
__ldg 不把数据搬到 shared memory。
__ldg 的结果仍然进入寄存器。
```

如果 `coeff` 来自：

```cpp
float* d_coeff;
cudaMalloc(&d_coeff, size);
```

那么 `coeff[k]` 的权威数据仍在：

```text
HBM/GDDR global memory
```

`__ldg` 只是改变 cache 查找路径：

```text
register 需要 coeff[k]
    -> 当前 SM 的 read-only/L1TEX cache
    -> L2 cache
    -> HBM/GDDR
    -> register
```

## 13. read-only cache 位于哪里

read-only cache 不是 HBM/GDDR，也不是全 GPU 共享的 L2。

它更接近：

```text
GPU die 上，每个 SM 私有的 read-only / texture / L1TEX cache path
```

不同架构实现不同：

- Kepler 时代有更明确的 read-only data cache / texture pipeline 说法。
- Maxwell/Pascal 之后许多架构把 L1 和 texture cache 统一。
- Volta/Ampere 之后常见说法是 unified L1/Texture cache 或 L1TEX。

因此现代 CUDA 优化中可以这样理解：

```text
__ldg 使用的是 SM-local 的 read-only/L1TEX 路径；
L2 仍然是全 GPU 共享的下一级 cache；
HBM/GDDR 仍然是数据主体所在。
```

## 14. `__ldg` 和普通 `const float*` 的区别

普通代码：

```cpp
float x = A[i];
```

如果 `A` 是：

```cpp
const float* A
```

这表示不能通过 `A` 指针写，但不一定证明没有别的指针会写同一块内存。

例如：

```cpp
__global__ void kernel(float* C, const float* A) {
    C[threadIdx.x] = 1.0f;
    float x = A[threadIdx.x];
}
```

从类型上看 `A` 是 `const`，但如果调用方让 `C == A`，那么同一个 kernel 内仍可能写到 `A` 指向的数据。

因此编译器未必敢自动使用 read-only path。

更强的写法是：

```cpp
__global__ void kernel(
    float* __restrict__ C,
    const float* __restrict__ A
) {
    float x = A[threadIdx.x];
    C[threadIdx.x] = x;
}
```

`const + __restrict__` 增加编译器判断只读、无别名的机会。

手写：

```cpp
float x = __ldg(&A[i]);
```

则是程序员明确告诉编译器：

```text
这个地址在当前 kernel 生命周期内按只读 load 处理。
```

如果数据在同一个 kernel 中可能被写，不应该使用 `__ldg`。

## 15. `__ldg` 适合什么场景

适合：

```text
数据在整个 kernel 生命周期内只读；
多个线程或多次访问会读到相同或邻近地址；
不适合或不值得手动搬到 shared memory；
希望利用 SM-local read-only/L1TEX cache。
```

典型场景：

### 15.1 查表

```cpp
__global__ void lookup_kernel(float* out, const float* table, const int* idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = idx[i];
    out[i] = __ldg(&table[k]);
}
```

如果 `table` 只读，并且不同线程访问有重复或局部性，`__ldg` 可能有效。

### 15.2 只读系数

```cpp
__global__ void stencil(float* out, const float* in, const float* coeff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float c0 = __ldg(&coeff[0]);
    float c1 = __ldg(&coeff[1]);
    float c2 = __ldg(&coeff[2]);

    out[i] = c0 * in[i - 1] + c1 * in[i] + c2 * in[i + 1];
}
```

如果 `coeff` 比较小且 warp 内访问同一地址，也可以考虑 constant memory。

### 15.3 大只读数组的不规则 gather

```cpp
float v = __ldg(&weights[index[i]]);
```

如果访问不规则但有重复或局部性，read-only cache 可能比普通 path 更合适。

## 16. `__ldg` 不适合什么场景

### 16.1 数据要搬到 shared 后复用

例如 matmul tile staging：

```cpp
tile[threadIdx.x] = A[i];
```

或者：

```cpp
tile[threadIdx.x] = __ldg(&A[i]);
```

这两者都是：

```text
global -> register -> shared
```

`__ldg` 只改变 global load 的 cache path，不能避免寄存器中转，也不能产生异步 overlap。

如果目标是 GM -> SMEM staging，优先考虑：

- coalesced load
- vectorized load
- shared memory layout
- `cp.async`
- double buffering / multi-stage pipeline

### 16.2 数据在同一个 kernel 中可能被写

错误示例：

```cpp
A[i] = 1.0f;
float x = __ldg(&A[i]);
```

read-only cache path 是 non-coherent 的只读路径。若同一 kernel 中有写入，可能读到旧值或产生非预期结果。

### 16.3 没有复用、没有局部性的一次性流式读取

如果每个数据只读一次，并且访问是良好 coalesced 的 streaming load，`__ldg` 往往没有明显收益。此时普通 load 或 `.cg`/`__ldcg` 控制 L1 污染更值得考虑。

## 17. `__ldg`、constant memory、texture memory 的区别

### 17.1 `__ldg`

```cpp
float x = __ldg(&A[i]);
```

特点：

- `A` 通常是普通 global memory 指针。
- 数据主体在 HBM/GDDR。
- load 使用 read-only/L1TEX cache path。
- 结果进入寄存器。
- 适合大只读数组、查表、不规则 gather、有局部性但不适合 shared 的场景。

### 17.2 constant memory

```cpp
__constant__ float coeff[1024];
```

特点：

- 数据主体在 device memory。
- 每个 SM 有 constant cache。
- 适合小型只读数据。
- 特别适合 warp 内线程读取同一个 constant 地址，因为 constant cache 支持广播式访问。
- 如果一个 warp 中 32 个线程读 32 个不同 constant 地址，请求会被拆分，吞吐下降。

典型适用：

```text
卷积核参数
少量配置参数
小 lookup table，且 warp 内访问较 uniform
```

### 17.3 texture memory

特点：

- 数据主体通常在 device memory。
- 通过 texture object/reference 访问。
- 使用 texture cache/L1TEX path。
- 支持 texture 专用功能，例如地址模式、归一化坐标、过滤、格式转换等。
- texture cache 对 2D 空间局部性有优化。

典型适用：

```text
图像/2D 数据
空间局部性强但 global coalescing 不理想的访问
需要硬件过滤或地址模式
```

## 18. constant cache、texture cache、read-only cache、L1TEX 的关系

历史上这些名字的边界随架构变化。

一个实用理解：

```text
constant cache:
    专门服务 constant memory，SM-local，适合 uniform/broadcast。

texture cache:
    服务 texture/surface fetch，SM-local，现代架构中常并入 L1TEX。

read-only cache:
    服务 __ldg/read-only global load，SM-local，现代架构中常映射到 L1TEX。

L1TEX:
    现代 NVIDIA profiling 和架构文档里常见的统一 L1/Texture 结构，
    负责多类近端数据访问路径。
```

不要把它们理解成所有架构上都完全独立的物理 SRAM。

更稳妥的说法是：

```text
constant cache 是专用 constant path；
texture/read-only/L1 data path 在现代架构中通常由 SM-local L1TEX/unified L1/Texture 结构承载；
L2 是所有 SM 共享的下一级 cache；
HBM/GDDR 是数据主体。
```

## 19. L1 cache、L2 cache 和 shared memory 的关系

### 19.1 L1 cache

L1 cache 位于每个 SM 内，当前 SM 私有。

作用：

- 缓存部分 global/local load。
- 降低访问延迟。
- 作为 warp 访存的近端数据路径。

是否使用 L1 取决于：

- GPU 架构。
- load/store 类型。
- cache operator。
- 编译选项。
- 具体指令。

### 19.2 L2 cache

L2 cache 位于 GPU die 上，由所有 SM 共享。

作用：

- global memory 访问的重要共享缓存。
- 跨 SM 复用主要依赖 L2，而不是某个 SM 私有 L1。
- 许多 memory consistency/coherency 行为以 L2 为关键层级。

如果多个 block/SM 读同一片 global 数据，L2 命中比 L1 更有意义，因为 L1 是 SM 私有的。

### 19.3 shared memory

shared memory 位于每个 SM 内，由程序显式分配和管理。

作用：

- block 内线程协同复用数据。
- 可预测、低延迟、高带宽。
- 需要考虑 bank conflict。
- 容量有限，会影响 occupancy。

在许多架构中，shared memory 和 L1/L1TEX 共享同一片物理片上 SRAM 的容量预算，通过 carveout 划分比例。但编程语义上：

```text
L1 是自动 cache；
shared 是显式 scratchpad。
```

不要因为它们可能共享物理 SRAM，就把二者语义混同。

## 20. 为什么不能“都用 cp.async 搬到 shared 当只读缓存”

这是一个很常见的疑问。

`cp.async + shared memory` 很强，但它只适合一类问题：

```text
数据访问范围可预测；
一个 block 内有大量复用；
数据块能装入 shared；
同步/搬运成本能被复用收益覆盖。
```

matmul 是典型适用场景：

```text
A/B tile 从 global 搬进 shared；
block 内很多线程反复使用同一 tile；
计算量足够大，可以隐藏搬运延迟。
```

但很多只读访问不适合 shared。

### 20.1 shared memory 是 block 私有的

如果一个大表被所有 block 读：

```text
每个 block 都要把同一份数据搬进自己的 shared memory。
```

这会重复占用带宽和 shared 容量。

L2 cache 可以让多个 block/SM 受益于跨 SM 的 cache 命中；read-only/L1TEX cache 则让同一个 SM 上的重复只读访问受益。

### 20.2 shared memory 容量有限

大 lookup table、embedding table、稀疏权重、不规则图数据通常放不进 shared。

### 20.3 不规则访问很难 staging

如果每个线程访问：

```cpp
table[index[i]]
```

其中 `index[i]` 分散且运行时才知道，把所有可能数据先搬到 shared 往往不可行。

### 20.4 shared 需要同步

搬到 shared 后，通常要等待：

```cpp
__syncthreads();
```

或者：

```ptx
cp.async.wait_group ...
```

如果数据只用一次，同步和搬运成本可能比 cache load 更高。

### 20.5 shared 可能引入 bank conflict 和布局成本

shared memory 的高性能依赖合理 layout。设计不好可能出现 bank conflict，反而降低性能。

### 20.6 `__ldg` 是轻量表达

`__ldg` 的优势是：

```text
不需要显式 staging；
不需要 shared 容量；
不需要 block 级同步；
不改变代码结构；
适合只读但不适合 shared tiling 的访问。
```

因此：

```text
block 内 tile 复用强：shared / cp.async
只读大表或不规则 gather：__ldg / read-only cache
小常量且 warp uniform：constant memory
2D 空间局部性或 texture 功能：texture memory
```

## 21. 对 `src/matmul.cu` 中 cooperative_ldst 的分析

`src/matmul.cu` 中有同步协同搬运：

```cpp
dst[i] = src[gm_row * max_cols + gm_col];
```

当用于：

```cpp
cooperative_ldst(... A -> A_tile ...);
cooperative_ldst(... B -> B_tile ...);
```

它的意图是：

```text
每个 block 协同把 A/B tile 从 global memory 搬进 shared memory；
之后计算阶段从 A_tile/B_tile 反复读取。
```

此时数据复用发生在 shared memory 中，而不是依赖 global load cache。

所以把它改成：

```cpp
dst[i] = __ldg(&src[gm_row * max_cols + gm_col]);
```

只能改变 global load 的 cache path，仍然是：

```text
global -> read-only/L1TEX -> register -> shared
```

不能变成：

```text
global -> shared
```

也不能自动产生异步 overlap。

因此，对这个函数更关键的优化是：

1. global load 是否 coalesced。
2. 是否能向量化，例如 `float4`。
3. shared memory layout 是否避免 bank conflict。
4. 是否用 `cp.async` 直接做 global -> shared。
5. 是否用 double buffering/multi-stage pipeline 隐藏延迟。
6. 是否选择 `.cg` 避免 tile staging 污染 L1。

## 22. 对 `cooperative_ldst_vector` 的分析

代码中有向量化版本：

```cpp
float4 value = *reinterpret_cast<const float4*>(&src[gm_row * max_cols + gm_col]);
*reinterpret_cast<float4*>(&dst[smem_offset]) = value;
```

这通常会形成类似：

```ptx
ld.global.v4.f32
st.shared.v4.f32
```

或 SASS 中的 128-bit global load + 128-bit shared store。

路径仍是：

```text
global -> register(s) -> shared
```

但相比标量版本有优势：

- 减少指令数。
- 更容易形成宽 load/store。
- 更好利用 memory transaction。
- 提高 GM -> SMEM staging 效率。

如果改用 `__ldg` 标量读四次：

```cpp
float x0 = __ldg(&src[idx + 0]);
float x1 = __ldg(&src[idx + 1]);
float x2 = __ldg(&src[idx + 2]);
float x3 = __ldg(&src[idx + 3]);
```

可能破坏原来的 `float4` 向量化加载，不一定更快。

`__ldg` 支持部分向量类型，如 `float2`、`float4`，但它解决的是 read-only cache path，不是 GM -> SMEM copy pipeline。

## 23. `cp.async` 在 matmul 中的定位

matmul 的数据复用结构：

```text
A_tile: M_tile x K_tile
B_tile: K_tile x N_tile
C_tile: M_tile x N_tile
```

每个 A/B tile 被 block 内多个线程重复使用。

因此理想路径是：

```text
global A/B tile -> shared A_tile/B_tile -> register compute
```

同步版本：

```text
global -> register -> shared
__syncthreads()
shared -> register
FMA
```

`cp.async` 版本：

```text
global -> shared
wait
shared -> register
FMA
```

双缓冲：

```text
stage 0: 当前计算
stage 1: 下一 tile 用 cp.async 预取
```

这就是 `cp.async` 对 matmul 有价值的原因：它匹配 tile staging，并且能和计算重叠。

## 24. 常见选择准则

| 场景 | 优先考虑 |
| --- | --- |
| block 内大量复用同一块 global 数据 | shared memory |
| shared tile staging，且希望隐藏 GM 延迟 | `cp.async` + double/multi-stage buffering |
| 只读大数组，不规则 gather，有局部性，不适合 shared | `__ldg` 或 `const __restrict__` 让编译器走 read-only path |
| 小只读参数，warp 内线程常读同一地址 | constant memory |
| 2D 空间局部性、图像采样、地址模式/过滤/格式转换 | texture memory |
| streaming 读一次，不想污染 L1 | `.cg` / `__ldcg` / `cp.async.cg` |
| 普通 coalesced 一次性读写 | 普通 global load/store，保证合并访存 |
| 需要跨 block/SM 复用 global 数据 | L2 cache 更关键，shared 不跨 block 共享 |

## 25. 常见误解澄清

### 25.1 `cp.async` 是 host 到 device 的异步拷贝吗

不是。

```text
cp.async: kernel 内 global -> shared
cudaMemcpyAsync: host/device 之间或 device/device 的 API 级异步拷贝
```

### 25.2 `__ldg` 会把数据放到只读内存吗

不会。

`__ldg` 不改变数据存储位置。数据主体仍在 global memory/HBM/GDDR，只是这次 load 走 read-only cache path。

### 25.3 read-only cache 是 L2 吗

不是。

read-only cache path 更接近每个 SM 私有的 L1TEX/texture/read-only 路径。L2 是所有 SM 共享的下一级 cache。

### 25.4 constant memory 和 constant cache 是一回事吗

不是。

```text
constant memory:
    CUDA 内存空间，数据主体在 device memory。

constant cache:
    每个 SM 上缓存 constant memory 的片上 SRAM/cache path。
```

### 25.5 texture memory 和 texture cache 是一回事吗

不是。

```text
texture memory:
    CUDA 访问路径/内存空间/对象抽象，数据主体通常在 device memory。

texture cache:
    GPU die 上缓存 texture fetch 的 SM-local cache path。
```

### 25.6 shared memory 是 L1 cache 吗

不是。

它们在某些架构上共享同一片物理片上 SRAM 容量预算，但语义不同：

```text
L1 cache:
    硬件自动管理。

shared memory:
    程序员显式管理。
```

### 25.7 `__ldg` 可以替代 `cp.async` 吗

不能。

```text
__ldg:
    global -> read-only cache -> register

cp.async:
    global -> shared
```

二者目标不同。

### 25.8 使用 `__ldg` 就一定更快吗

不一定。

它只有在只读数据存在 cache reuse 或局部性，且普通 load 没有已经被编译器优化到类似路径时，才可能带来收益。

对一次性 streaming load，`__ldg` 可能没有收益。

### 25.9 `const float*` 就一定会走 `__ldg` 吗

不一定。

`const` 只限制通过该指针写入，不保证没有别的 alias 指针写同一地址。

`const __restrict__` 更有利于编译器判断只读无别名，但最终仍要看编译器和目标架构。

## 26. 简短面试回答模板

### 26.1 `cp.async` 是什么

`cp.async` 是 kernel 内 SM 线程发起的异步 global-to-shared copy 指令，不是 host-to-device copy。它从 global memory 读取数据写入 shared memory，payload 数据不经过线程可见的通用寄存器，可以通过 `.ca` 或 `.cg` 控制 cache policy，并用 `commit_group`/`wait_group` 或 pipeline API 等待完成。它的主要价值是减少寄存器压力、减少显式 LDG+STS 搬运开销、避免 L1 污染，并通过双缓冲隐藏 GM 到 SMEM 的延迟。

### 26.2 `__ldg` 是什么

`__ldg` 是 read-only data cache load。它不会改变数据存储位置，也不会搬到 shared memory。它从 global memory 地址读取数据，使用 SM-local 的 read-only/L1TEX cache path，最终结果进入寄存器。适合 kernel 生命周期内只读的大数组、查表、不规则 gather 等不适合 shared tiling 的访问。

### 26.3 read-only cache 在哪里

read-only cache 不是显存，不是 L2，也不是一个全 GPU 共享的只读内存。它更接近每个 SM 私有的 read-only/texture/L1TEX cache path。数据主体仍在 HBM/GDDR global memory；L2 是所有 SM 共享的下一级 cache。

### 26.4 constant cache、L1、L2、texture、shared 的物理位置

global/constant/texture 数据主体通常在 HBM/GDDR。L2 在 GPU die 上，由所有 SM 共享。L1、L1TEX、texture cache、read-only cache、constant cache 都在 GPU die 上，每个 SM 私有。shared memory 也在 GPU die 上，每个 SM 内，但它是程序员显式管理的 SRAM，不是自动 cache。

## 27. 参考资料

- NVIDIA CUDA C++ Programming Guide: Memory Hierarchy, Device Memory Accesses, Read-Only Data Cache Load Function.
- NVIDIA PTX ISA: `cp.async`, cache operators `.ca` / `.cg`, `ld.global.nc`.
- NVIDIA Ampere GPU Architecture Tuning Guide: Asynchronous Data Copy from Global Memory to Shared Memory, Unified Shared Memory/L1/Texture Cache.

相关官方链接：

- https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- https://docs.nvidia.com/cuda/parallel-thread-execution/
- https://docs.nvidia.com/cuda/ampere-tuning-guide/

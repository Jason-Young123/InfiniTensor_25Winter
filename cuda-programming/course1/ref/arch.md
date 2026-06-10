# RTX 5090 vs A100 架构面试回答

## 一句话回答

RTX 5090 是 NVIDIA Blackwell 架构的消费级 GeForce GPU，CUDA compute capability 是 12.0，编译目标通常用 `sm_120`。A100 是 NVIDIA Ampere 架构的数据中心 GPU，CUDA compute capability 是 8.0，编译目标通常用 `sm_80`。二者最大的区别不是单纯新旧，而是定位不同：5090 面向图形、游戏、工作站和本地推理，A100 面向数据中心训练、HPC、多租户和高可靠部署。

## 基本信息

| GPU | 架构 | 产品定位 | Compute Capability | 常用 nvcc 架构参数 |
| --- | --- | --- | --- | --- |
| GeForce RTX 5090 | Blackwell | 消费级 / 工作站级 GeForce | 12.0 | `-arch=sm_120` |
| NVIDIA A100 | Ampere | 数据中心 / HPC / AI 训练 | 8.0 | `-arch=sm_80` |

## 主要区别

### 1. 架构代际

A100 属于 Ampere，是 NVIDIA 数据中心 GPU 里非常经典的一代，重点是 Tensor Core、HBM2/HBM2e、高带宽、MIG、多 GPU 互联和数据中心稳定性。

RTX 5090 属于 Blackwell，是更新一代架构。它在消费级产品线上强调更强的图形能力、RT Core、Tensor Core、本地 AI 推理能力、更高显存带宽和新一代 CUDA 指令能力。写 CUDA 时要注意它的目标架构是 `sm_120`，不能继续用默认的老架构，例如 `sm_52`。

### 2. 产品定位

A100 是数据中心卡，设计目标是长时间满载运行、大规模训练、HPC、云端虚拟化和多用户隔离。它通常没有显示输出，部署在服务器里，配合 ECC、MIG、NVLink、NVSwitch 等能力使用。

RTX 5090 是 GeForce 卡，设计目标包括游戏图形、内容创作、个人工作站、本地模型推理和 CUDA 开发。它的峰值算力很强，但平台能力、可靠性特性和多卡互联生态与 A100 不是同一类产品。

### 3. Tensor Core 和数值格式

A100 的 Tensor Core 重点支持 AI 训练和 HPC 常用格式，例如 FP16、BF16、TF32、INT8，以及数据中心场景需要的高吞吐矩阵乘。

RTX 5090 的 Blackwell Tensor Core 是更新一代，面向本地 AI 推理和图形 AI 功能会更激进，通常会支持更新的低精度格式和更高的消费级 AI 吞吐。但在训练场景中，是否比 A100 更好不能只看峰值 Tensor 算力，还要看显存容量、带宽、ECC、互联、软件栈和集群扩展能力。

### 4. 显存和内存系统

A100 使用 HBM 系列显存，特点是高带宽、面向数据中心、容量版本较大，并且通常带 ECC，适合大模型训练、HPC stencil、稀疏访存和长期稳定运行。

RTX 5090 使用 GDDR 系列显存，带宽很高，成本和消费级平台更友好，但和 A100 的 HBM 数据中心内存体系不同。做 CUDA kernel 优化时，5090 的峰值能力很强，但仍然要用 Nsight Compute 看真实瓶颈，例如 global memory throughput、shared memory bank conflict、L2 hit rate、occupancy 和 warp stall reason。

### 5. 多 GPU 和数据中心能力

A100 的优势之一是数据中心扩展能力，例如 NVLink、NVSwitch、MIG、多实例隔离、服务器散热和长期稳定运行。MIG 可以把一张 A100 切成多个隔离的 GPU 实例，适合云服务和多租户部署。

RTX 5090 更偏单机单卡或少量多卡使用，重点不是云端多租户隔离。即使单卡性能很强，也不能直接替代 A100 在数据中心集群里的角色。

### 6. CUDA 编译和性能分析

如果本机是 RTX 5090，编译 CUDA kernel 时应显式指定：

```bash
nvcc -arch=sm_120 ...
```

如果目标是 A100，则使用：

```bash
nvcc -arch=sm_80 ...
```

这一点很重要。比如在 RTX 5090 上如果 ptxas 输出显示：

```text
for 'sm_52'
```

说明编译目标不是当前 GPU 的真实架构。此时寄存器数量、指令选择、occupancy 判断和最终性能都可能不具备代表性。

## 面试可直接回答版本

RTX 5090 和 A100 的核心区别是架构代际和产品定位。RTX 5090 是 Blackwell 架构的消费级 GeForce GPU，compute capability 是 12.0，CUDA 编译一般用 `sm_120`。A100 是 Ampere 架构的数据中心 GPU，compute capability 是 8.0，CUDA 编译一般用 `sm_80`。

从定位上看，A100 是为数据中心训练、HPC 和云端多租户设计的，强调 HBM 高带宽显存、ECC、MIG、NVLink/NVSwitch、长期稳定运行和集群扩展。RTX 5090 则是消费级 Blackwell，强调图形、内容创作、本地 AI 推理和高单卡性能。

所以不能简单说 5090 一定全面强于 A100。5090 架构更新，单卡峰值能力和新特性很强；但 A100 在大规模训练、数据中心可靠性、多 GPU 通信、显存体系和部署生态上仍然有明显优势。做 CUDA 优化时，两者也要分别用对应的 `sm_120` 和 `sm_80` 编译，并结合 Nsight Compute 看寄存器、shared memory、occupancy、memory throughput 和 stall reason，而不是只看理论算力。

## 和 CUDA kernel 优化相关的记忆点

- RTX 5090: Blackwell, `sm_120`, 消费级/工作站级。
- A100: Ampere, `sm_80`, 数据中心/HPC/训练卡。
- 架构参数写错会影响 ptxas 输出和性能判断。
- Theoretical occupancy 主要受 registers/thread、threads/block、shared memory/block 和架构上限影响。
- 同一个 kernel 在 `sm_80` 和 `sm_120` 下可能生成不同指令，寄存器数量也可能不同。
- 判断性能瓶颈时不要只看 occupancy，还要看 achieved occupancy、eligible warps、issue slots、memory throughput、bank conflict、L2 hit rate 和 stall reasons。

## 参考来源

- NVIDIA GeForce RTX 5090 官方规格页：Blackwell 架构，CUDA Capability 12.0。
- NVIDIA A100 Tensor Core GPU 官方资料：Ampere 架构。
- NVIDIA CUDA GPUs 官方 compute capability 列表：A100 为 8.0，RTX 5090 为 12.0。

## 硬件层面的架构区别：Blackwell GB202 vs Ampere GA100

先澄清一个容易混淆的点：RTX 5090 用的是面向 GeForce/RTX 的 Blackwell GB202，A100 用的是面向数据中心的 Ampere GA100。它们不是同一个产品线里的相邻两代卡，所以硬件差异既包含架构代际差异，也包含消费级 GPU 和数据中心 GPU 的取舍差异。

### 1. 芯片组织和 SM 数量

RTX 5090 对应的 GB202 是 RTX Blackwell 家族的旗舰芯片。完整 GB202 包含 12 个 GPC、96 个 TPC、192 个 SM、512-bit GDDR7 memory interface。RTX 5090 实际启用 170 个 SM、21760 个 CUDA Core、680 个第五代 Tensor Core、170 个第四代 RT Core、96 MB L2 cache。

A100 对应的 GA100 是数据中心 Ampere 芯片。完整 GA100 包含 8 个 GPC、128 个 SM、8192 个 FP32 CUDA Core、512 个第三代 Tensor Core、6 组 HBM2 stack。A100 实际启用 108 个 SM、6912 个 FP32 CUDA Core、432 个第三代 Tensor Core、5 组 HBM2 stack。

面试里可以这样说：Blackwell GB202 的规模更偏图形和 AI 推理吞吐，SM 数量和 L2 很大；GA100 的组织更偏 HPC 和数据中心训练，SM 数量少于完整 GB202，但配套 HBM、FP64、MIG、NVLink 和 RAS 能力。

### 2. 单个 SM 的组成不同

RTX Blackwell GB202 的每个 SM 包含：

- 128 个 CUDA Core
- 4 个第五代 Tensor Core
- 1 个第四代 RT Core
- 4 个 Texture Unit
- 256 KB register file
- 128 KB L1/shared memory

A100 GA100 的每个 SM 主要面向计算：

- 64 个 FP32 CUDA Core
- 4 个第三代 Tensor Core
- 192 KB combined L1/shared memory
- 支持异步 global-to-shared copy、异步 barrier、L2 residency control、warp-level reduction
- 面向 HPC 的 FP64 路径明显更强

这说明两者 SM 的设计目标不同。GB202 的 SM 同时服务 raster、ray tracing、neural rendering、AI 推理和通用 CUDA；GA100 的 SM 更集中服务 GEMM、HPC、深度学习训练和数据中心吞吐。

### 3. Tensor Core 代际和数据类型

A100 是第三代 Tensor Core。它的代表性变化是引入 TF32，让很多原本 FP32 的深度学习训练可以直接走 Tensor Core；同时支持 FP16、BF16、TF32、FP64 Tensor Core、INT8、INT4、binary，以及 2:4 structured sparsity。A100 的 Tensor Core 对训练和 HPC 很关键，尤其是 TF32 和 FP64 Tensor Core。

RTX 5090 的 RTX Blackwell 是第五代 Tensor Core。它继续支持 FP16、BF16、TF32、INT8，并加入 FP8 Transformer Engine、FP6、FP4 等更低精度能力。Blackwell 的重点更偏现代 AI 推理和生成式 AI：用 FP4/FP6 降低模型显存占用，提高低精度推理吞吐。

面试里可以这样概括：Ampere A100 的 Tensor Core 关键字是 TF32、BF16、FP64 Tensor Core、structured sparsity，解决的是“训练和 HPC 怎么更快”；Blackwell RTX 5090 的 Tensor Core 关键字是第五代、FP8/FP6/FP4、Transformer Engine，解决的是“生成式 AI 和低精度推理怎么更快、更省显存”。

### 4. FP64 和 HPC 能力差异

A100 是数据中心/HPC GPU，FP64 是硬件设计重点之一。GA100 有强 FP64 CUDA Core 路径，也有 FP64 Tensor Core，用来加速科学计算、线性代数、仿真等双精度工作负载。

RTX 5090 虽然 GB202 也保留少量 FP64 能力用于程序兼容性，但 FP64 吞吐不是设计重点。官方资料中 GB202 的 FP64 rate 是 FP32 的 1/64，这意味着它不适合拿来替代 A100 做严肃 FP64 HPC。

所以如果面试官问“5090 更新，能不能替代 A100 做 HPC”，答案应当是：对 FP32/低精度推理类任务可能很强，但对 FP64 HPC、数据中心多卡训练和高可靠部署，不能简单替代。

### 5. 显存子系统：GDDR7 vs HBM2/HBM2e

RTX 5090 使用 GDDR7。GB202/RTX 5090 是 512-bit memory interface，32 GB GDDR7，峰值带宽约 1.792 TB/s。GDDR7 使用 PAM3 信号技术，相比 GDDR6/GDDR6X 提升了带宽和能效。RTX 5090 还具有较大的 L2 cache，具体到 RTX 5090 是 96 MB。

A100 使用 HBM2/HBM2e。A100 的 GA100 通过多个 HBM stack 和超宽 memory controller 提供高带宽、ECC 和数据中心可靠性。A100 40 GB 版本约 1.555 TB/s，80 GB 版本更高。HBM 的优势不是只看带宽数字，还包括数据中心级容量、可靠性、封装和能效。

对 CUDA 优化来说，这个差异会影响访存策略：5090 的大 L2 + GDDR7 对 cache-friendly 和高并发访问很重要；A100 的 HBM + 数据中心 cache/memory 控制更适合长期满载的大规模训练/HPC。

### 6. L1/shared memory 和异步数据搬运

A100 的 GA100 SM 有 192 KB combined L1/shared memory，并且 Ampere 引入了 `cp.async` 风格的 global-to-shared 异步拷贝能力，可以绕过中间寄存器，配合异步 barrier 做 software pipeline。这是手写高性能 GEMM kernel 时非常重要的硬件能力。

RTX Blackwell GB202 的每个 SM 有 128 KB L1/shared memory。Blackwell 属于更新的 compute capability 12.0，支持更新的 CUDA ISA 和调度能力，但在写普通 CUDA kernel 时仍然要回到具体指标：shared memory 容量、bank conflict、register pressure、occupancy、L2 hit rate 和 memory throughput。

面试里不要只说“新架构一定 shared memory 更强”。更准确的说法是：A100 的 Ampere 在数据搬运上已经有很成熟的异步 copy + shared memory pipeline；Blackwell 更新，但具体到 RTX 5090，它的 L1/shared 容量配置和数据中心 Blackwell 并不完全等价。

### 7. 图形硬件：RT Core、Raster、Texture 是 5090 的重点，A100 不是

RTX 5090 有第四代 RT Core、Raster Engine、ROP、Texture Unit 等完整图形管线硬件。Blackwell 还强化了 neural rendering、DLSS 4、Shader Execution Reordering、AI Management Processor 等图形/AI 混合工作负载相关能力。

A100 是计算卡，核心目标不是图形渲染。它通常没有显示输出，也不强调 RT Core、游戏图形管线和 DLSS 这类能力。A100 的硬件预算更多给了 HBM、FP64、Tensor Core、MIG、NVLink 和数据中心 RAS。

这也是为什么“Blackwell vs Ampere”不能脱离具体芯片。RTX Blackwell 的很多增强是图形 + AI 推理导向；A100 Ampere 是计算 + 训练 + HPC 导向。

### 8. 多实例、多卡互联和可靠性

A100 有 MIG，可以把一张 GPU 切成多个隔离的 GPU instance；还有第三代 NVLink、数据中心 RAS、ECC 和面向云服务的隔离能力。这些都是硬件/平台级能力，不是单个 CUDA kernel 能弥补的。

RTX 5090 更适合单机本地使用。它的单卡吞吐很强，但没有 A100 那种面向数据中心多租户和大规模多 GPU 拓扑的硬件定位。即使 5090 的某些低精度算力很高，训练集群里仍然要考虑互联、显存容量、一致性、可靠性和调度隔离。

### 9. 对手写 CUDA kernel 的实际影响

如果你写的是普通 SIMT CUDA kernel，例如手写 FP32 matmul，不使用 Tensor Core，那么要重点看：

- `sm_120` 和 `sm_80` 下 ptxas 生成的寄存器数量可能不同。
- Blackwell GB202 的每 SM CUDA Core 数量、L1/shared 配置、L2 容量和调度能力与 A100 不同。
- A100 对 FP64/HPC 更友好，RTX 5090 对 FP32、图形、AI 推理和低精度 Tensor 更友好。
- shared memory bank conflict、global memory coalescing、occupancy、tail effect 在两种架构上都要用 Nsight Compute 实测。
- 如果不用 Tensor Core，Blackwell 第五代 Tensor Core 的 FP4/FP6 优势不会自动体现在普通 `float` FMA kernel 上。
- 如果要发挥 A100 或 Blackwell 的矩阵乘硬件能力，应该进一步学习 WMMA、MMA、CUTLASS、Tensor Core、`cp.async`/pipeline，而不是只优化 scalar CUDA core 版本。

### 10. 面试回答：硬件版

硬件层面看，RTX 5090 的 Blackwell GB202 和 A100 的 Ampere GA100 差别很大。RTX 5090 是面向 GeForce/RTX 的 Blackwell，SM 数量更多，单 SM 有 128 个 CUDA Core、第五代 Tensor Core、第四代 RT Core、较大的 L2 cache 和 GDDR7 显存，重点是图形、neural rendering、本地 AI 推理和低精度 AI，例如 FP8、FP6、FP4。

A100 是面向数据中心的 Ampere GA100，单 SM 是 64 个 FP32 CUDA Core、第三代 Tensor Core、192 KB L1/shared memory，并且有强 FP64、TF32、BF16、structured sparsity、HBM、MIG、NVLink 和数据中心 RAS。它的硬件目标是训练、HPC、多租户和多 GPU 扩展。

所以 Blackwell 更新不等于 RTX 5090 在所有硬件维度都替代 A100。5090 在消费级图形、FP32/低精度推理和本地 AI 上很强；A100 在 FP64、HBM 显存体系、训练稳定性、多卡互联和数据中心部署上仍然是另一类硬件。

## CUDA 中 PTX、SASS、JIT 的区别

### 1. 一句话回答

PTX 是 NVIDIA CUDA 的虚拟 ISA，中间表示，面向 `compute_xx`；SASS 是针对具体 GPU 架构的真实机器指令，面向 `sm_xx`；JIT 是运行时由 NVIDIA driver 把 PTX 编译成当前 GPU 可执行 SASS 的过程。

### 2. CUDA 编译链路

一个 CUDA kernel 大致会经历：

```text
CUDA C++ source
    -> NVVM IR
    -> PTX
    -> SASS / cubin
    -> GPU execute
```

平时用 `nvcc` 编译时，可能同时生成两类代码并打包进 fatbin：

```text
PTX:  面向虚拟架构，例如 compute_120
SASS: 面向真实架构，例如 sm_120
```

如果 fatbin 里已经有当前 GPU 对应的 SASS，driver 通常直接加载执行。如果没有匹配的 SASS，但有兼容的 PTX，driver 会在运行时 JIT 编译 PTX，生成当前 GPU 的 SASS 再执行。

### 3. PTX 是什么

PTX 全称是 Parallel Thread Execution。它是 NVIDIA 定义的虚拟指令集，可以理解为 CUDA 的中间汇编语言。

PTX 的特点：

- 面向虚拟架构，例如 `compute_80`、`compute_120`。
- 不是最终硬件直接执行的机器码。
- 可读性比 SASS 高，能看到 load/store、fma、barrier、shared memory 等逻辑操作。
- 具有一定前向兼容性。老程序里如果包含 PTX，将来在新 GPU 上可能由新 driver JIT 成新架构的 SASS。
- PTX 仍然是 NVIDIA 平台相关的，不是跨厂商通用 IR。

常见命令：

```bash
nvcc -ptx kernel.cu -o kernel.ptx
```

或者：

```bash
nvcc -arch=compute_120 -ptx kernel.cu
```

面试里可以说：PTX 类似 CUDA 世界里的“虚拟汇编”或“中间表示”，保留了 GPU 线程、内存空间和同步语义，但还没有完全绑定到某一代 GPU 的真实指令编码和调度细节。

### 4. SASS 是什么

SASS 是 NVIDIA GPU 的真实机器指令，类似 CPU 上最终执行的 assembly/machine code。它是针对具体 SM 架构生成的，例如 `sm_80`、`sm_89`、`sm_120`。

SASS 的特点：

- 面向真实硬件架构，例如 A100 是 `sm_80`，RTX 5090 是 `sm_120`。
- 是 GPU 真正执行的指令形式。
- 包含更具体的指令选择、寄存器分配、调度信息、指令编码和硬件相关细节。
- 不保证跨架构兼容。`sm_80` 的 SASS 不能直接拿到 `sm_120` 上当作通用机器码使用。

常见查看方式：

```bash
cuobjdump --dump-sass ./your_binary
```

或者：

```bash
nvdisasm your_kernel.cubin
```

面试里可以说：SASS 是最终落到 GPU 上跑的机器码。分析极限性能、指令条数、是否用了 Tensor Core 指令、是否产生了特定 load/store 指令时，看 SASS 比看 PTX 更接近真实执行。

### 5. JIT 是什么

JIT 是 Just-In-Time compilation。在 CUDA 里，通常指 driver 在程序运行时把 PTX 编译成当前 GPU 对应的 SASS。

典型场景：

- binary 里没有当前 GPU 的 SASS。
- binary 里包含 PTX fallback。
- 程序第一次在某个 GPU 上运行时，driver JIT 编译 PTX。
- JIT 结果可能被缓存，后续运行不一定每次都重新编译。

例如程序只带了：

```text
compute_80 PTX
```

但运行在 RTX 5090 上，driver 可能把这份 PTX JIT 成：

```text
sm_120 SASS
```

前提是 PTX 版本和 driver 支持足够新。

JIT 的优点：

- 提供前向兼容性。
- 老 binary 如果带 PTX，可能能在新 GPU 上运行。
- 新 driver 可以针对新 GPU 生成更合适的 SASS。

JIT 的缺点：

- 首次运行有编译开销。
- JIT 结果受 driver 版本影响，可重复性比离线 cubin 差一些。
- 如果 PTX 太老，可能无法表达新架构的全部硬件能力。
- 性能分析时，看到的最终 SASS 可能不是你离线 ptxas 生成的那份。

### 6. `compute_xx` 和 `sm_xx` 的区别

`compute_xx` 表示虚拟架构，通常对应 PTX 目标。

```bash
-arch=compute_120
```

意思是生成面向 compute capability 12.0 的 PTX。

`sm_xx` 表示真实 GPU 架构，通常对应 SASS/cubin 目标。

```bash
-arch=sm_120
```

意思是直接为 compute capability 12.0 的真实 GPU 生成机器码。

更完整的写法常用 `-gencode`：

```bash
nvcc \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_120,code=compute_120 \
  kernel.cu
```

第一行生成 `sm_120` SASS，第二行保留 `compute_120` PTX 作为 fallback。

### 7. 为什么编译架构会影响 ptxas 输出

`ptxas` 是把 PTX 编成 SASS 的离线编译器。它针对不同 `sm_xx` 会做不同的指令选择、寄存器分配和调度。

所以同一个 kernel：

```bash
nvcc -arch=sm_80  -Xptxas -v kernel.cu
nvcc -arch=sm_120 -Xptxas -v kernel.cu
```

可能得到不同的：

- registers per thread
- shared memory usage
- spill stores / spill loads
- 指令条数
- 是否使用某些新架构指令

因此在 RTX 5090 上分析性能时，如果 ptxas 输出是：

```text
for 'sm_52'
```

这个结果就不适合作为 5090 的最终判断。应该显式用：

```bash
nvcc -arch=sm_120 -Xptxas -v ...
```

### 8. 面试可直接回答版本

CUDA 里的 PTX、SASS、JIT 可以按编译链路来理解。CUDA C++ 先被编译成 PTX，PTX 是 NVIDIA 的虚拟 ISA，面向 `compute_xx`，类似中间汇编，主要提供可移植性和前向兼容。然后 PTX 会被 ptxas 或 driver 编译成 SASS，SASS 是针对具体 GPU 的真实机器指令，面向 `sm_xx`，是真正被 GPU 执行的代码。

JIT 指的是运行时编译。如果程序里没有当前 GPU 对应的 SASS，但包含兼容 PTX，NVIDIA driver 会在程序运行时把 PTX 编译成当前 GPU 的 SASS。这样程序可以在未来 GPU 上运行，但首次运行会有 JIT 开销，而且性能和最终指令可能受 driver 版本影响。

所以做性能优化时，PTX 可以用来看逻辑上的内存访问、同步和大致指令形态；但最终性能要看 SASS、ptxas 输出和 Nsight Compute。编译时也要区分 `compute_120` 和 `sm_120`：前者是 PTX 虚拟架构，后者是 RTX 5090 这类 Blackwell GPU 的真实机器码目标。

### 9. 常见面试追问

Q: PTX 能不能直接在 GPU 上执行？

A: 不能。GPU 最终执行的是 SASS。PTX 需要先被 ptxas 或 driver JIT 编译成对应 GPU 架构的 SASS。

Q: 为什么 binary 里要同时放 SASS 和 PTX？

A: SASS 用于当前已知架构，启动快、结果稳定；PTX 用作 fallback，提供前向兼容，让程序有机会在未来 GPU 上通过 JIT 运行。

Q: 为什么我在 5090 上不能用 `sm_52` 的 ptxas 结果判断性能？

A: 因为 `sm_52` 是 Maxwell 时代的真实架构，不是 Blackwell。寄存器分配、指令选择、调度和硬件资源都不同。在 5090 上应该用 `sm_120` 编译和分析。

Q: PTX 和 SASS 哪个更适合性能分析？

A: 两者都可以看，但层次不同。PTX 适合看程序逻辑和编译器前端生成的大致操作；SASS 更接近真实执行，适合看最终指令、寄存器、load/store 形态和 Tensor Core 指令。真正判断瓶颈还要结合 Nsight Compute。

## CUDA 版本、CC、架构代号、PTX、SASS、fatbin、cubin 的关系

### 1. 一句话总览

CUDA Toolkit 版本是软件工具链版本；Compute Capability 是 GPU 硬件能力编号；Ampere、Ada Lovelace、Hopper、Blackwell 是 NVIDIA 的架构代号；`compute_xx` 是 PTX 虚拟架构目标；`sm_xx` 是真实 GPU 机器码目标；`.ptx` 是虚拟 ISA 文本；SASS 是真实机器指令；`.cubin` 是某个 `sm_xx` 的机器码文件；`.fatbin` 是把多个 PTX/cubin 打包在一起的容器。

如果把 CUDA 编译链路画出来：

```text
CUDA C++ source
    -> nvcc / NVVM
    -> PTX              对应 compute_xx，虚拟架构
    -> ptxas
    -> SASS in cubin    对应 sm_xx，真实机器码
    -> fatbin           打包多个 cubin 和 PTX
    -> CUDA driver 加载
    -> GPU 执行 SASS
```

注意：你写的 `.fabin` 一般应指 `.fatbin`，也就是 fat binary。

### 2. CUDA 版本到底指什么

面试中说 CUDA 版本时，至少要区分三件事：

| 概念 | 看什么命令 | 含义 |
| --- | --- | --- |
| CUDA Toolkit version | `nvcc --version` | 本机安装的编译器、头文件、库版本 |
| CUDA Driver support version | `nvidia-smi` 里的 CUDA Version | 当前 NVIDIA driver 最高支持的 CUDA runtime 能力 |
| CUDA Runtime library version | 程序链接/加载的 `libcudart` | 程序运行时用到的 CUDA runtime 库 |

常见误区：`nvidia-smi` 显示的 CUDA Version 不等于你安装的 CUDA Toolkit 版本。它表示当前 driver 最高支持到哪个 CUDA runtime 版本。例如 `nvidia-smi` 可能显示 CUDA 12.x，但你的 `/usr/local/cuda` 可能安装的是另一个版本。

CUDA 版本影响：

- `nvcc` 是否认识新的 `sm_xx`，例如 `sm_120`。
- PTX ISA 版本是否足够新。
- driver 是否能 JIT 这份 PTX。
- CUDA runtime/library API 是否可用。

但 CUDA 版本不是 GPU 架构本身。GPU 架构由硬件决定，例如 A100 是 Ampere，RTX 5090 是 Blackwell。

### 3. Compute Capability 是什么

Compute Capability，简称 CC，是 NVIDIA 给 GPU 硬件能力定义的数字编号，例如：

```text
A100       -> 8.0
RTX 4090   -> 8.9
H100       -> 9.0
RTX 5090   -> 12.0
```

CC 决定了 CUDA 编程中很多硬件能力：

- 支持哪些指令。
- 支持哪些 Tensor Core 数据类型。
- shared memory、register、warp、block 等资源上限。
- 是否支持某些异步 copy、barrier、cluster、TMA 等新特性。
- 编译时应该用什么 `sm_xx`。

例如 RTX 5090 的 CC 是 12.0，所以编译目标通常是：

```bash
nvcc -arch=sm_120 ...
```

A100 的 CC 是 8.0，所以编译目标通常是：

```bash
nvcc -arch=sm_80 ...
```

面试里可以说：CC 是 CUDA 层面对硬件能力的数字抽象，架构代号是 NVIDIA 产品/微架构命名，两者相关但不是完全一回事。

### 4. 架构代号不是虚拟架构

Ampere、Ada Lovelace、Hopper、Blackwell 是架构代号，不是 `nvcc` 所说的 virtual architecture。

大致对应关系可以这样记：

| 架构代号 | 典型 GPU | 常见 CC / `sm_xx` |
| --- | --- | --- |
| Ampere | A100 | 8.0 / `sm_80` |
| Ampere | RTX 30 系列部分 GPU | 8.6 / `sm_86` |
| Ada Lovelace | RTX 40 系列 | 8.9 / `sm_89` |
| Hopper | H100 | 9.0 / `sm_90` |
| Blackwell | RTX 50 系列，如 RTX 5090 | 12.0 / `sm_120` |

这张表只是帮助理解，实际项目里应该查 NVIDIA CUDA GPUs 官方列表。原因是同一个架构代号下可能有不同 CC，例如 Ampere 的 A100 是 `sm_80`，消费级 RTX 30 系列常见是 `sm_86`。同样叫 Blackwell，也可能因为产品线不同而有不同能力配置。

### 5. 虚拟架构 `compute_xx`

`compute_xx` 是 virtual architecture，通常用于生成 PTX。

例如：

```bash
nvcc -arch=compute_120 -ptx kernel.cu
```

会生成面向 compute capability 12.0 的 PTX。这里的 `compute_120` 不是具体 GPU 机器码，而是一个虚拟目标，表达“我需要 12.0 这一级别的 CUDA/PTX 能力”。

特点：

- 生成 PTX。
- 有一定前向兼容意义。
- 可以让未来 driver 在新 GPU 上 JIT。
- 不是最终 GPU 直接执行的代码。

面试里可以这样说：`compute_xx` 关注“PTX 层能用什么语义和指令能力”，而不是“哪块 GPU 直接执行它”。

### 6. 实际架构 `sm_xx`

`sm_xx` 是 real architecture，也就是针对具体 Streaming Multiprocessor 版本生成 SASS。

例如：

```bash
nvcc -arch=sm_120 kernel.cu
```

表示生成 RTX 5090 这类 CC 12.0 GPU 可以直接执行的机器码。

特点：

- 生成 SASS/cubin。
- 绑定具体硬件架构。
- 启动时通常不需要 JIT。
- 性能和指令选择更稳定。
- 跨大架构通常不能直接通用。

面试里可以这样说：`compute_xx` 是 PTX 虚拟目标，`sm_xx` 是真实硬件目标。性能优化时应该看目标 GPU 对应的 `sm_xx` 编译结果。

### 7. `.ptx` 文件

`.ptx` 是 PTX 文本文件。它是 NVIDIA 的虚拟 ISA，可以读到类似下面的信息：

- 使用了哪些 global/shared/local memory load/store。
- 是否有 barrier、sync、atomic。
- 大致算术指令形态，例如 `fma`。
- kernel 参数、寄存器声明、地址空间。

生成方式：

```bash
nvcc -arch=compute_120 -ptx kernel.cu -o kernel.ptx
```

PTX 的用途：

- 作为 driver JIT 的输入。
- 用于观察编译器前端生成的中间代码。
- 提供一定前向兼容。

限制：

- 不是最终执行代码。
- 不等于真实指令条数。
- 不能完全反映寄存器分配和最终调度。
- 新硬件的真实性能要看 SASS 和 Nsight Compute。

### 8. SASS

SASS 是 NVIDIA GPU 最终执行的机器指令。它通常存在于 cubin 里，可以用工具反汇编查看。

查看方式：

```bash
cuobjdump --dump-sass ./matmul
```

或者：

```bash
nvdisasm kernel.cubin
```

SASS 能看到更真实的信息：

- 最终 load/store 指令。
- FFMA、HMMA、MMA 等真实计算指令。
- 寄存器编号和使用情况。
- 分支、predicate、barrier 等真实指令。
- 是否真的使用了 Tensor Core 指令。

性能分析时，SASS 比 PTX 更可信。PTX 里看到一个操作，不代表最后 SASS 里就是一条同样的指令；编译器可能合并、拆分、重排或者替换指令。

### 9. `.cubin` 文件

`.cubin` 是 CUDA binary 文件，里面主要装的是针对某个 `sm_xx` 生成的机器码，也就是 SASS 以及相关元数据。

生成方式示例：

```bash
nvcc -arch=sm_120 -cubin kernel.cu -o kernel.cubin
```

特点：

- 面向具体 `sm_xx`。
- 通常启动更快，因为不需要 JIT。
- 适合固定部署环境，例如明确只跑 A100 或只跑 RTX 5090。
- 兼容性不如 PTX。不同大架构之间通常不能指望 cubin 通用。

面试里可以说：cubin 是离线编译好的 GPU 机器码容器，适合性能稳定和启动快；PTX 是 JIT fallback，适合前向兼容。

### 10. `.fatbin` 文件

`.fatbin` 是 fat binary，里面可以同时包含多份 cubin 和 PTX。实际可执行文件里常常会嵌入 fatbin。

例如可以同时打包：

```text
sm_80 cubin       给 A100 直接执行
sm_90 cubin       给 H100 直接执行
sm_120 cubin      给 RTX 5090 直接执行
compute_120 PTX   给未来 GPU JIT fallback
```

常见编译方式：

```bash
nvcc \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_120,code=compute_120 \
  kernel.cu
```

这里：

- `code=sm_80` 生成 A100 可直接执行的 SASS/cubin。
- `code=sm_120` 生成 RTX 5090 可直接执行的 SASS/cubin。
- `code=compute_120` 保留 PTX，作为未来架构的 JIT fallback。

面试里可以说：fatbin 解决的是“一个程序支持多代 GPU”的问题。运行时 driver 会从 fatbin 里挑最合适的代码：优先用匹配的 cubin，没有就尝试用 PTX JIT。

### 11. 运行时 driver 如何选择代码

假设程序 fatbin 里有：

```text
sm_80 cubin
sm_90 cubin
compute_120 PTX
```

运行在 A100 上：

```text
A100 是 sm_80 -> 直接加载 sm_80 cubin
```

运行在 H100 上：

```text
H100 是 sm_90 -> 直接加载 sm_90 cubin
```

运行在 RTX 5090 上：

```text
RTX 5090 是 sm_120
如果没有 sm_120 cubin，但有可用 PTX -> driver JIT PTX 到 sm_120 SASS
```

如果既没有匹配 cubin，也没有可 JIT 的 PTX，程序就会报类似：

```text
no kernel image is available for execution on the device
```

### 12. 面试可直接回答版本

CUDA 里这些概念可以按“软件版本、硬件能力、编译目标、代码产物”四层理解。

CUDA Toolkit 版本是软件工具链版本，决定 `nvcc`、库和 PTX 支持到什么程度；Compute Capability 是 GPU 的硬件能力编号，比如 A100 是 8.0，H100 是 9.0，RTX 5090 是 12.0；Ampere、Ada、Hopper、Blackwell 是 NVIDIA 的架构代号，不等于虚拟架构。真正的虚拟架构是 `compute_xx`，用于生成 PTX；真实机器码架构是 `sm_xx`，用于生成 SASS/cubin。

`.ptx` 是虚拟 ISA 文本，不能直接执行，但可以被 driver JIT 成目标 GPU 的机器码；SASS 是 GPU 最终执行的真实机器指令；`.cubin` 是某个 `sm_xx` 的 SASS 机器码容器；`.fatbin` 是把多份 cubin 和 PTX 打包在一起，让同一个程序可以支持多种 GPU。

运行时 driver 会优先找当前 GPU 匹配的 cubin，比如 RTX 5090 找 `sm_120`；如果找不到，但有兼容 PTX，就 JIT 编译 PTX；如果两者都没有，就会报 no kernel image。做性能优化时，PTX 适合看逻辑，SASS 和 Nsight Compute 才更接近真实性能。

### 13. 最可能被问到的问题

Q: CUDA version 和 Compute Capability 是一回事吗？

A: 不是。CUDA version 是软件工具链/driver/runtime 版本；Compute Capability 是 GPU 硬件能力编号。比如 RTX 5090 的 CC 是 12.0，但你可以用不同 CUDA Toolkit 版本编译，只要该版本支持 `sm_120` 且 driver 足够新。

Q: `nvidia-smi` 里的 CUDA Version 是不是我安装的 CUDA Toolkit？

A: 不是。`nvidia-smi` 显示的是当前 driver 最高支持的 CUDA runtime 版本。Toolkit 要看 `nvcc --version` 或 `/usr/local/cuda/version.txt`。

Q: Ampere、Hopper、Blackwell 是虚拟架构吗？

A: 不是。它们是 NVIDIA 的架构代号。CUDA 编译里的虚拟架构是 `compute_xx`，例如 `compute_80`、`compute_120`；真实机器码架构是 `sm_xx`，例如 `sm_80`、`sm_120`。

Q: `compute_120` 和 `sm_120` 有什么区别？

A: `compute_120` 表示面向 CC 12.0 的 PTX 虚拟目标；`sm_120` 表示面向 CC 12.0 GPU 的真实机器码目标。前者生成 PTX，后者生成 SASS/cubin。

Q: PTX 和 SASS 的区别是什么？

A: PTX 是虚拟 ISA，中间表示，不能直接在 GPU 上执行；SASS 是具体 GPU 的真实机器指令，GPU 最终执行的是 SASS。

Q: `.ptx`、`.cubin`、`.fatbin` 分别是什么？

A: `.ptx` 是 PTX 文本；`.cubin` 是某个 `sm_xx` 的机器码容器；`.fatbin` 是 fat binary，可以包含多份 cubin 和 PTX，用来支持多架构运行。

Q: JIT 什么时候发生？

A: 当程序没有当前 GPU 对应的 cubin，但 fatbin 里有兼容 PTX 时，driver 会在运行时把 PTX JIT 成当前 GPU 的 SASS。

Q: 为什么发布程序时通常既放 cubin 又放 PTX？

A: cubin 给已知 GPU 直接执行，启动快、性能稳定；PTX 作为 fallback，提供前向兼容，让未来 GPU 有机会通过 driver JIT 运行。

Q: 为什么不能拿 `sm_52` 的 ptxas 输出分析 RTX 5090？

A: 因为 RTX 5090 是 Blackwell，目标是 `sm_120`。不同 `sm_xx` 的寄存器分配、指令选择和调度都可能不同。分析 5090 应该用 `-arch=sm_120 -Xptxas -v`。

Q: 如果只有 `sm_80` cubin，能不能跑在 RTX 5090 上？

A: 不应该依赖这种跨大架构 cubin 兼容。正确做法是给 RTX 5090 编译 `sm_120` cubin，或者至少保留合适 PTX 让 driver JIT。否则可能出现 no kernel image 或性能不可控。

Q: 性能优化时应该看 PTX 还是 SASS？

A: PTX 适合看编译前端生成的逻辑形态；SASS 更接近真实执行。最终性能判断要看 SASS、ptxas 资源报告和 Nsight Compute 指标。

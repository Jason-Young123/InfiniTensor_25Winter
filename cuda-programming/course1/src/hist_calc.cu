/*
### 9. 并行直方图统计 (Histogram Computation)

题目：给定一个包含大量随机整数（取值范围 `0-255`）的数组，统计每个数字出现的频率。

面试官附加限制：

- 必须使用原子操作（Atomic Operations）。
- 面试官会问：如果直接在 Global Memory 上做 `atomicAdd`，会有什么性能灾难？你应该如何利用 Shared Memory 进行两阶段的局部直方图合并？
*/
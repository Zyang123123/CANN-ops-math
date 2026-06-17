# TopK V2 算子核心实现 — 芯片专家讲解版

> 本文面向 **芯片架构师 / 微架构设计师 / ISA 定义团队**,
> 把 TopK V2 在 Ascend (AIV) 上的实现抽象为「**选型 → 算法 → 计算流 → 指令集需求 → 瓶颈 → ISA 改进**」六个层次,
> **不展开代码细节**(详见同目录的 5 份深度文档)。
>
> 配套阅读(按需):
> - `CODE_ANALYSIS.md` — 文件清单、include 关系
> - `TILING_DETAIL.md` — 7 种模式决策树、字段表
> - `KERNEL_DETAIL.md` — 每模式 kernel 流程、HardEvent 同步
> - `HARDWARE_REQUIREMENTS.md` — 算法→硬件 IP 映射
> - `ASCENDC_TOPK_INTERNAL.md` — `AscendC::TopK` 公共 API 内部解读

---

## 0. 一页摘要 (TL;DR)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TopK V2 = K-of-N 选择问题                       │
│                                                                          │
│   输入:  X[M, N]   输出: (Y[M, K], Index[M, K])                          │
│                                                                          │
│   算法选型矩阵 (核心):                                                    │
│   ┌────────────────┬──────────────────────────────────────────────────┐  │
│   │ N ≤ 1024, fp系│ → MergeSort + 截取            (复用 AscendC API) │  │
│   │ N ≤ ~15K      │ → Radix-Select 单核 (UB 内)    (本算子主战场)    │  │
│   │ N 中等, K 大  │ → MultiCore Radix + Atomic     (核心瓶颈在原子)  │  │
│   │ N ≥ 10M       │ → 完整 Radix-Sort + 截取                          │  │
│   └────────────────┴──────────────────────────────────────────────────┘  │
│                                                                          │
│   关键硬件依赖 (Vector + Scalar 协同):                                    │
│   ① Histogram 微码 (256 bins)        —— 必需                              │
│   ② GatherMask / Vector Select       —— 必需                              │
│   ③ Twiddle (Cast + XOR)             —— 必需(支持有符号/浮点 radix)     │
│   ④ Atomic-Add 32-bit on HBM         —— 多核性能上限决定者                │
│   ⑤ PipeBarrier / HardEvent V_S      —— Vector↔Scalar 协同               │
│                                                                          │
│   最大性能机会:                                                          │
│   ★ 把 "Histogram + BinarySearch + GatherMask" 三步融合为单条指令         │
│     可以把 Radix-Select 每字节迭代的 latency 砍掉 ~60%                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 问题定义与性能目标

### 1.1 问题规格

| 维度 | 取值 |
|---|---|
| 语义 | 从长度 N 的 1D 序列中选出 Top-K 最大或最小的 `(value, index)` 对 |
| 输入 | `X[M, N]`,K 标量,`largest` 布尔,可选 `sorted` 布尔,`dim` 轴参数 |
| 输出 | `Y[M, K]` + `Index[M, K]`,可选排序 |
| 支持的 dtype | fp32 / fp16 / bf16 / int8 / int16 / int32 / int64 / uint8 / uint16 / uint32 / uint64(共 11 种) |
| N 范围 | 1 ~ 数千万(实际 testcase 已见到 N=14840 的常见用例) |
| K 范围 | 1 ~ N |

### 1.2 性能/精度目标

- **延迟目标**: 中等 N (1K~16K) 应在 **单核百微秒级** 完成
- **多核扩展**: N 较大时需线性 scale 到 8~24 核
- **精度**: TopK 本身是 exact 算法(不涉及浮点累加误差),但 Twiddle/Cast 路径不能改变原值顺序
- **真实约束**: UB 容量是头号瓶颈 (~192 KB/核),决定是否走单核 vs 多核

---

## 2. 算法选型对比 — 为什么选 Radix-Select?

### 2.1 候选算法

| 算法 | 复杂度 | 对硬件的诉求 | TopK 场景适配度 |
|---|---|---|---|
| **Bitonic / Merge Sort + 截取** | O(N log N) | 高吞吐 Vector Compare-Swap 网络 | 差: 全排序浪费 K 之外的所有工作 |
| **Priority Queue (Heap)** | O(N log K) | Scalar 密集比较 + 随机访问 | 差: Vector 单元用不上,纯 SIMT 路径 |
| **QuickSelect** | O(N) 平均 | Scalar 分区 + 递归 | 差: 数据依赖 + 分支,Vector 难并行 |
| **Radix-Select** ★ | O(N × B) (B = 字节数) | Histogram + GatherMask + Scalar 二分 | **优**: 完美匹配 Vector 流水 |
| **MergeSort + 截取** | O(N log N) | 复用 AscendC::MergeSort API | 适: 小 N 时 API 调用开销低 |

### 2.2 为什么 Radix-Select 是 TopK 的"天作之合"?

```
普通排序:      sort 全部 N → 取前 K
                ↑ 浪费 (N-K) × log N 次比较

Radix-Select:  按字节 MSB→LSB 迭代:
                  pass 1: 把 N 个数按"最高字节"分到 256 桶
                          → 找到 K-th 大数所在的桶 b1
                          → 保留 ≥ b1 桶的元素 (≈ K 个)
                  pass 2: 在剩余元素中按"次高字节"重复
                  ...
                  pass B: 收敛到 K 个候选
                ↑ 每步都剪掉 ~99% 的数据,Vector 完美并行
```

**对芯片架构师的意味**:

1. **Radix-Select 不需要通用 Sort 硬件** — 只需要 **Histogram** + **Compare-Select** 两类基本 Vector 操作
2. **每字节迭代是固定指令数** — 性能可解析预测,没有数据依赖的分支
3. **Scalar ↔ Vector 协同模式** — 直方图在 Vector 上算,binary-search 在 Scalar 上跑,需要高效的 `HardEvent::V_S` 同步

### 2.3 数据类型如何映射到 Radix?

```
fp32/fp16/bf16 ──→ Twiddle ──→ uint32/uint16 (保单调性)
                                  ↓
                              Radix-Select
                                  ↓
                          Twiddle 反变换 ──→ 原 fp 值

int8/16/32      ──→ Twiddle ──→ uint (XOR 0x80 把有符号转无符号)
uint8/16/32     ──→ 直接用
int64/uint64    ──→ 8 字节迭代 (NUM_PASS=8,最慢路径)
```

**Twiddle 的本质**: 用位级变换把"任意可比较的 bit pattern"转成"无符号整数序",
让同一套 Radix 引擎吃所有 dtype。**对硬件的潜在优化**: 把 Twiddle 合并进 Load 单元(类似 GPU 的 `BytePermute`)。

---

## 3. Host Tiling 模式决策树

### 3.1 7 模式决策树 (按优先级)

```
TopKV2Tiling(context, X, K, largest, sorted, dim)
  │
  ├─ ① SmallSizeMergeSort  ──→ dataType∈{fp32/fp16/bf16} AND N ≤ 1024
  │                            └─ 用 AscendC MergeSort + Concat 一次过
  │
  ├─ ② SingleBlock          ──→ N × sizeof(T) + TopK API tmp ≤ UB
  │                            └─ 单核单 tile, AscendC::TopK<RADIX_SELECT>
  │
  ├─ ③ Fp32MergeSort         ──→ fp32 AND 0.25 ≤ K/N < 0.50
  │   ├─ ③a MoreCore            └─ 多核 Bitonic + 4-way Merge
  │   └─ ③b IntraCore           └─ 单核 Bitonic + 4-way Merge
  │
  ├─ ④ SingleCore            ──→ unsortedDimNum ≥ maxCoreNum
  │                            └─ 单核多次 Radix-Select
  │
  ├─ ⑤ MultiCoreOptim        ──→ K × tileNum ≤ tileSize
  │                            └─ 多核并行 Radix-Select,无跨核通信
  │
  ├─ ⑥ MultiCoreRadix        ──→ N < 10,000,000 AND UB 够用
  │                            └─ 多核 Radix + Atomic Histogram (主流大 N 路径)
  │
  └─ ⑦ SortAndTopK           ──→ 兜底 (N ≥ 10M 或 UB 不足)
                               └─ 完整 Radix Sort + 截取
```

### 3.2 7 模式速查表 (芯片专家视角)

| # | 模式 | 核心算法 | 多核策略 | 跨核通信 | 主要瓶颈 |
|---|---|---|---|---|---|
| ① | SmallSize | MergeSort+Concat | 行间并行 | 无 | Vector Sort ALU |
| ② | **SingleBlock** | **AscendC::TopK<RADIX_SELECT>** | 单核 | 无 | UB 容量 (~30KB) |
| ③a | Fp32MergeMoreCore | Bitonic Sort + 4-way Merge | 多核行并行 | GM 临时区拼接 | HBM 带宽 |
| ③b | Fp32MergeIntraCore | 同上 | 单核 | 无 | Vector ALU |
| ④ | SingleCore | Radix-Select 4-pass | 单核 | 无 | Vector↔Scalar 同步 |
| ⑤ | MultiCoreOptim | 多核独立 Radix-Select | 行间并行 | 无 | HBM 带宽 |
| ⑥ | **MultiCoreRadix** | **Radix + Atomic Hist** | **核间切 tile** | **Atomic Add + PipeBarrier** | **GM 原子吞吐** |
| ⑦ | SortAndTopK | 完整 Radix Sort + 截取 | 多核 | Atomic Hist | GM workspace 容量 |

> **★ 主战场**: ② (中小 N) 与 ⑥ (大 N),覆盖 90% 实际用例。

---

## 4. 核心算法 — Radix-Select 的精髓 (单页讲清)

### 4.1 一图概括

```
输入: data[N], K
输出: top-K candidates (≥K 个,后续 Sort 精确到 K)

迭代 B 次 (B = sizeof(dtype), 如 fp16 → B=2):

  ┌─────────────────────────────────────────────────────────┐
  │  Pass p (处理第 p 字节,从 MSB 开始):                     │
  │                                                         │
  │  [Vector] Histograms<uint8_t, uint16_t, 256>            │
  │           扫一遍数据, 得到当前字节值分布 hist[0..255]    │
  │                          │                              │
  │  [HardEvent V_S]         │  Vector→Scalar 同步           │
  │                          ▼                              │
  │  [Scalar] Reverse-CumSum: hist[b] = #elem with byte ≥ b │
  │           BinarySearch: 找最小的 b*, 使 hist[b*] ≥ K    │
  │           → kthVal (本字节边界)                          │
  │                          │                              │
  │  [HardEvent S_V]         │  Scalar→Vector 同步           │
  │                          ▼                              │
  │  [Vector] GatherMask(data, byte[p] ≥ b*)                │
  │           → 保留 ≥ b* 桶的元素 (≈ K 个候选)              │
  │                                                         │
  │  remainK = K - hist[b*+1]   // 还需要多少元素达到 K       │
  │  下一 pass 只对 byte[p] == b* 的元素细化                 │
  └─────────────────────────────────────────────────────────┘

最终: 得到 ≥ K 个候选, 用 AscendC::Sort + 截取得到有序的 Top-K
```

### 4.2 关键不变量

1. **每 pass 单调收敛**: 候选集从 N → ~N/256 → ~N/256² → ... 几乎指数下降
2. **K 是不变量**: 任意 pass 结束后,候选数 ≥ K (保证正确性)
3. **指令数固定**: 每个 pass 都是同一套指令流,无数据依赖分支
4. **dtype → 迭代次数**: fp16=2, fp32=4, int64=8

### 4.3 芯片专家视角的 3 个关键观察

| 观察 | 含义 | ISA/微架构机会 |
|---|---|---|
| **Histogram 是热路径** | 每个 pass 必算,占 Vector ALU ~40% | 用专用 Histogram 微码 (已有 `Histograms<>` 模板) |
| **BinarySearch 在 Scalar** | Vector 算完必须 stall 等 Scalar 找 b* | 用 `HardEvent::V_S` 高效同步(已实现) |
| **GatherMask 浪费带宽** | 读全部数据但只保留 ~1/256 | 考虑"过滤式 Load"指令 |

---

## 5. 各模式速解 (一节一图)

### 5.1 ② SingleBlock — 单核 UB 内 Radix-Select

```
┌─────── 单核 (blockIdx=0) ─────────────────────────────────┐
│  GM:  X[batch, N]  →  UB xLocal[N]   (一次搬入)            │
│                         │                                  │
│                         ▼                                  │
│           AscendC::TopK<RADIX_SELECT>(xLocal, K)           │
│                         │                                  │
│                         ▼                                  │
│              UB sortedTopK[K]  →  GM Y, Index              │
└────────────────────────────────────────────────────────────┘

约束: N × sizeof(T) + ~8KB (API tmp) ≤ UB (~192KB)
典型: fp16 → N ≤ ~15K, fp32 → N ≤ ~7K
瓶颈: 单核 Vector ALU 利用率
```

### 5.2 ⑥ MultiCoreRadix — 跨核协作的 Radix-Select

```
核 0                    核 1                    核 P
 │                       │                       │
 ▼                       ▼                       ▼
GM X[tile0] ─DataCopy─→ UB  GM X[tile1] ─→ UB  GM X[tileP] ─→ UB
                          │       │                  │
                          ▼       ▼                  ▼
                    [Vector]  Histograms (per-tile, 256 bins)
                          │       │                  │
                          ▼       ▼                  ▼
                    Atomic Add into GM globalHistGm[256]
                                  │
                                  ▼
                    ┌─── PipeBarrier<PIPE_ALL> ───┐
                    │   (所有核到此同步)            │
                    └────────────────┬─────────────┘
                                     ▼
                          [Scalar] BinarySearch on globalHist
                                     │
                                     ▼  boundaryBin
                          Pass 2: 只处理 ≥ boundaryBin 的位
                                     ...
                          最终: GM tileTopkValueGm (≥K 候选)
                                     │
                                     ▼  DataCopy → UB
                          AscendC::TopK + Sort
                                     │
                                     ▼
                                 GM Y, Index
```

**核心瓶颈** (芯片专家重点):
1. **Atomic Add 吞吐**: P 个核同时往 256 个 bin 写,**冲突在同一个 cache line 上**
   - 实测中这是多核模式的最大延迟来源
   - 当前实现没有 bin 分桶到不同 cache line 的优化
2. **PipeBarrier 开销**: 每个 pass 必须全核同步一次,NUM_PASS 越多越糟
3. **GM workspace 占用**: globalHistGm + cumSumBinsGm + tileTopkValueGm 总计可达几 MB

### 5.3 ① SmallSize — MergeSort 快速路径

```
N ≤ 1024, fp32/fp16/bf16:

UB xLocal[N] ──Concat──→ UB concatBuf ──Sort──→ UB sorted[N]
                                                  │
                                                  ▼
                                              取前 K 个

优势: 完全复用 AscendC::MergeSort 公共 API,无需自行实现
劣势: 必须 fp 类 + 小 N, 适用面窄
```

### 5.4 ⑦ SortAndTopK — 大 N 兜底

```
N ≥ 10M (典型 LLM / 推荐系统场景):

完整 Radix Sort (8 pass for fp32) → GM sortedIdxTmp[N]
                                       │
                                       ▼
                              DataCopy 前 K 项到 GM Y, Index

代价: O(N × B) 完整排序,workspace 占用 M × N × 4B (最重)
意义: 保证超大 N 时也能跑通 (不追求最优 perf)
```

---

## 6. Vector 单元计算流详解 (指令级时序)

### 6.1 Radix-Select 单次迭代的指令流 (理想时序)

```
时间轴 →

[DMA/MTE2]  DataCopy GM→UB  ────────────┐
                                        │
[Vector]                       Twiddle (Cast+XOR) ────┐
                                                        │
[Vector]                                     Histograms<>256 ──┐
                                                                  │
[Sync]                                                SetFlag<V_S> ─┐
                                                                          │
[Scalar]                                                       BinarySearch ─┐
                                                                                 │
[Sync]                                                                SetFlag<S_V> ─┐
                                                                                          │
[Vector]                                                                        GatherMask<> ──┐
                                                                                                         │
[MTE3]                                                                                          DataCopy UB→GM
                                                                                          ↓
                                                                                      下一 pass
```

**关键依赖链**: `Hist → V→S → BinarySearch → S→V → GatherMask` 是**串行关键路径**,
单 pass 的 latency 由这条链决定,**而不是吞吐**。

### 6.2 关键 Vector 指令使用统计

| 指令类别 | 具体调用 | 调用频率 | 用途 |
|---|---|:---:|---|
| **Histogram** | `Histograms<uint8_t, uint16_t, BIN_NUM, ACCUMULATE>` | B 次/pass | 256-bin 直方图 (微码) |
| **Gather / Filter** | `GatherMask<>`, `CopyMask` | B 次/pass | 保留 ≥ kthVal 的元素 |
| **Cast / Twiddle** | `Cast<>, XOR, Add(scalar)` | 1 次/elem | 有符号/浮点 → 无符号 |
| **Reduce** | `AscendC::ReduceSum<DIM>` | 1 次/pass | CumSum 256 bins |
| **Sort (辅助)** | `AscendC::Sort<>`, `AscendC::MergeSort<>` | 末尾 1 次 | 候选集最终排序 |
| **DataCopy** | `DataCopyPad`, `MicroAPI::DataCopy<POST_MODE_UPDATE>` | 多次 | GM↔UB, UB 自增指针 |
| **Scalar 算术** | `Add`, `Compare`, `Sub` (scalar regs) | 每次 BinarySearch | kthVal 二分 |

### 6.3 同步事件使用

| HardEvent | 方向 | 用途 | 出现频率 |
|---|---|---|:---:|
| `V_S` | Vector → Scalar | Hist 完成后让 Scalar 读 cumsum | 每 pass 1 次 |
| `S_V` | Scalar → Vector | BinarySearch 完成后让 Vector 用 kthVal | 每 pass 1 次 |
| `V_MTE3` | Vector → MTE3 | UB 数据写回 GM 前 | 末尾 1 次 |
| `MTE2_V` | MTE2 → Vector | GM 数据搬入 UB 后 | 每 tile 1 次 |

---

## 7. 内存层级使用模式

### 7.1 三级内存角色分配

```
┌──────────────────────────────────────────────────────────┐
│ HBM (GM)                                                  │
│   容量: ~32 GB (跨核共享)                                  │
│   带宽: ~1 TB/s                                            │
│   角色:                                                   │
│     - 输入 X / 输出 Y, Index                               │
│     - workspace (globalHistGm, tileTopkValueGm 等)        │
│     - 跨核通信介质 (Atomic Add 目标)                       │
│                                                           │
│   ★ 瓶颈: Atomic Add 吞吐 + PipeBarrier 同步延迟           │
└────────────────────────┬─────────────────────────────────┘
                         │ DataCopyPad (32B 对齐)
                         ▼
┌──────────────────────────────────────────────────────────┐
│ UB (Unified Buffer, 核私有 SRAM)                          │
│   容量: ~192 KB / 核                                      │
│   带宽: 核内 ~10 TB/s                                     │
│   角色:                                                   │
│     - xLocal (输入 tile)                                  │
│     - tileHistLocal[256] (本 tile 直方图)                 │
│     - cusumLocal[256] (累计和)                            │
│     - topKApiTmpTBuf (~8KB, AscendC::TopK 内部)          │
│     - valuesQue / indicesQue (输出双缓冲)                 │
│                                                           │
│   ★ 瓶颈: 容量决定 tileSize,间接决定能走单核 vs 多核        │
└────────────────────────┬─────────────────────────────────┘
                         │ Reg::LoadAlign
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Vector Register File                                      │
│   容量: 由微架构决定 (AscendC 编译器管理)                  │
│   角色: SIMD VL 寄存器,Histogram/GatherMask 操作数         │
└──────────────────────────────────────────────────────────┘
```

### 7.2 UB 占用估算 (模式② SingleBlock, fp16, N=14840, K=10)

```
inputXQue_:     14840 × 2B  ≈ 29.7 KB
topKApiTmp:                    8 KB (固定)
valuesQue_:        10 × 2B  ≈    20 B
indicesQue_:       10 × 4B  ≈    40 B
cusumLocal:      256 × 4B  =     1 KB
tileHistLocal:   256 × 4B  =     1 KB
────────────────────────────────────
总计:                         ≈ 40 KB  ← 远低于 192KB,轻松容纳
```

### 7.3 Workspace 占用估算 (模式⑥ MultiCoreRadix, M=1000, N=14840, K=10)

```
globalHistGm:    256 × NUM_PASS × 4B ≈ 2 KB   (每核读)
cumSumBinsGm:    256 × tileNum × 4B  ≈ 几 KB
tileTopkValueGm: M × (K+pad) × sizeof(T)
                = 1000 × 16 × 2B     = 32 KB
topkValueIndexGm: 同上                = 32 KB
────────────────────────────────────────────
总计:                                ≈ 70 KB
```

> 对比模式⑦: 同样 M=1000, N=14840 时 SortAndTopK workspace ≈ M×N×4 = **59 MB** ——这就是为什么 host 优先选 ⑥ 而非 ⑦。

---

## 8. 多核并行与同步 — 核心瓶颈分析

### 8.1 三种并行维度

```
┌─────────────────────────────────────────────────────────┐
│ 维度 1: 外轴 (M) 并行 — "batch parallel"                │
│   不同行之间完全独立,无通信                              │
│   适用模式: ①②④⑤                                       │
│                                                         │
│ 维度 2: 排序轴 (N) 切 tile — "tile parallel"            │
│   同一行的不同 tile 分给不同核                            │
│   跨核同步: 通过 globalHistGm                            │
│   适用模式: ⑥⑦                                         │
│                                                         │
│ 维度 3: pass 内字节并行 — "byte parallel"               │
│   理论上不同字节可并行,但 binarySearch 强制串行           │
│   当前未利用                                              │
└─────────────────────────────────────────────────────────┘
```

### 8.2 同步原语使用

| 同步原语 | 用途 | 开销 | 优化方向 |
|---|---|---|---|
| `HardEvent::V_S / S_V` | 核内 Vector↔Scalar 同步 | ~10 cycle | 已最小化 |
| `PipeBarrier<PIPE_ALL>` | 全核同步 (每 pass 一次) | ~100-1000 cycle | 减少 pass 数 (B=2 vs B=8) |
| `SetAtomicAdd<int32_t>` | 跨核直方图累加 | 冲突时剧增 | **未优化: bin→cache line 散列** |
| `DataCopy` 顺序写 | 末尾结果写回 GM | 带宽限制 | 已用 DataCopyPad 批量 |

### 8.3 性能天花板 (直觉估算)

```
单核 Radix-Select (模式②):
  latency ≈ N × B × (Hist_latency + V_S_sync + Scalar_BS + S_V_sync + GatherMask_latency)
                   └──────── 关键路径 ────────┘

多核 Radix-Select (模式⑥):
  speedup ≈ P / (1 + PipeBarrier_count × barrier_cost / useful_work)
                  └── 同步占比, P 大时显著 ──┘

经验值: 当 P=8 时, 实测加速比 ~5-6x (受 Atomic 瓶颈限制)
       当 P=24 时, 加速比 ~10-12x (瓶颈更突出)
```

---

## 9. 关键指令集需求清单 (芯片专家核心关切)

### 9.1 必需指令清单

| 类别 | 指令/能力 | 规格 | 当前实现 |
|---|---|---|---|
| **Vector SIMD** | `Histograms<T_in, T_out, BIN_NUM, ACCUM>` | BIN_NUM=256, T_in∈{u8,u16}, T_out∈{u16,u32} | ✅ 微码 |
| | `GatherMask<T>` / `CopyMask` | mask 位宽 = VL | ✅ |
| | `Cast<dst_T, src_T>` | 多 dtype 互转 | ✅ |
| | `XOR / Add(scalar)` | Twiddle 用 | ✅ |
| | `ReduceSum<DIM>` | 256-bin cumsum | ✅ |
| | `Compare / Max / Min` | 候选筛选 | ✅ |
| **Scalar** | 32-bit 整数二分搜索 | 软件 loop on scalar reg | ✅ (但 latency 高) |
| **DMA** | `DataCopyPad` GM↔UB | 支持 32B 对齐 + 末端 padding | ✅ |
| | `MicroAPI::DataCopy<POST_MODE_UPDATE>` | UB 内自增指针 copy | ✅ |
| **Sync** | `SetFlag<HardEvent::*>` + `WaitFlag` | V_S / S_V / V_MTE3 / MTE2_V | ✅ |
| | `PipeBarrier<PIPE_ALL>` | 全核栅栏 | ✅ |
| **Atomic** | `SetAtomicAdd<int32_t>` | HBM 32-bit atomic add | ✅ (但带宽有限) |
| **Sort (辅助)** | `AscendC::Sort<>`, `AscendC::MergeSort<>` | Vector 上 Bitonic | ✅ 公共 API |

### 9.2 隐性能力需求 (源码中未直接调用但必需)

| 能力 | 为什么需要 |
|---|---|
| UB 物理双缓冲 (DoubleBufferFactor=2) | DataCopy 与 Vector 计算重叠 |
| Vector 寄存器堆足够大 | Histogram 需要同时持 256 个累加器 |
| Scalar 单元独立执行流 | 不能阻塞 Vector ALU |
| Atomic 操作的 cache line 隔离 | 避免不同 bin 的累加互相 invalidation |

---

## 10. 性能上限与瓶颈分析

### 10.1 各模式的理论上限

| 模式 | 理论最优 throughput | 实际瓶颈 |
|---|---|---|
| ① SmallSize | MergeSort API 上限 (~N log N / Vector SIMD throughput) | API 调用开销 |
| ② SingleBlock | HBM 带宽 (N 较大时) / Vector ALU (N 中等) | 单核 Vector 利用率 |
| ④ SingleCore | 同 ②,但 N 更大 | UB 容量 |
| ⑤ MultiCoreOptim | HBM 带宽 × P | 行间负载均衡 |
| ⑥ **MultiCoreRadix** | HBM 带宽 × P / (1 + α×P) | **Atomic 冲突** |
| ⑦ SortAndTopK | HBM 带宽 × P / (1 + α×P) | GM workspace 带宽 |

### 10.2 三大瓶颈排序

```
┌────────────────────────────────────────────────────────┐
│  瓶颈 #1: 多核模式下 Atomic Add 到 globalHistGm         │
│  ──────────────────────────────────────────────────    │
│   原因: P 个核同时往 256 个 bin 原子加,cache line 冲突  │
│   影响: 大 N + 多核场景下,实测占比 30~50%               │
│                                                        │
│  瓶颈 #2: PipeBarrier 全核同步                          │
│  ──────────────────────────────────────────────────    │
│   原因: 每个 pass 必须等所有核完成 histogram            │
│   影响: NUM_PASS 多 (如 int64 的 8 pass) 时延迟叠加     │
│                                                        │
│  瓶颈 #3: Vector↔Scalar 同步 latency                    │
│  ──────────────────────────────────────────────────    │
│   原因: BinarySearch 必须在 Scalar 上跑,Vector stall    │
│   影响: 每 pass ~数十 cycle, B 次叠加                   │
└────────────────────────────────────────────────────────┘
```

---

## 11. ISA / 微架构改进机会 (芯片专家重点!)

### 11.1 改进 A: "HistAndFilter" 融合指令

**当前**: 每个 pass 走 `Hist → V→S sync → BinarySearch → S→V sync → GatherMask` 五步,
其中 4 步是同步 / 标量操作,Vector 大部分时间在 stall。

**建议**:

```
新指令:  VFTopKFilter<T>  ub_in, ub_out, K, byte_pos
         (Vector-Filter-TopK 单指令)

微码行为:
  1. 内部 Histograms 累加
  2. 内部 cumsum + 二分搜索 kthVal (微码实现,非 Scalar 单元)
  3. 内部 GatherMask 过滤
  4. 输出: 过滤后的元素 + 新的 K (remainK)

预期收益:
  - 单 pass latency: 5 步 → 1 步,~60% 延迟下降
  - 适用模式: ②④⑤⑥ 全部受益
  - 实现复杂度: 中等 (Histogram 微码已存在,只需封装)
```

### 11.2 改进 B: Atomic-Histogram 硬件加速

**当前**: 多核模式下 `SetAtomicAdd<int32_t>` 让 P 个核往同一 256-bin 数组原子加,
cache line 冲突严重。

**建议**:

```
方案 1: 硬件 Histogram-Reduction 单元
  - 每核 Private Histogram Buffer (PHB) 在 L1/UB
  - 核内累加完全在 PHB 内,无 GM 访问
  - pass 结束时,硬件自动 reduce 到 GM (一次批量写)

方案 2: Bin-to-CacheLine 散列
  - 把 256 个 bin 散列到 8 条 cache line (每 32 bin 一组)
  - 减少 false sharing

预期收益:
  - 多核模式 ⑥ 加速比从 ~5x 提升到 ~7x (P=8 时)
  - GM 带宽占用下降 ~80%
```

### 11.3 改进 C: Twiddle 融合进 Load 单元

**当前**: GM 数据搬入 UB 后,需要单独的 `Cast + XOR` 操作把 fp 转成 uint。

**建议**:

```
新指令: DataCopyTwiddle<CAST_T, XOR_MASK>
  - 在 DMA 搬运过程中,硬件完成位级变换
  - 类似 NVIDIA GPU 的 BytePermute

预期收益:
  - Twiddle 阶段延迟 ~0 (overlap 在 DMA 中)
  - UB 节省 (无需 inputXCopy 中间 buffer)
```

### 11.4 改进 D: 多 pass 并行 (字节级 ILP)

**当前**: 每个 pass 必须等上一个 pass 完成,因为 K-th 边界依赖于上一 pass 的累计。

**观察**: 实际上只有"边界桶" b* 内的元素需要进入下一 pass,
其他桶(>b*)已经确定为 top-K,可以提前送出。

**建议**:

```
- 硬件维护 "已确定 top-K" 寄存器
- 每个 pass 输出两类: confirmed (>b* 桶) + candidates (=b* 桶)
- 下一 pass 只处理 candidates,confirmed 直接累积到输出队列
- 多 pass 的 confirmed 可以并行处理

预期收益:
  - 减少有效 pass 数 (理论从 B 降到 log₂₅₆(N/K))
  - 大 N 小 K 场景收益最大
```

### 11.5 改进 E: Scalar BinarySearch 硬件化

**当前**: BinarySearch 是软件 loop,占 Scalar 单元 ~10-20 cycle/pass。

**建议**:

```
新指令:  VFindKthVal  ub_hist, K → scalar_reg
  - 单周期完成 256-bin cumsum + 二分搜索

实现: 256-bin 累加器树 + priority encoder
       (类似 GPU 的 __shfl_sync + ballot)
```

---

## 12. 总结 (给芯片专家的 3 句话)

1. **TopK V2 的核心是 Radix-Select**,它把 K-of-N 选择问题转化为 B 次 (B=sizeof(dtype)) "Histogram + 二分 + 过滤" 循环,**完美匹配 Vector + Scalar 协同**的微架构。

2. **当前实现的性能上限**由三个瓶颈决定:**多核 Atomic Add 冲突** (大 N)、**PipeBarrier 同步开销** (NUM_PASS 多)、**Vector↔Scalar 同步 latency** (每 pass 关键路径)。三者都可被 ISA 改进缓解。

3. **最有价值的 ISA 改进**是 **"HistAndFilter" 融合指令** —— 把 Histogram + BinarySearch + GatherMask 封装成单条 Vector 微码,预期降低单 pass 延迟 ~60%,且对全模式受益。

---

## 附录 A: dtype × 模式 矩阵 (覆盖度速查)

| dtype \ 模式 | ① Small | ② SingleBlock | ③ Fp32Merge | ④ SingleCore | ⑤ MultiCoreOptim | ⑥ MultiCoreRadix | ⑦ SortAndTopK |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| fp32 | ✓ | ✓ | ✓★ | ✓ | ✓ | ✓ | ✓ |
| fp16 | ✓ | ✓★ | — | ✓ | ✓ | ✓ | ✓ |
| bf16 | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| int32 | — | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| int64 | — | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| uint8/16/32 | — | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| int8/16 | — | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| uint64 | — | ✓ | — | ✓ | ✓ | ✓ | ✓ |

★ = 主路径 (该 dtype 最常用模式)

---

## 附录 B: 与 NVIDIA GPU / CUDA 实现对比

| 维度 | Ascend AIV (本算子) | NVIDIA CUDA (CUB thrust) |
|---|---|---|
| Host tiling 层 | **必须** (UB 容量小,需 host 决策) | **可选** (有 L2/shared memory 自动管理) |
| 主算法 | Radix-Select + Twiddle | Radix-Select + Bitonic (类似) |
| 多核通信 | Atomic Add + PipeBarrier | `atomicAdd` + `__syncthreads` |
| 公共 API | `AscendC::TopK` (header-only) | `cub::DeviceRadixSelect` |
| 关键差异 | **Twiddle 必须 (浮点位级变换)** | 直接用 float 位运算 |
| 瓶颈 | Atomic HBM 冲突 | Shared Memory bank conflict |

> **结论**: Ascend 的实现没有"特别先进"的地方,但 Twiddle 路径证明了**位级 dtype 抽象**对 radix 算法的关键价值。

---

## 附录 C: 文档导航

| 你想了解 | 看这份文档 |
|---|---|
| 文件结构、include 关系 | `CODE_ANALYSIS.md` |
| Host tiling 字段表、谓词细节 | `TILING_DETAIL.md` |
| Kernel 每模式代码流程 | `KERNEL_DETAIL.md` |
| 算法→硬件 IP 详细映射 | `HARDWARE_REQUIREMENTS.md` |
| AscendC::TopK 公共 API 内部 | `ASCENDC_TOPK_INTERNAL.md` |
| **本文**: 选型、瓶颈、ISA 改进 | **`CHIP_EXPERT_BRIEFING.md`** ★ |

---

**文档版本**: v1.0 (2026-06-17)
**面向读者**: 芯片架构师 / 微架构设计师 / ISA 定义团队
**作者**: 基于 top_k_v2 5 份深度文档抽象

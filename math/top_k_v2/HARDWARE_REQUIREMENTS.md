# TopK V2 算子硬件需求抽象

> 本文把 TopK V2 算子的实现拆解为「步骤 / 数据流 / 内存层级 / Layout 变换」四维,
> 再抽象为「硬件能力清单」,用于回答一个问题:
> **「要高效执行 TopK V2,芯片必须提供哪些硬件能力?」**
>
> 配套阅读:
> - `CODE_ANALYSIS.md` — 代码组成
> - `TILING_DETAIL.md` — Tiling 调度策略 (含 §11 CUDA 对比 / §12 模式选择)
> - `KERNEL_DETAIL.md` — Kernel 7 种模式实现
>
> 风格: 中文 + 表格 + ASCII 数据流图,源码引用使用 `文件:行号` 格式。
>
> **重要声明**: 本文从源码反推硬件需求,具体的容量、带宽、频率参数需以
> 昇腾官方 datasheet (Ascend 910/910B/A2 等) 为准。文中标注 `【需查 datasheet】` 的位置
> 表示该参数需要交叉验证。

---

## 0. 视角与方法

### 0.1 黑盒视角

把 TopK V2 当成一个对硬件提出需求的黑盒:

```
   ┌────────────────────────────────────────────────────────────┐
   │                      TopK V2 算子                            │
   │  输入: X[M, N] (任意 dtype) + K + largest/smallest          │
   │  输出: Y[M, K] + Index[M, K]                                 │
   │                                                              │
   │  内部对硬件的需求:                                            │
   │   ① 计算单元: Vector/Scalar/SIMT/特殊指令                   │
   │   ② 内存层级: HBM + UB + 寄存器                              │
   │   ③ 并行度: 多核 + 多 tile + 双缓冲                          │
   │   ④ 同步: Pipe Barrier + Atomic                              │
   │   ⑤ DMA: DataCopy 对齐/Pad/PostUpdate                        │
   └────────────────────────────────────────────────────────────┘
```

### 0.2 抽象方法

```
源码 (op_host + op_kernel) ─┬─→ §1 执行步骤分解
                             ├─→ §2 数据流
                             ├─→ §3 内存层级
                             ├─→ §4 Layout 变换
                             └─→ §5 硬件能力清单 (抽象层)
```

每一节最后给出"对硬件的具体要求",在 §5 汇总成矩阵。

---

## 1. 执行步骤分解 (Step Decomposition)

### 1.1 端到端时间线

```
Host (CPU)                          Device (AIV Cores)
─────────────                       ────────────────────
[1] 解析输入 shape/dtype
[2] 计算 K, largest 标志
[3] 选模式 (7 选 1)
    ├ IsSmallSizeMergeSortMode?
    ├ IsSingleBlockMode?
    ├ IsTopkRadixMoreCoreMode?
    ├ IsSingleCoreMode?
    ├ IsMultiCoreOptimMode?
    ├ IsSortAndTopKMode?
    └ IsTopkMergeSort{More/Intra}CoreMode?
[4] 计算 tiling 字段
    ├ 核数 / 每核 tile 数
    ├ tileSize / loop 次数
    └ 各 buffer size
[5] 计算 workspace size
[6] SetTilingKey(dtype × mode)
[7] 序列化 tiling → GM
        │
        │  (Host→Device 边界)
        ▼
                                    [8] kernel 入口 top_k_v2()
                                    [9] 读 TilingKey,二级路由
                                    [10] 按 mode 实例化 OpObject
                                    [11] Init: GM 指针 + UB 分配
                                    [12] GetBlockIdx() 领任务
                                    [13] 主循环 (mode 特定)
                                         ├ GM→UB (DataCopyPad)
                                         ├ 计算 (Vector/SIMT)
                                         ├ UB→GM (DataCopyPad/Atomic)
                                         └ Pipe Barrier / Sync
                                    [14] 写最终结果到 Y, Index GM
                                    [15] (可选) SortWithIndex 后处理
```

### 1.2 Host 侧步骤详解

| 步骤 | 关键源码 | 产出 |
|---|---|---|
| 1. 解析输入 | `top_k_v2_tiling_arch35.cpp` 各 `Is*Mode()` | shape, dtype, K, largest |
| 2. 模式选择 | `top_k_v2_tiling_arch35.cpp:913-1586` | modeType ∈ {1..7} |
| 3. 核数计算 | `CalcCoreNum*`, `lastDimNeedCore` | lastDimRealCore_, unsortedDimParallel_ |
| 4. tile 切分 | `CalcTilingInfo*` | tileSize, tileNum, loop |
| 5. buffer size | `CalcTmpBufferSize*` | workspace 各分区大小 |
| 6. TilingKey | `context->SetTilingKey(...)` `:1008/1082/1518/1757` | 编译期常量 |
| 7. 序列化 | `WriteTilingData` | tiling GM 字节流 |

### 1.3 Device 侧步骤详解(以 mode⑥ MultiCore RadixSelect 为例)

```
┌──────────────────────────────────────────────────────────────────┐
│ Step 1: 入口路由 (top_k_v2_apt.cpp:323-427)                       │
│   读 TILING_KEY_VAR → 选模板参数 <T,UNSIGNED,NUM_PASS,...>         │
│   → 调用 generateOpObject → RadixSortTopK                         │
├──────────────────────────────────────────────────────────────────┤
│ Step 2: Init (radix_sort_top_k_base.h:75)                        │
│   - 拿到 GM 指针: inputXGm_, outputYGm_, outputIndexGm_           │
│   - 拿到 workspace 指针: globalHistGm_, cumSumBinsGm_, ...        │
│   - TPipe::InitBuffer 分配 UB (inQueueX_, valuesQue_, ...)        │
├──────────────────────────────────────────────────────────────────┤
│ Step 3: 任务领取 (radix_sort_top_k.h:500-506)                    │
│   unsortedAxisId = GetBlockIdx() / lastDimRealCore_               │
│   startTileId    = GetBlockIdx() % lastDimRealCore_               │
├──────────────────────────────────────────────────────────────────┤
│ Step 4: 主循环 ProcessMultiBlockTopK (radix_sort_top_k.h:496)     │
│   for pass = 0..NUM_PASS-1:                                       │
│     for tile in my tiles:                                         │
│       ├ DataCopyPad: GM→UB (inputXGm → xLocal)                    │
│       ├ PipeBarrier<MTE2_V>                                       │
│       ├ Twiddle (signed→unsigned)                                 │
│       ├ Compute histogram (256 bins, Vector)                      │
│       ├ CumSum on histogram                                       │
│       ├ asc_vf_call<CopyCumSumToGm*> (SIMT)                       │
│       └ SetAtomicAdd<int32_t> → 累加到 globalHistGm                │
│     PipeBarrier<PIPE_ALL> // 跨核全同步                            │
│     FindBoundary (BinarySearch on globalHistGm)                   │
│     for tile in my tiles:                                         │
│       └ 把 boundary bin 内的元素累加到 tileTopkValueGm             │
│     PipeBarrier<PIPE_ALL>                                         │
│     校准 K (若 boundary bin 多算了几个)                            │
├──────────────────────────────────────────────────────────────────┤
│ Step 5: 最终 TopK + Sort (radix_sort_top_k.h:845-951)             │
│   ├ DataCopyPad: tileTopkValueGm → UB                             │
│   ├ AscendC::TopK<T,...,TOPK_NORMAL> (Vector API)                 │
│   ├ AscendC::Cast<int64_t,int32_t> (index 加宽)                   │
│   ├ AscendC::Sort<T,T_INDEX_TO,...> (按值排序)                    │
│   └ DataCopyPad: UB→GM (outputYGm_, outputIndexGm_)               │
├──────────────────────────────────────────────────────────────────┤
│ Step 6: (可选) SortWithIndex 后处理                                │
│   若 K>2000 && sorted==true → 走 sortwithindexForTopK              │
└──────────────────────────────────────────────────────────────────┘
```

### 1.4 七种模式步骤差异表

| 模式 | Host 决策点 | Device 关键步骤 | 是否需要跨核同步 | 是否需要原子操作 |
|---|---|---|:---:|:---:|
| ① SmallSizeMergeSort | `IsSmallSizeMergeSortMode` (TILING §12.1) | DataCopy → VbsMergeSort → 输出 | ✗ (单核一行) | ✗ |
| ② SingleBlock | `IsSingleBlockMode` (`:913`) | DataCopy → RadixHistogram → TopK API | ✗ | ✗ |
| ③a Fp32MergeMoreCore | `IsTopkMergeSortMoreCoreFp32Mode` (`:1552`) | DataCopy → Concat+Sort → 跨核归并 | ✓ | ✗ |
| ③b Fp32MergeIntraCore | `IsTopkMergeSortIntraCoreFp32Mode` (`:1586`) | 单核内多 batch 归并 | ✗ | ✗ |
| ④ SingleCore | `IsSingleCoreMode` (`:950`) | Radix 多 pass 全程 UB | ✗ | ✗ |
| ⑤ MultiCoreOptim | `IsMultiCoreOptimMode` (`:533`) | 单核多 row + TopK API | ✗ | ✗ |
| ⑥ MultiCore (默认) | `IsTopkRadixMoreCoreMode` (`:844`) | Radix 多核分布式 + Atomic | ✓ | ✓ int32 |
| ⑦ SortAndTopK | size > 10M (`SORT_AND_TOP_K_THRESHOLD`) | 完整 radix sort + 截断 | ✓ | ✓ |

---

## 2. 数据流 (Data Flow)

### 2.1 三层数据通路

```
┌──────── HBM (Global Memory / GM) ────────┐
│  - 容量大 (~32GB+), 带宽高 (~1.2TB/s)    │
│  - 跨核共享                              │
│  - 分配: op_host 计算 offset             │
└──────────────┬───────────────────────────┘
               │ DataCopyPad / DataCopy<POST_UPDATE>
               ▼
┌──────── UB (Unified Buffer / SRAM) ──────┐
│  - 容量小 (~192KB / 核)                  │
│  - 核私有                                │
│  - 分配: TPipe::InitBuffer              │
└──────────────┬───────────────────────────┘
               │ Vector/SIMT compute
               ▼
┌──────── 寄存器 / Scalar Unit ────────────┐
│  - Scalar 用于循环/分支/offset 计算      │
│  - Vector 用于批量计算                   │
│  - SIMT (asc_vf_call) 用于 64 位原子    │
└──────────────────────────────────────────┘
```

### 2.2 mode⑥ 完整数据流 (RadixSelect MultiCore)

```
Pass=0..NUM_PASS-1 (B32 共 4 pass):

[HBM inputXGm] ──DataCopyPad──→ [UB xLocal]
                                      │
                                      │ Twiddle (signed→unsigned)
                                      ▼
                                [UB unsignedXData]
                                      │
                                      │ Compute Histogram (256 bins)
                                      ▼
                                [UB tileHistLocal]
                                      │
                                      │ CumSum + ReinterpretCast
                                      ▼
                                [UB cusumLocal]
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                           │
        int64 路径                                  int32 路径
   asc_vf_call<CopyCumSumToGmB64>            asc_vf_call<CopyCumSumToGmB8B16B32>
        (radix_sort_top_k.h:590)                  (radix_sort_top_k.h:596)
                │                                           │
                └─────────────┬─────────────────────────────┘
                              ▼
              SetAtomicAdd<int32_t> (radix_sort_top_k.h:603)
                              ▼
                    [HBM globalHistGm]  ← 跨核累加点 (256 bins)
                              │
                              ▼ (所有核到此同步)
                    PipeBarrier<PIPE_ALL> (:620)
                              │
                              ▼
                   BinarySearch on globalHistGm
                              │
                              ▼
                       boundaryBin (uint32)
                              │
                              ▼
                  Pass N+1: 仅处理 ≥ boundaryBin 的位

最后一 pass 结束:
                  [HBM tileTopkValueGm] (累计 ≥ K 个候选)
                              │
                              ▼ DataCopyPad
                       [UB topkValueLocal]
                              │
                              ▼ AscendC::TopK<>  (:877)
                       [UB top K value+index]
                              │
                              ▼ AscendC::Sort<>  (:951)
                       [UB sorted top K]
                              │
                              ▼ DataCopyPad
                  [HBM outputYGm + outputIndexGm]
```

### 2.3 各模式数据流对比

| 模式 | GM→UB 输入 | UB 内部变换 | UB→GM 输出 | 跨核数据交换 |
|---|---|---|---|---|
| ① Small | DataCopy X 单 row | VbsMergeSort (Concat+Sort) | DataCopy Y+Idx | 无 |
| ② SingleBlock | DataCopy X 多 batch 并行 | Histogram + TopK API | DataCopy Y+Idx | 无 |
| ③a MoreCore | DataCopy X 多 row 分核 | Concat+Sort | DataCopy 临时结果 | 临时区拼接 |
| ③b IntraCore | DataCopy X 多 row 单核 | 同上但单核 | 同上 | 无 |
| ④ SingleCore | DataCopy X 全 row | Radix 4-pass 全 UB | DataCopy Y+Idx | 无 |
| ⑤ MultiCoreOptim | DataCopy X 多 row | TopK API per row | DataCopy Y+Idx | 无 |
| ⑥ MultiCore | DataCopy X 分 tile | Histogram+CumSum+Atomic | DataCopy Y+Idx | **globalHistGm (atomic)** |
| ⑦ SortAndTopK | DataCopy X 分核 | 完整 Radix Sort | DataCopy 截断 | globalHistGm + outIdxTmpGm |

### 2.4 关键 DataCopy 调用点

| 调用 | 位置 | 作用 | 对齐要求 |
|---|---|---|---|
| `DataCopyPad(inputBuf, GM, params, padParams)` | `radix_sort_top_k.h:359, 693, 933, 944` | GM→UB 主输入 | `ROUND_UP_AGLIN(count*sizeof(T))` |
| `DataCopyPad(GM, outputBuf, params)` | `radix_sort_top_k.h:976, 988, 1069` | UB→GM 最终输出 | 同上 |
| `MicroAPI::DataCopy<uint32_t, POST_MODE_UPDATE>` | `radix_sort_top_k.h:378, 1173; radix_sort_topk_b32.h:66` | SIMT 自增指针 DataCopy | 32B 对齐 |
| `AscendC::Copy(sortTempBuffer, tmpUbInputs[0], ...)` | `top_k_merge_sort_more_core.h:433` | UB→UB copy (MrgSort 辅助) | sizeof(CONVERT_TYPE) |

### 2.5 吞吐瓶颈分析

| 阶段 | 瓶颈类型 | 是否可并行 | 优化手段 |
|---|---|:---:|---|
| GM→UB 输入 | HBM 带宽 | 跨核可 | 双缓冲 (`DOUBLE_BUFFER_FACTOR=2`) |
| Histogram 计算 | Vector ALU | 核内可 | 256 bins / tile 一次算完 |
| **Atomic 累加到 globalHistGm** | **GM 原子吞吐** | **跨核竞争** | **核心瓶颈,边界二分** |
| PipeBarrier | 同步开销 | 不可 | 减少 pass 数 (B8=1 < B16=2 < B32=4 < B64=8) |
| TopK API | Vector | 核内可 | 单次调用对 ~K 个候选 |
| Sort API | Vector | 核内可 | 单次排序 |
| UB→GM 输出 | HBM 带宽 | 跨核可 | DataCopyPad 批量 |

---

## 3. 内存层级 (Memory Hierarchy)

### 3.1 内存层级模型

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 0: HBM (off-chip DRAM)                                │
│   - 全芯片共享,跨核可见                                       │
│   - 容量: ~32GB (Ascend 910B)【需查 datasheet】              │
│   - 带宽: ~1.2TB/s【需查 datasheet】                          │
│   - 分配: 用户 tensor + workspace                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ PCIe / 片内 NoC
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: L1 Cache (per core)                                │
│   - 通常对软件透明                                            │
│   - TopK V2 源码不显式使用                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: UB (Unified Buffer, per-core SRAM)                 │
│   - 核私有 (~192KB / 核 on AIV) 【需查 datasheet】            │
│   - 由 TPipe::InitBuffer 分配                                │
│   - TQue / TBuf 抽象                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Vector Register File                               │
│   - Vector ALU 寄存器                                         │
│   - 由 AscendC 编译器管理                                      │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 HBM (GM) 使用清单

| GM 区域 | 用途 | 大小 (典型) | 跨核共享 | 源码位置 |
|---|---|---|:---:|---|
| `inputXGm` | 输入 X | M × N × sizeof(T) | 只读 | `radix_sort_top_k_base.h` |
| `outputYGm` | 输出 Y (TopK value) | M × K × sizeof(T) | 写 | 同上 |
| `outputIndexGm` | 输出 Index | M × K × sizeof(T_INDEX_TO) | 写 | 同上 |
| `tilingGm` | TilingData 字节流 | ~256B | 只读 | `top_k_v2_apt.cpp` |
| `globalHistGm` | 全局直方图 (256 bins × passes) | ~256 × NUM_PASS × 4B | **读写+原子** | workspace |
| `cumSumBinsGm` | 各核/各 tile 的 cumsum | ~256 × tileNum × 4B | 读写 | workspace |
| `tilesCusumGm` | 单核专用 cumsum | 同上 | 读写 | workspace |
| `tileTopkValueGm` | 候选值累积区 | M × (K+padding) × sizeof(T) | 读写 | workspace |
| `topkValueIndexGm` | 候选 index 累积区 | M × (K+padding) × sizeof(T_INDEX_TO) | 读写 | workspace |
| `sortOutIdxTmpGM` | SortAndTopK 临时 | M × N × sizeof(T_INDEX) | 读写 | workspace |
| `excusiveBinsGmWk` | 排他 bin sum | ~256 × numCore × 4B | 读写 | workspace |

### 3.3 UB (SRAM) 使用清单

UB 是核私有资源,每核一份。典型分配 (mode⑥ RadixSortTopK):

| UB 区域 | 类型 | 用途 | 大小 (典型) |
|---|---|---|---|
| `inputXQue_` | `TQue<VECIN, 1>` | 输入 X 缓冲 | `ROUND_UP_AGLIN(numTileData_) × sizeof(T)` |
| `valuesQue_` | `TQue<VECOUT, 1>` | 输出 value 缓冲 | `ROUND_UP_AGLIN(K) × sizeof(T)` |
| `indicesQue_` | `TQue<VECOUT, 1>` | 输出 index 缓冲 | `ROUND_UP_AGLIN(K) × sizeof(T_INDEX_TO)` |
| `inputXCopy_` | `TBuf<VECCALC>` | Twiddle 临时 | `numTileData × sizeof(UNSIGNED_TYPE)` |
| `cusumLocal_` | `TBuf<VECCALC>` | 256 bins cumsum | 256 × 4B = 1KB |
| `tileHistLocal_` | `TBuf<VECCALC>` | 单 tile 直方图 | 256 × 4B = 1KB |
| `topKApiTmpTBuf_` | `TBuf<VECCALC>` | AscendC::TopK API 内部临时 | ~8KB 【需查 API 文档】 |
| `indicesOutTbuf_` | `TBuf<VECCALC>` | index 类型转换临时 | K × sizeof(T_INDEX_TO) |

注: 模式不同,具体 UB 区域差异较大 (详见 `KERNEL_DETAIL.md §9 UB Layout`)。

### 3.4 寄存器 / Scalar

| 寄存器用途 | 源码体现 |
|---|---|
| 循环变量、tileId | `for (tileId = startTileId; tileId < ...)` 散布全文件 |
| offset 计算 | `topkValuesGmOffset = CeilAlignDivMul<uint64_t>(...)` `radix_sort_top_k.h:259` |
| Scalar 累加 | `Add(cusumLocal_.ReinterpretCast<int32_t>(), ...)` `radix_sort_top_k_single_core.h:311` |
| BinarySearch 标量 | `radix_sort_top_k.h:719` `BinarySearch` |

### 3.5 UB 容量约束表 (按模式)

| 模式 | UB 主要占用 | 估算 (sizeof) | 备注 |
|---|---|---|---|
| ① SmallSizeMergeSort | xLocal + sorted + concat tmp | ~6 × tileSize | `top_k_merge_sort.h` |
| ② SingleBlock | xLocal[] (parallelBatchNum) + TopK tmp | ~3 × tileSize | `radix_sort_top_k_single_block.h` |
| ③a Fp32MergeMoreCore | 输入 + 排序缓冲 | ~4 × sortBufferSize | `top_k_merge_sort_more_core.h` |
| ④ SingleCore | xLocal + unsigned + hist + cumsum | ~5 × tileSize | `radix_sort_top_k_single_core.h` |
| ⑥ MultiCore | xLocal + hist + cumsum + TopK tmp | ~3 × tileSize + 固定 8KB | `radix_sort_top_k.h` |
| ⑦ SortAndTopK | xLocal + sortIdx + globalHist local | ~4 × tileSize | `sort_and_top_k_more_core.h` |
| SortWithIndex 后处理 | `NEED_UB_SIZE_BYTE = 221184` (216KB) | 216KB | `sort_with_index_tiling.h:39` |

> 注意 SortWithIndex 模式预留了 32KB 给 SIMT 使用,实际可用 UB 为 216KB。

### 3.6 Workspace 容量约束表

| 模式 | 主要 workspace 项 | 估算公式 |
|---|---|---|
| ⑥ MultiCore | globalHistGm + cumSumBinsGm + tileTopkValueGm | `256×NUM_PASS×4 + 256×tileNum×4 + M×(K+pad)×sizeof(T)` |
| ⑦ SortAndTopK | globalHistGm + outIdxTmpGm + sortOutIdxTmpGm | `~M × N × sizeof(T_INDEX) (最重)` |
| ③a Fp32MergeMoreCore | sortTempBuffer on GM | `M × sortBufferSize` |

---

## 4. Layout 变换 (Layout Transformations)

### 4.1 Layout 变换一览

```
用户视角 (NHWC/MDType)                 Kernel 内部视角
─────────────────────                  ─────────────────
X[M, N] dtype=T                       ┌────────────────────┐
   │                                  │  1. Reformat/Twiddle │
   │                                  │     (signed→unsigned)│
   ▼                                  │     for radix sort   │
GM (T 类型连续存储)                    └──────────┬───────────┘
   │                                             │
   │ DataCopyPad                                 ▼
   ▼                                  ┌────────────────────┐
UB xLocal[T]                          │  2. Histogram layout │
   │                                  │     (256 bins)       │
   │ TwiddleIn*                       └──────────┬───────────┘
   ▼                                             │
UB unsignedXData                      ┌──────────▼───────────┐
   │                                  │  3. CumSum layout     │
   │ Histogram Compute                │     (prefix-sum on    │
   ▼                                  │      256-bins)        │
UB tileHist[256]                      └──────────┬───────────┘
   │                                             │
   │ CumSum + ReinterpretCast                   │
   ▼                                             ▼
UB cusum[256]                         ┌────────────────────┐
                                      │  4. Sort layout      │
                                      │     (value+index 对) │
                                      └──────────┬───────────┘
                                                 │
                                                 ▼
                                      Y[K], Index[K] (T 类型)
```

### 4.2 Twiddle (signed → unsigned)

**为什么需要**: Radix Sort 要求输入为无符号整数 (按字节切分进行 histogram)。但 TopK 输入可能是:

| dtype | 处理函数 | 源码位置 | 变换内容 |
|---|---|---|---|
| FP32 | `TwiddleInFp32` | `radix_sort_top_k.h:1121; top_k_radix_block_sort_*.h` | IEEE754 浮点 → 可排序无符号 |
| FP16/BF16 | `TwiddleInFp16` | `radix_sort_top_k.h:1117` | 同上 (16 位版本) |
| INT8 | `TwiddleInB8` | `radix_sort_top_k.h:1129` | 补码 → 偏移无符号 |
| INT16 | `TwiddleInB16` | `radix_sort_top_k.h:1125` | 同上 |
| INT32 | `TwiddleInB32` | `radix_sort_top_k.h:1113` | 同上 |
| INT64 | `TwiddleInB64` | `radix_sort_top_k.h:1109` | 同上 (64 位) |

变换公式 (FP32 示例):

```
对于 largest=true:
  if (sign_bit == 0)  out = in | 0x80000000;   // 正数: 翻转符号位
  else                out = ~in;                // 负数: 全部取反

对于 largest=false: 反之
```

这样变换后,无符号比较的顺序与原始 IEEE754 顺序一致。

### 4.3 Histogram Layout

```
单 tile 内 (256 bins,共 4 字节/bin):
┌──────────────────────────────────────────────┐
│ bin[0] | bin[1] | ... | bin[255]              │  ← 在 UB 中
└──────────────────────────────────────────────┘
   4B      4B            4B

跨核累加到 GM (globalHistGm):
Pass 0:                              Pass 1:                              ...
┌────────────────────────┐          ┌────────────────────────┐
│ bin[0]   bin[1] ... bin[255] │    │ bin[0]   bin[1] ... bin[255] │
│  (累加 atomic)               │    │                              │
└────────────────────────┘          └────────────────────────┘
   offset 0                            offset 256*4 = 1024
```

### 4.4 Sort 输入/输出 Layout

`AscendC::Sort<>` 输入要求 `Concat(value, index)` 对:

```
输入 layout (Concat 后):
┌─────────────────────────────────────────┐
│ values[0..N-1] | indices[0..N-1]        │  ← 在 UB
└─────────────────────────────────────────┘
   N × sizeof(T)    N × sizeof(T_INDEX_TO)

输出 layout:
┌─────────────────────────────────────────┐
│ sorted_values[0..N-1] | sorted_idx[0..N-1] │
└─────────────────────────────────────────┘

源码: top_k_merge_sort_more_core.h:300-321
  AscendC::Concat(concatLocal, inputLocal, concatTempLocal, repeatTimes);
  AscendC::Sort<CONVERT_TYPE, true>(sortedValueLocal, concatLocal,
                                     sortedValueIndexLocal, sortTempLocal, repeatTimes);
```

### 4.5 Index Layout (int32 / int64)

| 阶段 | 类型 | 原因 |
|---|---|---|
| 单 tile 内 index | `int32_t` | 节省 UB / GM 空间 |
| 跨核累加用 index | `int32_t` | SetAtomicAdd<int32_t> 仅支持 32 位 |
| 最终输出 index | `T_INDEX_TO` (可 int64) | 大数组需要 64 位寻址 |
| 类型转换 | `AscendC::Cast<int64_t, int32_t>` `radix_sort_top_k.h:892` | 在 Sort 前加宽 |

### 4.6 对齐 (Alignment)

| 资源 | 对齐要求 | 源码体现 |
|---|---|---|
| GM 地址 | 32 字节 (oneBlock_) | `CeilAlignDivMul<uint64_t>(bytes, oneBlock_)` `:252, 259` |
| UB 元素数 | `UB_AGLIN_VALUE` (通常 32B) | `ROUND_UP_AGLIN(count * sizeof(T)) / sizeof(T)` |
| DataCopy 长度 | block 对齐,不足部分用 `padParams.rightPadding` | `radix_sort_top_k.h:350-359` |
| Sort 输入 | repeat 边界对齐 (通常 256B) | `concatRepeatTimes` `top_k_merge_sort_more_core.h:300` |

---

## 5. 硬件能力清单 (Hardware Capability Matrix)

### 5.1 能力总表

| 类别 | 具体需求 | 现有代码使用位置 | 不满足时的退化 |
|---|---|---|---|
| **计算-Vector** | 256-bin histogram, CumSum, Add, Cast | `radix_sort_top_k.h:568-617` | 退化为标量循环 |
| **计算-Scalar** | BinarySearch, offset 计算 | `:674, 719` | 退化为线性搜索 (慢 N×) |
| **计算-SIMT** | 64-bit atomic add on GM | `:590 asc_vf_call<CopyCumSumToGmB64>` | 退化为 32-bit + 锁 |
| **特殊指令** | AscendC::TopK (radix-select 引擎) | `:877` | 用 Sort+Truncate (慢) |
| **特殊指令** | AscendC::Sort (radix-sort 引擎) | `:951` | 软件实现 (极慢) |
| **特殊指令** | AscendC::Concat | `top_k_merge_sort_more_core.h:300` | 软件拼接 |
| **HBM 容量** | ≥ M×N×sizeof(T) × 2 (输入+输出) | 全局 GM | 越界 OOM |
| **HBM 带宽** | ≥ 1TB/s (避免成为瓶颈) | 所有 DataCopyPad | 计算被 IO 隐藏不掉 |
| **UB 容量** | ≥ 192KB / 核 (SortWithIndex 需 216KB) | `TPipe::InitBuffer` | 模式不可用,降级 |
| **并行-核数** | ≥ lastDimRealCore × unsortedDimParallel | `GetBlockIdx/GetBlockNum` | 单核串行,慢 10-50× |
| **并行-Vector 宽** | ≥ 256 lanes/batch (histogram 用) | Vector API 隐式 | 串行循环 |
| **同步-PipeBarrier** | 跨 PIPE 全屏障 | `PipeBarrier<PIPE_ALL>` (`:450, 461, 541, 620, 643, 659, 823, 840, 907, 954`) | 不能跨核并行 |
| **同步-HardEvent** | MTE2/MTE3/V/S 之间事件 | `SetFlag<HardEvent::*>` 散布全文件 | 必须串行执行 |
| **同步-AtomicAdd** | GM 上 int32 atomic add | `SetAtomicAdd<int32_t>` `:603` | 必须用锁/单核串行 |
| **DMA-DataCopyPad** | 支持 pad 模式 (非整块) | `DataCopyPad(GM, UB, params, padParams)` | 软件填充 0 |
| **DMA-PostUpdate** | DataCopy 自增指针 (POST_MODE_UPDATE) | `MicroAPI::DataCopy<..., POST_MODE_UPDATE>` | 软件管理指针 |
| **DMA-DoubleBuffer** | TQue 双缓冲支持 | `DOUBLE_BUFFER_FACTOR=2` `top_k_constant_var_simd.h` | 计算与 IO 不能重叠 |

### 5.2 计算单元详细需求

#### 5.2.1 Vector ALU

| 指令类别 | 需要的操作 | 调用位置 | 频率 |
|---|---|---|---|
| 算术 | Add, Sub, Mul (CumSum 用) | `:311 single_core` | 每 pass × 每 tile 一次 |
| 比较 | Greater, Less (Histogram 用) | Vector API 隐式 | 每 pass × 每 element |
| 类型转换 | Cast (int32↔int64, FP↔INT) | `:892, 951; top_k_merge_sort_more_core.h:566, 581` | 每次 Sort/TopK 前 |
| 位运算 | XOR, NOT, OR (Twiddle 用) | `TwiddleIn*` 系列 | 输入预处理一次 |
| Reduction | Sum (CumSum 内部) | AscendC 内部 | 每 256-bins 一次 |
| Duplicate | 常量填充 (init hist=0) | `Duplicate(xLocal, 0, countAlign)` `single_core:236` | 每 tile 开始 |

#### 5.2.2 Scalar Unit

| 操作 | 源码 |
|---|---|
| 循环控制 | `for (pass = 0; pass < NUM_PASS; pass++)` 散布 |
| offset 计算 | `gmOffset = blockIdx * tileSize + ...` 散布 |
| BinarySearch | `radix_sort_top_k.h:719` 标量循环 |
| K 校准 | `if (currCount > K) ...` 边界处理 |

#### 5.2.3 SIMT (Vector Microcode)

**关键**: 64 位 atomic add 是 mode⑥ 跨核累加的关键路径。

```
路径 A (int64 index):  asc_vf_call<CopyCumSumToGmB64<T_INDEX, T_INDEX_TO>>
                       radix_sort_top_k.h:590
                       → 调用 SIMT 微码,执行 64 位 GM 原子加

路径 B (int8/16/32):   asc_vf_call<CopyCumSumToGmB8B16B32<T_INDEX, T_INDEX_TO>>
                       radix_sort_top_k.h:596
                       → 32 位 atomic,SetAtomicAdd<int32_t> (:603)
```

### 5.3 内存层级详细需求

| 层级 | 容量需求 | 带宽需求 | 一致性 |
|---|---|---|---|
| HBM | ≥ 输入+输出+workspace (典型 100MB-1GB) | ≥ 1TB/s (避免 IO 瓶颈) | 跨核一致 (atomic 保证) |
| UB | ≥ 192KB / 核,SortWithIndex 需 216KB | ≥ 10TB/s (核内带宽)【需查 datasheet】 | 核私有,无一致性问题 |
| L1 | (透明) | (透明) | (透明) |
| 寄存器 | Vector RF 充足 | N/A | N/A |

### 5.4 并行度详细需求

| 维度 | 需求 | 实际使用 |
|---|---|---|
| 核数 (AIV) | ≥ `lastDimRealCore × unsortedDimParallel` | `GetBlockNum()` 拿到硬件核数 |
| 每核 Vector 宽 | ≥ 256B (单次 histogram) | AscendC API 隐式 |
| 双缓冲 | ≥ 2 (TQue depth) | `DOUBLE_BUFFER_FACTOR=2` |
| 多 pass 并行 | 不允许 (pass 之间有数据依赖) | 串行 `for pass = 0..NUM_PASS-1` |
| 多 tile 并行 | 允许 (核内串行 tile,跨核并行 tile) | `for tileId in [startTileId, endTileId)` |

### 5.5 同步原语详细需求

#### 5.5.1 Pipe-Level Events (HardEvent)

| 事件对 | 含义 | 使用频率 |
|---|---|---|
| `MTE2_V` | GM→UB 完成 → Vector 可用 | 极高 |
| `V_MTE3` | Vector 完成 → UB→GM 可发 | 极高 |
| `MTE3_V` | UB→GM 完成 → Vector 可再读 | 高 |
| `V_S` | Vector 完成 → Scalar 可读 | 中 |
| `S_V` | Scalar 完成 → Vector 可用 | 中 |
| `MTE2_S` | GM→UB 完成 → Scalar 可读 | 中 |
| `MTE3_S` | UB→GM 完成 → Scalar 可读 | 中 |

源码体现: `SetFlag<HardEvent::X_Y>(eventId)` + `WaitFlag<HardEvent::X_Y>(eventId)` 配对,
散布于 `radix_sort_top_k.h:362-363, 392-393, 424-425, 533-534, 557-558, ...`。

#### 5.5.2 PipeBarrier

`PipeBarrier<PIPE_ALL>()` 用于跨所有 PIPE 的全屏障,在以下关键点使用:

| 位置 | 用途 |
|---|---|
| `:450, 461` | histogram 计算前后 |
| `:541` | 跨核 atomic 累加前 |
| `:620, 643, 659` | Pass 内关键同步点 |
| `:823, 840, 907, 954` | TopK/Sort API 前后 |

**注意**: PipeBarrier<PIPE_ALL> 是最重的同步,代码尽量减少使用次数。

#### 5.5.3 Atomic Operations

| API | 数据宽度 | 用途 | 源码 |
|---|---|---|---|
| `SetAtomicAdd<int32_t>()` | 32-bit | histogram 跨核累加 | `:603` |
| `asc_vf_call<CopyCumSumToGmB64>` | 64-bit | int64 path cumsum 累加 | `:590` |
| `asc_vf_call<CopyCumSumToGmB8B16B32>` | 8/16/32-bit | 通用 path cumsum 累加 | `:596` |

**没有显式的 SyncBlockAll**: 跨核同步通过 `globalHistGm` 上的 atomic + `PipeBarrier<PIPE_ALL>` 组合实现。

### 5.6 DMA (DataCopy) 详细需求

| 能力 | 需求 | 源码体现 |
|---|---|---|
| 基础 DataCopy | 支持 T 类型 stride copy | `DataCopyPad` 全文 |
| Pad 模式 | 不足整块时右侧补 0 | `DataCopyPadExtParams.rightPadding` |
| PostUpdate | 拷贝同时自增指针 | `MicroAPI::DataCopy<..., POST_MODE_UPDATE>` |
| Broadcast | DataCopy 时 1→N 广播 | `MicroAPI::DataCopy<uint32_t, LoadDist::DIST_BRC_B32>` `:1172` |
| 双缓冲 | TQue depth=2 自动管理 | `TQue<POS, 2>` (但本算子多采用 depth=1) |
| Non-blocking | SetFlag + WaitFlag 配对实现流水 | 全文散布 |

### 5.7 特殊指令 / IP 块

| 指令 / IP | 提供能力 | 源码体现 | 不存在时的退化 |
|---|---|---|---|
| **AscendC::TopK** | 单次 Top-K 选取 (硬件 radix-select 引擎) | `:877, 888; single_block:160; single_core:461, 477; inter_core_opt:176, 182, 229` | 用 Sort + 截断 (慢 5-10×) |
| **AscendC::Sort** | 硬件 radix-sort 引擎 | `:951; single_core:582; merge_sort_more_core:301, 321; simd:115, 150` | 软件归并排序 |
| **AscendC::Concat** | 硬件张量拼接 | `merge_sort_more_core:300, 320; simd:113, 149` | 软件 memcpy 拼接 |
| **AscendC::Cast** | 硬件类型转换 | `:892; merge_sort_more_core:566, 581; simd:142, 156` | 软件逐元素 |
| **asc_vf_call (SIMT)** | Vector 微码调用 (atomic) | `:590, 596` | 退化为锁 |

---

## 6. 与 NVIDIA GPU 的硬件差异

> 本节是对 `TILING_DETAIL.md §11` 的硬件视角补充。

### 6.1 一句话对比

> **Ascend**: 显式分层 (HBM/UB/Reg) + 显式同步 (Pipe Barrier/Atomic) + 软件调度 (Tiling)
> **GPU**: 隐式分层 (Global/L1/Reg) + 隐式同步 (Memory Fence/Atomics) + 硬件调度 (Warp Scheduler)

### 6.2 详细对照表

| 维度 | Ascend (AIV) | NVIDIA CUDA | TopK V2 体现 |
|---|---|---|---|
| 计算单元 | Vector ALU + Scalar + SIMT + TopK/Sort IP | CUDA Core + Tensor Core | Ascend 有专用 TopK IP,GPU 没有 |
| 内存层级 | HBM → UB (192KB) → Reg | Global → L2 → L1 → Reg | UB 是显式资源,GPU L1 透明 |
| 内存分配 | TPipe::InitBuffer (静态) | malloc / shared (动态) | Ascend 静态分配,编译期决定 |
| 并行粒度 | Block (AI Core) | Block → Warp → Thread | Ascend Block 内部 SIMT 透明 |
| 同步-核内 | HardEvent + PipeBarrier | __syncthreads() + __warp_sync() | Ascend 显式 PIPE 划分 |
| 同步-跨核 | Atomic on GM + PipeBarrier<PIPE_ALL> | atomicAdd + __threadfence_system | 两者都用 GM atomic,Ascend 多一层 PipeBarrier |
| 原子操作 | SetAtomicAdd + asc_vf_call | atomicAdd / atomicCAS | Ascend 有专用 SIMT 微码 |
| 软件 Tiling | **必须** (TilingData 序列化到 GM) | 可选 (硬件调度) | Ascend 把 schedule 显式化 |
| 调度单位 | TilingKey 决定模板 | CUDA Grid/Block 启动配置 | Ascend 模式选择在 Host 完成 |
| DMA | DataCopyPad + HardEvent | __ldg / memcpy_async | Ascend 显式 PIPE 依赖 |
| 特殊指令 | AscendC::TopK / Sort | (无对应,GPU 用 thrust::sort) | Ascend 有硬件加速 |

### 6.3 设计取舍

**为什么 Ascend 选择"显式"路径?**

| 取舍 | 优点 | 代价 |
|---|---|---|
| 显式 Tiling | 性能可预测,适合推理场景 | 算子开发复杂 |
| 显式 UB 分配 | 避免 cache miss,延迟可控 | 程序员负担重 |
| 专用 TopK/Sort IP | TopK 类算子极致性能 | 通用性下降 |
| PipeBarrier + Atomic | 同步语义清晰 | 调试困难 |
| HardEvent 流水 | Vector/Scalar/MTE 真并行 | 编程模型复杂 |

**为什么 NVIDIA 选择"隐式"路径?**

| 取舍 | 优点 | 代价 |
|---|---|---|
| 硬件 Warp Scheduler | 编程简单 (CUDA C++ 接近主机 C++) | 性能预测难 |
| L1/L2 Cache 透明 | 程序员不关心 cache | cache 行为不确定 |
| 无专用 TopK IP | 通用性强 | TopK 类算子性能上限低 |
| thrust/CUB 库 | 开箱即用 | 库版本绑死 |

### 6.4 哪种更好?

| 场景 | Ascend 更优 | GPU 更优 |
|---|:---:|:---:|
| 大规模推理 (固定 shape) | ✓ | |
| TopK / Sort 密集型 | ✓ (有专用 IP) | |
| 快速原型开发 | | ✓ |
| Shape 多变 (训练) | | ✓ |
| 极致性能调优 | ✓ (可控) | |
| 通用计算 (任意算法) | | ✓ |

---

## 7. 总结

### 7.1 硬件能力"必须项"

TopK V2 算子的运行**硬性依赖**以下能力,缺一不可:

1. **Vector ALU** — histogram / CumSum / Cast
2. **Scalar Unit** — BinarySearch / 循环控制
3. **SIMT 微码** — 64-bit atomic add (`asc_vf_call`)
4. **AscendC::TopK** API — 最终 K 选取
5. **AscendC::Sort** API — 最终排序
6. **HBM** — 输入/输出/workspace 存储
7. **UB ≥ 192KB / 核** — TQue / TBuf 分配
8. **DataCopyPad** — 支持 pad 模式的 DMA
9. **HardEvent** — PIPE 间事件同步
10. **PipeBarrier<PIPE_ALL>** — 跨 PIPE 屏障
11. **SetAtomicAdd<int32_t>** — GM 32 位原子加
12. **多核 (≥ 8 AIV)** — lastDimRealCore 并行

### 7.2 硬件能力"加分项"

以下能力可显著提升性能:

1. **更多核数** — 跨核并行度提升
2. **更大 UB** — 允许更大 tileSize,减少 loop 次数
3. **更高 HBM 带宽** — 减少输入/输出 IO 时间
4. **更宽 Vector** — histogram 一次算更多 bins
5. **64-bit atomic 原语** — INT64 path 加速

### 7.3 设计要点 (从硬件视角)

1. **显式同步是性能的关键**: 每一个 `PipeBarrier<PIPE_ALL>` 都意味着一次跨核停顿,代码尽量减少其使用次数。
2. **Atomic 是 mode⑥ 的瓶颈**: 跨核累加到 `globalHistGm` 时,256 个 bin 上原子操作竞争是主要性能损失点。
3. **专用 IP 是核心优势**: AscendC::TopK / Sort 让 TopK 类算子性能远超 GPU 的 thrust 实现。
4. **UB 容量决定模式选择**: Host 端 Tiling 的模式决策本质是"在 192KB UB 限制下如何最大化吞吐"。
5. **PASS 数 = 主要延迟来源**: B8=1, B16=2, B32=4, B64=8 → INT64 TopK 比 INT8 慢 8 倍的 radix pass 部分。
6. **跨核数据流只有 GM**: Ascend 没有 GPU 的 shared memory 概念,核间通信必须走 GM。
7. **Tiling = 软件 GPU 调度器**: Host Tiling 代码承担了 GPU 硬件 Warp Scheduler 的工作。

---

## 附录 A: 关键源码行号速查

### A.1 入口路由
- `top_k_v2_apt.cpp:323` — `top_k_v2()` 入口
- `top_k_v2_apt.cpp:198` — `generateOpObject`
- `top_k_v2_apt.cpp:269-321` — MergeSort 路径路由

### A.2 关键算法
- `radix_sort_top_k.h:496` — `ProcessMultiBlockTopK` 主循环
- `radix_sort_top_k.h:674` — `FindBoundary`
- `radix_sort_top_k.h:719` — `BinarySearch`
- `radix_sort_top_k.h:845` — `StoreFinalAnswer2Gm` (TopK + Sort)
- `radix_sort_top_k.h:913` — `SortTopKRes`
- `radix_sort_top_k.h:1103` — `PreProcess` (Twiddle 入口)

### A.3 同步原语
- `radix_sort_top_k.h:362-363` — MTE2→V event
- `radix_sort_top_k.h:533-534` — V→MTE3 event
- `radix_sort_top_k.h:590, 596` — SIMT atomic path
- `radix_sort_top_k.h:603` — `SetAtomicAdd<int32_t>`
- `radix_sort_top_k.h:620, 643, 659` — `PipeBarrier<PIPE_ALL>`

### A.4 内存分配
- `radix_sort_top_k_base.h:59-65` — TQue/TBuf 声明
- `radix_sort_top_k_base.h:75-91` — `BaseInit` 函数
- `radix_sort_top_k_inter_core_template_optimization.h:86-91` — InitBuffer 调用

### A.5 关键常量
- `top_k_constant_var_simd.h` — `RADIX_SORT_BIN_NUM=256, SHIFT_BIT_NUM=8`
- `radix_topk_constant.h:39` — `SUPPORT_SORT_MAX_BYTE_SIZE=8000`
- `sort_with_index_tiling.h:39` — `NEED_UB_SIZE_BYTE=221184`
- Host: `top_k_v2_tiling_arch35.cpp` — `SORT_AND_TOP_K_THRESHOLD=10000000` 等

---

## 附录 B: 与本文档相关的其他文档导航

| 想了解... | 看哪份文档 |
|---|---|
| 代码整体组成 | `CODE_ANALYSIS.md` |
| Tiling 调度策略 (Host 侧) | `TILING_DETAIL.md` |
| CUDA 对比 (软件视角) | `TILING_DETAIL.md §11` |
| 7 种模式选择策略 | `TILING_DETAIL.md §12` |
| Kernel 7 种模式实现细节 | `KERNEL_DETAIL.md` |
| **硬件能力需求 (本文档)** | `HARDWARE_REQUIREMENTS.md` |

---

**文档状态**: 完成。
**作者**: Claude (基于源码反推,不替代昇腾官方 datasheet)。
**最后更新**: 2026-06-15。

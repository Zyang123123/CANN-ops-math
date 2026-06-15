# TopK V2 Kernel 侧详细分析

> 本文聚焦 `op_kernel/` 目录下的 device 侧实现，对照 Tiling 层 7 种执行模式分别说明。
> 配合 `TILING_DETAIL.md`（host 侧调度）和 `CODE_ANALYSIS.md`（整体架构）阅读。
> 风格：中文 + 章节表格 + 伪代码，关键代码引用使用 `文件:行号` 格式。

---

## 1. Kernel 侧全局架构

### 1.1 调用链总览

```
extern "C" top_k_v2()                                [top_k_v2_apt.cpp:323]
  ├─ SetSysWorkspace / GetUserWorkspace              // 取出 workspace GM
  ├─ 按 ORIG_DTYPE_X + TILING_KEY_VAR 二级路由        [apt.cpp:334-427]
  │     ↓
  ├─ generateOpObject<T,UNSIGNED,NUM_PASS,IDX,IDX_TO> [apt.cpp:198]
  │     ├─ isSingleBlock   → RadixSortTopKSingleBlockOpObject → RadixSortTopKSingleBlock
  │     ├─ isSortAndTopK   → SortAndTopKOpObject              → SortAndTopKMoreCore
  │     ├─ isSingleCore    → RadixSortTopKSingleCoreOpObject  → RadixSortTopKSingleCore
  │     ├─ isMultiCoreOptim→ RadixSortTopKMultiCoreOpObject   → RadixSortTopKMultiCoreOptimization
  │     └─ else(多核默认)  → RadixSortTopKOpObject             → RadixSortTopK
  │
  └─ generateMergeTopK*（TilingKey 偏移路由）         [apt.cpp:269-321]
        ├─ 普通偏移 10000 → MergeSort                              (Small)
        ├─ 23003 偏移      → TopKMergeSortMoreCore                 (MoreCore)
        └─ 33003 偏移      → TopKMergeSortIntraCore                (IntraCore)
```

### 1.2 Host ↔ Kernel 通信载体

| 通道 | 携带内容 |
|---|---|
| `tiling` GM | `TopKV2TilingDataSimd` 结构体序列化字节流（modeType、tileSize、核数、loop 次数等） |
| `TILING_KEY_VAR` | 编译期常量，决定模板实例化（dtype × Merge 偏移） |
| `workspace` GM | 中间直方图、tileTopkValue、SortWithIndex 临时数据 |

---

## 2. 文件清单（按职责分类）

### 2.1 入口与路由

| 文件 | 行数 | 职责 |
|---|---:|---|
| `top_k_v2_apt.cpp` | 429 | kernel C 入口 `top_k_v2()`，按 dtype + TilingKey 路由 |

### 2.2 Radix 系列核心（5 种子模板）

| 文件 | 行数 | 对应模式 | 关键类 |
|---|---:|:---:|---|
| `radix_sort_top_k_base.h` | 92 | 共享基类 | `RadixSortTopKBase` |
| `radix_sort_top_k.h` | 1196 | ⑥ MultiCore | `RadixSortTopK` |
| `radix_sort_top_k_single_block.h` | 204 | ② SingleBlock | `RadixSortTopKSingleBlock` |
| `radix_sort_top_k_single_core.h` | 733 | ④ SingleCore | `RadixSortTopKSingleCore` |
| `radix_sort_top_k_inter_core_template_optimization.h` | 366 | ⑤ MultiCoreOptim | `RadixSortTopKMultiCoreOptimization` |

### 2.3 Merge 系列核心（3 种子模板）

| 文件 | 行数 | 对应模式 | 关键类 |
|---|---:|:---:|---|
| `top_k_merge_sort.h` | 265 | ① SmallSizeMergeSort | `topkV2::MergeSort` |
| `top_k_merge_sort_more_core.h` | 629 | ③a Fp32MergeMoreCore | `topkV2::TopKMergeSortMoreCore` |
| `top_k_merge_sort_intra_core.h` | 356 | ③b Fp32MergeIntraCore | `topkV2::TopKMergeSortIntraCore` |
| `top_k_merge_sort_simd.h` | 174 | ① 的 SIMD 辅助 | `KernelVbsMergeSort` |

### 2.4 SortAndTopK（兜底）

| 文件 | 行数 | 对应模式 | 关键类 |
|---|---:|:---:|---|
| `sort_and_top_k_more_core.h` | 320 | ⑦ SortAndTopK | `SortAndTopK::SortAndTopKMoreCore`（继承 `Sort::SortRadixMoreCore`） |

### 2.5 SortWithIndex（后处理）

| 文件 | 行数 | 职责 |
|---|---:|---|
| `sort_with_index_entry.h` | 135 | 入口 `sortwithindexForTopK()`，按 tilingKeyForSort 路由 |
| `sort_with_index_single_block.h` | 73 | 单 Block radix sort（小 K） |
| `sort_with_index_multi_block.h` | 174 | 多 Block radix sort（大 K） |
| `sort_with_index_merge_sort.h` | 79 | Merge Sort 路径（F16/F32/BF16） |

### 2.6 位宽辅助（SIMD twiddle/reverse）

| 文件 | 行数 | 职责 |
|---|---:|---|
| `radix_sort_topk_b8.h` | 85 | int8/uint8 twiddle |
| `radix_sort_topk_b16.h` | 95 | int16/uint16 twiddle |
| `radix_sort_topk_b32.h` | 101 | int32/uint32/float twiddle |
| `radix_sort_topk_b64.h` | 111 | int64/uint64 twiddle |
| `top_k_radix_block_sort_b8.h` | 316 | b8 RadixBlockSort SIMD |
| `top_k_radix_block_sort_b16.h` | 471 | b16 RadixBlockSort SIMD |
| `top_k_radix_block_sort_b32.h` | 435 | b32 RadixBlockSort SIMD |
| `top_k_radix_block_sort_b64.h` | 321 | b64 RadixBlockSort SIMD |

### 2.7 常量与工具

| 文件 | 行数 | 关键内容 |
|---|---:|---|
| `top_k_constant_var_simd.h` | 70 | `RADIX_SORT_BIN_NUM=256`、`SHIFT_BIT_NUM=8`、`B*_BITE_SIZE`、双缓冲因子 |
| `radix_topk_constant.h` | 41 | 11 种 dtype 的 max/min 值 |
| `radix_topk_util.h` | 84 | `GetTypeMaxValue/Min`、`TopkGetMax/Min` |
| `top_k_util_type_simd.h` | 78 | `ROUND_UP_AGLIN`、`CeilDivMul` |

---

## 3. 入口分发：TILING_KEY 路由

### 3.1 TilingKey 编码规则

| TilingKey | dtype | 路由函数 | Kernel 模板 |
|---:|---|---|---|
| 1001 | int8 | `generateOpObject` | Radix 系列 5 选 1 |
| 1002 | int16 | 同上 | 同上 |
| 1003 | int32 | 同上 | 同上 |
| 1004 | int64 | 同上 | 同上 |
| 2001 | uint8 | 同上 | 同上 |
| 2002 | uint16 | 同上 | 同上 |
| 2003 | uint32 | 同上 | 同上 |
| 2004 | uint64 | 同上 | 同上 |
| 3003 | float | 同上 | 同上 |
| 3002 | float16 | 同上 | 同上 |
| 4002 | bfloat16 | 同上 | 同上 |
| **13002** | float16 | `generateMergeTopKObject` | `MergeSort`（Small） |
| **13003** | float | 同上 | `MergeSort`（Small） |
| **14002** | bfloat16 | 同上 | `MergeSort`（Small） |
| **23003** | float | `generateMergeTopKMoreCoreObject` | `TopKMergeSortMoreCore` |
| **33003** | float | `generateMergeTopKIntraCoreObject` | `TopKMergeSortIntraCore` |

> 编码规律：`1xxxx`=MergeSmall（+10000）、`2xxxx`=MergeMoreCore、`3xxxx`=MergeIntraCore。

### 3.2 generateOpObject 内部 5 选 1

| 判定字段 | 字段值 | 路由到 | 模板类型 |
|---|---|---|---|
| `lastDimNeedCore == 1` | — | `RadixSortTopKSingleBlockOpObject` | `RadixSortTopKSingleBlock` |
| `modeType == 5 (SORT_AND_TOP_K_MODE)` | — | `SortAndTopKOpObject` | `SortAndTopKMoreCore` |
| `modeType == 1 (SINGLE_CORE_MODE)` | — | `RadixSortTopKSingleCoreOpObject` | `RadixSortTopKSingleCore` |
| `modeType == 4 (MULT_CORE_OPTIM_MODE)` | — | `RadixSortTopKMultiCoreOpObject` | `RadixSortTopKMultiCoreOptimization` |
| `else（modeType==2 MULT_CORE_MODE） `| — | `RadixSortTopKOpObject` | `RadixSortTopK` |

> 进一步分支：`isInInt32Range` 决定 `T_INDEX` 为 `int32_t` 还是 `int64_t`，影响 GM 累加原子操作选 `SetAtomicAdd<int32_t>` 还是 SIMT `asc_vf_call`。

### 3.3 分发伪代码（精简）

```cpp
extern "C" __global__ void top_k_v2(x, k, values, indices, workspace, tiling) {
    SetSysWorkspace(workspace);
    GM_ADDR globalWorkGm = GetUserWorkspace(workspace);
    
    #if ORIG_DTYPE_X == DT_FLOAT
        TILING_KEY_IS(TOPK_COMMON_TILING_KEY_FLOAT);       // 3003
        TILING_KEY_IS(TOPK_MERGE_SORT_TILING_KEY_FLOAT);   // 13003
        TILING_KEY_IS(TOPK_MERGE_SORT_MORE_CORE_TILING_KEY_FLOAT);   // 23003
        TILING_KEY_IS(TOPK_MERGE_SORT_INTRA_CORE_TILING_KEY_FLOAT);  // 33003
        #if TILING_KEY_VAR == 3003
            generateOpObject<float, uint32_t, B32_BITE_SIZE, DTYPE_INDICES>(...);
        #elif TILING_KEY_VAR == 13003
            generateMergeTopKObject<float, float, DTYPE_INDICES>(...);
        #elif TILING_KEY_VAR == 23003
            generateMergeTopKMoreCoreObject<float, float, DTYPE_INDICES>(...);
        #elif TILING_KEY_VAR == 33003
            generateMergeTopKIntraCoreObject<float, float, DTYPE_INDICES>(...);
        #endif
    #endif
    // ... 其他 dtype 类似
}
```

---

## 4. 七种模式 × Kernel 实现对照表

| 模式 | 实现文件 | 类名 | 模板参数 | 入口函数 | 主要 AscendC API |
|:---:|:---|:---|:---|:---|:---|
| ① SmallSizeMergeSort | `top_k_merge_sort.h` | `topkV2::MergeSort<T,CONVERT,TILING,IS_LARGEST,INDEX>` | T=F16/F32/BF16 | `ProcessSort()` | `Concat` + `Sort`（经 `KernelVbsMergeSort`） |
| ② SingleBlock | `radix_sort_top_k_single_block.h` | `RadixSortTopKSingleBlock<T,U,IS_LARGEST,IS_SORT,IDX,IDX_TO>` | 全类型 | `Process()` | `AscendC::TopK<RADIX_SELECT>` |
| ③a Fp32MergeMoreCore | `top_k_merge_sort_more_core.h` | `TopKMergeSortMoreCore<T,CONVERT,IS_DESCEND,INDEX>` | float 专用 | `Process()` | `Sort` + `MrgSort`（4-way） |
| ③b Fp32MergeIntraCore | `top_k_merge_sort_intra_core.h` | `TopKMergeSortIntraCore<T,INDEX,IS_DESCEND>` (extends `Sort::SortMergeIntraCore`) | float 专用 | `Process()` | `Sort` + `MrgSort` + Extract |
| ④ SingleCore | `radix_sort_top_k_single_core.h` | `RadixSortTopKSingleCore<T,U,NUM_PASS,IS_LARGEST,IS_SORT,IDX,IDX_TO>` | 全类型 | `Process()` | Radix histogram + `AscendC::TopK` |
| ⑤ MultiCoreOptim | `radix_sort_top_k_inter_core_template_optimization.h` | `RadixSortTopKMultiCoreOptimization<T,IS_LARGEST,IS_SORT,IDX_TO>` | 全类型 | `Process()` | Radix + 集中 TopK |
| ⑥ MultiCore（Medium/Big） | `radix_sort_top_k.h` | `RadixSortTopK<T,U,NUM_PASS,IS_LARGEST,IS_SORT,IDX,IDX_TO>` | 全类型 | `ProcessTopK()` | Radix + `TopK`/`Sort` |
| ⑦ SortAndTopK | `sort_and_top_k_more_core.h` | `SortAndTopK::SortAndTopKMoreCore<T,T_TO,UT,IDX,isDescend>` (extends `Sort::SortRadixMoreCore`) | 全类型 | `ProcessTopK()` | 完整 Radix Sort + 截取 |

---

## 5. Radix Select 核心流程（模式⑥ 主路径）

### 5.1 NUM_PASS 计算规则

| 数据类型 | 位宽 | UNSIGNED_TYPE | NUM_PASS | 单 pass 桶数 |
|---|:---:|---|:---:|:---:|
| int8 / uint8 | 8 | uint8_t | `B8_BITE_SIZE=1` | 256 |
| int16 / uint16 / float16 / bfloat16 | 16 | uint16_t | `B16_BITE_SIZE=2` | 256 |
| int32 / uint32 / float | 32 | uint32_t | `B32_BITE_SIZE=4` | 256 |
| int64 / uint64 | 64 | uint64_t | `B64_BITE_SIZE=8` | 256 |

> 每个 pass 处理 8 bit（`SHIFT_BIT_NUM=8`），共 `sizeof(T)` 个 pass。

### 5.2 ProcessMultiBlockTopK 主循环（`radix_sort_top_k.h:496`）

```
输入: totalDataNum_(N), numTileData_(tileSize), lastDimRealCore_, K=topkValueInput_

for round = NUM_PASS-1 down to 0:           // MSB → LSB
    1. 清空 UB 的 tileCusumBuffer（256-bin × lastDimTileNumTimes_）
    2. 第 0 号核清空 GM 的 cumSumBinsGm_（256-bin × unsortedAxisId）
    3. SyncAll()
    4. 各核并行 for tileId = startTileId; tileId < tileCount; tileId += lastDimRealCore_:
         a. CopyDataIn / CopyDataInWithReuseBuffer  // GM tile → UB, 第一个NUM_PASS以及所有核第一批数据时开辟buffer，之后复用
         b. PreProcess (Twiddle)                    // 有符号 → 无符号
         c. GetTileExcusive                         // 计算 256-bin 直方图（掩码按 round）
         d. DataCopyPad + SetAtomicAdd<int32_t>     // UB 直方图 原子累加到 GM
            （int64 走 asc_vf_call<CopyCumSumToGmB64>）
    5. SyncAll()
    6. FindBoundary (BinarySearch)                   // 找 boundaryBin，使用二分法
    7. 更新 involvedDataMask、andDataMask
    8. topkValueInput_ -= boundaryBinPrevCuSum        // 剩余 K
    9. 各核更新 tileTopkValue_(tileTopkValueIndex)
    10. SyncAll()

最后 (按 lastDimTileNum_ vs platformCoreNum_):
   ├─ medium mode: StoreAnswer2Gm
   └─ big data mode: StoreBigSizeData2Gm

if IS_SORT && startTileId==0 && !needSortWithIndex_:
   if NUM_PASS != B64 || K <= SUPPORT_SORT_MAX_SIZE/2:
       SortTopKRes(xLocal)                          // AscendC::Sort
```

### 5.3 关键子函数表

| 函数 | 行号 | 作用 |
|---|---:|---|
| `InitPara` | 295 | 从 tilingData 读取 N、K、tileSize、核数等 |
| `Init` | 207 | 设置 GM buffer、分配 UB、初始化 Queue |
| `ProcessTopK` | 323 | 外层 sortLoopTimes 循环；末尾可选调 SortWithIndex |
| `ProcessMultiBlockTopK` | 496 | Radix Select 主体（NUM_PASS 轮） |
| `PreProcess` | 1103 | Twiddle：signed → unsigned（按 dtype 调 B8/B16/B32/B64） |
| `GetTileExcusive` | 1073 | 单 tile 的 256-bin 直方图（带掩码） |
| `FindBoundary` | 674 | 调用 `BinarySearch`，找 K 落在哪个 bin |
| `BinarySearch` | 719 | 在 cumSumLocal 上二分查找 |
| `StoreAnswer2Gm` | 429 | medium 模式：调用 `UpdateTileTopkValue` + `StoreFinalAnswer2Gm` |
| `StoreBigSizeData2Gm` | 744 | big 模式：`UpdateTileRealTopk` + 重新读 GM + `StoreFinalAnswer2Gm` |
| `StoreFinalAnswer2Gm` | 845 | **核心**：调 `AscendC::TopK<RADIX_SELECT>` 精确提取 K |
| `SortTopKRes` | 913 | 调 `AscendC::Sort<RADIX_SORT>` 做最终排序 |
| `CopyTopkValue2Gm` | 963 | 把结果从 UB 写回 GM |

### 5.4 数据流（NUM_PASS=4 示例）

```
GM inputValueGm_  ─┐
                   │ CopyDataIn
                   ▼
UB inQueueX_      ─┐
                   │ PreProcess(Twiddle)
                   ▼
UB inputXCopy_    ─┐
                   │ GetTileExcusive (256-bin histogram, 掩码按 round)
                   ▼
UB tileCusumBuffer ─┐
                    │ DataCopyPad + SetAtomicAdd<int32_t>
                    ▼
GM cumSumBinsGm_   ─┐
                    │ FindBoundary (BinarySearch)
                    ▼
              boundaryBin (K 落在此 bin)
                    │
                    ▼  StoreFinalAnswer2Gm
              AscendC::TopK<RADIX_SELECT>
                    │
                    ▼
GM topkValueGm_ / topkValueIndexGm_
```

### 5.5 int32 vs int64 累加路径

| 条件 | 路径 | 实现 |
|---|---|---|
| `T_INDEX == int32_t` | `SetAtomicAdd<int32_t>` + `DataCopyPad` | 硬件原子加 |
| `T_INDEX == int64_t` 且 `NUM_PASS == B64` | `asc_vf_call<CopyCumSumToGmB64>` | SIMT 64-bit 原子加微码 |
| `T_INDEX == int64_t` 且 `NUM_PASS < B64` | `asc_vf_call<CopyCumSumToGmB8B16B32>` | SIMT 32-bit 累加再写回 |

---

## 6. Merge Sort 三变体差异

### 6.1 三变体对照表

| 维度 | ① Small | ③a MoreCore | ③b IntraCore |
|---|---|---|---|
| 文件 | `top_k_merge_sort.h` | `top_k_merge_sort_more_core.h` | `top_k_merge_sort_intra_core.h` |
| 基类 | 独立 struct | 独立 struct | 继承 `Sort::SortMergeIntraCore` |
| 入口 | `ProcessSort()` | `Process()` | `Process()` |
| 数据类型 | F16/F32/BF16 | float32 专用 | float32 专用 |
| 排序轴规模 | ≤ 1024 | 中等，splitCoreNum > 1 | 中等，blocksPerRow > 1 |
| 核间并行 | 外轴并行（oneCoreRowNum/核） | splitCoreNum 切排序轴 | 单核内多 block |
| 核心 API | `KernelVbsMergeSort.VbsMergeSort` (Concat+Sort) | `Sort` + `MrgSort`（4-way 多轮） | `Sort` + `MrgSort` + `Extract` |
| Workspace | 无（全 UB） | unsortedDim × alignInput × 8B × 2(DoubleBuffer) | 较少 |
| 最终截取 | Sort 时直接输出 K | `FinalSortAndCopy` | `ExtractAndCopyOut` |
| K/N 触发 | 任意 | 0.25 ≤ K/N < 0.50 | 0.25 ≤ K/N < 0.50 |

### 6.2 Small (top_k_merge_sort.h) 流程

```
ProcessSort():
  for i in [0, sortLoopTimes):
    ProcessSingleBlockSort(inputValueGm_[i*unsortedDimParallel*oneCoreRowNum*numTileData])
      ├─ tileId = GetBlockIdx()
      ├─ nowCoreRealRowNum = min(unsortedDimNum - unsortedDimIndex, oneCoreRowNum)
      ├─ CopyDataIn (GM → UB, 双缓冲)
      ├─ xLocal = inQueueX_.DeQue
      ├─ if BF16: vbsSort.VbsMergeSortBf16(xLocal, sortedValueLocal, sortedValueIndexLocal, ...)
      │   else:  vbsSort.VbsMergeSort    (xLocal, sortedValueLocal, sortedValueIndexLocal, ...)
      ├─ if INDEX_TYPE==int64: Cast int32 索引 → int64
      ├─ EnQue + CopyValue2Gm (UB → GM)
```

### 6.3 MoreCore (top_k_merge_sort_more_core.h) 流程

```
Process():
  ├─ MultiCoreInit                       // 各核分到一段排序轴
  ├─ for tileNum in [numTileData, tailNum]:
  │     SortInSingleCore(tileNum, offsetPerCore)
  │       ├─ CopyInData (GM → UB)
  │       ├─ InitIndexLocal               // 生成 0..N 索引
  │       ├─ if BF16: DoSortBf16
  │       │   else:    DoSort             // AscendC::Sort + FlipSignBit
  │       └─ CopyOutWorkSpace             // 各核排序结果写 GM
  ├─ SyncAll()
  ├─ CopyInMultiCore                      // GM 排序结果 → UB
  ├─ SortInMultiCore
  │     └─ DealingMergeSort               // 多轮 4-way MrgSort
  └─ FinalSortAndCopy                     // 最终 Sort + 提取 K
        └─ ExtractAndCopyOut
```

### 6.4 IntraCore (top_k_merge_sort_intra_core.h) 流程

```
Process():                                  // 三阶段流水
  for batchIdx in [0, batchNum):
    === Phase 1: SortSingleBatchInUb ===
       pipe.InitBuffer(inQueueX_, DOUBLE_BUFFER, blockSortSize * sizeof(T))
       pipe.InitBuffer(sortedOutQueue_, DOUBLE_BUFFER, sortBufferSize)
       Base::SortSingleBatchInUb(inputXGm_, batchIdx * sortAxisNum)

    === Phase 2: MergeSingleBatch (4-way) ===
       pipe.InitBuffer(mergeInQueue_, 1, mergeBufferSize)
       pipe.InitBuffer(mergeOutQueue_, 1, mergeBufferSize)
       for i in [0, numBlocks, MERGE_LIST_MAX_NUM):
           MergeOneGroup(i, groupBlockCount, ...)   // 4-way MrgSort

    === Phase 3: ExtractAndCopyOut ===
       pipe.InitBuffer(extractInQueue_, DOUBLE_BUFFER, extractInSize)
       pipe.InitBuffer(outValueQueue_, DOUBLE_BUFFER, extractChunkSize * sizeof(T))
       ExtractAndCopyOut(batchIdx, resultRegion)
```

> override 的 4 个关键函数：`SortSingleBatchInUb`、`MergeSingleBatch`、`MergeOneGroup`、`ExtractAndCopyOut`。
> 关键优化：`dstElemCount >= topKValue_` 时提前终止（不再继续归并）。

---

## 7. SortAndTopK 流程（模式⑦ 兜底）

### 7.1 调用链

```
SortAndTopKMoreCore::ProcessTopK()                     [sort_and_top_k_more_core.h:308]
  for i in [0, sortLoopTimes):
    ProcessSortAndTopK(inputXGm_[loopOffset], loopOffset, i)
      ├─ ClearWorkSapce + SyncAll
      ├─ SetDoubleBuffer (inputXDbGm_, idxDbGm_)       // 双缓冲 workspace
      ├─ for round in [0, sizeof(T)):                  // 完整 radix sort
      │     GetGlobalExcusiveSum(round, ...)            // 全局前缀和
      │     SyncAll
      │     ComputeOnePass(round, ...)                  // 单 pass 重排
      │     SyncAll
      └─ GetTopKRes(sortLoopRound)                     // 截取前 K
            ├─ GetValueResData(...)                     // 输出 value
            └─ GetIndexResData(...)                     // 输出 index
```

### 7.2 与模式⑥ 的根本差异

| 维度 | 模式⑥ Radix Select | 模式⑦ SortAndTopK |
|---|---|---|
| pass 数 | NUM_PASS（按位宽） | sizeof(T)（等同） |
| 目标 | 找到 boundary bin 后用 TopK API 精确提取 | 完整排序后再截取 |
| Workspace | 小（仅 histogram） | 大（双缓冲 sortedOutValueGM/IdxGM） |
| 是否需要 boundary | 是 | 否 |
| 性能 | 通常更快 | N ≥ 10M 或 UB 装不下时使用 |
| 父类 | `RadixSortTopKBase` | `Sort::SortRadixMoreCore`（共享 Sort 模块） |

---

## 8. SortWithIndex 后处理

### 8.1 触发条件

```cpp
if (IS_SORT) {
    needSortWithIndex_ = (topkValueInput_ > SUPPORT_SORT_MAX_SIZE);  // 2000
}
if (needSortWithIndex_) {
    pipe.Reset();
    sortwithindexForTopK<T_INDEX_TO>(topkValuesGmAddr_, topkIndicesGmAddr_,
                                     valueAddr_, indicesAddr_,
                                     sortWithIndexWorkspace_, tilingDataPtr_, &pipe);
}
```

> 即：用户要求 `sorted=true` 且 K > 2000 时，主 Radix/Merge 完成后还需要单独排序。

### 8.2 tilingKeyForSort 路由表

| tilingKeyForSort | dtype | 路由 |
|---:|---|---|
| 1001/1002/1003/1004 | int8/16/32/64 | `generateOpForTopK` → MultiBlock（int32/int64 范围决定 IDX） |
| 2001/2002/2003/2004 | uint8/16/32/64 | 同上 |
| 3003 | float | `generateOpForTopK` (UT=uint32) |
| 3002 | float16 | `generateOpForTopK` (UT=uint16) |
| 4002 | bfloat16 | `generateOpForTopK` (UT=uint16, Convert=float) |
| **13002** | float16 | `generateMergeSortOpForTopK` → `SortWithIndexMergeSort` |
| **13003** | float | 同上 |
| **14002** | bfloat16 | 同上 |

### 8.3 三种子实现

| 子类 | 文件 | 触发条件 | 算法 |
|---|---|---|---|
| `SortWithIndexSingleBlock` | `sort_with_index_single_block.h` | `isSingleBlock && modeType==1` | 单 Block radix sort |
| `SortWithIndexMultiBlock` | `sort_with_index_multi_block.h` | `isInt32Range` 或 int64 | 多核 radix sort |
| `SortWithIndexMergeSort` | `sort_with_index_merge_sort.h` | Merge tilingKey（13xxx/14xxx） | 复用 `topkV2::MergeSort` |

### 8.4 调用链

```
RadixSortTopK::ProcessTopK() 末尾:
  if (needSortWithIndex_):
      pipe.Reset()                                              // 重置 UB
      sortwithindexForTopK<T_INDEX_TO>(topkValues, topkIndices,
                                       outValues, outIndices,
                                       sortWithIndexWorkspace, tiling, &pipe)
        └─ 按 tilingKeyForSort 选 XType/ConvertType/UnsignedType
              └─ generateOpForTopK / generateMergeSortOpForTopK
                    └─ Init + Process / ProcessSort
```

---

## 9. UB 布局对比

### 9.1 RadixSortTopK（模式⑥）UB 分配

| TBuf/Queue | 大小 | 用途 |
|---|---|---|
| `inQueueX_` | `numTileData * sizeof(T)` | 输入数据缓冲 |
| `blockCumSumTbuf_` | `256 * sizeof(T_INDEX) * lastDimTileNumTimes` | tile 直方图累加 |
| `inputXCopyTbuf_` | `numTileData * sizeof(UNSIGNED_TYPE)` | Twiddle 后无符号数据 |
| `tileTopkValueTbuf_` | `lastDimTileNumTimes * 4` | 每 tile 的 topk 计数 |
| `remainTileTopkValueTbuf_` | 同上 | 剩余 topk |
| `dataSetCumSumTbuf_` | `256 * sizeof(T_INDEX)` | 数据集级前缀和 |
| `topkSrcIndexTbuf_` | `numTileData * sizeof(int32)` | TopK API 输入索引 |
| `topkSrcIndexCopyTbuf_` | `numTileData * sizeof(T_INDEX_TO)` | SortTopKRes 用 |
| `sortedShareMemTbuf_` | `topkAcApiTmpBufferSize` | TopK/Sort API 临时空间 |
| `topkValueQueue_` | `min(numTileData, K) * sizeof(T)` | 输出值 |
| `topkValueIndexQueue_` | 同上 | 输出索引 |

### 9.2 SingleBlock（模式②）UB 分配（精简版）

| TBuf/Queue | 大小 | 用途 |
|---|---|---|
| `inputXQue_` | `batchNumInUb * ROUND_UP_AGLIN(numTileData) * sizeof(T)` | 多行输入 |
| `valuesQue_` | `batchNumInUb * ROUND_UP_AGLIN(K * sizeof(T))` | 多行 K 输出值 |
| `indicesQue_` | `batchNumInUb * ROUND_UP_AGLIN(K * sizeof(T_INDEX_TO))` | 多行 K 输出索引 |
| `indicesOutTbuf_` | `batchNumInUb * ROUND_UP_AGLIN(K * sizeof(int32))` | TopK API 内部 int32 索引 |
| `topKApiTmpTBuf_` | `ROUND_UP_AGLIN(topKApiTmpSize)` | TopK API 临时 |

### 9.3 MergeSort Small（模式①）UB 分配

| Queue | 大小 | 用途 |
|---|---|---|
| `inQueueX_` | `DOUBLE_BUFFER * ROUND_UP_AGLIN(numTileData) * oneCoreRowNum * sizeof(T)` | 双缓冲输入 |
| `outValueQueue_` | 同上 | 排序后值 |
| `outIndexQueue_` | 同上 × `sizeof(INDEX_TYPE)` | 排序后索引（支持 int64） |

---

## 10. 关键 AscendC API 使用对照

| API | 使用位置 | 作用 |
|---|---|---|
| `AscendC::TopK<T,...,RADIX_SELECT>` | SingleBlock、StoreFinalAnswer2Gm | 在 boundary bin 内精确提取 K |
| `AscendC::Sort<T,...,RADIX_SORT>` | SortTopKRes、MoreCore、SortAndTopK | 完整排序 |
| `AscendC::MrgSort` | MoreCore、IntraCore | 4-way 归并排序 |
| `AscendC::Concat` | Small (经 VbsMergeSort) | 拼接相邻段 |
| `AscendC::DataCopyPad` | 所有 CopyDataIn/CopyOut | GM ↔ UB 带 padding 传输 |
| `AscendC::SetAtomicAdd<int32_t>` | Radix 累加阶段 | int32 直方图原子累加 |
| `asc_vf_call<CopyCumSumToGmB64>` | Radix 累加（int64） | SIMT 64-bit 原子加 |
| `AscendC::Cast<int64_t,int32_t>` | int64 索引输出 | int32 → int64 类型转换 |
| `AscendC::Adds` | StoreFinalAnswer2Gm | 局部索引 → 全局索引（+tileOffset） |
| `AscendC::Duplicate` | Init 阶段 | UB 清零 |
| `PipeBarrier<PIPE_ALL>` / `SyncAll` | 多处 | 流水同步 + 核间同步 |

---

## 11. 完整调用链伪代码（以最复杂的模式⑥ 为例）

```cpp
// === Host: Tiling 已确定 modeType=2(MULT_CORE), lastDimNeedCore>1 ===

// === Device ===
extern "C" top_k_v2(x, k, values, indices, workspace, tiling) {
    SetSysWorkspace(workspace);
    GM_ADDR globalWorkGm = GetUserWorkspace(workspace);
    
    // TILING_KEY 路由（以 float 为例）
    TILING_KEY_IS(3003);  // TOPK_COMMON_TILING_KEY_FLOAT
    #if TILING_KEY_VAR == 3003
        generateOpObject<float, uint32_t, B32_BITE_SIZE=4, DTYPE_INDICES>(
            x, k, values, indices, globalWorkGm, tiling);
    #endif
}

generateOpObject<float, uint32_t, 4, IDX, IDX_TO>(...) {
    GET_TILING_DATA(tilingData, tiling);
    // modeType==2 → 走默认 RadixSortTopKOpObject
    RadixSortTopKOpObject<float, uint32_t, 4, IDX, IDX_TO>(...);
}

RadixSortTopKOpObject(...) {
    // 按 isLargest × isSort 选 4 个模板特化之一
    RadixSortTopK<float, uint32_t, 4, true, true, IDX, IDX_TO> obj;
    obj.Init(...);
    obj.ProcessTopK();
}

RadixSortTopK::ProcessTopK() {
    for (i = 0; i < sortLoopTimes_; i++) {
        topkValueInput_ = topkValueInitInput_;   // 重置 K
        ProcessMultiBlockTopK(inputValueGm_[loopOffset]);
    }
    if (needSortWithIndex_) {                    // K>2000 且 sorted=true
        pipe.Reset();
        sortwithindexForTopK<T_INDEX_TO>(...);
    }
}

RadixSortTopK::ProcessMultiBlockTopK(inputX) {
    for (round = NUM_PASS-1=3; round >= 0; round--) {       // 4 轮 radix
        // 1. 清 UB/GM 直方图
        // 2. 各核并行 for tileId:
        //      CopyDataIn → PreProcess(Twiddle) → GetTileExcusive
        //      DataCopyPad + SetAtomicAdd → cumSumBinsGm_
        // 3. FindBoundary → boundaryBin
        // 4. topkValueInput_ -= boundaryBinPrevCuSum
    }
    // 输出阶段
    if (lastDimTileNum_ <= platformCoreNum_) {
        StoreAnswer2Gm(...);                    // medium
    } else {
        StoreBigSizeData2Gm(...);               // big
    }
    // 可选最终排序
    if (IS_SORT && !needSortWithIndex_ && NUM_PASS != B64) {
        SortTopKRes(xLocal);                    // AscendC::Sort
    }
}
```

---

## 12. 关键差异速查

### 12.1 Radix vs Merge 路径选择

| 维度 | Radix 系（②④⑤⑥⑦） | Merge 系（①③a③b） |
|---|---|---|
| 算法核心 | 多 pass 直方图 + 二分 boundary | Sort + MrgSort 归并 |
| 数据类型 | 全 11 种 | 仅 F16/F32/BF16 |
| 是否需要 Twiddle | 是（转无符号） | 否（直接 float 比较） |
| K 大小敏感 | 不敏感（任何 K 都 OK） | 敏感（K/N 需在 [0.25, 0.5)） |
| K ≤ 1024 + F16/BF16 | — | 走 ① Small |
| K 中等 + F32 + 0.25≤K/N<0.5 | — | 走 ③a/③b |
| 其他 | 走 Radix | — |

### 12.2 单核 vs 多核

| 模式 | 核数策略 |
|---|---|
| ② SingleBlock | 满核并行外轴，单核 UB 装多行 |
| ④ SingleCore | 单核串行处理整行（外轴 ≥ 核数时反而更稳） |
| ⑤ MultiCoreOptim | 满核 + 末轮 K 末位集中策略 |
| ⑥ MultiCore | 满核 + lastDimRealCore 切排序轴 |
| ⑦ SortAndTopK | 满核 + DoubleBuffer GM |

### 12.3 输出索引类型转换

| `T_INDEX_TO` | 处理 |
|---|---|
| int32 | TopK API 直接输出 int32 |
| int64 | TopK API 输出 int32 → `AscendC::Cast<int64_t,int32_t>` |

### 12.4 atomic 累加路径选择

| `T_INDEX` | NUM_PASS | 路径 |
|---|---|---|
| int32 | 任意 | `SetAtomicAdd<int32_t>` + `DataCopyPad` |
| int64 | B64（=8） | `asc_vf_call<CopyCumSumToGmB64>`（SIMT 微码） |
| int64 | B8/B16/B32 | `asc_vf_call<CopyCumSumToGmB8B16B32>`（SIMT 微码） |

---

## 13. 调试与排错指引

| 现象 | 可能原因 | 排查路径 |
|---|---|---|
| Kernel 未命中预期模式 | TilingKey 或 modeType 错误 | 检查 host tiling 输出 → 对照 §3.1/3.2 |
| int64 索引结果错误 | asc_vf_call 路径异常 | 对照 §5.5，确认 SIMT 微码调用 |
| sorted=true 时结果未排序 | needSortWithIndex_ 未触发 | 检查 K > 2000 阈值（§8.1） |
| 大数据模式（lastDimTileNum_ > 核数）异常 | StoreBigSizeData2Gm 路径 | 对照 §5.3，检查 UpdateTileRealTopk |
| Merge Sort 模式 K 截取错误 | dstElemCount >= topKValue 提前终止 | 检查 IntraCore §6.4 Phase 2 |
| UB 空间不足 | Tiling 算错 tileSize | 参考 TILING_DETAIL.md §6 |

---

## 14. 总结

TopK V2 Kernel 侧的设计要点：

1. **二级路由**：TilingKey（dtype × Merge 偏移）→ modeType/lastDimNeedCore → 具体模板
2. **7 种模板各司其职**：从最快的 Small MergeSort 到兜底的 SortAndTopK，复杂度递增
3. **Radix Select 是核心**：MSB→LSB 多 pass 直方图 + 二分 boundary + TopK API 精确提取
4. **int32/int64 双路径**：原子累加分别走硬件原子加和 SIMT 微码
5. **SortWithIndex 后处理**：K > 2000 的排序需求独立成模块，复用 Sort 库
6. **UB 精算**：每个模板的 UB 分配严格匹配 Tiling 层计算的 tileSize
7. **流水与同步**：`PipeBarrier<PIPE_ALL>` + `SyncAll` + `SetFlag/WaitFlag` 确保核间/流水正确性

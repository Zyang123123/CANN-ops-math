# TopK V2 Tiling 层核心调度策略详解

> Tiling（分块策略）是 Ascend CANN 算子在 Host 侧执行的核心环节。它在编译期/运行时根据输入Shape、数据类型、硬件资源（UB大小、核数），计算出一组参数传递给 Device Kernel，决定 Kernel 采用哪种算法模板、如何分块、分配多少核、UB如何划分。

---

## 1. 全局入口与参数校验

### 1.1 入口调用链

```
Tiling4TopKV2()                          # 框架入口
  └─ TopKV2Tiling(context, coreNum)      # 核心Tiling函数
       ├─ IsValidParam()                  # 参数合法性校验
       ├─ 构造 computeNowTileSizeInfo     # 收集当前计算的上下文信息
       ├─ 模式判断与选择（7种模式优先级决策）
       ├─ Workspace大小计算
       ├─ SortWithIndex Tiling（可选）
       └─ 保存TilingData、设置BlockDim/Workspace
```

### 1.2 TilingParse 阶段

在算子编译时，`TilingPrepareForTopKV2()` 从平台信息获取 AI Vector 核心数量，存入 `TopKV2CompileInfo::coreNum`。后续运行时 Tiling 直接使用该预编译信息。

### 1.3 参数校验 (`IsValidParam`)

| 校验项 | 条件 |
|--------|------|
| 输入x数据类型 | 必须是11种支持的类型之一 |
| 输入k数据类型 | 必须是 int32 或 int64 |
| 输出values类型 | 必须与输入x类型一致 |
| 输出indices类型 | 必须是 int32 或 int64 |
| 输出Shape一致性 | values 和 indices 的Shape必须相同 |
| dim属性 | 必须在 `[-inputDimNum, inputDimNum-1]` 范围内 |

### 1.4 核心上下文结构 `TopkComputeNowTileSizeInfo`

Tiling 过程中所有决策都基于这个结构体收集的信息：

```cpp
struct TopkComputeNowTileSizeInfo {
    ge::DataType dataType;           // 输入数据类型
    ge::DataType indicesDType;       // 输出索引类型
    bool isLargest;                  // 取最大(true)/最小(false)
    bool isSort;                     // 结果是否排序
    bool isInInt32Range;             // 排序轴是否在int32范围内
    int64_t lastAxisNum;             // 排序轴（最后一维）大小
    int64_t kValue;                  // K值（输出最后一维大小）
    uint32_t maxCoreNum;             // 平台最大可用核数
    uint64_t ubSizePlatForm;         // 可用UB大小（已减去32K SIMT预留）
    uint64_t ubBlockAlignSize;       // UB块对齐大小（32字节）
    uint32_t unsortedDimNum;         // 外轴总大小（所有非排序维的乘积）
};
```

---

## 2. 七种执行模式决策树

### 2.1 优先级判断流程

`TopKV2Tiling()` 按严格优先级依次尝试 7 种模式，命中即返回：

```
TopKV2Tiling()
  │
  ├─① IsSmallSizeMergeSortMode? ──→ TileModeSmallSizeOptim
  │   条件: dataType∈{float/float16/bfloat16} 且 lastAxisNum ≤ 1024
  │
  ├─② IsSingleBlockMode? ──→ TileModeSmallSize
  │   条件: UB能一次性装下 lastAxisNum 条数据（含TopK API临时空间）
  │
  ├─③ IsFp32MergeSortMode? ──→ TileModeFp32MergeSort
  │   条件: float32 且 0.25 ≤ K/N < 0.50
  │   ├─ IsTopkMergeSortMoreCoreFp32Mode → 多核Merge Sort
  │   └─ IsTopkMergeSortIntraCoreFp32Mode → 核内Merge Sort
  │
  ├─④ IsSingleCoreMode? ──→ TileModeSingleCore
  │   条件: unsortedDimNum ≥ maxCoreNum 且 UB能算出有效tileSize
  │
  ├─⑤ IsMultiCoreOptimMode? ──→ TileMultiCoreOptimSize
  │   条件: K × lastDimTileNum ≤ tileSize 且tileSize ≥ lastAxisNum
  │
  ├─⑥ IsTopkRadixMoreCoreMode? ──→ TileTopkMoreCoreMode
  │   条件: lastAxisNum < 1000万 且 UB能算出有效tileSize
  │
  └─⑦ 兜底: TileModeSortAndTopK
      条件: 以上全部不命中（超大排序轴场景）
```

### 2.2 模式总览表

| 优先级 | 模式名 | modeType | 数据类型限制 | lastAxisNum范围 | 算法 |
|--------|--------|----------|-------------|----------------|------|
| 1 | SmallSizeOptim | 3(implicit) | float/float16/bf16 | ≤ 1024 | Merge Sort(Concat+Sort) |
| 2 | SingleBlock | 3 | 全类型 | ≤ UB容量/一行数据大小 | 单Block Radix Select |
| 3a | Fp32MergeMoreCore | 6 | float32 | 适中，0.25≤K/N<0.50 | 多核Sort→4-way归并→Extract |
| 3b | Fp32MergeIntraCore | 7 | float32 | 适中，0.25≤K/N<0.50 | 核内Sort→4-way归并→Extract |
| 4 | SingleCore | 1 | 全类型 | 中等 | 单核多次Radix Select |
| 5 | MultiCoreOptim | 4 | 全类型 | 较大 | 多核Radix Select+集中TopK |
| 6 | MultiCore(Medium/Big) | 2 | 全类型 | 大 | 多核Radix Select |
| 7 | SortAndTopK | 5 | 全类型 | ≥ 1000万或UB不足 | 完整Radix Sort→截取TopK |

---

## 3. 各模式详细分析

### 3.1 模式① SmallSizeMergeSort (Merge Sort优化)

**判断条件** (`IsSmallSizeMergeSortMode`)：
```
dataType ∈ {DT_FLOAT, DT_FLOAT16, DT_BF16}  AND  lastAxisNum ≤ 1024
```

**为什么选择 Merge Sort：** 排序轴小，可以将整行数据一次加载到UB，使用 AscendC 的 Concat + Sort API 一次性完成带索引排序，再截取前K个。对于浮点小数据场景比 Radix Select 更快。

**Tiling流程** (`TileModeSmallSizeOptim`)：

```
1. 调用 SetMergeSortTmpSize() 计算 Concat API 需要的临时UB大小
2. 调用 ComputeMergeSortTileData() 计算每个核能处理的最大数据量
   ├─ 从UB总量中减去 Sort API 临时空间
   ├─ 计算对齐后的单行数据大小 aglinNum
   ├─ 计算每核最多能处理的行数 oneCoreRowNumMax
   └─ 调用 GetTileDataForMergeSort() 优化核利用率
3. 计算外轴并行参数：
   ├─ oneCoreRowNum = tileData / (bufferNum × aglinNum)
   ├─ sortLoopTimes = ceil(virtualDimNeedCore / maxCoreNum)
   └─ coreNumNeed = min(virtualDimNeedCore, maxCoreNum)
```

**关键参数计算 — `GetTileDataForMergeSort`：**

```
目标: 找到一个tileData，使得：
  - 所有核尽量满载（利用率 ≥ 70%）
  - tileData不超过UB能容纳的最大数据量

算法:
  1. 初始 tileData = 7680（默认值）
  2. 计算 oneCoreRowNum = tileData / (bufferNum × aglinNum)
  3. 计算需要的虚拟核数 virUnsortedDimNeedCoreNum
  4. 若虚拟核数 < 总核数 → 外轴能均摊到所有核 → 返回tileData
  5. 若虚拟核数 ≥ 总核数 → 逐步增大tileData（每次+256，但保证不超过最大tiledata），直到虚拟核数 < 总核数
  6. 检查最后一个loop的核利用率:
     - 若 lastLoopDimNeedCore / maxCoreNum < 0.7 → tileData -= 256，重新计算
     - 确保每个loop至少70%的核在工作
```

**Double Buffer策略：**
```
lastAxisNum ≥ 2048 → bufferNum = 1（关闭双缓冲，UB空间不够）
lastAxisNum < 2048 → bufferNum = 2（启用双缓冲，数据搬运与计算并行）
```

**TilingKey：** 原始dataTypeKey + 10000（偏移量区分Radix模板和Merge模板）

---

### 3.2 模式② SingleBlock

**判断条件** (`IsSingleBlockMode`)：
```
ComputeSingleBlockTileData() 返回 tileSize > 0
  AND  tileSize >= lastAxisNum
```

含义：一个Block的UB可以一次性装下一行数据+TopK API临时空间。

**tileSize计算流程** (`ComputeSingleBlockTileData`)：

```
1. 初始tileSize = 15360（非64位类型）或 10240（64位类型）
2. 计算 TopK API 临时空间大小 (topkAcApiNeedBuffer)
3. 计算运行时所需空间:
   needSpace = batchNumInUb × (
     alignTileData × dtypeSize +      // 输入数据
     alignKValue × dtypeSize +         // TopK值输出
     alignKValue × indexDtypeSize +    // TopK索引输出
     alignKValue × sizeof(int32_t)     // 索引转换缓冲
   )
4. while (topkAcApiNeedBuffer + needSpace > ubSizePlatForm):
     tileSize -= 256
     if tileSize < lastAxisNum → 返回0（不适用）
     重新计算needSpace
5. 返回tileSize
```

**核分配** (`TileModeSmallSize`)：

```
batchNumInUb = tileSize / lastAxisNum    // 单核UB能同时装多少行
batchNumSingleLoop = maxCoreNum × batchNumInUb  // 一轮loop处理的总行数
sortLoopTimes = ceil(unsortedDimNum / batchNumSingleLoop)
coreNumNeed = maxCoreNum                  // 满核运行
```

**为什么能单Block处理：** `lastDimNeedCore = 1`（排序轴不需要分核），外轴用所有核并行。

---

### 3.3 模式③ Fp32MergeSort

专门为 float32 类型优化的模式，仅在 `0.25 ≤ K/N < 0.50` 时触发。

#### 3.3a Fp32MergeMoreCore (多核Merge Sort)

**判断条件** (`IsTopkMergeSortMoreCoreFp32Mode`)：
```
1. dataType == DT_FLOAT
2. kValue > 0
3. 0.25 ≤ K / lastAxisNum < 0.50
4. onceMaxElements ≥ 32（UB能容纳的最小归并单元）
5. splitCoreNum = ceil(lastAxisNum / 2048) > 1
6. splitCoreNum ≤ maxCoreNum
7. unsortedDimNum × splitCoreNum ≤ maxCoreNum（核数够用）
```

**Tiling计算** (`GetTopkMergeMoreCoreFp32`)：

```
splitCoreNum = ceil(lastAxisNum / 2048)     // 排序轴分成多少核
numTileDataSize = lastAxisNum / splitCoreNum // 每核处理的数据量
coreNumNeed = unsortedDimNum × splitCoreNum  // 总需要核数

Workspace = unsortedDimNum × alignInput × 8(Bytes/element) × 2(DoubleBuffer)
```

**Kernel执行流程：**
1. 每核用Sort API排序自己分到的数据段
2. 多轮4-way MrgSort合并各核排序结果
3. 最终单核Extract截取TopK

#### 3.3b Fp32MergeIntraCore (核内Merge Sort)

**判断条件** (`IsTopkMergeSortIntraCoreFp32Mode`)：
```
1. MoreCore 路由未命中
2. dataType == DT_FLOAT 且 unsortedDimNum ≥ maxCoreNum/2
3. kValue > 0 且 0.25 ≤ K/N < 0.50
4. 若 sorted=false 且 kValue ≤ 2000 → 不适用（走Radix更快）
5. blocksPerRow = ceil(N/blockSortSize) ∈ (1, 256]
```


**Tiling计算** ：
```
blockSortSize = UB大小 / Phase2每元素字节数     // 对齐到32
extractChunkSize = UB大小 / Phase3每元素字节数   // 对齐到32
```
**Tiling计算** (`GetTopkMergeIntraCoreFp32`)：
```
blocksPerRow = ceil(lastAxisNum / blockSortSize)
actualCoreNum = min(maxCoreNum, unsortedDimNum)
batchPerCore = ceil(unsortedDimNum / actualCoreNum)
```

**Kernel执行三阶段流水线：**
1. SortSingleBatchInUb — UB内块排序
2. MergeSingleBatch — 4-way增量归并（TopK提前终止）
3. ExtractAndCopyOut — 提取TopK结果

---

### 3.4 模式④ SingleCore

**判断条件** (`IsSingleCoreMode`)：
```
1. unsortedDimNum ≥ maxCoreNum（外轴够大，可以每核分到至少一行）
2. ComputeSingleCoreTileData() 返回 tileSize > 0
```

**设计意图：** 当排序轴不是特别大，但外轴足够多时，让每个核独立处理一行数据（排序轴可能需要多次tile循环），避免核间同步开销。

**tileSize计算** (`ComputeSingleCoreTileData`)：

```
初始 tileSize = 15360 或 10240(64位)

计算运行时空间 (GetSingleCoreTopkRunTimeNeedSpace):
  = 输入Queue(ceilAlign(tileNum) × dtypeSize)
  + 输出ValueQueue(ceilAlign(min(tileNum, K) × dtypeSize))
  + 输出IndexQueue(ceilAlign(min(tileNum, K) × indexDtypeSize))
  + 索引转换(ceilAlign(min(tileNum, K) × sizeof(int32_t)))
  + 直方图累加(BIN_NUM × sizeof(int32_t))
  + 全局直方图(BIN_NUM × indexTypeSize)
  + tileTopK计数(ceilAlign(lastDimTileNum × sizeof(int32_t)))
  + 无符号数据(ceilAlign(tileNum × sizeof(int32_t)))
  + int64直方图(BIN_NUM × indexTypeSize)
  + 排序索引(可选，当sorted且K*dtypeSize ≤ 8000)

while (topkAcApiNeedBuffer + needSpace > ubSizePlatForm):
    tileSize -= 256
```

**核分配** (`TileModeSingleCore`)：
```
lastDimTileNum = ceil(lastAxisNum / tileSize)  // 排序轴分块数
sortLoopTimes = unsortedDimNum / maxCoreNum     // 外轴循环次数
unsortedDimParallel = maxCoreNum                 // 外轴核并行数
```

---

### 3.5 模式⑤ MultiCoreOptim

**判断条件** (`IsMultiCoreOptimMode`)：

```
1. 初始 tileSize ≥ kValue（K值不超过一个tile）
2. 能在UB中找到满足条件的tileSize
3. tileSize < lastAxisNum（确保需要多核）
4. K × lastDimTileNum ≤ tileSize（所有tile的TopK结果能集中到一个核再做一次TopK）
```

**核心思想：**

```
假设 lastAxisNum = 10000, tileSize = 2500, K = 100:
  lastDimTileNum = 4 个tile
  每个tile做一次TopK(100) → 每个tile产出100个候选
  4 × 100 = 400 < 2500 = tileSize
  → 可以将4个tile的候选集中到一个核，再做一次TopK(100)

传统多核Radix: 每核都要做多趟Radix Pass + 全核同步
优化模式: 每核独立TopK → 最终一次TopK，减少同步次数
```

**tileSize搜索：**
```
从默认 tileSize 开始，每次减少32
确保: tileSize ≥ kValue 且 tileSize < lastAxisNum
验证: kValue × ceil(lastAxisNum/tileSize) ≤ tileSize
```

**Tiling计算** (`TileMultiCoreOptimSize`)：
```
modeType = MULT_CORE_OPTIM_MODE (4)
其余参数与标准多核模式相同
```

---

### 3.6 模式⑥ MultiCore (Medium/Big)

**判断条件** (`IsTopkRadixMoreCoreMode`)：
```
1. lastAxisNum < 10,000,000（排序轴不超过一千万）
2. ComputeTopkRadixMoreCoreTileData() 返回 tileSize > 0
```

这是最通用的多核模式，进一步细分为 Medium 和 Big 两种子模式。

**tileSize计算** (`ComputeTopkRadixMoreCoreTileData`)：

```
初始tileSize = 7680(非64位) 或 5120(64位)

计算运行时空间 (GetTopkMultiCoreRunTimeNeedSpace):
  = BIN_NUM × indexDtypeSize × lastDimTileNumTimes  // tile直方图--BIN_NUM桶个数
  + 2 × ceilAlign(sizeof(uint32_t) × lastDimTileNumTimes)  // tileTopK值--每个tile内的topk的个数
  + (dtypeSize×2 + indexDtypeSize + indexToDtypeSize) × tileSize  // 主要数据空间

while (topkAcApiNeedBuffer + needSpace > ubSizePlatForm):
    tileSize -= 256
```

#### 子模式判断 (`TileTopkMoreCoreMode`)：

```
sortedDimParallelData = (tileSize × maxCoreNum) / 2

if lastAxisNum ≤ sortedDimParallelData:
    → Medium模式: 排序轴不算特别大，外轴和排序轴都能分配核（几行一起所有核处理）
else:
    → Big模式: 排序轴很大，所有核都用来处理排序轴（一行所有核处理）
```

#### Medium模式 (`TileModeMediumSize`)

```
lastDimTileNum = ceil(lastAxisNum / tileSize)   //  内轴需要几个核
unsortedDimParallel = maxCoreNum / lastDimTileNum    // 外轴几行一起干活
coreNumNeed = lastDimTileNum × unsortedDimParallel    //  同一时刻多少核在干活
sortLoopTimes = ceil(unsortedDimNum / unsortedDimParallel)  //  需要多少轮
```

核分配示意（假设 maxCoreNum=8, lastDimTileNum=2）：
```
核0: tile0行0,  核1: tile1行0
核2: tile0行1,  核3: tile1行1
核4: tile0行2,  核5: tile1行2
核6: tile0行3,  核7: tile1行3
→ unsortedDimParallel=4, lastDimNeedCore=2
```

#### Big模式 (`TileModeBigSize`)

```
lastDimTileNum = ceil(lastAxisNum / tileSize)
coreNumNeed = min(maxCoreNum, lastDimTileNum) //  每一行单独同时处理
sortLoopTimes = unsortedDimNum    // 每行独立处理
unsortedDimParallel = 1           // 所有核只处理排序轴
```

核分配示意（假设 maxCoreNum=8, lastDimTileNum=10）：
```
行0: 核0-7 处理 tile0-7, 然后核0-1 处理 tile8-9
行1: 同上重复...
→ sortLoopTimes = unsortedDimNum, unsortedDimParallel = 1
```

**K值过大时的处理：** 当K > 2000且处于Big模式时，tileSize减半，因为需要更多空间存放TopK结果。

---

### 3.7 模式⑦ SortAndTopK

**触发条件：** 以上6种模式全部不命中（通常因为 lastAxisNum ≥ 10,000,000 或 UB空间完全不够分配）。

**算法思想：** 与其做多轮Radix Select，不如直接做完整的Radix Sort，然后从排序结果中截取前K个。

**Tiling流程** (`TileModeSortAndTopK`)：

```
1. SortCheckParams() — 校验参数，获取UB大小、Shape信息
2. GetRadixSortMoreCore() — 复用Sort算子的Radix Sort Tiling
   ├─ 预留32K给SIMT
   ├─ ComputeTileData() — 计算排序轴的分块大小
   │   ├─ ubExtra = int32:4096, int64:7168（固定预留）
   │   ├─ tileFactor = 10/14 + dtypeSize（每数据元素占用的UB系数）
   │   ├─ tileData = (ubSize - ubExtra) / tileFactor
   │   ├─ 对齐到256的整数倍
   │   ├─ while (sortApiTmpSize > remainUb):
   │   │     tileData -= 256（减小分块直到Sort API临时空间够用）
   │   └─ NeedAdjTileData() — 根据B轴/H轴比例微调分块
   ├─ 计算核间通信参数(keyParams0-5)
   └─ 计算TopK截取参数(tileDataSize, blockTileNum, tailTileNum)
3. ComputeWorkSpace() — 计算GM Workspace大小
4. 设置 tilingKey = dataType原始Key
```

**核间通信参数计算：**

```
// 全局直方图分块：每个核处理一部分
allNumGloblHist = 256 × lastDimTileNum × dtypeSize × unsortedDimParallel
oneCoreSize = ceil(allNumGloblHist / coreNumNeed)
keyParams5 = max(oneCoreSize, blockUbSize)     // 每核处理的GM大小
keyParams0 = ceil(allNumGloblHist / keyParams5) // 需要几轮
keyParams3 = ceil(keyParams5 / ubSizeNum)       // 每轮GM到UB的次数
keyParams2 = min(keyParams5, ubSizeNum)         // 每次UB处理大小

// 排他前缀和分块
allNumExcusiveBin = 256 × dtypeSize × unsortedDimParallel
keyParams4 = max(ceil(allNumExcusiveBin / coreNumNeed), blockUbSize)
keyParams1 = ceil(allNumExcusiveBin / keyParams4)
```

---

## 4. SortWithIndex 后处理 Tiling

### 4.1 触发条件 (`needSortWithIndex`)

```cpp
当 sorted=true 且:
  - MULT_CORE_MODE 且 K > 2000  → 需要 SortWithIndex
  - SINGLE_CORE_MODE 且 K > 2000 或 K × dtypeSize > 8000 → 需要 SortWithIndex
```

**为什么需要 SortWithIndex：** Radix Select 只筛选出 TopK 元素，但保证不了它们有序。当 K 较小时可以直接用 Sort API 排序，但当 K > 2000 时，Kernel 侧需要独立的排序流程。

### 4.2 SortWithIndex Tiling 流程 (`RadixSortTilingOfIdx`)

位于 `sort_with_index_tiling.h`，是一个独立的 Tiling 计算，复用 TopK 的 TilingData：

```
1. 计算排序轴和外轴大小
2. 判断 isInInt32Range（决定索引用 int32 还是 int64）
3. 根据排序轴大小选择子模式:
   ├─ ≤ 512 且为浮点类型 → SMALL_SIZE_OPTIM_MODE
   │   → 使用 Merge Sort (TileModeSmallSizeOptimOfIdx)
   ├─ ≤ tileSize → SMALL_SIZE_MODE
   │   → 外轴并行，排序轴不分块
   └─ > tileSize → MULT_CORE_MODE
       → 多核Radix Sort (TileMoreCoreModeOfIdx)
4. 计算 Workspace（累加到TopK的usrSize上）
5. 补充TopK Values/Indices的GM中间空间
```

---

## 5. Workspace 大小计算

### 5.1 TopK Workspace

根据不同模式，Workspace 组成不同：

| 模式 | Workspace组成 |
|------|-------------|
| SINGLE_CORE_MODE | `lastDimTileNum × 256 × unsortedDimParallel × indexTypeSize` |
| MULT_CORE_OPTIM_MODE | `ceilAlign(dtypeSize × factor × K) + ceilAlign(sizeof(int32_t) × factor × K)` ；`Factor = lastDimTileNum * unsortedDimParallel;`|
| 其他模式(MULT_CORE等) | `256 × indexTypeSize × unsortedDimParallel + 2 × ceilAlign(lastDimTileNum × indexTypeSize × unsortedDimParallel)` |

### 5.2 SortAndTopK Workspace

```
Workspace = excusiveBinsGm + globalHistGm + outIdxDb + sortOutIdxGm
          + histTileGm + xB8Gm + outValueDb
+ 16MB (SYS_WORK_SPACE_SIZE)
```

### 5.3 Fp32MergeMoreCore Workspace

```
Workspace = unsortedDimNum × alignInput × 8(Bytes/element) × 2(DoubleBuffer)
+ 16MB (SYS_WORK_SPACE_SIZE)
```

### 5.4 SortWithIndex Workspace（叠加到TopK Workspace上）

```
额外增加:
+ topkValuesGm (lastAxisNum × unsortedDimNum × dtypeSize)
+ topkIndicesGm (lastAxisNum × unsortedDimNum × indexDtypeSize)
+ Sort自身的Workspace
```

---

## 6. UB空间分配策略

### 6.1 UB预留

```
总UB = 平台UB大小
预留 32KB = SIMT使用（向量微码运行需要）
可用UB = 总UB - 32KB
```

### 6.2 tileSize搜索的核心约束

所有模式的 tileSize 搜索都遵循同一模式：

```
1. 从默认最大值开始（7680/5120/15360/10240，根据模式和类型）
2. 计算该tileSize下的总UB需求 = API临时空间 + 运行时数据空间
3. while (总需求 > 可用UB):
      tileSize -= 256 (BIN_NUM)
4. 如果tileSize降到0或低于lastAxisNum → 该模式不可行
```

### 6.3 SortAndTopK模式的UB因子计算

```
int32索引: tileFactor = 10 + dtypeSize
                        ↑              ↑
                   固定开销(10个缓冲)  每个数据元素

int64索引: tileFactor = 14 + dtypeSize
                        ↑
                   固定开销(14个缓冲,含int64扩展)

ubExtra(int32) = 4096 字节
ubExtra(int64) = 7168 字节

tileData = (ubSize - ubExtra) / tileFactor
```

### 6.4 Sort API临时空间分配

排序轴的Sort API需要的临时空间通过 `AdjTmpUb` 动态分配：

```
remainUb = ubSize - ubExtra - tileFactor × tileData
sortApiTmpSize = GetSortMaxMinTmpSize(tileData)  // 查询API需求

若 sortApiTmpSize < remainUb:
    将剩余UB全部分配给sortApiTmpSize（AdjTmpUb）
    sortApiTmpSize += floor(remainUb / blockUbSize) × blockUbSize
```

---

## 7. 核利用率优化策略

### 7.1 Merge Sort的核利用率平衡

`IsLastLoopCoreUtilizationSuccess()` 确保每个loop的核利用率 ≥ 70%：

```
假设 maxCoreNum=8, 某个tileSize下:
  oneCoreRowNum = 3
  unsortedDimNum = 20
  virUnsortedDimNeedCoreNum = 20/3 = 7
  sortLoopTimes = 7/8 = 1
  lastLoopDimNum = 20 % (8×3) = 20
  lastLoopDimNeedCoreNum = 20/3 = 6
  利用率 = 6/8 = 75% > 70% ✓

若 oneCoreRowNum = 7:
  virUnsortedDimNeedCoreNum = 20/7 = 3
  sortLoopTimes = 3/8 = 1
  lastLoopDimNum = 20 % 56 = 20
  lastLoopDimNeedCoreNum = 20/7 = 2
  利用率 = 2/8 = 25% < 70% ✗ → 减小tileSize
```

### 7.2 SortAndTopK的核间均衡

`NeedAdjTileData()` 根据B轴和H轴的比例，动态调整分块策略：

```
B=1, H_tile=1:     tileSize = ceil(sortAxisNum / maxCoreNum)
B=1, H_tile ≥ 核数: tileSize = ceil(sortAxisNum / allCore)
B小, H_tile=1:     按B轴分核, H轴均摊
B大, H_tile > 1:   先按H轴切，再按B轴分配剩余核
```

---

## 8. TilingData 保存与传递

### 8.1 数据序列化

```cpp
topkTilingData.SaveToBuffer(
    context->GetRawTilingData()->GetData(),
    context->GetRawTilingData()->GetCapacity());
context->GetRawTilingData()->SetDataSize(topkTilingData.GetDataSize());
```

### 8.2 Kernel侧读取

```cpp
// 在Kernel中通过宏获取TilingData
GET_TILING_DATA(tilingData, tiling);
// 使用 tilingData.isLargest, tilingData.numTileDataSize 等
```

### 8.3 关键设置

```cpp
context->SetTilingKey(dataTypeKey);           // 决定Kernel走哪个模板
context->SetBlockDim(topkTileInfo.coreNumNeed); // 启动多少个核
context->SetLocalMemorySize(ubSizePlatForm);    // 每核的UB大小
context->SetScheduleMode(1);                     // 调度模式
```

---

## 9. 完整参数流转图

```
┌───────────────────────────────────────────────────────────────┐
│                    Tiling 输入                                │
│  inputShape, dataType, k值, attrs(sorted/largest/dim)        │
│  platformInfo(UB大小, 核数)                                   │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────────┐
│              computeNowTileSizeInfo 构建                       │
│  lastAxisNum, kValue, unsortedDimNum, maxCoreNum,             │
│  ubSizePlatForm, dataType, isLargest, isSort, isInInt32Range  │
└──────────────────────┬────────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │ 7级优先级模式判断       │
          └────────────┬────────────┘
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
   TopK Tiling    Sort Tiling    MergeSort Tiling
   (Radix Select) (SortAndTopK)  (Concat+Sort)
         │             │              │
         └─────────────┼──────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────────┐
│                   TilingData 填充                              │
│  isLargest, isSort, modeType, lastAxisNum, unsortedDimNum,    │
│  sortLoopTimes, unsortedDimParallel, lastDimTileNum,           │
│  numTileDataSize, topKRealValue, oneCoreRowNum,               │
│  topkAcApiTmpBufferSize, isInInt32Range, keyParams0-5,        │
│  [sort相关字段], batchNumInUb, tailBatchNum, ...               │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────────┐
│                    运行时设置                                  │
│  SetTilingKey → 决定Kernel模板分支                             │
│  SetBlockDim → 启动核数                                       │
│  SetWorkspaceSizes → GM Workspace大小                          │
│  SetLocalMemorySize → 每核UB大小                               │
└───────────────────────────────────────────────────────────────┘
```

---

## 10. 总结

TopK V2 的 Tiling 层是一个精密的调度引擎，核心设计要点：

1. **7级优先级决策**：从最快的小数据路径（Merge Sort）到兜底的超大数据路径（SortAndTopK），层层递进
2. **UB空间精算**：每种模式都精确计算 tileSize，通过迭代搜索确保 UB 空间不溢出
3. **核利用率优化**：通过 `IsLastLoopCoreUtilizationSuccess` 等机制确保每个loop至少70%核在工作
4. **SortWithIndex后处理**：对需要排序的大规模结果，独立计算排序Tiling并叠加Workspace
5. **K/N比例自适应**：Fp32 MergeSort模式根据K占排序轴的比例选择最优算法
6. **int32/int64索引自动切换**：排序轴超过int32范围时自动使用int64，影响UB空间计算和SIMT原子操作

---

## 11. 附录：为什么 CUDA 中看不到 Tiling 层

> 一个常见的疑问：NVIDIA CUDA 编程中为什么没有类似 CANN Tiling 层这样的显式调度模块？是编译器代劳了吗？
> 答案不是"编译器做了"，而是**两边硬件架构和编程模型根本不同**。

### 11.1 核心区别：数据搬运谁负责

#### Ascend：Host 端预先规划一切

```
Host (CPU)                        Device (AI Core)
┌──────────────┐                  ┌──────────────┐
│  Tiling层    │                  │              │
│  - 算好每个核 │  TilingData      │  Kernel      │
│    处理哪些数据│ ──────────────► │  - 按tiling   │
│  - UB怎么分   │                  │    参数执行   │
│  - 开几个核   │                  │  - 被动执行   │
│  - 循环几次   │                  │              │
└──────────────┘                  └──────────────┘
```

Ascend AI Core 的特点：
- **没有硬件调度器**，核不会自己去取任务
- UB（Unified Buffer）很小且需要**显式 DataCopy** 管理数据搬运
- 执行是"被动式"的 —— host 必须提前算好每个核干什么

因此，Tiling 层本质上是**用软件实现了一个调度器**。

#### CUDA：硬件调度器 + Kernel 内自主管理

```
Host                              Device (GPU SM)
┌──────────────┐                  ┌──────────────┐
│  启动kernel   │  grid/block维度  │  Hardware     │
│  <<<grid,    │ ──────────────► │  Scheduler    │
│    block>>>   │                  │  ┌────────┐  │
│              │                  │  │自动把   │  │
│              │                  │  │block分  │  │
│              │                  │  │配到SM   │  │
│              │                  │  └────┬───┘  │
│              │                  │       ▼      │
│              │                  │  Thread Code │
│              │                  │  - 自己算索引 │
│              │                  │  - 自己管理   │
│              │                  │    shared mem │
└──────────────┘                  └──────────────┘
```

CUDA 的特点：
- **G80 开始就有硬件 Giga Thread Engine**，动态把 thread block 调度到 SM
- 程序员只定义 grid/block 维度，硬件自动分配
- Shared memory 的分块（tiling into shared mem）是在 **kernel 代码内部**做的，不是 host 端单独一层

### 11.2 职责对比

| 方面 | Ascend CANN | NVIDIA CUDA |
|------|------------|-------------|
| **核分配** | Host 端 Tiling 层算好 | 硬件调度器动态分配 |
| **数据分块** | Host 端 Tiling 层算好 tileSize | Kernel 内部用 `__shared__` 手动管理 |
| **UB/Shared Mem 规划** | Host 端算好偏移量 | Kernel 内部声明、编译器分配地址 |
| **循环次数** | Host 端算好写进 TilingData | Kernel 内部用 threadIdx/blockIdx 计算 |
| **多核协调** | Host 端显式分配各核任务 | 原子操作 + warp 级原语 |

### 11.3 代码对比：矩阵乘法的分块

```c
// CUDA: tiling 在 kernel 内部完成
__global__ void matmul(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE][TILE];  // kernel内部声明shared mem
    __shared__ float Bs[TILE][TILE];  // tile大小是编译时常量或模板参数

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;

    // kernel内部自己计算要处理哪些数据
    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0;
    for (int p = 0; p < (N + TILE - 1) / TILE; p++) {
        // kernel内部自己搬数据到shared mem
        As[ty][tx] = A[row * N + p * TILE + tx];
        Bs[ty][tx] = B[(p * TILE + ty) * N + col];
        __syncthreads();

        for (int i = 0; i < TILE; i++)
            sum += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

```cpp
// Ascend: tiling 在 host 端完成，kernel只是被动执行
// Host端:
void ComputeTiling(int totalRows, int totalCols, int K, int ubSize) {
    int tileSize = FindMaxTileSize(ubSize, totalCols); // 算UB能装多少
    int coreNum = CeilDiv(totalRows, tileSize);         // 算需要几个核
    int loopCount = CeilDiv(totalRows, coreNum * tileSize); // 算每个核循环几次
    // ... 写入 TilingData
}

// Kernel端:
// 按TilingData被动执行，不需要自己算这些
```

### 11.4 为什么设计选择不同

**GPU 选择硬件调度**，因为：
- SM 是通用的，任何 thread block 都能跑在任何 SM 上
- Warp 调度是零开销的（硬件上下文切换）
- 设计目标：大规模并发线程，硬件自动均衡负载

**Ascend 选择软件调度**，因为：
- AI Core 是为矩阵/向量运算定制的，资源更加固定和受限
- UB 空间紧张（通常几十KB），必须精打细算
- 数据搬运路径固定（GM → UB → 计算单元 → GM），需要精确编排
- 换来的是**更高的计算密度和能效比** —— 不浪费晶体管在调度器上

### 11.5 补充：cuDNN 内部其实也有类似的 tiling

NVIDIA 的库（cuDNN、cuBLAS、TensorRT）内部**确实有类似 Ascend Tiling 层的东西**，比如卷积的各种 im2col/winograd/implicit GEMM 策略选择、split-K 的 tiling 等。只是这些被封装在库内部，对用户不可见。而 CANN 把这层开放出来，让算子开发者可以自己控制。

### 11.6 小结

不是编译器代劳，而是：
1. **硬件调度器**承担了核分配 —— CUDA 有，Ascend 没有
2. **Kernel 代码内部**承担了数据分块 —— CUDA 的 `__shared__` + threadIdx，Ascend 则交给 host 端 Tiling
3. Ascend 因为**没有硬件调度器 + UB 更受限**，所以必须把调度逻辑提到 host 端作为显式的一层

---

## 12. 各模式选择策略详细对照表

> 本节以代码实际判断顺序为基准，给出每种模式的精确触发条件、关键参数和适用场景。
> 决策严格按优先级 ① → ⑦ 顺序，命中即返回，不会再走下一个模式。

### 12.1 主决策表（精确触发条件）

| 优先级 | 模式名 | modeType 常量 | TilingKey 规则 | 数据类型限制 | 精确触发条件（按代码 AND 关系） | 失败兜底条件 |
|:---:|:---|:---:|:---:|:---|:---|:---|
| ① | **SmallSizeMergeSort** | 3（隐式） | `dataTypeKey + 10000` | `optDataTypeBitMap` 命中：float / float16 / bfloat16 | `lastAxisNum ≤ 1024` **AND** `dataType ∈ {F32, F16, BF16}` | 不命中类型或 N > 1024 时进入② |
| ② | **SingleBlock** | 3（同上） | `dataTypeKey`（无偏移） | 全部 11 种类型 | `ComputeSingleBlockTileData() > 0` **AND** `tileSize ≥ lastAxisNum`（UB 能装下整行 + TopK API 临时空间） | UB 装不下整行 → 进入③ |
| ③a | **Fp32MergeMoreCore** | 6 | `23003`（float 专用） | 仅 `DT_FLOAT` | `dataType == F32` **AND** `kValue > 0` **AND** `0.25 ≤ K/N < 0.50` **AND** `onceMaxElements ≥ 32` **AND** `splitCoreNum = ⌈N/2048⌉ ∈ (1, maxCoreNum]` **AND** `unsortedDimNum × splitCoreNum ≤ maxCoreNum` | 核数不够 → 进入③b |
| ③b | **Fp32MergeIntraCore** | 7 | `33003`（float 专用） | 仅 `DT_FLOAT` | ③a 未命中 **AND** `dataType == F32` **AND** `unsortedDimNum ≥ maxCoreNum/2` **AND** `kValue > 0` **AND** `0.25 ≤ K/N < 0.50` **AND** `!(isSort==false && kValue ≤ 2000)` **AND** `1 < blocksPerRow = ⌈N/blockSortSize⌉ ≤ 256` | K 太小且不需排序 → 进入④ |
| ④ | **SingleCore** | 1 | `dataTypeKey` | 全部 11 种类型 | `unsortedDimNum ≥ maxCoreNum`（外轴 ≥ 核数） **AND** `ComputeSingleCoreTileData() > 0`（UB 能找到合适 tileSize） | 外轴 < 核数或 UB 算不出 → 进入⑤ |
| ⑤ | **MultiCoreOptim** | 4 | `dataTypeKey` | 全部 11 种类型 | `tileData 默认值 > kValue` **AND** 迭代搜索到合法 tileSize **AND** `kValue × lastDimTileNum ≤ tileSize` **AND** `tileSize ≥ lastAxisNum` | K 太大或搜索失败 → 进入⑥ |
| ⑥ | **MultiCore（Medium/Big）** | 2 | `dataTypeKey` | 全部 11 种类型 | `lastAxisNum < 10,000,000`（SORT_AND_TOP_K_THRESHOLD） **AND** `ComputeTopkRadixMoreCoreTileData() > 0`（UB 能算出 tileSize） | N ≥ 10M 或 UB 不够 → 进入⑦ |
| ⑦ | **SortAndTopK**（兜底） | 5 | `dataTypeKey` | 全部 11 种类型 | 以上 6 种全部未命中 | 无（最终路径） |

> 注：`N` = `lastAxisNum`，`K` = `kValue`，`unsortedDimNum` = 外轴总大小。

### 12.2 资源使用对比表

| 模式 | 默认起始 tileSize | tileSize 递减步长 | Workspace 占用 | 双缓冲 | 核数策略 |
|:---|:---:|:---:|:---|:---:|:---|
| ① SmallSizeMergeSort | 7680（TMP_DATA_NUM） | +256 增大到合适数（向 70% 核利用率看齐） | 无（全在 UB 内） | N < 2048 时开启 | 按行均摊，最多满核 |
| ② SingleBlock | 15360 / 10240（b64） | 256 | 无（仅 UB+TopK API 临时） | N < 2048 时开启 | 满核运行 |
| ③a Fp32MergeMoreCore | — | — | `unsortedDimNum × alignInput × 8B × 2`（DoubleBuffer） | 启用 | `unsortedDimNum × splitCoreNum` |
| ③b Fp32MergeIntraCore | blockSortSize / extractChunkSize | 32 对齐 | 较少 | 启用 | `min(maxCoreNum, unsortedDimNum)` |
| ④ SingleCore | 15360（SINGLE_CORE_DATA_NUM） | 256（BIN_NUM） | 无 | N < 2048 时开启 | 单核串行处理整行 |
| ⑤ MultiCoreOptim | 7680 / 5120（b64） | 32（TILE_SIZE_DECREASING_FACTOR） | 无 | 启用 | 满核 + 末轮 K 末位集中 |
| ⑥ MultiCore | 7680（TMP_DATA_NUM） | 256 | 无 | 启用 | 满核 + 末轮集中 |
| ⑦ SortAndTopK | 由 SortRadix 子模式决定 | — | 较大（radix 全排序） | 由 Sort 决定 | 满核 |

### 12.3 模式互斥关系矩阵

> 决策是 if-elif 顺序结构，下表表示「如果命中本模式，会跳过哪些后续模式」。

| 命中模式 | 跳过的后续模式 | 原因 |
|:---|:---|:---|
| ① SmallSizeMergeSort | ②③④⑤⑥⑦ | 小数据 + 浮点 = 最优路径，无需再判 |
| ② SingleBlock | ③④⑤⑥⑦ | UB 能装整行，没必要走 Merge/Radix |
| ③a Fp32MergeMoreCore | ③b④⑤⑥⑦ | F32 + K/N ∈ [0.25, 0.5) + 核数够 → 多核归并最优 |
| ③b Fp32MergeIntraCore | ④⑤⑥⑦ | F32 + K/N ∈ [0.25, 0.5) + 核数不够 → 单核内归并 |
| ④ SingleCore | ⑤⑥⑦ | 外轴 ≥ 核数 → 单核串行更稳定 |
| ⑤ MultiCoreOptim | ⑥⑦ | K 末位集中策略能省一次 TopK API |
| ⑥ MultiCore | ⑦ | 经典多核 Radix Select |
| ⑦ SortAndTopK | — | 兜底，无下一级 |

### 12.4 典型场景触发示例

假设硬件为 `coreNum = 32`、UB ≈ 192KB（已扣 SIMT 32KB）：

| 输入场景 | N (lastAxis) | K | unsortedDim | dataType | sorted | 命中模式 | 触发原因 |
|:---|:---:|:---:|:---:|:---|:---:|:---|:---|
| Transformer 注意力小尾轴 | 128 | 5 | 4096 | float16 | true | **①** | N ≤ 1024 + F16 |
| 小 LLM logits | 512 | 10 | 2048 | float | false | **①** | N ≤ 1024 + F32 |
| 嵌入层检索 | 4096 | 100 | 8192 | float | true | **②** | UB 能装 4096 × 4B = 16KB |
| 推荐排序 | 10000 | 50 | 100000 | float | false | **③a 或 ③b** | K/N = 0.005 不命中（< 0.25），落入 ④/⑤ |
| 中等规模检索 | 5000 | 1500 | 5000 | float | true | **③a** | K/N = 0.3 ∈ [0.25, 0.5) + splitCoreNum = 3 ≤ 32 |
| 同上但 unsorted=10 | 5000 | 1500 | 10 | float | true | **③b** | ③a 核数不够（10×3=30，但 unsortedDimNum=10 偏小走核内） |
| 普通多核场景 | 8192 | 100 | 65536 | int32 | false | **④** | unsortedDim(65536) ≥ 32 → 单核更稳 |
| 大 K 占比 | 20000 | 5000 | 1000 | int8 | true | **⑤** | K=5000, tileSize 需 ≥ K × lastDimTileNum |
| 经典 TopK | 100000 | 100 | 1000 | float16 | false | **⑥** | N < 10M + UB 算得出 tileSize |
| 超大词表 | 20000000 | 10 | 1 | int32 | false | **⑦** | N ≥ 10M → 兜底完整排序 |

### 12.5 关键常量速查表

| 常量名 | 值 | 含义 | 影响模式 |
|:---|:---:|:---|:---|
| `SMALL_MAX_DATA_SZIE` | 1024 | SmallSize 模式的 N 上限 | ① |
| `SINGLE_BLOCK_DATA_NUM` | 15360 | SingleBlock 默认 tileSize | ② |
| `SINGLE_BLOCK_DATA_NUM_B64` | 10240 | 64 位类型默认 tileSize | ② |
| `SINGLE_CORE_DATA_NUM` | 15360 | SingleCore 默认 tileSize | ④ |
| `TMP_DATA_NUM` | 7680 | MultiCoreOptim / MergeSort 默认 tileSize | ① ⑤ ⑥ |
| `TMP_DATA_NUM_B64` | 5120 | 64 位类型默认 tileSize | ⑤ |
| `TILE_SIZE_DECREASING_FACTOR` | 32 | MultiCoreOptim 迭代递减步长 | ⑤ |
| `BIN_NUM` | 256 | Radix 桶数（也作递减步长） | ④ ⑥ |
| `FP32_K_LAST_AXIS_LOWER_RATIO` | 0.25 | F32 Merge 模式 K/N 下限 | ③ |
| `FP32_K_LAST_AXIS_UPPER_RATIO` | 0.50 | F32 Merge 模式 K/N 上限 | ③ |
| `MERGE_MORE_CORE_ONE_CORE_DATA_SIZE` | 2048 | F32 多核 Merge 单核数据量 | ③a |
| `MERGE_INTRA_CORE_SORT_ALIGN` | 32 | F32 核内 Merge 对齐粒度 | ③a ③b |
| `SORT_AND_TOP_K_THRESHOLD` | 10,000,000 | SortAndTopK 兜底触发线 | ⑥ ⑦ |
| `SUPPORT_SORT_MAX_SIZE` | 2000 | 不需排序的 K 上限（F32 核内走 Radix 更快） | ③b |
| `CONST_SIMT_SPACE` | 32768 | UB 预留给 SIMT 的空间 | 全部 |
| `MERGE_SORT_TILING_OFFSET` | 10000 | Merge 模式 TilingKey 偏移 | ① |

### 12.6 决策流程伪代码（精简版）

```cpp
TopKV2Tiling() {
    if (IsSmallSizeMergeSortMode(dataType, N))       return TileModeSmallSizeOptim();   // ①
    if (IsSingleBlockMode(...))                      return TileModeSmallSize();        // ②
    if (IsFp32MergeSortMode(...)) {
        if (IsTopkMergeSortMoreCoreFp32Mode(...))    return GetTopkMergeMoreCoreFp32(); // ③a
        if (IsTopkMergeSortIntraCoreFp32Mode(...))   return GetTopkMergeIntraCoreFp32();// ③b
    }
    if (IsSingleCoreMode(...))                       return TileModeSingleCore();       // ④
    if (IsMultiCoreOptimMode(...))                   return TileMultiCoreOptimSize();   // ⑤
    if (IsTopkRadixMoreCoreMode(...))                return TileTopkMoreCoreMode();     // ⑥
    return TileModeSortAndTopK();                                                       // ⑦ 兜底
}
```

### 12.7 表格使用指引

- **要查 N 多大走哪个模式** → 看 12.1 表的「精确触发条件」列 + 12.4 示例
- **要查为什么没走到某模式** → 看 12.3 互斥关系
- **要查 UB 算到多少 tileSize** → 看 12.2 资源使用表
- **要查常量数值** → 看 12.5 速查表
- **要看完整调用顺序** → 看 12.6 伪代码

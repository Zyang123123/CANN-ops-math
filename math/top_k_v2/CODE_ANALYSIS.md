# TopK V2 算子代码架构详解

> Ascend CANN (Compute Architecture for Neural Networks) 中的 TopK V2 算子实现，用于在张量的最后一维中找到最大/最小的 K 个元素及其索引。

---

## 1. 目录结构总览

```
top_k_v2/
├── CMakeLists.txt                          # 构建配置
├── README.md
├── op_graph/                               # 算子图注册层
│   ├── top_k_v2_proto.h                    # 算子原型声明（GE注册）
│   ├── top_k_v2_graph_infer.cpp            # 图推理（暂为空）
│   └── fusion_pass/.gitkeep
├── op_host/                                # Host侧（CPU端）代码
│   ├── top_k_v2_def.cpp                    # 算子定义：输入/输出/属性声明
│   ├── top_k_v2_infershape.cpp             # 输出Shape推导
│   ├── arch35/
│   │   ├── top_k_v2_tiling_arch35.h        # Tiling数据结构定义
│   │   ├── top_k_v2_tiling_arch35.cpp      # Tiling策略实现（~800行）
│   │   └── sort_with_index_tiling.h        # SortWithIndex的Tiling策略
│   └── config/ascend950/                   # 芯片二进制/配置
│       ├── top_k_v2_binary.json
│       └── top_k_v2_simplified_key.ini
├── op_kernel/                              # Device侧（NPU端）代码
│   ├── top_k_v2_apt.cpp                    # 算子Kernel入口（根据tiling分发）
│   └── arch35/
│       ├── radix_sort_top_k.h              # 核心：Radix Sort TopK主算法
│       ├── radix_sort_top_k_base.h         # Radix Sort TopK基类
│       ├── radix_sort_top_k_single_block.h # 单Block模式
│       ├── radix_sort_top_k_single_core.h  # 单核多Block模式
│       ├── radix_sort_top_k_inter_core_template_optimization.h  # 多核优化模板
│       ├── radix_sort_topk_b8.h            # 8位数据Radix Sort直方图
│       ├── radix_sort_topk_b16.h           # 16位数据Radix Sort直方图
│       ├── radix_sort_topk_b32.h           # 32位数据Radix Sort直方图
│       ├── radix_sort_topk_b64.h           # 64位数据Radix Sort直方图
│       ├── top_k_radix_block_sort_b8.h     # 8位数据Block排序预处理
│       ├── top_k_radix_block_sort_b16.h    # 16位数据Block排序预处理
│       ├── top_k_radix_block_sort_b32.h    # 32位数据Block排序预处理
│       ├── top_k_radix_block_sort_b64.h    # 64位数据Block排序预处理
│       ├── radix_topk_constant.h           # 各数据类型极值常量
│       ├── radix_topk_util.h               # 工具函数（最大/最小值获取）
│       ├── top_k_constant_var_simd.h       # SIMD常量定义
│       ├── top_k_util_type_simd.h          # SIMD工具函数与类型特性
│       ├── top_k_merge_sort.h              # Merge Sort模式（小排序轴）
│       ├── top_k_merge_sort_simd.h         # Merge Sort的SIMD实现
│       ├── top_k_merge_sort_more_core.h    # Merge Sort多核模式
│       ├── top_k_merge_sort_intra_core.h   # Merge Sort核内模式
│       ├── sort_and_top_k_more_core.h      # SortAndTopK模式（超大排序轴）
│       ├── sort_with_index_entry.h         # SortWithIndex入口分发
│       ├── sort_with_index_single_block.h  # SortWithIndex单Block
│       ├── sort_with_index_multi_block.h   # SortWithIndex多Block
│       └── sort_with_index_merge_sort.h    # SortWithIndex Merge Sort
├── examples/
│   └── test_geir_topkv2.cpp                # 测试用例
└── tests/ut/
    └── op_host/
        ├── arch35/test_top_k_v2_tiling.cpp  # Tiling单元测试
        └── test_top_k_v2_infershape.cpp     # InferShape单元测试
```

---

## 2. 架构分层

### 2.1 整体架构

```
┌─────────────────────────────────────────────────┐
│                   用户调用层                      │
│         torch.topk / tensorflow TopKV2           │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              算子注册与定义层 (op_graph)           │
│    proto.h → GE引擎注册算子名称/输入/输出/属性     │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│            Host编译时处理层 (op_host)              │
│  def.cpp → 声明数据类型支持矩阵                    │
│  infershape.cpp → 推导输出张量形状                 │
│  tiling_arch35.cpp → 计算分块策略(Tiling)          │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│          Device Kernel执行层 (op_kernel)          │
│  top_k_v2_apt.cpp → 根据Tiling Key分发到具体算法   │
│  ┌─────────────┬──────────────┬────────────────┐ │
│  │ Radix Sort  │  Merge Sort  │ SortAndTopK    │ │
│  │ (通用场景)   │  (小排序轴)   │ (超大排序轴)    │ │
│  └─────────────┴──────────────┴────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 3. 各层详细分析

### 3.1 算子注册层 — `op_graph/top_k_v2_proto.h`

使用华为GE (Graph Engine) 框架的 `REG_OP` 宏注册算子。

**算子规格：**
| 项目 | 说明 |
|------|------|
| 算子名 | `TopKV2` |
| 输入 x | 1D-8D张量，支持 float16/float32/int8/int16/int32/int64/uint8/uint16/uint32/uint64/bfloat16 |
| 输入 k | 0D张量，int32/int64，表示要取的Top-K个数 |
| 输出 values | 排序后的值，类型与输入相同 |
| 输出 indices | 值的索引，int32/int64 |
| 属性 sorted | bool，默认true，输出是否有序 |
| 属性 dim | int，默认-1（保留用，当前只在最后维操作） |
| 属性 largest | bool，默认true，取最大还是最小 |
| 属性 indices_dtype | int，默认DT_INT32，索引输出类型 |

### 3.2 算子定义层 — `op_host/top_k_v2_def.cpp`

通过 `OpDef` 类声明算子的完整类型支持矩阵。核心是4个向量，枚举所有合法的 (x类型, k类型, values类型, indices类型) 组合：

- **11种数据类型 × 4种组合** = 44种组合
- 4个组合为：{indices=int32, k=int32}、{indices=int64, k=int32}、{indices=int32, k=int64}、{indices=int64, k=int64}

关键配置：
```cpp
aicoreConfig.DynamicCompileStaticFlag(true)    // 支持动态编译静态
             .DynamicRankSupportFlag(true)      // 支持动态维度
             .DynamicShapeSupportFlag(true)     // 支持动态Shape
```

目标芯片：`ascend950`

### 3.3 Shape推导层 — `op_host/top_k_v2_infershape.cpp`

**核心逻辑：** 输出Shape与输入Shape相同，仅将排序轴（最后一维）替换为K值。

```
输入Shape: [D0, D1, ..., Dn-1, Dn]
输出Shape: [D0, D1, ..., Dn-1, K]
```

处理流程：
1. 从运行时Tensor获取k值（支持int32/int64两种dtype）
2. 计算排序轴位置（默认最后一维）
3. 构造输出Shape，将排序轴维度设为K

### 3.4 数据类型推导层 — `op_host/top_k_v2_def.cpp` (InferDataType)

- `values` 输出类型 = `x` 输入类型
- `indices` 输出类型由 `indices_dtype` 属性决定（int32 或 int64）

### 3.5 Tiling层 — 核心调度策略

#### 3.5.1 Tiling数据结构 — `top_k_v2_tiling_arch35.h`

`TopKV2TilingDataSimd` 结构体包含两大组字段：

**TopK字段** — 用于Radix Select + TopK模式：
| 字段 | 含义 |
|------|------|
| isLargest | 取最大还是最小 |
| isSort | 结果是否需要排序 |
| sortLoopTimes | 外轴循环次数 |
| unsortedDimParallel | 外轴并行核数 |
| unsortedDimNum | 外轴总大小 |
| lastDimNeedCore | 排序轴需要的核数 |
| numTileDataSize | UB一次处理的数据量 |
| platformCoreNum | 平台总核数 |
| topKRealValue | 实际K值 |
| modeType | 执行模式标识 |

**Sort字段** — 用于SortWithIndex后处理：
| 字段 | 含义 |
|------|------|
| tilingKeyForSort | SortWithIndex的Tiling Key |
| modeTypeForSort | SortWithIndex模式类型 |
| lastAxisNumForSort | SortWithIndex的排序轴大小 |
| keyParams0-5 | Radix Sort核间通信参数 |

#### 3.5.2 Tiling策略 — `top_k_v2_tiling_arch35.cpp`

Tiling阶段根据数据规模选择不同的执行模式：

| 模式 | modeType值 | 适用场景 | 描述 |
|------|-----------|----------|------|
| SINGLE_BLOCK_MODE | 3 | 排序轴 ≤ UB容量 | 数据量小，单Block直接处理 |
| SINGLE_CORE_MODE | 1 | 排序轴较大但不需要多核 | 单核多次循环处理 |
| MULT_CORE_MODE | 2 | 排序轴大，需多核协作 | 多核Radix Select |
| MULT_CORE_OPTIM_MODE | 4 | 多核场景性能优化 | 优化版多核模板 |
| SORT_AND_TOP_K_MODE | 5 | 排序轴超大(>10M) | 先完整排序再截取 |
| FP32_MERGE_MORE_CORE_MODE | 6 | FP32, K/LastAxis比例适中 | 多核Merge Sort |
| FP32_MERGE_INTRA_CORE_MODE | 7 | FP32, K/LastAxis比例较高 | 核内Merge Sort |

**Tiling核心决策逻辑：**
1. 计算UB可用空间（预留32K给SIMT）
2. 根据数据类型位宽计算单次可处理的数据量
3. 判断排序轴是否需要分块（tileDataSize vs lastAxisNum）
4. 计算需要的核数和循环次数
5. 对于FP32且K/LastAxis比例在特定范围内的场景，选择Merge Sort模式

#### 3.5.3 SortWithIndex Tiling — `sort_with_index_tiling.h`

当TopK结果需要排序（sorted=true）且数据量较大时，使用SortWithIndex做后处理排序。其Tiling策略独立计算：
- **小规模优化模式** (SMALL_SIZE_OPTIM_MODE)：排序轴 ≤ 512 且为浮点类型
- **小规模模式** (SMALL_SIZE_MODE)：排序轴 ≤ 单次tile数据量
- **多核模式** (MULT_CORE_MODE)：需要多核Radix Sort

### 3.6 Kernel执行层

#### 3.6.1 入口分发 — `top_k_v2_apt.cpp`

入口函数 `top_k_v2()` 是一个 `__global__ __aicore__` 函数，根据两个维度进行分发：

**维度一：数据类型（通过TILING_KEY匹配）**
```
int8 → B8 (1趟排序)
int16/uint16/half/bfloat16 → B16 (2趟排序)
int32/uint32/float → B32 (4趟排序)
int64/uint64 → B64 (8趟排序)
```

**维度二：执行模式（通过modeType判断）**
```
generateOpObject() → Radix Select模式（通用）
  ├── RadixSortTopKSingleBlock     (单Block)
  ├── RadixSortTopKSingleCore      (单核多Block)
  ├── RadixSortTopKMultiCoreOptimization (多核优化)
  ├── SortAndTopKMoreCore          (超大轴)
  └── RadixSortTopK               (多核标准)

generateMergeTopKObject() → Merge Sort模式（小轴/浮点优化）
generateMergeTopKMoreCoreObject() → Merge Sort多核模式
generateMergeTopKIntraCoreObject() → Merge Sort核内模式
```

#### 3.6.2 核心算法 — Radix Sort TopK (`radix_sort_top_k.h`)

**算法原理：** 基数选择(Radix Select) — 不做完整排序，仅通过直方图(Histogram)逐步缩小范围找到Top-K元素。

**模板参数：**
```cpp
template <typename T,           // 数据类型
          typename UNSIGNED_TYPE, // 无符号数据类型（用于位操作）
          int32_t NUM_PASS,      // Radix排序趟数（1/2/4/8）
          bool IS_LARGEST,       // 取最大/最小
          bool IS_SORT,          // 是否排序结果
          typename T_INDEX,      // 索引计算类型（int32/int64）
          typename T_INDEX_TO>   // 索引输出类型（int32/int64）
```

**核心流程 (`ProcessMultiBlockTopK`)：**

```
对每趟 radix pass (从高位到低位):
  1. 将数据分块(tile)加载到UB
  2. 对每个tile计算256-bin直方图 (GetTileExcusive)
  3. 将tile直方图累加到全局直方图 (atomic add to GM)
  4. 全核同步 (SyncAll)
  5. 在全局直方图上二分查找分界桶 (FindBoundary)
  6. 更新每个tile的TopK计数
  7. 缩小下一趟的处理范围

对每个tile:
  8. 使用AscendC TopK API精确提取TopK元素
  9. 转换索引为全局索引
  10. 写回GM

如果需要排序(sorted=true)且数据量适中:
  11. 使用Sort API对TopK结果排序
  12. 或调用SortWithIndex对大数据排序
```

**关键方法说明：**

| 方法 | 功能 |
|------|------|
| `InitPara()` | 初始化参数，设置GM缓冲区 |
| `Init()` | 初始化UB缓冲区、Queue、Pipe |
| `ProcessTopK()` | 外轴循环入口 |
| `ProcessMultiBlockTopK()` | 单行数据的多核Radix Select |
| `FindBoundary()` | 在全局直方图上二分查找分界桶 |
| `BinarySearch()` | 精确查找分界点 |
| `GetTileExcusive()` | 计算单个tile的256-bin直方图 |
| `PreProcess()` | 有符号数据转无符号（Twiddle操作） |
| `StoreFinalAnswer2Gm()` | 调用TopK API提取最终结果 |
| `SortTopKRes()` | 对TopK结果调用Sort API排序 |
| `CopyCumSumToGmB*()` | 使用SIMT（向量微码）做原子加 |

#### 3.6.3 数据预处理 — `top_k_radix_block_sort_b*.h`

不同位宽的数据需要不同的预处理（Twiddle操作），将有符号数转换为可进行Radix排序的无符号格式：

- **B8** (int8)：XOR翻转符号位
- **B16** (half/bfloat16/int16)：XOR翻转符号位，负数取反
- **B32** (float/int32)：XOR翻转符号位，负数取反
- **B64** (int64)：XOR翻转符号位，负数取反

对于取最小值（`IS_LARGEST=false`）的场景，使用 `ReverseInputData` 翻转数据后按最大值处理。

#### 3.6.4 Merge Sort模式 (`top_k_merge_sort.h`)

用于小排序轴场景（如排序轴 ≤ 4096），特别是 float/float16/bfloat16 类型。

**算法流程：**
1. 将排序轴数据完整加载到UB
2. 使用 AscendC Concat + Sort API 做带索引的排序
3. 截取前K个结果写回GM

**特点：** 使用双缓冲(Double Buffer)提高数据搬运效率。

#### 3.6.5 Merge Sort多核模式 (`top_k_merge_sort_more_core.h`)

用于中等规模排序轴的多核场景。

**算法流程：**
1. **Phase 1 — 核内排序**：每个核处理一段数据，用 Sort API 排序
2. **Phase 2 — 多核归并**：多轮4-way MrgSort，逐步合并各核的排序结果
3. **Phase 3 — 最终提取**：单核执行最终归并，Extract + CopyOut 截取TopK

**4-way归并策略：**
```
Round 1: [核0][核1][核2][核3] → 归并 → [合并0]
         [核4][核5][核6][核7] → 归并 → [合并1]
         ...
Round 2: [合并0][合并1][合并2][合并3] → 归并 → [最终合并0]
         ...
Final:   最后的 ≤4 个列表 → Extract TopK
```

#### 3.6.6 Merge Sort核内模式 (`top_k_merge_sort_intra_core.h`)

继承自 `Sort::SortMergeIntraCore`，重写4个关键函数以支持TopK截断。

**三阶段流水线：**
1. **SortSingleBatchInUb** — UB内块排序
2. **MergeSingleBatch** — 4-way增量归并（支持TopK提前终止）
3. **ExtractAndCopyOut** — 从排序缓存中提取TopK结果

关键优化：在归并过程中，一旦收集够了TopK个元素就提前终止（`dstElemCount < topKValue_`）。

#### 3.6.7 SortAndTopK模式 (`sort_and_top_k_more_core.h`)

用于超大排序轴（>10,000,000），先完整排序再截取TopK。

**继承关系：** `SortAndTopKMoreCore` → `Sort::SortRadixMoreCore`

**流程：**
1. 执行完整的Radix Sort（`sizeof(T)` 趟pass）
2. 排序完成后，从排序结果中截取前K个元素
3. 使用 `GetValueResData` / `GetIndexResData` 分块拷贝结果

#### 3.6.8 SortWithIndex后处理 (`sort_with_index_entry.h`)

当 `sorted=true` 且 TopK 结果量较大（>2000）时，调用 SortWithIndex 对 Radix Select 的结果做排序。

入口函数 `sortwithindexForTopK()` 根据数据类型分发：
- 小规模单Block → `SortWithIndexSingleBlock`
- 多Block Radix Sort → `SortWithIndexMultiBlock`
- 浮点Merge Sort → `SortWithIndexMergeSort`

---

## 4. 内存模型

### 4.1 存储层次

```
GM (Global Memory, HBM)
  │  输入数据 x / 输出 values,indices / Workspace
  │
  ├─ UB (Unified Buffer, 片上SRAM)
  │   └─ Pipe管理: TQue(VECIN/VECOUT) + TBuf(VECCALC)
  │       ├─ inQueueX_        输入数据缓冲
  │       ├─ topkValueQueue_  TopK值输出缓冲
  │       ├─ topkValueIndexQueue_ TopK索引输出缓冲
  │       ├─ blockCumSumTbuf_ 直方图累加缓冲
  │       ├─ inputXCopyTbuf_  数据拷贝缓冲
  │       ├─ tileTopkValueTbuf_ 每tile的TopK计数
  │       ├─ sortedShareMemTbuf_ Sort/TopK API临时空间
  │       └─ ...
  │
  └─ GM Workspace (通过SetSysWorkspace设置)
      ├─ cumSumBinsGm_     全局直方图
      ├─ tileTopkValueGm_  每tile的TopK值(GM)
      ├─ sortWithIndex空间  排序后处理中间结果
      └─ SortAndTopK空间   完整排序中间结果
```

### 4.2 数据流

```
输入x(GM) → DataCopyPad → UB(inQueueX) → 预处理(Twiddle)
    → 直方图计算 → atomic_add → 全局直方图(GM)
    → 二分查找分界桶 → TopK API → UB(topkValueQueue/IndexQueue)
    → DataCopyPad → 输出values/indices(GM)

[可选] SortWithIndex后处理:
TopK结果(GM) → Sort API → 排序后values/indices(GM)
```

---

## 5. 多核并行策略

### 5.1 核分配

```
总核数 = platformCoreNum
  ├── lastDimNeedCore = 分配给排序轴的核数
  └── unsortedDimParallel = 分配给外轴的核数
总使用核数 = lastDimNeedCore × unsortedDimParallel
```

### 5.2 核间同步

关键同步点：
- `PipeBarrier<PIPE_ALL>()` — 核内流水线屏障
- `SyncAll()` — 全核同步
- `asc_atomic_add()` — 原子累加（SIMT方式或硬件原子操作）

### 5.3 索引范围处理

排序轴数据量超过 int32 范围（> 1073741823）时，直方图累加和索引计算切换到 int64：
```cpp
if (isInInt32Range) {
    // 使用 int32_t 做直方图累加
} else {
    // 使用 int64_t 做直方图累加
    // 使用 asc_vf_call (SIMT) 做原子加
}
```

---

## 6. 性能优化策略

### 6.1 算法选择优化

```
排序轴 N 的大小:
  N ≤ 4096 (浮点)  → Merge Sort (单核，利用Concat+Sort API)
  N ≤ UB容量       → Single Block (一次加载，Radix Select)
  N 较大           → Single Core (单核多次循环) 或 多核Radix Select
  N > 10,000,000   → SortAndTopK (先完整Radix Sort再截取)
```

### 6.2 数据搬运优化

- **Double Buffer**：Merge Sort模式使用双缓冲，在计算的同时搬运数据
- **UB复用**：大tile场景下复用UB缓冲区，避免重复分配
- **对齐访问**：所有UB操作按32字节对齐（`ROUND_UP_AGLIN`）

### 6.3 计算优化

- **SIMT微码**：`asc_vf_call` 调用向量微码做原子加，比软件循环高效
- **MicroAPI**：`ReduceSum`、`DataCopy`等使用底层向量指令
- **提前终止**：Merge Sort归并时收集够TopK元素即停止
- **批量处理**：外轴并行处理多行数据（`oneCoreRowNum`）

### 6.4 K/LastAxis比例优化

针对 FP32 类型，根据 K/N 比例选择更优策略：
- K/N ≤ 0.25 → Radix Select（适合K较小时快速筛选）
- 0.25 < K/N ≤ 0.50 → Merge Sort（适合K较大时排序更高效）
- K/N > 0.50 → 核内Merge Sort（数据大部分需要保留，直接排序）

---

## 7. 数据类型处理细节

### 7.1 类型支持矩阵

| 数据类型 | 字节数 | Radix趟数 | Tiling Key |
|---------|--------|----------|------------|
| int8    | 1      | 1        | 1001       |
| int16   | 2      | 2        | 1002       |
| int32   | 4      | 4        | 1003       |
| int64   | 8      | 8        | 1004       |
| uint8   | 1      | 1        | 2001       |
| uint16  | 2      | 2        | 2002       |
| uint32  | 4      | 4        | 2003       |
| uint64  | 8      | 8        | 2004       |
| float   | 4      | 4        | 3003       |
| float16 | 2      | 2        | 3002       |
| bfloat16| 2      | 2        | 4002       |

### 7.2 有符号转无符号（Twiddle）

为确保Radix Sort正确处理有符号数和浮点数：

```
int8:    XOR 0x80               → 翻转符号位
int16:   XOR 0x8000             → 翻转符号位
int32:   XOR 0x80000000         → 翻转符号位
int64:   XOR 0x8000000000000000 → 翻转符号位
float:   XOR 0x80000000 + 条件取反 → IEEE754保序
half:    XOR 0x8000 + 条件取反
bfloat16:XOR 0x8000 + 条件取反
```

对于取最小值的场景，翻转所有比特后按最大值处理即可。

---

## 8. 构建系统 — `CMakeLists.txt`

```cmake
set(SUPPORT_COMPUTE_UNIT "ascend950" "mc62")
set(SUPPORT_TILING_DIR "arch35" "arch35")
add_all_modules_sources(
    OPTYPE top_k_v2
    ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR}
    DISABLE_IN_OPP TRUE
    DEPENDENCIES sort sort_with_index)
```

关键配置：
- 支持芯片：`ascend950` 和 `mc62`
- Tiling策略：统一使用 `arch35` 目录
- 依赖：`sort` 和 `sort_with_index` 算子（共享排序实现）
- 不生成 OPP 包（`DISABLE_IN_OPP TRUE`）

---

## 9. 总结

TopK V2 算子是一个高度优化的 Ascend NPU 算子实现，核心设计特点：

1. **多算法自适应**：根据数据规模和K/N比例，自动选择最优算法（Radix Select / Merge Sort / SortAndTopK）
2. **多级并行**：支持单Block、单核、多核、多核优化等多种并行模式
3. **全类型覆盖**：支持11种数据类型，44种合法输入输出组合
4. **内存精细管理**：通过Tiling策略精确计算UB空间分配，最大化数据吞吐
5. **SortWithIndex后处理**：对需要排序的大规模结果，使用独立的排序流程
6. **SIMT/MicroAPI加速**：利用底层向量指令做直方图累加等关键操作

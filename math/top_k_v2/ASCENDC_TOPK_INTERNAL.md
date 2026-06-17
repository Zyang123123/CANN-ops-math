# AscendC::TopK 内部实现深度解析

> 本文是对 **AscendC 框架 `AscendC::TopK` API** 源码的解读,基于本地 devkit 仓库:
>
> - 头文件: `C:/Users/ziyang/Desktop/CANN/asc-devkit/include/adv_api/sort/`
> - 实现:   `C:/Users/ziyang/Desktop/CANN/asc-devkit/impl/adv_api/detail/sort/topk/`
>
> **重要修正**: 之前在 `KERNEL_DETAIL.md §15.2` 中推断 AscendC::TopK 是"闭源 API,源码不在仓库"
> ——这是错误的。源码就在 `asc-devkit/` 下,公开声明 + 模板包装 + 内部分派 + 硬件 intrinsic
> 全部可见。真正的"黑盒"只剩最底层的 `Histograms` / `GatherMask` 等向量微码。
>
> 配套阅读:
> - `CODE_ANALYSIS.md` / `TILING_DETAIL.md` / `KERNEL_DETAIL.md` / `HARDWARE_REQUIREMENTS.md`
> - 本文 §6 直接回答: 「既然有 AscendC::TopK,为什么 top_k_v2 还要自己写 RadixSelect?」
>
> 风格: 中文 + 表格 + ASCII 数据流图 + `文件:行号` 引用。

---

## 0. 修正声明

| 之前的陈述 (KERNEL_DETAIL §15.2) | 真相 |
|---|---|
| "AscendC::TopK 是闭源 API" | **错**。源码在 `asc-devkit/` 完整可见 |
| "本仓库只能看到调用方式" | **错**。devkit 提供完整头文件 + impl |
| "实现位于 CANN SDK 的 .so / .a 中" | **部分错**。Header-only 模板,编译期展开 |
| "看不到 C++ 算法层,只能看 stub" | **错**。算法层全部是 inline template |
| "硬件微码完全闭源" | **基本对**。`Histograms<>` / `GatherMask<>` 等最终落到 vector 微码,这部分确实不开源 |

**结论**: AscendC 是 **header-only template library**,源码完全可读。开发者可以学习、
调试、改造,但**不能**用它绕过硬件 IP (核心计算仍由 AIV Vector 微码执行)。

---

## 1. 文件清单

### 1.1 公共头文件目录 (`include/adv_api/sort/`)

| 文件 | 行数 | 职责 | 公开符号 |
|---|---:|---|---|
| `topk.h` | 334 | **主入口** `AscendC::TopK` 模板重载 | `TopK<>` × 4 重载 |
| `topk_utils_constants.h` | 52 | 枚举 + 配置结构 | `TopKMode`, `TopKAlgo`, `TopKOrder`, `TopKConfig` |
| `topk_utils.h` | 37 | 形状描述 | `struct TopKInfo { outter, inner, n }` |
| `topk_tilingdata.h` | 49 | Tiling 字段 (Host 序列化) | `optiling::TopkTiling` (28 字段) |
| `topk_tiling.h` | 89 | Host 侧 tiling 计算接口 | `GetTopKMaxMinTmpSize`, `TopKTilingFunc` |
| `topk_tiling_intf.h` | 18 | **废弃**别名 | `TopKDeprecatedHeader` |
| `kernel_operator_topk_intf.h` | 34 | **废弃**别名 | `LibTopKInterface` |
| `sort.h` | 213 | Sort 主入口 (TopK 内部复用) | `AscendC::Sort<>` |
| `sort_utils.h` / `sort_utils_constants.h` | 42/35 | Sort 配置 | `SortConfig`, `SortType`, `DEFAULT_SORT_CONFIG` |
| `sort_tiling_intf.h` | 61 | Sort tiling 接口 | `GetSortMaxMinTmpSize` |

### 1.2 实现目录 (`impl/adv_api/detail/sort/topk/`)

| 文件 | 行数 | 职责 | 服务的芯片 |
|---|---:|---|---|
| `topk_common_impl.h` | 166 | **公共入口** `TopKNormal` / `TopKNSmall` 包装层 (MergeSort 路径) | v200 / v220 |
| `topk_common_utils.h` | 75 | 跨芯片共享常量 + `defaultTopKConfig` | 全部 |
| `topk_3510_impl.h` | **1417** | **核心 radix-select** 实现 + 3510 公共入口 | 3510 / 5102 / 3003 / 3113 |
| `topk_v200_impl.h` | 458 | RpSort16 + MergeSort 路径 | v200 (`__NPU_ARCH__==2002`) |
| `topk_v220_impl.h` | 364 | Sort32 + MergeSort 路径 | v220 (`__NPU_ARCH__==2201`) |

### 1.3 文件包含关系

```
include/adv_api/sort/topk.h            ← 用户唯一入口
    ├─ include/.../topk_utils.h           (TopKInfo)
    ├─ include/.../topk_utils_constants.h (TopKConfig/Mode/Algo/Order)
    ├─ impl/.../topk_common_utils.h       (常量 + defaultTopKConfig)
    └─ 按 __NPU_ARCH__ 条件编译:
        ├─ 3510/5102/3003/3113 → impl/.../topk_3510_impl.h     (radix-select)
        ├─ 2201                → impl/.../topk_v220_impl.h      (Sort32+MrgSort)
        ├─ 2002                → impl/.../topk_v200_impl.h      (RpSort16+MrgSort)
        └─ 2201/2002 还共用    → impl/.../topk_common_impl.h    (MergeSort 包装)
```

源码引用: `topk.h:34-48` 是核心 `#if defined(__NPU_ARCH__)` 派发块。

---

## 2. 顶层 API 表面

### 2.1 `AscendC::TopK` 模板签名

共有 **4 个重载**,按 (是否传入 tmpLocal) × (芯片架构) 分:

#### 重载 1: 显式 tmpLocal + 3510 系列架构 (`topk.h:81-127`)

```cpp
template <typename T,
          bool isInitIndex = false,
          bool isHasfinish = false,
          bool isReuseSrc  = false,
          enum TopKMode topkMode = TopKMode::TOPK_NORMAL,
          const TopKConfig& config = defaultTopKConfig>
__aicore__ inline void TopK(
    const LocalTensor<T>&      dstValueLocal,    // [out] K 个 value
    const LocalTensor<int32_t>& dstIndexLocal,   // [out] K 个 index
    const LocalTensor<T>&      srcLocal,         // [in]  输入数据
    const LocalTensor<int32_t>& srcIndexLocal,   // [in]  输入 index (isInitIndex=true 时用)
    const LocalTensor<bool>&   finishLocal,      // [in]  每行是否完成 (isHasfinish=true 时用)
    const LocalTensor<uint8_t>& tmpLocal,        // [in]  显式临时缓冲
    const int32_t              k,
    const TopkTiling&          tilling,
    const TopKInfo&            topKInfo,
    const bool                 isLargest = true);
```

#### 重载 2: 隐式 stack buffer + 3510 系列 (`topk.h:155-218`)

去掉 `tmpLocal` 参数,内部用 `PopStackBuffer<T, TPosition::LCM>` 从编译器栈取一块。
带 `__ASC_USE_RESERVED_UBUF__(3510, ...)` 标记 —— 必须保留 3510 的 reserved UB 区。

#### 重载 3 & 4: v200 / v220 架构 (`topk.h:247-323`)

不带 `TopKConfig` 模板参数,内部走 `CHECK_FUNC_HIGHLEVEL_API` 宏派发,只能用 MergeSort
路径 (这两代芯片没有 radix-select 硬件支持)。

### 2.2 模板参数详解

| 参数 | 类型 | 默认 | 取值 | 含义 |
|---|---|---|---|---|
| `T` | typename | — | half / float / bfloat16_t / int8/16/32/64 / uint8/16/32/64 | 数据类型,RADIX_SELECT 支持 11 种,MERGE_SORT 仅 half/float |
| `isInitIndex` | bool | false | true / false | 是否由调用方提供 srcIndex |
| `isHasfinish` | bool | false | true / false | 是否启用 per-row finish flag (RADIX_SELECT 必须 false) |
| `isReuseSrc` | bool | false | true / false | 是否复用 src 内存 (预留) |
| `topkMode` | enum | TOPK_NORMAL | TOPK_NORMAL / TOPK_NSMALL | NSMALL 仅当 inner=32 时用,性能更高 |
| `config` | struct | `defaultTopKConfig` | `{algo, order, sorted}` | 算法选择 + 排序方向 + 是否输出已排序 |

### 2.3 关键结构体

#### `TopKInfo` (`topk_utils.h:27-31`)

```cpp
struct TopKInfo {
    int32_t outter = 1;   // 外层 (batch) 维度
    int32_t inner;        // 内层 (必须 32 字节对齐)
    int32_t n;            // 实际数据长度 (n <= inner)
};
```

#### `TopKConfig` (`topk_utils_constants.h:41-45`)

```cpp
struct TopKConfig {
    TopKAlgo  algo  = TopKAlgo::MERGE_SORT;  // 默认 MergeSort
    TopKOrder order = TopKOrder::UNSET;       // 排序方向 (UNSET 表示用 isLargest 参数)
    bool      sorted = true;                  // 输出是否已排序
};
```

> 注意: **默认是 MERGE_SORT**,但 top_k_v2 算子显式指定 `{RADIX_SELECT, UNSET, false}`。
> 见 `radix_sort_top_k.h:862`。

#### `defaultTopKConfig` (`topk_common_utils.h:65`)

```cpp
constexpr TopKConfig defaultTopKConfig = { TopKAlgo::MERGE_SORT, TopKOrder::UNSET, true };
```

仅在 3510/5102/3003/3113 架构下定义。v200/v220 不支持 RADIX_SELECT。

### 2.4 枚举完整定义

| 枚举 | 文件:行 | 取值 | 用途 |
|---|---|---|---|
| `TopKMode` | `topk_utils_constants.h:25-28` | `TOPK_NORMAL`, `TOPK_NSMALL` | Normal vs 内层=32 的小模式 |
| `TopKAlgo` | `topk_utils_constants.h:30-33` | `RADIX_SELECT`, `MERGE_SORT` | 算法选择 |
| `TopKOrder` | `topk_utils_constants.h:35-39` | `UNSET`, `LARGEST`, `SMALLEST` | 排序方向 |

### 2.5 `TopkTiling` 字段表 (`topk_tilingdata.h:15-48`)

Host 侧计算、序列化到 GM、Kernel 侧反序列化读取。共 28 个字段:

| 字段类别 | 字段 | 用途 |
|---|---|---|
| **尺寸** | `tmpLocalSize`, `allDataSize`, `innerDataSize` | 各 buffer 大小 |
| **对齐** | `kAlignFourBytes`, `kAlignTwoBytes`, `maskOffset` | K 值按 dtype 对齐 |
| **Sort 重复** | `sortRepeat`, `mrgSortRepeat` | 32元素 Sort / 归并排序的 repeat 次数 |
| **Merge 偏移** | `mrgSortSrc1offset`, `mrgSortSrc2offset`, `mrgSortSrc3offset`, `mrgSortTwoQueueSrc1Offset`, `mrgFourQueueTailPara1`, `mrgFourQueueTailPara2` | 4 路归并的 buffer offset |
| **Mask** | `maskVreducev2FourBytes`, `maskVreducev2TwoBytes`, `vreduceValMask0/1`, `vreduceIdxMask0/1`, `vreducehalfValMask0..7` | GatherMask 用的硬件 mask |
| **其它** | `srcIndexOffset`, `topkNSmallSrcIndexOffset`, `copyUbToUbBlockCount`, `topkMrgSrc1MaskSizeOffset` | 各种 offset / block 参数 |

### 2.6 dtype 支持矩阵

| Algo \ dtype | half | float | bfloat16_t | int8 | uint8 | int16 | uint16 | int32 | uint32 | int64 | uint64 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RADIX_SELECT** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **MERGE_SORT** | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

源码: `topk.h:99-102` (RADIX_SELECT static_assert) 和 `topk.h:117` (MERGE_SORT static_assert)。

---

## 3. 调用链与数据流

### 3.1 整体派发树 (3510 系列架构)

```
AscendC::TopK<T, ...>                                      [topk.h:81]
   │
   ├─ if AIC (Cube Core): return (TopK 仅在 Vector 核跑)    [:89-91]
   │
   ├─ CPU_DEBUG: TopkInputCheck<...>(k, topKInfo)           [:94]
   │     ├─ k ∈ [1, n]
   │     ├─ n ∈ [1, inner]
   │     ├─ inner % 32 == 0
   │     ├─ MERGE_SORT: T 必须 half/float
   │     ├─ TOPK_NORMAL: inner ≤ 4096          ★硬上限
   │     └─ TOPK_NSMALL: inner == 32
   │
   ├─ tempBuffer = tmpLocal.ReinterpretCast<T>()            [:96]
   │
   └─ 按 config.algo 派发:
       │
       ├─ RADIX_SELECT (top_k_v2 用的就是这条):
       │     static_assert(!isHasfinish)                    [:103]
       │     │
       │     ├─ TOPK_NORMAL → Reg::RadixSelectTopK::TopKNormal<...>
       │     │                                          [topk_3510_impl.h:1289]
       │     │     ├─ CreateVecIndex (如 !isInitIndex)   [:1299]
       │     │     ├─ for (i in 0..outter):
       │     │     │     └─ TopKRaidxSelect<...>          [:1307, 1158]
       │     │     │           ├─ InitializeTempBuffer    [:215/255]
       │     │     │           ├─ TwiddleInData           [:1194/1203]
       │     │     │           ├─ for byte MSB→LSB:       [:1215]
       │     │     │           │     ├─ GenerateAccumulateData  [:544, __simd_vf__]
       │     │     │           │     │   └─ Histograms<>  (硬件指令! `CHISTv2`)
                                            └─ dhistv2/chistv2()
                                                └─ `__builtin_cce_dhistv2_m`
                                                    └─ `CHISTv2`
       │     │     │           │     ├─ SetFlag/WaitFlag<V_S>
       │     │     │           │     └─ 二分查找 boundary  (Scalar)
       │     │     │           ├─ GatherGreaterAndEqualKData   [:579, __simd_vf__]
       │     │     │           ├─ GatherGreaterAndEqualKIndex  [:790, __simd_vf__]
       │     │     │           ├─ if (sorted) Sort<...>        [:1265]
       │     │     │           │     else SaveData
       │     │     │           └─ TwiddleOutData               [:1273/1278]
       │     │     │
       │     └─ TOPK_NSMALL → Reg::RadixSelectTopK::TopKNSmall<...>
       │                                            [topk_3510_impl.h:1318]
       │
       └─ MERGE_SORT:
             ├─ TOPK_NORMAL → AscendC::TopKNormal<...>      [topk_common_impl.h:45]
             └─ TOPK_NSMALL → AscendC::TopKNSmall<...>      [topk_common_impl.h:96]
                   ├─ ArithProgression 生成 index
                   ├─ if (!isLargest) Muls×(-1)  反转
                   ├─ TopKCompute / TopKNSmallCompute
                   │     (3510 在 topk_3510_impl.h:120/166; v200/v220 在各自 impl)
                   └─ if (!isLargest) Muls×(-1)  恢复
```

### 3.2 单次调用数据流 (RADIX_SELECT, 3510)

```
输入 (UB 内):
[srcLocal: T × inner]   [srcIndexLocal: int32 × inner]   [tmpLocal: uint8 × N]

   │
   ▼ InitializeTempBuffer [:215/255]
[tmpSrcData]    ← 与 src 同尺寸,存"≥ kthValue 的候选值"
[tmpSrcIndex]   ← 候选 index
[tmpHistData]   ← 256 bins × 2 (low/high) uint16 直方图,共 512B+512B
[realWorkData]  ← 工作数据 (初次=src,后续=tmpSrcData)
[sortTmpBuffer] ← Sort API 的临时区

   │
   ▼ TwiddleInData (signed → unsigned, FP→可排序)
[realWorkData] (unsigned,可比较)

   │  ┌─────────── Radix-Select 主循环 (MSB byte → LSB byte) ─────────┐
   │  │ for i = sizeof(T) .. 1:                                       │
   │  │     GenerateAccumulateData<byte i>                            │
   │  │       ├─ 提取第 i 字节 (UpdateMask + FilterDataAndGivenByte) │
   │  │       └─ Histograms<BIN0/BIN1, ACCUMULATE> ← 硬件指令!        │
   │  │     [tmpHistData: 256 uint16 × 2 (low/high)]                  │
   │  │     SetFlag<V_S> + WaitFlag<V_S>  (Scalar 等直方图完成)        │
   │  │     二分查找: 找 boundaryByte 使 hist[255]-hist[boundary]==remainK │
   │  │     kthValue |= (boundary+1) << ((i-1)*8)                     │
   │  │     remainK -= (hist[255] - hist[boundary+1])                 │
   │  └────────────────────────────────────────────────────────────────┘
   │
   ▼ GatherGreaterAndEqualKData + GatherGreaterAndEqualKIndex  (__simd_vf__)
[tmpSrcData: 实际 ≥ kthValue 的所有元素]   [tmpSrcIndex: 对应 index]
   │
   ▼ if (config.sorted)
   │     Sort<ConvType, int32_t, false, sortConfig>            [:1265]
   │       (递归调用 AscendC::Sort!)
   │ else
   │     SaveData(dst, dstIndex, tmpSrcData, tmpSrcIndex, k)   [:1268]
   │
   ▼ TwiddleOutData (unsigned → signed/FP)
[dstValueLocal: T × K]   [dstIndexLocal: int32 × K]
```

### 3.3 关键派发点 (`topk.h:34-48`)

```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 2002 || __NPU_ARCH__ == 3510 ||
    __NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)

  #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 2002)
    #include "topk_common_impl.h"   // MergeSort 公共包装
  #endif

  #if __NPU_ARCH__ == 2201
    #include "topk_v220_impl.h"     // Sort32 + MrgSort
  #elif __NPU_ARCH__ == 2002
    #include "topk_v200_impl.h"     // RpSort16 + MrgSort
  #elif __NPU_ARCH__ == 3510 || __NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113
    #include "topk_3510_impl.h"     // 完整 radix-select
  #endif
#endif
```

| `__NPU_ARCH__` | 芯片 | 实现路径 |
|---|---|---|
| `3510` | Ascend 910B / 类似 | `topk_3510_impl.h` (radix-select) |
| `5102` | (衍生) | 同上 |
| `3003` | (衍生) | 同上 |
| `3113` | (衍生) | 同上 |
| `2201` | Ascend 220 / 310P | `topk_v220_impl.h` (Sort32+MrgSort) |
| `2002` | Ascend 200 / 310 | `topk_v200_impl.h` (RpSort16+MrgSort) |

---

## 4. 核心算法实现 (3510 radix-select)

### 4.1 主函数 `TopKRaidxSelect` (`topk_3510_impl.h:1157-1284`)

#### 4.1.1 入口准备 (1157-1212)

```cpp
template <typename T, bool isInitIndex, bool isReuseSrc, const TopKConfig& config>
__aicore__ inline void TopKRaidxSelect(...)
{
    using ConvType = typename AscendC::Internal::ExtractTypeBySize<sizeof(T)>::T;
    //                                              ↑ 按 byte size 选 uint8/16/32/64

    // 把 LocalTensor 转为 raw __ubuf__ 指针
    __ubuf__ ConvType* src           = (__ubuf__ ConvType*)srcLocal.GetPhyAddr();
    __ubuf__ ConvType* dst           = (__ubuf__ ConvType*)dstValueLocal.GetPhyAddr();
    __ubuf__ int32_t*  srcIndex      = ...;
    __ubuf__ int32_t*  dstIndex      = ...;
    __ubuf__ uint8_t*  tmp           = ...;

    // 在 tmp 里切出 5 个子区
    __ubuf__ ConvType* tmpSrcData;       // 候选值
    __ubuf__ int32_t*  tmpSrcIndex;     // 候选 index
    __ubuf__ uint16_t* tmpHistData;     // 256 bins 直方图 (uint16)
    __ubuf__ ConvType* realWorkData = src;
    __ubuf__ ConvType* sortTmpBuffer;   // Sort API 临时区
}
```

#### 4.1.2 Twiddle (signed → unsigned) (1188-1208)

```cpp
if (NeedPreProcess<T>(isLargest)) {
    if (isLargest) {
        Internal::TwiddleInData<T, ConvType, false>(src, realWorkData, count);
    } else {
        Internal::TwiddleInData<T, ConvType, true>(src, realWorkData, count);
    }
}
```

**逻辑等价于 top_k_v2 算子的 `radix_sort_top_k.h:1109-1129` TwiddleIn 系列**:
- FP32 正数: 翻符号位 (`x | 0x80000000`)
- FP32 负数: 全取反 (`~x`)
- INT: 补码偏移

#### 4.1.3 **Radix-Select 主循环** (1215-1246)

```cpp
constexpr uint16_t typeBytes = sizeof(T);
int32_t remainK = k;
ConvType kthValue = 0;
event_t eventVS = ...->FetchEventID(HardEvent::V_S);

for (uint16_t i = typeBytes; i > 0 && remainK > 0; --i) {
    // (1) 计算第 i 字节的 256-bin 直方图
    GenerateAccumulateData<ConvType>(realWorkData, tmpHistData, tmpSrcData,
                                     realCount, kthValue, i);

    // (2) 等 Vector 完成,Scalar 才能读 tmpHistData
    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);

    // (3) 标量二分查找 boundary byte
    int32_t expValue = tmpHistData[255] - remainK;
    int16_t left = 0, right = 255;
    bool found = false;
    while (left <= right) {
        int16_t mid = left + (right - left) / 2;
        if (tmpHistData[mid] == expValue) {
            kthValue |= (static_cast<ConvType>(mid + 1) << ((i - 1) * 8));
            remainK = 0;
            found = true;
            break;
        } else if (tmpHistData[mid] > expValue) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    // (4) 找不到精确匹配,取最接近的边界
    if (!found) {
        if (right >= 0) {
            kthValue |= (static_cast<ConvType>(right + 1) << ((i - 1) * 8));
            remainK -= (tmpHistData[255] - tmpHistData[right + 1]);
        } else {
            remainK = tmpHistData[0] - expValue;
        }
    }
}
```

**算法关键点**:

| 步骤 | 计算 | 硬件资源 |
|---|---|---|
| (1) | byte-i 的 256-bin 累加直方图 | Vector (`Histograms` 硬件指令) |
| (2) | 等直方图完成 | HardEvent V→S 同步 |
| (3) | 二分查找 boundary byte | Scalar Unit |
| (4) | 计算 remainK | Scalar Unit |
| 循环 | `typeBytes` 次 (B8=1, B16=2, B32=4, B64=8) | 串行 |

#### 4.1.3.1 Radix-Select 算法示意图

> 下面 3 张图分别展示 **整体流水线**、**单次迭代的硬件/数据流**、**一个具体例子**。

**(图 A) 整体外层循环 —— 从 MSB 字节向 LSB 字节逐字节细化**

```
      Outer loop:  for (i = typeBytes ; i > 0 && remainK > 0 ; --i)
                                    MSB                          LSB
   byte index    [typeBytes-1]  [typeBytes-2]       [1]       [0]
   ─────────────────────────────────────────────────────────────►
                          (每次只看一个字节, 不动其它位)

        ┌─────────────┐        kthValue 累积 (逐字节填进去)
   i=4  │  iteration  │ ──►  kthValue = [ b3 |  ? |  ? |  ? ]
   (MSB)│  histogram  │      候选集 realWorkData 缩减到 "前缀==b3" 的子集
        │  + search   │
        └──────┬──────┘
               │  remainK > 0 ?  ──► 是 ──► 进入下一字节
               ▼
        ┌─────────────┐
   i=3  │  iteration  │ ──►  kthValue = [ b3 | b2 |  ? |  ? ]
        │  histogram  │      候选集继续缩减到 "前缀==b3b2" 的子集
        │  + search   │
        └──────┬──────┘
               ▼
        ┌─────────────┐
   i=2  │  iteration  │ ──►  kthValue = [ b3 | b2 | b1 |  ? ]
        └──────┬──────┘
               ▼
        ┌─────────────┐
   i=1  │  iteration  │ ──►  kthValue = [ b3 | b2 | b1 | b0 ]   完整确定
   (LSB)└──────┬──────┘
               │
       remainK == 0  或  所有字节已遍历 ──► 退出循环
               ▼
         GatherGreaterAndEqualK  (§4.1.4)
```

关键性质: kthValue 的"已确定字节"在每次循环里只增不减,候选集 `realWorkData` 单调递减,
保留的都是「**前缀字节 == kthValue 已确定字节**」的样本 —— 也就是同分(tie)候选项。

---

**(图 B) 单次迭代内的硬件 / 数据流 —— Vector 与 Scalar 的握手**

```
              ┌──────────────────────────  Vector Unit (SIMD)  ──────────────────────────┐
              │                                                                          │
   realWorkData ──► FilterDataAndGivenByteFromOri   (只保留前缀==kthValue 的样本)         │
   (N 个候选值)      │                                                                 │
                     │  对 byte i 抽取 → colWorkBits (uint8 序列, 长度 = N)               │
                     ▼                                                                 │
                ┌──────────────────────────────────────────────────┐                     │
                │  Histograms<uint8_t,uint16_t,BIN0,ACCUMULATE>    │  ◄── 硬件 IP       │
                │  Histograms<uint8_t,uint16_t,BIN1,ACCUMULATE>    │      (AIV Vector  │
                │                                                  │       microcode)  │
                └────────────────────┬─────────────────────────────┘                     │
                                     │                                                   │
                          累加直方图 hist[0..255]  (uint16 × 256)                       │
                                     │                                                   │
              Reg::StoreAlign ──► tmpHistData (UB)                                       │
                                     │                                                   │
              └──────────────────────┼──────────────────────────────────────────────────┘
                                     │
            ┌────────────────────────┼───────────────────────────  Scalar Unit  ─────────┐
            │                        ▼                                                   │
            │  SetFlag<HardEvent::V_S>   ◄── 通知 Scalar: 直方图已就绪                  │
            │  WaitFlag<HardEvent::V_S>  ◄── Scalar 阻塞等待 (一次硬同步)               │
            │                                                                          │
            │   expValue = tmpHistData[255] - remainK                                  │
            │                                                                          │
            │   left = 0, right = 255                                                  │
            │   while (left <= right):                                                 │
            │       mid = (left+right)/2                                               │
            │       if hist[mid] == expValue:    ◄── 精确命中, 此字节完全确定          │
            │           kthValue |= (mid+1) << (8*(i-1));  remainK = 0; break          │
            │       elif hist[mid] > expValue:  right = mid-1                          │
            │       else:                      left  = mid+1                          │
            │                                                                          │
            │   if !found (二分没精确命中):                                            │
            │       kthValue |= (right+1) << (8*(i-1))                                 │
            │       remainK -= hist[255] - hist[right+1]   (差额留到下字节细化)        │
            └──────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                候选集 realWorkData 在下一轮 GenerateAccumulateData 里
                被 FilterDataAndGivenByteFromOri 进一步过滤 (前缀必须匹配)
```

---

**(图 C) 256-bin 累加直方图 + 二分查找可视化**

```
   tmpHistData[0..255]  (uint16, cumulative)

   hist[255] │                                                █ ◄─ = N'  (本轮候选总数)
             │                                              █ █
             │                                            █ █ █
             │                                          █ █ █ █
             │                                        █ █ █ █ █
             │                                      █ █ █ █ █ █
   hist[b+1] │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ █ █ █ █ █ █ ─ ─ ─  ◄── 边界 byte = b+1
             │                                    █ █ █ █ █ █ █ █ █          (kthValue 的第 i 字节)
             │                                  █ █ █ █ █ █ █ █ █ █
             │                                █ █ █ █ █ █ █ █ █ █ █
             │                              █ █ █ █ █ █ █ █ █ █ █ █
             │                            █ █ █ █ █ █ █ █ █ █ █ █ █
             │                          █ █ █ █ █ █ █ █ █ █ █ █ █ █
   hist[0]   │ ───────────────────────────────────────────────────────
             └────────────────────────────────────────────────────────────► byte value
             0                        b  b+1                                    255

       hist[255] - hist[b+1]   =   "byte_i ≥ b+1" 的样本数  =  本轮能保进 top-K 的个数
                                    └─────────────────────┘
                                              │
                                              ▼
                                       与 remainK 比较
                                       ┌────────────┴────────────┐
                                       │                         │
                                  ≤ remainK                   > remainK
                                  (不够, 全收,             (够, 缩小右边界,
                                   差额留到下字节)           继续向左二分)
```

---

**(图 D) 一个具体例子 —— uint16, K=3, 输入 8 个样本**

```
   Twiddle 后(当作 uint16, 直接比较即可), realCount = 8:

       index :   0      1      2      3      4      5      6      7
       value : 0x0517 0x1234 0xABCD 0x05E2 0x1230 0xAB10 0x0001 0x1239
       high  :  0x05   0x12   0xAB   0x05   0x12   0xAB   0x00   0x12

   目标: 取最大的 3 个 ──► 直观答案 = {0xABCD, 0xAB10, 0x1239}

   ─── 第 1 轮: i = 2 (高字节) ────────────────────────────────────────────
     GenerateAccumulateData (前缀过滤: kthValue=0, 全保留):
        Histograms 后累加直方图:
           hist[0x00] = 1
           hist[0x05] = 3        ← 0x0517 + 0x05E2 + (0x0001) 累加进 0x05? ── 实际上 bin 是按 byte
           hist[0x12] = 6
           hist[0xAB] = 8
           hist[0xFF] = 8

     Scalar 二分:  expValue = hist[0xFF] - remainK = 8 - 3 = 5
        hist[0x05] = 3 < 5  ─► left  = 0x06
        hist[0x12] = 6 > 5  ─► right = 0x11
        (没有精确等于 5 的 bin)
     未命中 ─► kthValue |= (0x11+1) << 8 = 0x1200
                remainK -= hist[0xFF] - hist[0x12] = 8 - 6 = 2
                remainK = 3 - 2 = 1

     含义: 高字节 ≥ 0x12 的有 2 个 (0xABCD, 0xAB10) ── 全部进 top-K
            还差 1 个, 必须从 "高字节 == 0x12" 的样本里继续找

   ─── 第 2 轮: i = 1 (低字节) ────────────────────────────────────────────
     FilterDataAndGivenByteFromOri 只保留 (value >> 8) == 0x12 的样本:
        候选 = {0x1234, 0x1230, 0x1239}      realCount = 3

     低字节累加直方图:
        hist[0xFF] = 3
     Scalar 二分:  expValue = hist[0xFF] - remainK = 3 - 1 = 2
        hist[0x30] = 1
        hist[0x34] = 2  ── 精确命中!  mid = 0x34
     命中 ─► kthValue |= (0x34+1) << 0 = 0x0035
              但 mid+1 = 0x35 写进 kthValue ──►  kthValue = 0x1235
              remainK = 0  ─► 退出外层循环

   ─── §4.1.4: GatherGreaterAndEqualK ─────────────────────────────────────
     在原始 realWorkData 上做 GatherMask:
        保留 value ≥ kthValue(0x1235) 的样本
        ──► {0x1239, 0xAB10, 0xABCD}    (0x1234 和 0x1230 都 < 0x1235)
     正好 K=3 个

   ─── 收尾 ──────────────────────────────────────────────────────────────
     Sort<...>(3 个候选) ──► [0xABCD, 0xAB10, 0x1239]
     TwiddleOutData (unsigned → signed) ──► 还原原 dtype
     写回 dstValueLocal / dstIndexLocal
```

> **图 D 的教学价值**: 它把 `remainK` 的语义、`hist[255]-hist[b+1]` 的几何含义
> (即"图 C 阴影面积")、以及**同分(tie)样本**如何逐字节细化,在一次 trace 里全部串起来。

#### 4.1.4 收集候选 + 后处理 (1248-1283)

```cpp
GatherGreaterAndEqualKData(realWorkData, tmpSrcData, kthValue, realCount);
GatherGreaterAndEqualKIndex(realWorkData, srcIndex, tmpSrcIndex, kthValue, realCount);

if constexpr (config.sorted) {
    // 递归调用 AscendC::Sort 完成最终排序
    Sort<ConvType, int32_t, false, sortConfig>(
        dstValueTensor, dstIndexTensor, sortDataSrc, sortIndexSrc,
        sortBufferTensor, static_cast<uint32_t>(k));
} else {
    SaveData<T>(dst, dstIndex, tmpSrcData, tmpSrcIndex, k);
}

// Twiddle out (unsigned → signed)
TwiddleOutData<T, ConvType, ...>(dst, dst, k);
```

### 4.2 与 top_k_v2 算法对照

| 维度 | AscendC::TopK (3510) | top_k_v2 mode⑥ RadixSelect |
|---|---|---|
| 直方图硬件指令 | `Histograms<uint8_t, uint16_t, BIN0/BIN1, ACCUMULATE>` | Vector 计算 + `asc_vf_call` 累加 |
| 跨核累加 | 不支持 (单核算法) | `SetAtomicAdd<int32_t>` on `globalHistGm` |
| 输入规模上限 | **`inner ≤ 4096`** (`topk.h:59`) | 数百万 (分 tile) |
| 同步 | `SetFlag/WaitFlag<V_S>` | `PipeBarrier<PIPE_ALL>` + `SyncAll` |
| 主循环位置 | `topk_3510_impl.h:1215-1246` | `radix_sort_top_k.h:496-680` |
| 算法核心思想 | **完全相同** (MSB→LSB byte 二分) | **完全相同** |

---

## 5. 底层 intrinsic / 硬件调用

这一节是文档的核心: AscendC::TopK 最终落到哪些**硬件微码**。

### 5.1 函数属性 (编译期)

| 属性 | 含义 | 典型用法 |
|---|---|---|
| `__aicore__` | 标记 AIV 核函数 | 所有 kernel 函数 |
| `__simd_vf__` | Vector SIMD 微码函数 | `GenerateAccumulateData`, `GatherGreaterAndEqualKData/Index` |
| `__simd_callee__` | 可被 SIMD 调用的函数 | (较少见) |
| `__ubuf__` | UB (Unified Buffer) 指针限定符 | raw pointer 参数 |

源码: `topk_3510_impl.h:544, 579, 663, 706, 749, 790, 879, 926, 989` 都是 `__simd_vf__`。

### 5.2 寄存器级 intrinsic

#### 5.2.1 Mask (掩码) 操作

| API | 作用 | 源码位置 |
|---|---|---|
| `MaskReg fullMask = CreateMask<uint32_t>()` | 全 1 掩码 | `:596, 337` |
| `MaskReg zeroMask = CreateMask<uint32_t, MaskPattern::ALLF>()` | 全 0 掩码 | `:597, 338` |
| `MaskReg halfMask = CreateMask<uint32_t, MaskPattern::H>()` | 半掩码 (低半=1) | `:598, 339` |
| `MaskAnd(dst, m1, m2, full)` | 掩码与 | `:540, 621, 626` |
| `MaskOr(dst, m1, m2, full)` | 掩码或 | `:622` |
| `MaskInterleave<uint32_t>(dst1, dst2, m1, m2)` | 掩码交织 | `:601-602, 625` |
| `UpdateMask<uint8_t/uint32_t>(count)` | 运行时按元素数更新掩码 | `:562, 606, 634` |
| `SetVectorMask<T, MaskMode::COUNTER>(0, n)` | 设置向量掩码 (计数模式) | `topk_common_impl.h:66, 118, 129` |

#### 5.2.2 寄存器 Load/Store

| API | 作用 | 源码位置 |
|---|---|---|
| `Reg::LoadAlign(regTensor, ptr)` | **对齐** load 到 RegTensor | `:350, 609, 637` |
| `Reg::StoreAlign(ptr, regTensor, mask)` | **对齐** store | `:354, 575-576` |
| `Reg::StoreUnAlign<uint32_t>(ptr, regTensor, unalignReg)` | **不对齐** store (用 UnalignReg 跟踪偏移) | `:630` |
| `RegTensor<T>` | 寄存器张量类型 (编译期类型) | `:555, 608, 611, 628` |
| `UnalignReg unalignReg` | 不对齐写指针 (硬件自动累加) | `:604` |

#### 5.2.3 数据提取 (Gather/Filter)

| API | 作用 | 源码位置 |
|---|---|---|
| `GatherMask<T>(dst, src, mode, dropMask, mask, params, rsvdCnt)` | 按掩码收集元素 (核心指令!) | `:82, 88, 612, 613, 629` |
| `GatherMask<T, GatherMaskMode::STORE_REG>(dst, src, mask)` | Gather 到寄存器 (不写回 UB) | `:629` |
| `FilterDataAndGivenByteFromOri(mask, dst, src, ...)` | 按字节提取 + 过滤 | `:566-567` |
| `CompareScalar<uint32_t, CMPMODE::GT>(mask, src, value, half)` | 大于标量比较 → mask | `:616` |
| `CompareScalar<uint32_t, CMPMODE::EQ>(mask, src, value, half)` | 等于标量比较 → mask | `:617` |

#### 5.2.4 **`Histograms` —— AscendC::TopK 的灵魂指令**

```cpp
Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(
    acumHistLow, colWorkBits, filterMask);
Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::ACCUMULATE>(
    acumHistHigh, colWorkBits, filterMask);
```

源码: `topk_3510_impl.h:569-572`。

| 模板参数 | 含义 |
|---|---|
| `uint8_t` | 输入元素类型 (字节级) |
| `uint16_t` | 直方图 bin 计数类型 |
| `HistogramsBinType::BIN0` / `BIN1` | 取低 4 位 / 高 4 位 (256 bins 拆成 16+16) |
| `HistogramsType::ACCUMULATE` | 累加模式 (多次调用累加) |

**硬件含义**: 输入 64 个字节 (一个 Vector lane),硬件在一个 cycle 内把它们分到 256 个 bin
并累加。这是 Ascend AIV 独有的硬件 IP,**CPU/GPU 没有对应物**。

#### 5.2.5 特殊寄存器 (SPR)

| API | 作用 | 源码位置 |
|---|---|---|
| `ClearSpr<SpecialPurposeReg::AR>()` | 清 AR (Address Register?) | `:595` |
| `event_t eventVS = ...->FetchEventID(HardEvent::V_S)` | 取事件 ID | `:1213` |

#### 5.2.6 流水同步

| API | 作用 | 出现位置 |
|---|---|---|
| `SetFlag<HardEvent::V_S>(eventVS)` | V→S 事件设置 | `topk_3510_impl.h:1218` |
| `WaitFlag<HardEvent::V_S>(eventVS)` | V→S 事件等待 | `topk_3510_impl.h:1219` |
| `SetFlag<HardEvent::S_V>(eventID)` | S→V 事件设置 | `topk_common_impl.h` 等 |
| `WaitFlag<HardEvent::S_V>(eventID)` | S→V 事件等待 | 同上 |
| `PipeBarrier<PIPE_V>()` | Vector 流水屏障 | 散布全文件 |
| `SetMaskCount()` / `SetMaskNorm()` | 切换 mask 计数模式 | `topk_common_impl.h:59, 70` |
| `ResetMask()` | 复位 mask | `topk_common_impl.h:71` |

### 5.3 公共工具 intrinsic

| API | 作用 | 源码位置 |
|---|---|---|
| `PopStackBuffer<T, TPosition::LCM>(stackTensor)` | 从编译器栈弹出一块 UB | `topk.h:169`, `topk_common_impl.h:81` |
| `ArithProgression(dst, start, step, count)` | 生成等差数列 (用于 index 初始化) | `topk_common_impl.h:55, 107` |
| `CopyData(dst, topKInfo)` | 复制 index 到多行 | `topk_common_impl.h:111` |
| `CreateVecIndex(dst, 0, count)` | 生成 [0, 1, 2, ..., count-1] | `topk_3510_impl.h:1299` |
| `Duplicate(reg, value, mask)` | 寄存器批量赋值 | `:558-559` |
| `Muls<T, false>(dst, src, T(-1), MASK_PLACEHOLDER, ...)` | 标量乘 (用于反转符号) | `topk_common_impl.h:67, 119, 130` |
| `Sort<ConvType, int32_t, false, sortConfig>(...)` | **递归调用** AscendC::Sort | `topk_3510_impl.h:1265` |

### 5.4 Intrinsic 调用点统计

| 文件 | intrinsic 调用次数 (粗略) | 核心指令 |
|---|---|---|
| `topk_3510_impl.h` | ~150 处 | `Histograms`, `GatherMask`, `Reg::Load/Store`, `Compare` |
| `topk_common_impl.h` | ~20 处 | `Muls`, `PipeBarrier`, `SetVectorMask` |
| `topk_v200_impl.h` | ~40 处 | `MrgSort4`, `TmpLocalSort16`, `GatherMask` |
| `topk_v220_impl.h` | ~30 处 | `MrgFourQueueSort`, `TmpLocalSort32`, `GatherMask` |

### 5.5 真正的"黑盒"在哪里

```
AscendC::TopK          ← 公开 API (本文档解读)
   ↓
TopKNormal/NSmall      ← 模板包装 (本文档解读)
   ↓
TopKRaidxSelect        ← 算法主循环 (本文档解读)
   ↓
GenerateAccumulateData ← SIMD helper (本文档解读)
   ↓
Histograms<>           ← ★ 这里开始是闭源 ★
   ↓
AIV Vector 微码 ROM    ← 完全闭源 (华为 IP)
```

`Histograms<>` / `GatherMask<>` / `Reg::LoadAlign` 等的 C++ 声明在更深层的 impl 头文件里
(本文未深入),它们的实际执行是 AIV Vector 微码,**这部分不开源**。

---

## 6. 与 top_k_v2 算子的对照

### 6.1 直接回答那个问题

**Q: 既然有 AscendC::TopK,为什么 top_k_v2 还要自己写 RadixSelect?**

**A: 因为 AscendC::TopK 有 `inner ≤ 4096` 的硬上限。** 大规模 TopK 必须分 tile。

源码证据 (`topk.h:59` CPU_DEBUG 检查):
```cpp
ASCENDC_ASSERT((topKInfo.inner <= TOPK_NORMAL_INNER_MAX_LEN), {
    KERNEL_LOG(KERNEL_ERROR, "The maximum value supported by inner is 4096.");
});
```

### 6.2 协作关系

top_k_v2 的策略是 **"大循环 RadixSelect + 收尾 AscendC::TopK"**:

```
top_k_v2 mode⑥ 完整流程
─────────────────────────────────
1. 多 pass radix 跨核分布式:
   - 输入 N (可达数百万)
   - 多核累加到 globalHistGm (atomic)
   - 输出: tileTopkValueGm 上的 M 个候选 (M > K, 但 M << N)
                                 ↓
2. StoreFinalAnswer2Gm 调用 AscendC::TopK (radix_sort_top_k.h:877):
   - 输入 M 个候选 (M ≤ inner ≤ 4096 ✓)
   - 单核 radix-select 精确截到 K
   - 输出: 最终 K 个 (value, index)
```

### 6.3 算法层级对照表

| 步骤 | top_k_v2 (粗筛) | AscendC::TopK (精筛) | 是否相同算法 |
|---|---|---|:---:|
| Twiddle (signed→unsigned) | `TwiddleInB8/B16/B32/B64/Fp16/Fp32` | `Internal::TwiddleInData` | ✓ 同思路 |
| 主循环字节 | MSB→LSB byte 二分 | MSB→LSB byte 二分 | ✓ 同思路 |
| 直方图计算 | 软件 (Vector Add + asc_vf_call) | 硬件 (`Histograms<>` IP) | ✗ 不同实现 |
| 跨核累加 | `SetAtomicAdd<int32_t>` on GM | 不支持 (单核算法) | ✗ |
| 边界二分 | `BinarySearch` (`:719`) Scalar | Scalar 二分 (`:1225`) | ✓ 同思路 |
| 候选收集 | 软件循环 + `Adds` | `GatherGreaterAndEqualKData/Index` | ✗ 不同实现 |
| 最终排序 | `AscendC::Sort` (`:951`) | `Sort<>` (`:1265`) | ✓ 都是 Sort API |
| Twiddle out | 软件恢复 | `Internal::TwiddleOutData` | ✓ 同思路 |

### 6.4 top_k_v2 调用 AscendC::TopK 的位置

| 调用点 | 文件:行 | K 值 | 用途 |
|---|---|---|---|
| mode⑥ 收尾 | `radix_sort_top_k.h:877, 888` | `tileTopkValue_(idx)` | 多核分布式 → 精确 K |
| mode② SingleBlock | `radix_sort_top_k_single_block.h:160` | K | 单 block 直接调 |
| mode④ SingleCore | `radix_sort_top_k_single_core.h:461, 477` | K | 单核直接调 |
| mode⑤ InterCoreOptim | `radix_sort_top_k_inter_core_template_optimization.h:176, 182, 229` | K | 单核多 row 直接调 |

### 6.5 模板参数对应关系

```cpp
// top_k_v2 调用方式 (radix_sort_top_k.h:862, 877)
static constexpr TopKConfig topkConfig{
    TopKAlgo::RADIX_SELECT,   // ← 显式指定 RADIX_SELECT
    TopKOrder::UNSET,
    false                      // ← 不要求输出已排序 (后面单独 Sort)
};
AscendC::TopK<T, false, false, false, TopKMode::TOPK_NORMAL, topkConfig>(
    topkValueOutLocal, topkValueOutIndexLocal, inputLocalTensor, topkSrcIndexLocal,
    emptyFinishLocal, shareTmpBuffer,
    static_cast<int32_t>(tileTopkValue_(tileTopkValueIndex)),
    emptyTopkTiling, topKInfo,
    IS_LARGEST);
```

| 模板参数 | top_k_v2 传值 | 含义 |
|---|---|---|
| `T` | (运行时 dtype) | 11 种 dtype 之一 |
| `isInitIndex` | `false` | 调用方不提供,API 内部生成 |
| `isHasfinish` | `false` | RADIX_SELECT 强制必须 false |
| `isReuseSrc` | `false` | 默认 |
| `topkMode` | `TOPK_NORMAL` | (因为 inner 可能 > 32) |
| `config` | `{RADIX_SELECT, UNSET, false}` | 关键: sorted=false,因为 top_k_v2 之后会单独 Sort |

---

## 7. 设计观察与潜在陷阱

### 7.1 算法选择规则

| 触发条件 | 选择的 algo | 性能特征 |
|---|---|---|
| `config.algo == RADIX_SELECT` + dtype ∈ 11 种 | radix-select | 适合 N > 256 的大数据 |
| `config.algo == MERGE_SORT` + dtype ∈ {half, float} | merge sort | 适合小数据,但仅 2 种 dtype |
| `topkMode == TOPK_NSMALL` + `inner == 32` | 小模式特化 | 性能最优 (无需排序 32 个元素) |
| `topkMode == TOPK_NORMAL` + `inner ≤ 4096` | 普通模式 | inner=32 也支持,但比 NSMALL 慢 |

### 7.2 dtype / algo / mode 合法组合

| algo | mode | dtype | inner | isHasfinish |
|---|---|---|---|---|
| RADIX_SELECT | NORMAL | 11 种 | ≤ 4096 且 % 32==0 | 必须 false |
| RADIX_SELECT | NSMALL | 11 种 | == 32 | 必须 false |
| MERGE_SORT | NORMAL | half/float | ≤ 4096 且 % 32==0 | true/false |
| MERGE_SORT | NSMALL | half/float | == 32 | true/false |

源码: `topk.h:99-126` (static_assert 检查)。

### 7.3 UB 临时空间要求

| 数据类型 | tmpLocalSize 公式 | 例子 (inner=4096) |
|---|---|---|
| `float` | `inner * 16 / sizeof(float) = inner * 4` | 16K 元素 = 64KB |
| `half` | `inner * 16 / sizeof(half) = inner * 8` | 32K 元素 = 64KB |
| `int8_t` | `inner * 16` | 64K 元素 = 64KB |
| `int64_t` | `inner * 16 / sizeof(int64_t) = inner * 2` | 8K 元素 = 64KB |

源码注释: `topk.h:171-172`。**所有 dtype 都要求 ~64KB 量级 tmpLocal**,因为内部要存:
- 候选值 (×1)
- 候选 index (×1)
- 直方图 (256 × 2 × 2B = 1KB)
- 工作数据 (×1)
- Sort tmp (×1)

### 7.4 性能建议 (基于源码推断)

| 场景 | 建议 |
|---|---|
| `inner == 32` | 用 `TOPK_NSMALL`,性能最优 |
| `inner ≤ 4096` + 大 dtype (int64) | 用 RADIX_SELECT |
| `inner ≤ 4096` + half/float + 需要稳定排序 | MERGE_SORT 更稳 |
| `inner > 4096` | **不能直接用 AscendC::TopK**,要像 top_k_v2 那样分 tile |
| `K == 1` | RADIX_SELECT 仍走完 sizeof(T) 个 pass,可以考虑其它优化 |
| 多 batch (outter > 1) | RADIX_SELECT 内部 for 循环串行处理,不并行 |

### 7.5 潜在陷阱

| 陷阱 | 后果 | 规避 |
|---|---|---|
| 忘记 `topkConfig.algo = RADIX_SELECT` | 走 MERGE_SORT,dtype 受限 | 显式指定 (像 top_k_v2 那样) |
| `inner` 不是 32 倍数 | CPU_DEBUG assert,Release 行为未定义 | 调用前对齐 padding |
| `inner > 4096` | assert 失败 / 越界 | 自己分 tile |
| 同时设 `isHasfinish=true` + `RADIX_SELECT` | static_assert 失败 | 只能 MERGE_SORT 支持 finish |
| `K > inner` | assert 失败 | 调用前 clamp |
| `tmpLocal` 不足 | stack buffer 版会 assert;显式版会越界 | 用 `GetTopKMaxMinTmpSize` 算 |
| `K` 不是 dtype 对齐倍数 | 输出 index 可能含 padding | 用 `kAlignFourBytes/TwoBytes` 修正 |
| 多线程并发调用 | UB 是核私有,但 `__ASC_USE_RESERVED_UBUF__` 要求保留区 | 检查编译选项 |

### 7.6 与 Sort API 的关系

AscendC::TopK 在 `config.sorted=true` 时**递归调用 `AscendC::Sort`** (`topk_3510_impl.h:1265`)。
这意味着:

1. Sort API 必须可访问 (同一 `sort.h` 头文件)
2. Sort 的 tmp buffer 需求要算进 TopK 的 tmpLocal
3. 如果用户只想"找 K 个不要求排序",设 `config.sorted=false` 可省一次 Sort

源码:
```cpp
if constexpr (config.sorted) {
    Sort<ConvType, int32_t, false, sortConfig>(
        dstValueTensor, dstIndexTensor, sortDataSrc, sortIndexSrc,
        sortBufferTensor, static_cast<uint32_t>(k));
} else {
    SaveData<T>(dst, dstIndex, tmpSrcData, tmpSrcIndex, k);
}
```

---

## 8. 总结

### 8.1 核心设计要点

1. **Header-only 模板库**: 完全 inline,无 .so / .a 链接,源码可读。
2. **架构派发**: `__NPU_ARCH__` 决定 include 哪个 impl 文件,编译期决定代码路径。
3. **三种芯片路径**: 3510 系列 (radix-select)、v220 (Sort32+MrgSort)、v200 (RpSort16+MrgSort)。
4. **`Histograms<>` 是灵魂指令**: AIV 独有硬件 IP,256-bin 直方图单 cycle 完成。
5. **算法本质 = top_k_v2 mode⑥**: 同样 MSB→LSB byte 二分,但单核 + 硬件加速。
6. **`inner ≤ 4096` 硬上限**: 这就是 top_k_v2 必须自己做 tile 分布式的原因。
7. **递归复用 Sort**: `config.sorted=true` 时调 `AscendC::Sort` 完成最终排序。

### 8.2 关键文件:行号速查

| 内容 | 位置 |
|---|---|
| `AscendC::TopK` 主入口 | `topk.h:81` |
| 显式 tmpLocal 重载 | `topk.h:81-127` |
| 隐式 stack buffer 重载 | `topk.h:155-218` |
| 架构派发 | `topk.h:34-48` |
| dtype static_assert | `topk.h:99-102` (RADIX_SELECT), `topk.h:117` (MERGE_SORT) |
| `TopKConfig` 定义 | `topk_utils_constants.h:41-45` |
| `defaultTopKConfig` | `topk_common_utils.h:65` |
| `TopKInfo` | `topk_utils.h:27-31` |
| `TopkTiling` 字段 | `topk_tilingdata.h:15-48` |
| TopKNormal 包装 | `topk_common_impl.h:45-93` |
| TopKNSmall 包装 | `topk_common_impl.h:96-156` |
| **Radix-Select 主循环** | `topk_3510_impl.h:1157-1284` |
| `GenerateAccumulateData` (硬件直方图) | `topk_3510_impl.h:544-577` |
| `GatherGreaterAndEqualKData` | `topk_3510_impl.h:579-662` |
| `GatherGreaterAndEqualKIndex` | `topk_3510_impl.h:790-989` |
| 二分查找 boundary byte | `topk_3510_impl.h:1225-1237` |
| `Sort<>` 递归调用 | `topk_3510_impl.h:1265` |
| `TwiddleInData` / `TwiddleOutData` | `topk_3510_impl.h:1194, 1273-1281` |
| `TopkInputCheck` (约束) | `topk_3510_impl.h:41-68` |

### 8.3 与既有文档的关系

| 想了解... | 看 |
|---|---|
| TopK API 公开签名和用法 | 本文 §2 |
| AscendC::TopK 内部算法 | 本文 §3, §4 |
| 硬件 intrinsic 列表 | 本文 §5 |
| top_k_v2 为什么不直接用 AscendC::TopK | 本文 §6 |
| dtype/algo/mode 合法组合 | 本文 §7.2 |
| UB 临时空间需求 | 本文 §7.3 |
| top_k_v2 整体架构 | `CODE_ANALYSIS.md` |
| top_k_v2 Tiling 调度 | `TILING_DETAIL.md` |
| top_k_v2 Kernel 7 种模式 | `KERNEL_DETAIL.md` |
| 硬件能力需求矩阵 | `HARDWARE_REQUIREMENTS.md` |

### 8.4 一句话总结

`AscendC::TopK` 是一个 **header-only、模板化、按芯片架构派发** 的 TopK 库,核心是
基于 `Histograms<>` 硬件指令的 radix-select 算法,与 top_k_v2 算子的 mode⑥ 算法
**思想完全一致**,但限于单核 + `inner ≤ 4096`,所以大规模 TopK 仍需要 top_k_v2 自己
做跨核 tile 分布式。

---

**文档状态**: 完成。
**基于源码版本**: `asc-devkit/` (本地副本,2025 Huawei Technologies CANN OSL v2.0)。
**最后更新**: 2026-06-16。

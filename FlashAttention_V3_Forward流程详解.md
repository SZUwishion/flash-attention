# Flash Attention V3 Forward 流程详解

> 版本：`hopper/` 目录 (SM90/SM80 双路径)  
> 重点：Hopper SM90 路径，TMA + WGMMA + 持久化调度

---

## 目录

1. [V3 vs V2 核心差异一览](#1-v3-vs-v2-核心差异一览)
2. [整体调用链路](#2-整体调用链路)
3. [Python 入口层](#3-python-入口层)
4. [核心参数结构 Flash_fwd_params](#4-核心参数结构-flash_fwd_params)
5. [C++ API 层：mha_fwd](#5-c-api-层mha_fwd)
6. [编译期分发链路](#6-编译期分发链路)
7. [内核启动模板 run_flash_fwd](#7-内核启动模板-run_flash_fwd)
8. [持久化 Tile 调度器](#8-持久化-tile-调度器)
9. [SM90 Warp 专用化内核 FlashAttnFwdSm90](#9-sm90-warp-专用化内核-flashattnfwdsm90)
10. [Producer Warpgroup：TMA 异步加载](#10-producer-warpgroupTMA-异步加载)
11. [Consumer Warpgroup：WGMMA 矩阵计算](#11-consumer-warpgroupwgmma-矩阵计算)
12. [在线 Softmax 与注意力分数计算](#12-在线-softmax-与注意力分数计算)
13. [Epilogue：输出归一化与写回](#13-epilogue输出归一化与写回)
14. [Split-KV 合并 Kernel](#14-split-kv-合并-kernel)
15. [关键优化技术汇总](#15-关键优化技术汇总)
16. [完整执行时序图](#16-完整执行时序图)

---

## 1. V3 vs V2 核心差异一览

| 特性 | V2 (`csrc/flash_attn/`) | V3 (`hopper/`) |
|------|------------------------|----------------|
| 目标架构 | SM80 (Ampere) | SM80 + **SM90 (Hopper)** |
| 矩阵乘法单元 | Warp-level MMA | **WGMMA** (Warpgroup MMA, 4 warp 协作) |
| 数据加载 | `cp.async` | **TMA** (Tensor Memory Accelerator) |
| Warp 职责 | 所有 warp 都做计算 | **生产者/消费者 Warpgroup 专用化** |
| 调度器 | 静态网格 (blockIdx) | **持久化调度器** + 信号量 |
| FP8 支持 | ❌ | ✅ `e4m3`，含 descale |
| V headdim | 必须 = Q/K headdim | **可以不同** (如 MLA: QK=192, V=128) |
| PackGQA | ❌ | ✅ 将 GQA 组折叠进 seqlen |
| 动态 Split-KV | ❌ | ✅ `num_splits_dynamic_ptr` |
| Paged KV + TMA | ❌ | ✅ `pagedkv_tma` |
| PDL 调度 | ❌ | ✅ Programmatic Dependent Launch |
| Cluster MMA | ❌ | ✅ `ClusterM=2` 多播 K/V |

---

## 2. 整体调用链路

```
Python: flash_attn_varlen_func / flash_attn_func
         ↓ torch.ops.flash_attn_3.fwd()
         ↓ flash_attn_3._C (torch extension)
C++:    mha_fwd()                             # hopper/flash_api.cpp
         ↓ set_params_fprop()
         ↓ ARCH_SWITCH / SPLIT_SWITCH / ...   # 编译期多维度分发
         ↓ run_mha_fwd_constexpr<Arch, ...>()
         ↓ run_mha_fwd_<Arch, T, kHeadDim, kHeadDimV, ...>()
         ↓ run_flash_fwd<...>()               # hopper/flash_fwd_launch_template.h
         |   ├─ 选择 CollectiveMainloopFwdSm90
         |   ├─ 选择 CollectiveEpilogueFwd
         |   ├─ 选择 Scheduler（静态/动态/Varlen动态）
         |   └─ 构建 FlashAttnFwdSm90<Mainloop, Epilogue, Scheduler>
         ↓ prepare_varlen_num_blocks()         # Varlen 预处理 PDL kernel
         ↓ AttnKernel<<<grid, block, smem>>>() # SM90 主 kernel
              ↓
              FlashAttnFwdSm90::operator()
                ├─ [WG0: 生产者]  load() 循环: TMA 加载 Q/K/V
                └─ [WG1/WG2: 消费者]  mma() 循环: WGMMA 计算 QK^T/softmax/PV
                     ↓
                     CollectiveEpilogueFwd::store()  # 写 O + LSE
         ↓ (if num_splits > 1) run_mha_fwd_combine()  # Split-KV 合并
```

---

## 3. Python 入口层

> 文件：[hopper/flash_attn_interface.py](hopper/flash_attn_interface.py)

### 3.1 函数签名扩展

V3 的 `_flash_attn_forward` 相比 V2 新增了大量参数：

```python
@torch.library.custom_op("flash_attn_3::fwd", mutates_args={"k", "v", "out"})
def _flash_attn_forward(
    q, k, v,
    # ── 新增：KV 缓存追加 ──────────────────────────────────
    k_new, v_new,
    # ── 新增：MLA 专用 Q_v（V 空间的 Q） ──────────────────
    qv,
    # ── 新增：Paged KV 缓存 ────────────────────────────────
    out, cu_seqlens_q, cu_seqlens_k, cu_seqlens_knew,
    seqused_q, seqused_k, leftpad_k, seqlens_rotary,
    page_table, kv_batch_idx,
    # ── 新增：旋转位置编码 ─────────────────────────────────
    rotary_cos, rotary_sin,
    # ── 新增：FP8 量化 descale ─────────────────────────────
    q_descale, k_descale, v_descale,
    # ── 调度与分块控制 ─────────────────────────────────────
    softmax_scale, causal, window_size_left, window_size_right,
    attention_chunk, softcap,
    rotary_interleaved,
    scheduler_metadata, num_splits, pack_gqa, sm_margin,
    ...
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # 返回: (out, softmax_lse, out_accum, softmax_lse_accum)
```

### 3.2 返回值变化

| 返回值 | 形状 | 说明 |
|--------|------|------|
| `out` | `[B, seqlen_q, H, dv]` | 最终输出 |
| `softmax_lse` | `[B, H, seqlen_q]` | log-sum-exp，用于反向传播 |
| `out_accum` | `[num_splits, B, H, seqlen_q, dv]` | Split-KV 时各分块的 float32 累加器 |
| `softmax_lse_accum` | `[num_splits, B, H, seqlen_q]` | Split-KV 时各分块的 LSE |

> 当 `num_splits == 1` 时，`out_accum` 和 `softmax_lse_accum` 为空张量，节省内存。

---

## 4. 核心参数结构 Flash_fwd_params

> 文件：[hopper/flash.h](hopper/flash.h)

V3 在 V2 的 `Flash_fwd_params` 基础上新增了以下关键字段：

```cpp
struct Flash_fwd_params : public Flash_fwd_bwd_params {
    // ── FP8 支持 ──────────────────────────────────────────────
    bool     is_e4m3;               // 是否为 FP8 e4m3 格式
    float   *q_descale_ptr;         // Q 的反量化缩放因子
    float   *k_descale_ptr;         // K 的反量化缩放因子
    float   *v_descale_ptr;         // V 的反量化缩放因子

    // ── V headdim 与 Q/K 可以不同（MLA 特性） ──────────────────
    int      dv;                    // V 的 headdim（可 ≠ d）
    int      dv_rounded;            // 上对齐后的 dv
    index_t  v_dim_stride;          // V 第一维步长（用于 colmajor 检测）

    // ── PackGQA：将 GQA 组折叠进 seqlen ────────────────────────
    bool     pack_gqa;              // 是否启用 PackGQA

    // ── 持久化调度器信号量 ──────────────────────────────────────
    int     *tile_count_semaphore;  // 全局原子计数器，分配 tile

    // ── 动态 Split-KV ───────────────────────────────────────────
    int     *num_splits_dynamic_ptr; // 每个 batch 的实际 split 数
    int     *num_m_blocks_ptr;       // Varlen 每个 batch 的 m-blocks 数
    int     *varlen_batch_idx_ptr;   // Varlen 排序后的 batch 索引
    int     *num_nheads_in_l2_ptr;   // L2 感知的 head swizzle

    // ── 硬件信息 ────────────────────────────────────────────────
    int      arch;                  // GPU 架构 (80/86/89/90)
    int      num_sm;                // 有效 SM 数 = total_SM - sm_margin

    // ── 高级特性 ────────────────────────────────────────────────
    int      attention_chunk;       // 分块 attention（chunkwise）
    bool     pagedkv_tma;           // Paged KV 是否使用 TMA 路径
    bool     prepare_varlen_pdl;    // 是否用 PDL 重叠预处理 kernel
    bool     skip_scheduler_metadata_computation;
    bool     head_swizzle;
    bool     varlen_sort_batches;
};
```

---

## 5. C++ API 层：mha_fwd

> 文件：[hopper/flash_api.cpp](hopper/flash_api.cpp)，主函数约 700 行

### 5.1 参数检验与形状推导

```
输入校验
├─ 支持架构：SM80/SM86/SM89/SM90（不支持 SM70）
├─ 数据类型：fp16 / bf16 / fp8_e4m3
├─ dv != d 仅在 SM90（Hopper）支持
├─ dv > 256 时只支持 fp16/bf16
├─ Varlen 模式：输入形状为 [total_q, H, D]
└─ Paged KV：page_size 必须是 kBlockN 的倍数
```

### 5.2 核心 Heuristics

#### `get_num_splits` —— Split-KV 分块数决策

```cpp
// 文件：hopper/flash_api.cpp
int get_num_splits(Flash_fwd_params& params) {
    // 1. 根据架构获取 tile 尺寸
    auto [kBlockM, kBlockN] = (params.arch >= 90)
        ? tile_size_fwd_sm90(...)  // SM90 专用 tile 尺寸
        : tile_size_fwd_sm8x(...); // SM80 通用 tile 尺寸

    // 2. KV 头的内存大小（是否会超出 L2 缓存）
    long long size_one_kv_head = seqlen_k * (d + dv) * element_size;

    // 3. 调用 heuristic 函数（见 hopper/heuristics.h）
    return num_splits_heuristic(
        total_mblocks, num_sm, num_n_blocks, num_m_blocks,
        size_one_kv_head, is_causal_or_local, max_splits
    );
}
```

**`num_splits_heuristic` 核心逻辑**（[hopper/heuristics.h](hopper/heuristics.h)）：

```
策略 1：若 total_mblocks >= 0.8 * num_SMs
  → 基本够用，不需要 Split（除非 KV 超过 L2=50MB 且 query 足够多）

策略 2：若 num_n_blocks <= 4
  → 太小，不值得 Split

策略 3：否则，寻找最优 splits
  for num_splits in [1, max_splits]:
      n_waves = total_mblocks * num_splits / num_SMs
      efficiency = n_waves / ceil(n_waves)    ← 波次效率
  返回第一个 efficiency >= 0.85 * max_efficiency 的 splits
```

**"波次效率"示例**：
- 48 个 m-blocks，108 个 SM
  - 1 split → n_waves=0.44，eff=44%（只用了 44% 的 SM）
  - 2 splits → n_waves=0.89，eff=89%
  - 3 splits → n_waves=1.33，eff=67%（需要 2 波）
  - **选 2 splits**（89% > 85% × 89%）

#### `get_pack_gqa` —— GQA Packing 决策

```cpp
bool get_pack_gqa(Flash_fwd_params& params) {
    // SM8x 始终 PackGQA（因为 SM8x 没有 TMA，PackGQA 更高效）
    if (params.arch < 90) return true;
    // 分 Split 时也需要
    if (params.num_splits > 1) return true;
    // PagedKV 非 TMA 路径
    if (page_table && !pagedkv_tma) return true;
    // 否则用 heuristic 判断
    return should_pack_gqa(varlen_q, seqlen_q, qhead_per_khead, kBlockM);
}
```

`should_pack_gqa` 比较效率：
$$\text{PackGQA 有利的条件} = \frac{\text{seqlen}_q}{\lceil \text{seqlen}_q / B_M \rceil} < 0.9 \times \frac{\text{seqlen}_q \times r}{\lceil \text{seqlen}_q \times r / B_M \rceil}$$

其中 $r$ 是 `qhead_per_khead`（每个 KV head 对应的 Q head 数）。

### 5.3 mha_fwd 主体流程

```
① 分配输出张量
   out       [B, seqlen_q, H, dv]       fp16/bf16 (FP8 输入 → bf16 输出)
   softmax_lse [B, H, seqlen_q]          fp32

② set_params_fprop(...)
   填充 Flash_fwd_params 所有字段

③ 调度器信号量分配（持久化调度器用）
   scheduler_metadata = torch::zeros({...}, dtype=int32)
   params.tile_count_semaphore = scheduler_metadata.data_ptr<int>()
   params.num_splits_dynamic_ptr → metadata[...]
   params.num_m_blocks_ptr       → metadata[...]
   params.varlen_batch_idx_ptr   → metadata[...]
   params.num_nheads_in_l2_ptr   → metadata[...]

④ 可选：配置 AppendKV 参数
   params.knew_ptr / params.vnew_ptr
   params.seqlen_knew / params.total_knew

⑤ 可选：配置旋转位置编码
   params.rotary_cos_ptr / params.rotary_sin_ptr

⑥ 可选：配置 FP8 descale
   params.q_descale_ptr / params.k_descale_ptr / params.v_descale_ptr

⑦ Split-KV 累加器分配（仅 num_splits > 1）
   out_accum    [num_splits, B, H, seqlen_q, dv]   fp32
   lse_accum    [num_splits, B, H, seqlen_q]        fp32

⑧ run_mha_fwd(params, stream)      ← 主 kernel

⑨ (if num_splits > 1)
   run_mha_fwd_combine(params, stream, enable_pdl=true)

⑩ 返回 {out, softmax_lse, out_accum, softmax_lse_accum}
```

---

## 6. 编译期分发链路

> 文件：[hopper/flash_api.cpp](hopper/flash_api.cpp) + [hopper/static_switch.h](hopper/static_switch.h)

V3 的分发维度比 V2 多出许多：

```cpp
void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream) {
    ARCH_SWITCH(params.arch, Arch, [&] {             // SM80/86/89/90
      SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
        PAGEDKV_SWITCH(page_table && !pagedkv_tma, PagedKVNonTMA, [&] {
          PACKGQA_SWITCH(params.pack_gqa, PackGQA, [&] {
            SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
              run_mha_fwd_constexpr<Arch, Split, PagedKVNonTMA, PackGQA, Has_softcap>(params, stream);
            });
          });
        });
      });
    });
}
```

`run_mha_fwd_constexpr` 再继续按数据类型和 headdim 分发：

```cpp
template <int Arch, bool Split, bool PagedKVNonTMA, bool PackGQA, bool Has_softcap>
void run_mha_fwd_constexpr(Flash_fwd_params& params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_e4m3, [&] {         // fp16/bf16 或 fp8_e4m3
      HEADDIM_SWITCH(params.d, kHeadDim, [&] { // 32/64/96/128/192/256
        HEADDIMV_SWITCH(params.dv, kHeadDimV, [&] { // 可以不同于 kHeadDim
          run_mha_fwd_<Arch, T, kHeadDim, kHeadDimV, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
        });
      });
    });
}
```

每一个 `<Arch, T, kHeadDim, kHeadDimV, ...>` 组合都对应一个**独立编译的 kernel**，实现零运行时开销的多态。

---

## 7. 内核启动模板 run_flash_fwd

> 文件：[hopper/flash_fwd_launch_template.h](hopper/flash_fwd_launch_template.h)

### 7.1 Tile 尺寸选择

```cpp
template <int Arch, int kHeadDim, int kHeadDimV, ...>
void run_flash_fwd(Flash_fwd_params& params, cudaStream_t stream) {
    // SM90 的 tile 尺寸
    constexpr auto [kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap] =
        tile_size_fwd_sm90(kHeadDim, kHeadDimV, Is_causal, Is_local,
                           sizeof(Element), V_colmajor, PagedKVNonTMA, Has_softcap);
    // SM8x 的 tile 尺寸
    constexpr auto [kBlockM, kBlockN, kNWarps, kStages, Q_in_regs] =
        tile_size_fwd_sm8x(Arch==86||Arch==89, kHeadDim, kHeadDimV, ...);
    
    // kStages = 2（SM90），流水线深度
    // MmaPV_is_RS: P 矩阵是否保留在寄存器（而非写回 SMEM）
    // IntraWGOverlap: 是否启用 WarpGroup 内 QK gemm 和 PV gemm 的流水线重叠
```

> **典型 tile 尺寸示例**（SM90, hdim=128, fp16）：
> - kBlockM=128, kBlockN=128, kStages=2, MmaPV_is_RS=false

### 7.2 类型系统组装

```cpp
// 主 loop 类型（SM90 路径）
using CollectiveMainloop = flash::CollectiveMainloopFwdSm90<
    kStages, ClusterShape, TileShape_MNK, kHeadDimV, Element, float,
    SM90, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA,
    AppendKV, HasQv, MmaPV_is_RS, IntraWGOverlap, PackGQA, Split, V_colmajor>;

// Epilogue 类型
using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
    TileShape_MNK_PV, ClusterShape, ElementOut, SM90,
    CollectiveMainloop::NumMmaThreads, Varlen, PackGQA, Split, FP8_TransposeV>;

// 调度器类型选择
using Scheduler = std::conditional_t<
    !UsePersistentScheduler,
    SingleTileScheduler<...>,          // 非持久化：单 tile 对应一个 CTA
    std::conditional_t<Varlen,
        VarlenDynamicPersistentTileScheduler<...>,  // Varlen 动态持久化
        std::conditional_t<!Is_causal && !Is_local,
            StaticPersistentTileScheduler<...>,     // 非因果：静态持久化
            DynamicPersistentTileScheduler<...>     // 因果：动态（LPT 调度）
        >
    >
>;
```

**调度器选择逻辑：**

```
SM90：
  if (Split && !Varlen)         → SingleTileScheduler
  else if (Varlen)              → VarlenDynamicPersistentTileScheduler
  else if (!Is_causal&&!local)  → StaticPersistentTileScheduler
  else                          → DynamicPersistentTileScheduler（LPT）

SM80：
  if (Is_causal && !Varlen)     → DynamicPersistentTileScheduler
  else if (Varlen && Split)     → VarlenDynamicPersistentTileScheduler
  else                          → SingleTileScheduler
```

### 7.3 Varlen 预处理：prepare_varlen_num_blocks

```cpp
if (Varlen && !params.skip_scheduler_metadata_computation) {
    prepare_varlen_num_blocks(
        params, stream, PackGQA, kBlockM, kBlockN,
        Arch >= 90 && params.prepare_varlen_pdl /*enable_pdl*/
    );
}
```

这个**预处理 kernel** 计算：
- `num_m_blocks_ptr[b]` = 每个 batch `b` 的 m-block 数量
- `num_splits_dynamic_ptr[b]` = 每个 batch 的动态 split 数
- `varlen_batch_idx_ptr` = 按序列长度排序的 batch 索引（使长序列先处理）

当 `prepare_varlen_pdl=true` 时，使用 **PDL（Programmatic Dependent Launch）** 与主 kernel 重叠执行，隐藏预处理延迟。

### 7.4 Kernel 启动

```cpp
// 获取 grid/block/smem
dim3 grid_dims  = AttnKernel::get_grid_shape(kernel_params);   // = num_sm（持久化时）
dim3 block_dims = AttnKernel::get_block_shape();               // = 128*NumWarpGroups
int  smem_size  = AttnKernel::SharedStorageSize;               // ~200KB（SM90）

// 实际启动
cutlass::kernel_launch<AttnKernel>(
    grid_dims, block_dims, smem_size, stream, kernel_params,
    Arch>=90 && Varlen && PDL_enabled /*launch_with_pdl*/
);
```

---

## 8. 持久化 Tile 调度器

> 文件：[hopper/tile_scheduler.hpp](hopper/tile_scheduler.hpp)

V3 引入了持久化调度（Persistent Scheduler）：**每个 SM 上的 CTA 不退出，而是持续从调度队列中领取新 tile**。

### 8.1 StaticPersistentTileScheduler（非因果场景）

```
所有 tile 均匀分配给 SM：

tile_idx = blockIdx.x, blockIdx.x + gridDim.x, blockIdx.x + 2*gridDim.x, ...

┌───────────────────────────────────────────────┐
│ SM 0: tile 0 → tile num_sm → tile 2*num_sm    │
│ SM 1: tile 1 → tile num_sm+1 → ...            │
│ ...                                            │
└───────────────────────────────────────────────┘
```

grid.x = num_sm（而非 num_tiles），**减少 kernel 启动开销**。

### 8.2 DynamicPersistentTileScheduler（因果/local 场景）

因果 attention 中，不同 tile 计算量差异巨大（靠近对角线的 tile 更重）。V3 采用 **Longest-Processing-Time-First (LPT) 调度**：

```cpp
// 使用全局原子计数器 tile_count_semaphore 分配 tile：
void prefetch_next_work(Params& params, WorkTileInfo& work_info) {
    if (threadIdx.x % NumProducerThreads == 0) {
        // 原子操作抢占下一个 tile
        work_info.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + gridDim.x;
    }
}
```

同时进行 **L2 感知 Swizzle**（head-batch 两级排列），确保 K/V 数据在 L2 中复用：

```
总 (num_head × num_batch) 个 head-batch 对，分为若干 "section"
section 大小 = floor(L2_size(32MB) / size_one_kv_head)，取最近 2 的幂次

Section 内按 LPT 倒序分配 tile（最长的序列先计算）：
tile_idx → L2_major → L2_minor → block_idx（反转后）→ bidh, bidb
```

### 8.3 VarlenDynamicPersistentTileScheduler（变长序列）

针对变长序列（decode 阶段最常见，每个序列长度差异巨大）：

```
预处理阶段（prepare_varlen_num_blocks）：
  1. 计算每个 batch 的 num_m_blocks
  2. 按序列长度排序，填写 varlen_batch_idx_ptr（优先处理长序列）
  3. 可选：计算 num_splits_dynamic（每 batch 独立 split 数）

主 kernel 调度：
  每个 CTA 从 tile_count_semaphore 原子抢占 tile_idx
  tile_idx → (bidb, bidh, m_block) 的映射通过 num_m_blocks_ptr 快速查找
```

---

## 9. SM90 Warp 专用化内核 FlashAttnFwdSm90

> 文件：[hopper/flash_fwd_kernel_sm90.h](hopper/flash_fwd_kernel_sm90.h)

### 9.1 线程块结构

```
Block Dims: (NumMmaThreads + 128, 1, 1)
例如：NumMmaWarpGroups=2 时，block_size = 256 + 128 = 384 线程

线程布局：
┌─────────────────────────────────────────────────────────────────┐
│ Warpgroup 0  (thread 0..127)   : Producer    → 加载数据        │
│ Warpgroup 1  (thread 128..255) : Consumer 1  → WGMMA QK^T + PV│
│ Warpgroup 2  (thread 256..383) : Consumer 2  → WGMMA PV       │
└─────────────────────────────────────────────────────────────────┘

说明：
- "Warpgroup" = 4 个 warp = 128 线程
- SM90 WGMMA 指令要求整个 Warpgroup 协同执行
- 寄存器分配差异化：
    Producer: LoadRegisterRequirement  (约 24 个寄存器/线程)
    Consumer: MmaRegisterRequirement   (约 240 个寄存器/线程)
```

寄存器差异化通过 CUDA 内置 API 实现：

```cpp
if (warp_group_idx == 0) {  // Producer
    cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
    // 释放多余寄存器，为 Consumer 留出资源
} else {                    // Consumer
    cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
    // 申请更多寄存器用于存储 WGMMA 累加器
}
```

### 9.2 共享内存布局

```cpp
struct SharedStorage {
    // ── 主 tensor 区（约 196KB）──────────────────────────────
    union {
        // Mainloop 用
        struct {
            array<uint32_t, padding> padding_;
            TensorStorage mainloop; // smem_q, smem_k, smem_v, [smem_p]
        };
        // Epilogue 复用 smem_v 的区域存放 smem_o
        TensorStorage epilogue;  // smem_o
    } tensors;

    // ── Pipeline 同步区（若干字节）──────────────────────────
    struct {
        ClusterTransactionBarrier barrier_Q;    // Q 就绪信号
        ClusterTransactionBarrier barrier_Qv;   // Qv 就绪信号（MLA）
        ClusterBarrier            barrier_O;    // O 写回完成信号
        PipelineK::SharedStorage  pipeline_k;   // K 流水线信号量
        PipelineV::SharedStorage  pipeline_v;   // V 流水线信号量
        PipelineVt::SharedStorage pipeline_vt;  // Vt 转置流水线
        PipelineKVNew::SharedStorage pipeline_k_new; // AppendKV
        PipelineKVNew::SharedStorage pipeline_v_new;
        TileScheduler::SharedStorage smem_scheduler;
    } pipelines;
};
```

**关键优化：smem_o 与 smem_v 内存复用**

```
epilogue 的 smem_o 与 mainloop 的 smem_v 指向相同起始地址
→ 节省约 kBlockM × kHeadDimV × sizeof(Element) 的共享内存
→ Epilogue 写 O 时，Mainloop 的 V 已经读完，可以安全复用
```

### 9.3 Producer-Consumer 协作模式

```cpp
void operator()(Params& params, char* smem_buf) {
    // ── 9.3.1 初始化 ───────────────────────────────────────────────
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // 获取 lane predicate（判断当前线程是否有效）
    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();

    // 预取 TMA 描述符（仅 thread 0 执行）
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
        CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    // 获取 warpgroup 索引
    int warp_group_idx = cutlass::canonical_warp_group_idx();

    // 初始化所有 Pipeline 和 Barrier
    if (warp_idx == 0 && lane_predicate) {
        // Q 屏障：TMA 加载 Q 完成后通知 Consumer
        shared_storage.pipelines.barrier_Q.init(Use_TMA_Q ? 1 : NumProducerThreads);
        // Qv 屏障（MLA 专用）
        if constexpr (HasQv) {
            shared_storage.pipelines.barrier_Qv.init(Use_TMA_Q ? 1 : NumProducerThreads);
        }
        // O 屏障：Consumer 写回 O 完成后通知 Producer
        shared_storage.pipelines.barrier_O.init(
            size(ClusterShape{}) * (Use_TMA_O ? 1 : NumMmaThreads)
        );
    }

    // 等待 Cluster 内所有 CTA 初始化完成
    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }

    // ── 9.3.2 Producer 分支：TMA 异步加载 ─────────────────────────
    if (warp_group_idx == 0) {
        // 释放多余寄存器，为加载留出资源
        cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

        // 创建 Pipeline 状态
        PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipelineK>();
        PipelineState smem_pipe_write_new = cutlass::make_producer_start_state<MainloopPipelineKVNew>();
        int work_idx = 0;

        // 等待依赖 grid 完成（PDL 用）
        cutlass::arch::wait_on_dependent_grids();

        // 主加载循环
        for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler);
             work_tile_info.is_valid(params.scheduler);
             work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info)) {

            auto block_coord = work_tile_info.get_block_coord(params.scheduler);

            // 追加 KV（可选）
            if constexpr (AppendKV) {
                bool tile_new_valid = mainloop.load_kv_new(
                    params.mainloop, pipeline_k_new, pipeline_v_new,
                    smem_pipe_write_new, shared_storage, seqlen_info, block_coord, work_idx);
                if (tile_new_valid) {
                    cutlass::arch::NamedBarrier::sync(
                        NumMmaThreads + NumProducerThreads,
                        static_cast<uint32_t>(FwdNamedBarriers::AppendKV)
                    );
                }
            }

            // 预取下一个 tile 信息
            auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() {
                scheduler.prefetch_next_work(params.scheduler, work_tile_info);
            };

            // 核心加载函数
            mainloop.load(params.mainloop, pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write,
                          shared_storage, scheduler_prefetch, seqlen_info, block_coord, work_idx);
        }

        // 清空流水线
        mainloop.load_tail(pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write, shared_storage, work_idx);

    // ── 9.3.3 Consumer 分支：WGMMA 计算 ──────────────────────────
    } else {
        // 申请更多寄存器用于 MMA 累加器
        cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

        // 初始化 MMA 对象
        TiledMmaPV tiled_mma_pv;
        PipelineState smem_pipe_read;
        PipelineState smem_pipe_read_new;

        scheduler.init_consumer();
        mainloop.mma_init();

        int work_idx = 0;

        // 主计算循环
        for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
             work_tile_info.is_valid(params.scheduler);
             ) {
            auto block_coord = work_tile_info.get_block_coord(params.scheduler);

            // 处理 AppendKV（可选）
            if constexpr (AppendKV) {
                bool tile_new_valid = mainloop.store_kv_new(
                    params.mainloop, pipeline_k_new, pipeline_v_new, smem_pipe_read_new,
                    threadIdx.x - MmaThreadOffset, shared_storage, seqlen_info, block_coord);
                if (tile_new_valid) {
                    asm volatile ("fence.proxy.async.global;");
                    cutlass::arch::NamedBarrier::arrive(
                        NumMmaThreads + NumProducerThreads,
                        static_cast<uint32_t>(FwdNamedBarriers::AppendKV)
                    );
                }
            }

            // FP8 时合并 q/k descale 到 softmax_scale
            float softmax_scale_log2 = params.mainloop.softmax_scale_log2;
            if constexpr (Is_FP8 && !Has_softcap) {
                int const bidh = get<1>(block_coord);
                int const bidh_kv = !PackGQA ? params.mainloop.qhead_per_khead_divmod.divide(bidh) : bidh;
                float const q_descale = params.mainloop.ptr_q_descale == nullptr
                    ? 1.0f : params.mainloop.ptr_q_descale[bidb * get<0>(params.mainloop.stride_q_descale) + bidh_kv * get<1>(params.mainloop.stride_q_descale)];
                float const k_descale = params.mainloop.ptr_k_descale == nullptr
                    ? 1.0f : params.mainloop.ptr_k_descale[bidb * get<0>(params.mainloop.stride_k_descale) + bidh_kv * get<1>(params.mainloop.stride_k_descale)];
                softmax_scale_log2 *= q_descale * k_descale;
            }

            // 创建 Softmax 对象
            flash::Softmax<...> softmax(softmax_scale_log2);

            // 分配输出累加器
            Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_MNK_PV{}));

            // 执行 QK^T → softmax → PV
            bool tile_valid = mainloop.mma(
                params.mainloop, pipeline_k, pipeline_v, smem_pipe_read,
                tOrO, softmax, threadIdx.x - MmaThreadOffset, work_idx,
                seqlen_info, block_coord, shared_storage);

            // 获取下一个 tile（在 epilogue 前）
            work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info);

            // Split-KV + Varlen 时触发合并 kernel
            if constexpr (Split && Varlen) {
                if (!work_tile_info.is_valid(params.scheduler)) {
                    cutlass::arch::launch_dependent_grids();
                }
            }

            // 写回输出
            if (tile_valid) {
                epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma_pv,
                               threadIdx.x - MmaThreadOffset, block_coord);
            } else {
                epilogue.store_zero(params.epilogue, threadIdx.x - MmaThreadOffset, block_coord);
            }
        }
        epilogue.store_tail();
    }
}
```

**Producer 核心加载流程（mainloop.load 内部）**：

```cpp
// mainloop.load() 内部关键步骤：
void load(Params& params, ...) {
    // 1. 确定 KV 块范围
    auto [n_block_min, n_block_max] = BlockMN_t::get_n_block_min_max(...);

    // 2. 等待 Consumer 读完上一个 Q
    barrier_Q.sync(QueryEmpty);

    // 3. TMA 加载 Q（单次）
    if (elect_one_sync() && warp0) {
        barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
        copy(tma_load_Q.with(barrier_Q), gQ → sQ);
    }

    // 4. 等待上一个 tile 的 O 写回
    barrier_O.wait((work_idx + 1) % 2);

    // 5. 循环加载 K/V
    for (n_block = n_block_max-1; n_block >= n_block_min; --n_block) {
        pipeline_k.producer_acquire(smem_pipe_write);
        copy(tma_load_K.with(barrier, mcast_mask_kv), gK[n_block] → sK[stage]);

        if (IntraWGOverlap) {
            copy(tma_load_V, gV[n_block_prev] → sV[prev_stage]);  // 重叠
        } else {
            copy(tma_load_V, gV[n_block] → sV[same_stage]);
        }
    }

    // 6. 预取下一个 tile
    scheduler_prefetch();
}
```

**Consumer 核心计算流程（mainloop.mma 内部）**：

```cpp
// mainloop.mma() 内部关键步骤：
void mma(...) {
    // 1. 等待 Q 加载完成
    barrier_Q.wait(work_idx % 2);

    // 2. 主循环：对每个 n_block 执行
    for (n_block = n_block_max-1; n_block >= n_block_min; --n_block) {
        // GEMM-I: Q × K^T → S
        consumer_wait(pipeline_k, smem_pipe_read);
        flash::gemm<zero_init=true, wg_wait=-1>(tiled_mma_qk, tSrQ, sK[cur], tSrS);

        // GEMM-II: P × V → O
        consumer_wait(pipeline_v, smem_pipe_read_prev);
        flash::gemm<zero_init=false, wg_wait=-1>(tiled_mma_pv, tOrP_prev, sV[prev], tOrO);

        // 等待 GEMM-I 完成
        warpgroup_wait<1>();
        pipeline_k.consumer_release(smem_pipe_read);

        // Softmax 计算
        scoremod_premask_fn(tSrS);      // softcap
        mask.apply(tSrS, m_block, n_block);  // causal/local mask
        scores_scale = softmax.max_get_scale(tSrS);
        softmax.online_softmax(tSrS);
        convert_type_out(tSrS, tOrP);   // float32 → fp16/bf16

        // 等待 GEMM-II 完成
        warpgroup_wait<0>();
        pipeline_v.consumer_release(smem_pipe_read_prev);
        softmax.rescale_o(tOrO, scores_scale);  // O *= correction_factor
    }
}

---

## 10. Producer Warpgroup：TMA 异步加载

> 文件：[hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp](hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp)，`load()` 函数

### 10.1 TMA 概述

TMA（Tensor Memory Accelerator）是 SM90 的硬件单元，支持：
- 从 GMEM **批量**拷贝多维 Tensor 到 SMEM
- **异步**执行（不占用 CUDA core）
- 支持**多播**（Cluster 内 KV 广播）
- 通过 `ClusterTransactionBarrier` 同步

TMA 描述符在 `to_underlying_arguments()` 阶段在 CPU 端构建：

```cpp
TMA_Q tma_load_Q = make_tma_copy_A_sm90(SM90_TMA_LOAD{},
    make_tensor(ptr_Q, shape_Q, stride_Q), SmemLayoutQ{}, TileShape_MNK{}, ClusterShape{});
TMA_K tma_load_K = make_tma_copy_B_sm90(SM90_TMA_LOAD_MULTICAST{},
    make_tensor(ptr_K, shape_K, stride_K), SmemLayoutK_2d{}, TileShape_MNK{}, ClusterShape{});
TMA_V tma_load_V = make_tma_copy(SM90_TMA_LOAD_MULTICAST{},
    make_tensor(ptr_V, shape_V_transposed, stride_Vt), SmemLayoutVt_2d{}, ...);
```

### 10.2 TMA 流水线工作原理

V3 使用深度为 2 的 TMA 流水线（`kStages=2`）：

```
Stage 0                 Stage 1
┌────────────────────┐  ┌────────────────────┐
│ smem_k[0] smem_v[0]│  │ smem_k[1] smem_v[1]│
└────────────────────┘  └────────────────────┘
      ↑ TMA Write                ↑ TMA Write
      ↓ WGMMA Read               ↓ WGMMA Read

Producer 和 Consumer 通过 PipelineState（phase, index, count）交替使用两个 Stage：
- Producer 在 Stage i 完成写入 → signal producer_commit(smem_pipe_write)
- Consumer 等待 consumer_wait(smem_pipe_read) → 执行 WGMMA → consumer_release(smem_pipe_read)
```

### 10.3 load() 详细流程

```cpp
template <typename SchedulerPrefetch, typename SharedStorage>
CUTLASS_DEVICE void
load(Params const& params,
     MainloopPipelineK pipeline_k,
     MainloopPipelineV pipeline_v,
     MainloopPipelineVt pipeline_vt,
     PipelineState& smem_pipe_write,
     SharedStorage& shared_storage,
     SchedulerPrefetch& scheduler_prefetch,
     SeqlenInfo_t const& seqlen_info,
     cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
     int const work_idx) {

    // 1. 解析 block 坐标
    int const m_block = get<0>(block_coord);
    int const bidh = get<1>(block_coord);
    int const bidb = get<2>(block_coord);
    int const split_idx = get<3>(block_coord);

    // 2. 确定本 tile 需要处理的 KV 块范围 [n_block_min, n_block_max)
    auto [n_block_min, n_block_max] = BlockMN_t::get_n_block_min_max(
        seqlen_info, m_block, bidb, split_idx, params.num_splits,
        params.window_size_left, params.window_size_right,
        params.attention_chunk_divmod, params.qhead_per_khead_divmod);

    // 空 tile，直接跳过
    if constexpr (Is_causal || Is_local || Varlen || Split) {
        if (n_block_max <= n_block_min) { return; }
    }

    // 3. 构建 TMA 分区视图
    // gQ: [kBlockM, kHeadDim]
    // gK_TMA: [num_heads, seqlen_k, kHeadDim] (通过 n_block 索引)
    // gVt_TMA: [kHeadDim, seqlen_k] (转置视图)

    // 4. 创建 K/V 加载 lambda
    auto load_K = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
        pipeline_k.producer_acquire(smem_pipe_write);
        if constexpr (!PagedKVNonTMA) {
            auto [n_block_idx, bidb_kv_idx] = paged_kv_manager.get_indices_for_K_TMA();
            copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write),
                     mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                 tKgK_TMA(_, n_block_idx, bidb_kv_idx), tKsK_TMA(_, smem_pipe_write.index()));
        } else {
            // cp.async 路径
            paged_kv_manager.template load_K<Seqlenk_mask>(n_block, sK_pi(_, _, smem_pipe_write.index()));
            pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
        }
    };

    auto load_V = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
        auto pipeline_v_load = cute::conditional_return<!Transpose_V>(pipeline_v, pipeline_vt);
        pipeline_v_load.producer_acquire(smem_pipe_write);
        if constexpr (!PagedKVNonTMA) {
            auto [n_block_idx, bidb_kv_idx] = paged_kv_manager.get_indices_for_V_TMA();
            copy(params.tma_load_V.with(*pipeline_v_load.producer_get_barrier(smem_pipe_write),
                     mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                 tVgVt_TMA(_, n_block_idx, bidb_kv_idx), tVsVt_TMA(_, smem_pipe_write.index()));
        } else {
            paged_kv_manager.template load_V<Seqlenk_mask>(n_block, sVcpasync(_, _, smem_pipe_write.index()));
            pipeline_v_load.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
        }
    };

    // 5. 计算多播 mask（Cluster 特性）
    uint16_t mcast_mask_kv = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<ClusterShape>{};
        for (int m = 0; m < size<0>(block_layout); ++m) {
            mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
        }
    }

    // 6. 首轮 K 加载
    int n_block = n_block_max - 1;
    bool should_load_KV = !Use_TMA_KV || ((SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync());

    if (should_load_KV) {
        if constexpr (PagedKVNonTMA) {
            paged_kv_manager.template load_page_table<true, true>(n_block);
        } else {
            paged_kv_manager.template load_page_table_TMA<true>(n_block);
        }
        if constexpr (Transpose_V) { load_V(n_block, smem_pipe_write, cute::true_type{}); }
        load_K(n_block, smem_pipe_write, cute::true_type{});
    }

    // 7. TMA 加载 Q
    if constexpr (Use_TMA_Q) {
        // 等待 Consumer 读完上一个 Q
        cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK + cutlass::NumThreadsPerWarp,
            static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty));

        // 发起 Q 的 TMA 加载
        if ((SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync()) {
            shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
            copy(params.tma_load_Q.with(...), tQgQ, tQsQ);
            // MLA 专用 Qv
            if constexpr (HasQv) {
                shared_storage.pipelines.barrier_Qv.arrive_and_expect_tx(TmaTransactionBytesQv);
                copy(params.tma_load_Qv.with(...), tQvgQv, tQvsQv);
            }
        }
    } else {
        // cp.async 加载 Q（PackGQA 路径）
        cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK + NumProducerThreads,
            static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty));
        PackGQAt::load_Q(mQ, sQ_pi, params.qhead_per_khead_divmod, thread_idx, seqlen_info.seqlen_q, m_block);
        cutlass::arch::cpasync_barrier_arrive(reinterpret_cast<uint64_t*>(&barrier_Q));
    }

    // 8. 等待上一个 tile 的 O 写回完成
    shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);

    // 9. 循环加载 K/V
    if constexpr (!Transpose_V && !IntraWGOverlap) {
        if (should_load_KV) { load_V(n_block, smem_pipe_write, cute::true_type{}); }
    }
    int n_block_prev = n_block;
    --n_block;

    #pragma unroll (!Transpose_V && Use_TMA_KV ? 2 : 1)
    for (; n_block >= n_block_min; --n_block) {
        PipelineState smem_pipe_write_v = smem_pipe_write;
        ++smem_pipe_write;
        if (should_load_KV) {
            if constexpr (PagedKVNonTMA) {
                paged_kv_manager.template load_page_table<false>(n_block);
            } else {
                paged_kv_manager.load_page_table_TMA(n_block);
            }
            if constexpr (Transpose_V) { load_V(n_block, smem_pipe_write, cute::false_type{}); }
            load_K(n_block, smem_pipe_write, cute::false_type{});
            if constexpr (!Transpose_V) {
                if constexpr (IntraWGOverlap) {
                    load_V(n_block_prev, smem_pipe_write_v, cute::true_type{});
                } else {
                    load_V(n_block, smem_pipe_write, cute::false_type{});
                }
            }
        }
        n_block_prev = n_block;
        if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write_v); }
    }

    // 10. 预取调度器下一个 tile 信息
    scheduler_prefetch();

    if constexpr (!Transpose_V && IntraWGOverlap) {
        if (should_load_KV) { load_V(n_block_prev, smem_pipe_write, cute::true_type{}); }
    }
    if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write); }
    ++smem_pipe_write;
    ++work_idx;
}
```

**load_tail() 清空流水线**：

```cpp
template <typename SharedStorage>
CUTLASS_DEVICE void
load_tail(MainloopPipelineK pipeline_k, MainloopPipelineV pipeline_v,
          MainloopPipelineVt pipeline_vt, PipelineState& smem_pipe_write,
          SharedStorage &shared_storage, int const work_idx) {
    // 等待最后一个 O 写回
    shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);

    // 发出 pipeline 收尾指令
    if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {
        pipeline_k.producer_tail(smem_pipe_write);
        pipeline_v.producer_tail(smem_pipe_write);
        if constexpr (Transpose_V) { pipeline_vt.producer_tail(smem_pipe_write); }
    }
}
```

**K/V TMA 多播（Cluster 特性）：**

当 `ClusterM=2` 时，Cluster 内有 2 个 CTA 共享同一份 K/V：

```cpp
// Producer CTA 0 发送 K/V，CTA 1 接收（via SM90_TMA_LOAD_MULTICAST）
uint16_t mcast_mask_kv = (1 << CTA_0) | (1 << CTA_1);
copy(tma_load_K.with(barrier, mcast_mask_kv, CacheHint::EVICT_LAST),
     gK[n_block] → sK[stage]);
// 两个 CTA 的 smem_k 同时被填充，减少 GMEM 带宽一半
```

### 10.4 PackGQA 下的 Q 加载

当 `PackGQA=true` 时，Q 不使用 TMA 而改用 `cp.async`（因为 Q 的内存布局需要重整）：

```cpp
// Q 从 [B, seqlen_q, H, D] 重整为 [(H/Hkv, seqlen_q), D, Hkv, B]
// 使 每个 tile 处理连续的 (qhead_per_khead × seqlen_q_block) 个 Q token
PackGQAManager::load_Q(mQ_packed, sQ_pi, qhead_per_khead_divmod, ...);
cpasync_barrier_arrive(&barrier_Q);
```

---

## 11. Consumer Warpgroup：WGMMA 矩阵计算

> 文件：[hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp](hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp)，`mma()` 函数

### 11.1 WGMMA 与 MMA 的区别

| 特性 | SM80 MMA | SM90 WGMMA |
|------|----------|------------|
| 执行单位 | 1 warp (32 线程) | 1 warpgroup (128 线程) |
| 输入来源 | A: 寄存器/共享内存，B: 共享内存 | A: 共享内存，B: 共享内存 |
| 同步方式 | `__syncwarp()` | `warpgroup_wait<N>()` |
| 矩阵大小 | 16×8×16 | 64×8×16 (或更大) |
| 异步性 | 同步 | **异步**（可与下一条指令并发） |

### 11.2 两种 WGMMA 操作

计算过程涉及两个矩阵乘法：

**GEMM-I：Q × K^T → S（注意力分数）**
```
TiledMmaQK: SS-mode（Q 和 K 都在 SMEM）
TileShape_MNK = (kBlockM, kBlockN, kHeadDim)

S[kBlockM, kBlockN] = Q[kBlockM, kHeadDim] × K^T[kHeadDim, kBlockN]
```

**GEMM-II：P × V → O（注意力输出）**
```
TiledMmaPV: RS-mode 或 SS-mode（P 在寄存器或 SMEM）
TileShape_MNK_PV = (kBlockM, kHeadDimV, kBlockN)

O[kBlockM, kHeadDimV] += P[kBlockM, kBlockN] × V^T[kBlockN, kHeadDimV]
  （注意 V 在 SMEM 中是转置存储的）
```

### 11.3 IntraWG Overlap（Warpgroup 内 GEMM 重叠）

当 `IntraWGOverlap=true` 时，实现 GEMM-I 和 GEMM-II 的流水线重叠：

```
时间轴（概念）：
┌──────────────────────────────────────────────────────────────────────────┐
│ Iter n   │ GEMM-I(n)  │ Softmax(n) │                    │              │
│ Iter n-1 │            │            │ GEMM-II(n-1)        │ Rescale O    │
└──────────────────────────────────────────────────────────────────────────┘
     ↑
     GEMM-I 先行一步，不等 GEMM-II 结束
```

**完整实现代码**：

```cpp
// IntraWGOverlap=true 时的首次迭代（单独处理）
if (IntraWGOverlap) {
    Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));

    // 1. 等待 K 就绪 → 启动 GEMM-I
    consumer_wait(pipeline_k, smem_pipe_read);
    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(
        tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);

    // 2. 等待 V 就绪 → 启动 GEMM-II（用上一轮的 P）
    warpgroup_wait<0>();  // 等待 GEMM-I 完成
    pipeline_k.consumer_release(smem_pipe_read);

    if constexpr (HasQv) {
        shared_storage.pipelines.barrier_Qv.wait(work_idx % 2);
        consumer_wait(pipeline_v, smem_pipe_read);
        flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(
            tiled_mma_qv, tSrQv, tSrV(_, _, _, smem_pipe_read.index()), tSrS);
    }

    // 3. Softmax 处理
    scoremod_premask_fn(tSrS);
    mask.template apply<true, Is_causal, Is_local>(tSrS, m_block, n_block);

    Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
    softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);

    // 4. 类型转换
    if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
    Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
    Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
    convert_type_out(tOrP_acc, tOrP);
    if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }

    // 5. 写 P 到共享内存（如果使用 RS 模式则跳过）
    if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
    if constexpr (!MmaPV_is_RS) { arrive_on_P_write_barrier(); }

    --n_block;
    clear(tOrO);  // 初始化 O 累加器

    // 6. 循环迭代（fwd_step lambda）
    auto fwd_step = [&](int const n_block, auto mask_fn, auto check_inf_type) {
        static constexpr bool Check_inf = decltype(check_inf_type)::value;

        // 创建新的 pipeline 状态
        PipelineState smem_pipe_read_v(smem_pipe_read.index(), smem_pipe_read.phase(), smem_pipe_read.count());
        ++smem_pipe_read;

        Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));

        // 等待 K 就绪 → GEMM-I
        if (!UseSchedulerBarrier || warp_group_idx == 0) {
            consumer_wait(pipeline_k, smem_pipe_read);
        }
        warp_scheduler_barrier_sync();  // Warpgroup 同步
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(
            tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);

        // Rescale O（如果需要）
        if constexpr (RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }

        // 等待 V 就绪 → GEMM-II
        if constexpr(!HasQv) {
            if (!UseSchedulerBarrier || warp_group_idx == 0) {
                consumer_wait(pipeline_v, smem_pipe_read_v);
            }
        }
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
            tiled_mma_pv,
            cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP),
            tOrV(_, _, _, smem_pipe_read_v.index()),
            tOrO);

        warp_scheduler_barrier_arrive();  // 通知下一个 warpgroup
        warpgroup_wait<1>();  // 等待 GEMM-I 完成
        pipeline_k.consumer_release(smem_pipe_read);

        if constexpr (HasQv) {
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read_v);
            consumer_wait(pipeline_v, smem_pipe_read);
            flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(
                tiled_mma_qv, tSrQv, tSrV(_, _, _, smem_pipe_read.index()), tSrS);
        }

        // Softmax 计算
        scoremod_premask_fn(tSrS);
        mask_fn(tSrS, n_block);
        cute::copy(softmax.template max_get_scale</*Is_first=*/false, Check_inf>(tSrS), scores_scale);
        if constexpr (LargeHeadDimV) { store_scales(scores_scale, smem_pipe_read_v.index()); }
        softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);

        if constexpr (!HasQv) {
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read_v);
        }

        // 类型转换
        if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
        convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);
        if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }

        if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
        if constexpr (!RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
        if constexpr (!MmaPV_is_RS) { arrive_on_P_write_barrier(); }
    };

    // 分离不同类型的 mask 迭代
    if constexpr (Is_causal || Is_local) {
        auto mask_fn = [&](auto& tSrS, int n_block) {
            mask.template apply<false, Is_causal, Is_local>(tSrS, m_block, n_block);
        };
        int const n_block_min_causal_local_mask = BlockMN_t::get_n_block_min_causal_local_mask(...);
        #pragma unroll 1
        for (; n_block >= n_block_min_causal_local_mask; --n_block) {
            fwd_step(n_block, mask_fn, cute::true_type{});
        }
    }

    // 无 mask 或 local mask 的迭代...
}
```

**无 IntraWGOverlap 的简化版本**：

```cpp
// 无 IntraWGOverlap 时，每个迭代是独立的
auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
    static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
    static constexpr bool Check_inf = decltype(check_inf_type)::value;

    auto smem_pipe_read_prev = smem_pipe_read;
    if constexpr (!Is_first_iter) { ++smem_pipe_read; }

    Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));

    // 等待 K → GEMM-I
    consumer_wait(pipeline_k, smem_pipe_read);
    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(
        tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);

    // 等待 GEMM-I 完成
    warpgroup_wait<0>();
    pipeline_k.consumer_release(smem_pipe_read);

    // 等待 V → GEMM-II
    consumer_wait(pipeline_v, smem_pipe_read);
    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
        tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP),
        tOrV(_, _, _, smem_pipe_read.index()), tOrO);

    warpgroup_wait<0>();
    pipeline_v.consumer_release(smem_pipe_read);

    // Softmax
    scoremod_premask_fn(tSrS);
    mask_fn(tSrS, n_block);
    Tensor scores_scale = softmax.template max_get_scale<Is_first_iter, Check_inf>(tSrS);
    softmax.template online_softmax<Is_first_iter, Check_inf>(tSrS);

    // Rescale O
    if constexpr (!Is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
};
```

### 11.4 FP8 特殊处理

当 `is_e4m3=true` 时：

1. **descale 合并到 softmax scale**：
   ```cpp
   softmax_scale_log2 *= q_descale * k_descale;
   // v_descale 在最后的 finalize 中处理
   ```

2. **V 的 colmajor/rowmajor 问题**：
   - FP8 WGMMA 要求 V 是列主序（K-major = row-major in the K dimension）
   - 若 V 是行主序（rowmajor），Producer 需要用 LDSM.T + 字节排列 + STSM 进行转置：
   ```cpp
   // Producer WG 执行 V 的原地转置
   void transpose_V(int stage) {
       // 从 smem_vt（行主序）读取，写到 smem_v（列主序）
       s2r_tiled_copy_vt.copy(smem_vt → rV_tmp);     // LDSM.T
       __byte_perm(rV_tmp, 0x6420, 0x7531);            // 字节重排（FP8 特有）
       r2s_tiled_copy_v.copy(rV_tmp → smem_v);        // STSM
   }
   ```

3. **Permute 输出**：FP8 的 WGMMA 输出布局与 fp16 不同，需要额外的寄存器排列：
   ```cpp
   flash::permute_Cregs_fp8(tSrS);    // 调整 S 的寄存器布局
   flash::permute_Aregs_fp8(tOrP);    // 调整 P 的寄存器布局（colmajor V）
   ```

---

## 12. 在线 Softmax 与注意力分数计算

> 文件：[hopper/softmax.h](hopper/softmax.h)

### 12.1 在线算法回顾

Flash Attention 使用在线 softmax，每次只看一块 K（块大小 = kBlockN），不存储完整的 S 矩阵：

$$m^{(i)} = \max(m^{(i-1)},\ \max_j S^{(i)}_{:,j})$$
$$\ell^{(i)} = e^{m^{(i-1)} - m^{(i)}} \cdot \ell^{(i-1)} + \sum_j e^{S^{(i)}_{:,j} - m^{(i)}}$$
$$O^{(i)} = e^{m^{(i-1)} - m^{(i)}} \cdot O^{(i-1)} + e^{S^{(i)} - m^{(i)}} \cdot V^{(i)}$$

V3 中使用 `exp2` 而非 `exp`（因为 CUDA 有原生 `exp2` 指令，更快）：

$$\text{softmax\_scale\_log2} = \text{scale} \times \log_2(e)$$
$$S_{\text{scaled}} = S \times \text{softmax\_scale\_log2}$$
$$P = \exp_2(S_{\text{scaled}} - m_{\text{log2}})$$

### 12.2 Softcap 处理

当 `Has_softcap=true` 时，先对分数应用 tanh 截断：

$$S' = \text{softcap} \times \tanh\left(\frac{S \times \text{scale}}{\text{softcap}}\right)$$

在 V3 中，提前合并常数以减少指令数：
```cpp
// CPU 端预计算
params.softmax_scale_log2 = softcap_val * log2(e);    // 用于 tanh 之后的 exp2
params.softcap_val         = scale / softcap_val;       // 用于 tanh 之前的缩放
```

### 12.3 Mask 类型

```
1. Seqlenk_mask:  屏蔽超出序列长度的 token（Varlen/Paged 边界）
2. Causal mask:   S[i,j] 置 -inf，若 j > i（token i 不能看 j）
3. Local mask:    滑动窗口，j < i - window_left 或 j > i + window_right 时置 -inf
4. Chunk mask:    分块 attention，j < (i/chunk)*chunk 时置 -inf

优化：掩码 iteration 与无掩码 iteration 分开循环（避免每次都判断）：
  - n_block_max-1 → n_block_min_causal_local_mask: 带掩码迭代
  - n_block_min_causal_local_mask → n_block_min_before_local_mask: 无掩码
  - n_block_min_before_local_mask → n_block_min: 仅 local 左侧掩码
```

---

## 13. Epilogue：输出归一化与写回

> 文件：[hopper/epilogue_fwd.hpp](hopper/epilogue_fwd.hpp)

### 13.1 流程概述

```
Consumer WG 完成最后一轮 GEMM-II → 持有累加器 tOrO（未归一化）

Epilogue.store():
  1. 类型转换（可选 FP8 permute）
     tOrO_out = convert_fp32_to_fp16/bf16(tOrO)

  2. 同步所有 epilogue 线程
     named_barrier_sync(EpilogueBarrier)

  3a. 若使用 TMA（!Split && !Varlen && !PackGQA）：
       rmem → smem（STSM）→ TMA 写 gmem
  3b. 否则：
       rmem → smem → 向量化存储到 gmem（128-bit store）

  4. 写 LSE = m + log(sum_exp) 到 gmem
     对每个 Q token（由 MMA 布局计算 row index）：
         LSE[bidb, bidh, seqlen_q_offset + row] = softmax.row_sum[row]
```

**详细实现代码**：

```cpp
template <typename SharedStorage, typename FrgTensorO, typename FrgTensorLSE, typename TiledMma>
CUTLASS_DEVICE void
store(Params const& params,
      FrgTensorO& tOrO,
      FrgTensorLSE const& lse,
      SharedStorage& shared_storage,
      TiledMma tiled_mma,
      int thread_idx,
      cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord) {

    auto [m_block, bidh, bidb, split_idx] = block_coord;

    // 1. FP8 输出 permute（可选）
    static constexpr bool NeedFP8Permute = FP8PermuteCol && (sizeof(Element) == 2 || sizeof(Element) == 4);
    if constexpr (NeedFP8Permute && Split) {
        flash::permute_output_fp8_Vcolmajor(tOrO);
    }

    // 2. 类型转换：float32 → fp16/bf16
    Tensor tOrO_out = make_tensor_like<Element>(tOrO);
    flash::convert_type_out(tOrO, tOrO_out);
    if constexpr (NeedFP8Permute && !Split) {
        flash::permute_output_fp8_Vcolmajor(tOrO_out);
    }

    // 3. 同步所有 epilogue 线程
    flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    // 4. Step 1: rmem → smem
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_o.data()), SmemLayoutO{});

    if constexpr (Use_smem) {
        auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

        if constexpr (Use_TMA_O) {
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        } else {
            flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        }
    }

    // 5. Step 2: 写 LSE 到 gmem
    flash::SeqlenInfo<Varlen, kBlockM> seqlen_info{bidb, size<0>(params.shape_O), params.cu_seqlens, params.seqused};
    int offset_o = seqlen_info.offset;
    int seqlen_o = seqlen_info.seqlen;

    Tensor mLSE = make_tensor(make_gmem_ptr(ptr_LSE + offset_o * stride_LSE),
                              shape_LSE_packed, stride_LSE_packed)(_, bidh, bidb, 0);

    if (!LargeHeadDimV || warp_group_idx == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            int const row = m_block * kBlockM + get<0>(taccOcO_row(mi));
            if (get<1>(taccOcO_row(_0{})) == 0 && row < seqlen_o) {
                mLSE(row) = lse(mi);
            }
        }
    }

    // 6. Step 3: smem → gmem（O 输出）
    if constexpr (Use_TMA_O) {
        // TMA 路径：单 CTA 写回
        Tensor mO = params.tma_store_O.get_tma_tensor(params.shape_O)(_, _, bidh, bidb, split_idx);
        Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
        auto block_tma_O = params.tma_store_O.get_slice(_0{});
        Tensor tOgO = block_tma_O.partition_D(gO);
        Tensor tOsO = block_tma_O.partition_S(sO);

        if (warp_idx_sync == NumEpilogueThreads / 32 - 1) {
            cutlass::arch::NamedBarrier::sync(...);
            if (cute::elect_one_sync()) {
                cute::copy(params.tma_store_O, tOsO, tOgO);
                tma_store_arrive();
                tma_store_wait<0>();
            }
        }
    } else {
        // 向量化存储路径
        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset_o * stride_O),
                                params.shape_O_packed, params.stride_O_packed)(_, _, bidh, bidb, _0{});
        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
        Tensor tOrO = make_fragment_like(tOsO);
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

        // 写入 gmem（处理边界情况）
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false,
                   /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM
        );
    }
}
```

### 13.2 Split-KV 时写入 Partial 结果

当 `Split=true` 时，epilogue 写的是部分结果而非最终结果：

```
out_accum[split_idx, bidb, bidh, seqlen_q, dv]  ← float32 累加器（未归一化）
lse_accum[split_idx, bidb, bidh, seqlen_q]       ← 当前 split 的 log-sum-exp
```

### 13.3 smem_v / smem_o 复用

```
Mainloop     Epilogue
smem_v ──────────────────── smem_o

时序保证：
  Consumer 在最后一次 GEMM-II 完成后（pipeline_v.consumer_release），
  才进入 Epilogue，此时 smem_v 内容已被读取完毕，
  可以安全地将 tOrO 写入该内存区域。
```

---

## 14. Split-KV 合并 Kernel

> 文件：[hopper/flash_fwd_combine_kernel.h](hopper/flash_fwd_combine_kernel.h)，由 `run_mha_fwd_combine()` 启动

当 `num_splits > 1` 时，主 kernel 生成 `num_splits` 份部分结果，最后需要合并：

### 14.1 合并公式

对每个 Q token，有 $S$ 个分块的结果 $(O_i, L_i)$，其中 $L_i = m_i + \log(\sum_j \exp(s_{ij} - m_i))$：

$$L^* = \log\sum_{i=1}^{S} e^{L_i}$$

$$O^* = \sum_{i=1}^{S} e^{L_i - L^*} \cdot O_i$$

**这等价于对所有分块执行一次 online softmax 的 rescale 步骤。**

### 14.2 PDL 启动

当 `enable_pdl=true` 时，合并 kernel 通过 PDL 启动：

```cpp
// 主 kernel 结束前（最后一个 tile）：
cutlass::arch::launch_dependent_grids();
// 这会触发合并 kernel 开始，而无需显式 cudaStreamWaitEvent
// → 减少 CPU 介入，降低延迟
```

**合并 kernel 详细实现**：

```cpp
CUTLASS_DEVICE void
operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // 解析 block 坐标
    int const m_block = blockIdx.x;   // Q tile 索引
    int const k_block = blockIdx.y;   // head 索引
    int const batch = params.varlen_batch_idx_ptr
        ? params.varlen_batch_idx_ptr[blockIdx.z] : blockIdx.z;

    // 获取当前 batch 的 split 数量
    int const num_splits = params.num_splits_dynamic_ptr
        ? params.num_splits_dynamic_ptr[batch]
        : get<1>(params.shape_LSE_partial);

    if (num_splits <= 1) { return; }

    // 共享内存张量
    Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.smem_lse_partial.data()), SmemLayoutLSE{});
    Tensor sMaxValidSplit = make_tensor(make_smem_ptr(shared_storage.smem_max_valid_split.data()), Shape<Int<kBlockM>>{});
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o_partial.data()), SmemLayoutO{});

    // ── Step 1: load LSE_partial from gmem → smem ──
    Tensor mLSEpartial = make_tensor(make_gmem_ptr(params.ptr_LSE_partial + offset * stride_LSE_partial),
                                    select<1, 0, 2, 3>(params.shape_LSE_partial),
                                    select<1, 0, 2, 3>(params.stride_LSE_partial))(_, _, _, !Varlen ? batch : 0);

    GmemTiledCopyLSE gmem_tiled_copy_LSE;
    auto gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_thread_slice(thread_idx);
    Tensor tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE);

    // 加载 LSE 数据
    #pragma unroll
    for (int m = 0; m < size<2>(tLSEcLSE); ++m) {
        int mi = int(get<1>(tLSEcLSE(_0{}, _0{}, m)));
        int idx = m_block * kBlockM + mi;
        if (idx < max_idx) {
            // 计算 head 索引
            int m_idx, bidh;
            seqlen_divmod_dynamic.divmod(m_idx, idx);
            Tensor mLSEpartial_cur = mLSEpartial_copy(_, _, m_idx, bidh);

            #pragma unroll
            for (int s = 0; s < size<1>(tLSEcLSE); ++s) {
                int si = get<0>(tLSEcLSE(_0{}, s, _0{}));
                if (si < num_splits) {
                    cute::copy(gmem_tiled_copy_LSE, mLSEpartial_cur(_, si), tLSEsLSE(_, s, m));
                } else {
                    cute::fill(tLSEsLSE(_, s, m), -INFINITY);  // 填充 -inf
                }
            }
        }
    }

    // ── Step 2: Load O_partial from gmem → smem ──
    // 异步加载，与 LSE 计算重叠
    GmemTiledCopyAccum gmem_tiled_copy_O_partial;
    auto gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_thread_slice(thread_idx);
    Tensor mOpartial = make_tensor(make_gmem_ptr(params.ptr_O_partial + offset * stride_O_partial),
                                   params.shape_O_partial, params.stride_O_partial)(_, _, _, _, !Varlen ? batch : 0);

    // ── Step 3: Online Softmax 合并 ──
    // 计算全局 max
    float maxLSE = -INFINITY;
    #pragma unroll
    for (int s = 0; s < num_splits; ++s) {
        maxLSE = fmaxf(maxLSE, sLSE(s, 0));  // 取每 split 的 max
    }

    // 写入 sMaxValidSplit
    #pragma unroll
    for (int mi = 0; mi < kBlockM; ++mi) {
        sMaxValidSplit(mi) = maxLSE;
    }

    // 计算 exp(L_i - L*)
    float sumExp = 0;
    Tensor tOrO_acc = make_fragment_like(...);  // float32 累加器

    #pragma unroll
    for (int s = 0; s < num_splits; ++s) {
        float exp_diff = exp2f(sLSE(s, 0) - maxLSE);  // exp2(L_i - L*)
        sumExp += exp_diff;

        // 加载 O_partial[s] 并累加
        cute::copy(gmem_tiled_copy_O_partial, ...);
        // O_acc += exp_diff * O_partial[s]
    }

    // ── Step 4: 归一化并写回 ──
    float logSumExp = maxLSE + log2f(sumExp);  // L* = log(sum(exp(L_i)))

    // 归一化：O* = O_acc / sumExp
    #pragma unroll
    for (int i = 0; i < size(tOrO_acc); ++i) {
        tOrO_acc(i) /= sumExp;
    }

    // 写回最终结果
    Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset * stride_O), ...);
    cute::copy(tOrO_acc, mO);

    // 写回最终 LSE
    params.ptr_LSE[batch, bidh, offset + mi] = logSumExp;
}

---

## 15. 关键优化技术汇总

### 15.1 TMA（Tensor Memory Accelerator）

**本质**：SM90 专用的硬件 DMA 引擎，绕过 CUDA core 进行数据搬运。

```
传统 cp.async（V2）：
  每个线程发出自己的内存请求 → 多线程协调 → 较多同步开销

TMA（V3）：
  仅需 1 个线程发出 TMA 指令 → 硬件完成整块 Tensor 传输
  → 完全不占用 CUDA core 的执行带宽
  → 支持多维 Tensor（非连续维度也可以 TMA）
```

| 指标 | cp.async | TMA |
|------|----------|-----|
| 发起线程数 | NumMmaThreads | 1 |
| 非连续维度 | 需要手动计算偏移 | 硬件自动处理 |
| 多播 | ❌ | ✅（ClusterM=2 时 K/V 只传一次）|
| 同步原语 | `cpasync_barrier_arrive` | `ClusterTransactionBarrier` |

### 15.2 WGMMA（Warpgroup MMA）

```
SM80 mma.sync：
  每个 warp 独立计算 16×8×16 矩阵块
  → warp 之间结果需要归约（shuffle）

SM90 wgmma：
  128 线程协同计算 (kBlockM×64) × kHeadDim 矩阵
  → 消除 warp 间 shuffle，提升吞吐量约 2×
  → 支持直接从 SMEM 读取（SS mode），减少寄存器压力
```

### 15.3 Warpgroup 专用化（Producer-Consumer 分工）

```
传统流程（V2）：所有 warp 交替做加载和计算

V3 专用化：
  WG0 (Producer)  ──── 专注 TMA 加载，拥有很少寄存器（24）
  WG1/2 (Consumer) ─── 专注 WGMMA，拥有大量寄存器（240）
  
好处：
  1. 加载和计算真正并行（不同 warpgroup 在不同 CUDA 功能单元）
  2. Consumer 有更多寄存器，可缓存更多累加器（O 和 softmax 状态）
  3. Consumer 完成 epilogue 时，Producer 已在加载下一个 tile 的数据
```

### 15.4 PackGQA

```
传统 GQA（V2）：
  Q heads: H，KV heads: Hkv，qhead_per_khead = H/Hkv
  每个 KV-head 对应多个 Q-head，每个 tile 只处理 1 个 Q-head

PackGQA（V3）：
  将 Q 重排为 [(qhead_per_khead, seqlen_q), D, Hkv, B]
  每个 tile 同时处理 qhead_per_khead 个 Q-head（针对同一个 KV-head）
  
好处：
  1. 对于短序列，增加每个 tile 的工作量，提升 SM 利用率
  2. Varlen 场景下避免大量空 tile
  3. Split-KV 时必须开启（共享同一份 K/V tile）
```

### 15.5 IntraWGOverlap（Warpgroup 内 GEMM 重叠）

```
关键思想：WGMMA 是异步的，可以同时执行两个 GEMM

优化前：
  GEMM-I(n) → Softmax(n) → GEMM-II(n) → rescale O → GEMM-I(n+1) ...

优化后（IntraWGOverlap=true）：
  GEMM-I(n)  ─────────────────────────────────────
  GEMM-II(n-1)  ──────────────
  Softmax(n)         ──────────────
  rescale O(n)                           ──────────

即：GEMM-I(n) 与 GEMM-II(n-1) 并发执行
条件：kHeadDim <= 128 且有足够寄存器
```

### 15.6 V 列主序（FP8 优化）

```
FP8 WGMMA 要求 B 矩阵（V）是列主序存储：

传统（fp16, 行主序）：V[seqlen_k, headdim_v]
FP8 优化（列主序）：V[headdim_v, seqlen_k]

好处：
  - WGMMA 可以直接从 SMEM 以列主序读取 V
  - 节省 STSM/LDSM 转置开销

代价：
  - 在线转置（Producer 执行 LDSM.T + __byte_perm + STSM）
  - 约增加 5-10% 的 Producer 负载
```

### 15.7 L2 感知调度（Head Swizzle）

```
问题：因果 attention 中，不同 tile 的计算量差异大（靠近对角线重）

解决：DynamicPersistentTileScheduler 的 L2 swizzle：

1. 计算 L2 section 大小：swizzle = floor(32MB / size_one_kv_head)
   （确保 K/V 数据留在 L2 中被复用）

2. 在同一 section 内，按 LPT 调度（最长处理时间优先）
   → 较长的序列块先被计算
   → SM 空闲时间减少（负载均衡）

3. Section 切换时，L2 中的旧 K/V 被新数据替换
   → 减少 L2 miss
```

---

## 16. 完整执行时序图

### 16.1 单 SM 上的执行时序（SM90，无 Split）

```
Time →

WG0 (Producer):
│  TMA Prefetch  │ barrier_O.wait │ TMA_K[n] │ TMA_V[n]│ TMA_K[n-1]│ TMA_V[n-1]│ ... │ load_tail │
                                               pipeline_k.commit  pipeline_v.commit

WG1 (Consumer):
│ mma_init │ barrier_Q.wait │ GEMM-I[n] │ Softmax[n] │ GEMM-II[n] │ rescale O │ GEMM-I[n-1] │ ... │ Epilogue │
                                                         pipeline_k.release   pipeline_v.release

同步点：
  barrier_Q    ← Producer TMA 完 Q 后 arrive，Consumer 等待
  pipeline_k   ← Producer TMA K 完成 commit；Consumer wait/release
  pipeline_v   ← Producer TMA V 完成 commit；Consumer wait/release
  barrier_O    ← Consumer Epilogue TMA 完 O 后 arrive；Producer 等待（用于下一 tile）
```

### 16.2 多 SM 的 Pipeline 时序（持久化调度）

```
SM 0:  [Tile 0] → [Tile num_sm] → [Tile 2*num_sm] → ...
SM 1:  [Tile 1] → [Tile num_sm+1] → ...
...
SM n:  [Tile n] → [Tile num_sm+n] → ...

每个 SM 上（WG0 Producer / WG1 Consumer）：
┌─────────────────────────────────────────────────────────────────────────┐
│ Producer │  Load(Tile_i+1) 预加载  │  Load(Tile_i+2)  │ ...            │
│ Consumer │ Compute(Tile_i) │ Epilogue(Tile_i) │ Compute(Tile_i+1) │... │
└─────────────────────────────────────────────────────────────────────────┘
                    ↑ 真正的 Overlap：下一个 tile 的数据加载与当前 tile 的计算重叠
```

### 16.3 Split-KV 完整流程

```
输入: Q[B, seqlen_q, H, D], K[B, seqlen_k, Hkv, D], V[B, seqlen_k, Hkv, Dv]

Step 1: 主 kernel（num_splits 个并行 KV 分片）

  Split 0 处理 K[0 : seqlen_k/S, :]         → out_accum[0], lse_accum[0]
  Split 1 处理 K[seqlen_k/S : 2*seqlen_k/S, :] → out_accum[1], lse_accum[1]
  ...
  Split S-1 处理 K[(S-1)*seqlen_k/S :, :]   → out_accum[S-1], lse_accum[S-1]

Step 2: Combine kernel

  for each (batch, head, seqlen_q position):
    L* = logsumexp(lse_accum[0..S-1])
    O* = sum_i( exp(lse_accum[i] - L*) * out_accum[i] )
    
  → 写入最终 out[B, seqlen_q, H, Dv] 和 softmax_lse[B, H, seqlen_q]
```

---

## 附录：V3 关键类/函数索引

| 组件 | 文件 | 功能 |
|------|------|------|
| `mha_fwd` | `hopper/flash_api.cpp` | C++ 主入口 |
| `Flash_fwd_params` | `hopper/flash.h` | 所有运行时参数 |
| `run_mha_fwd` | `hopper/flash_api.cpp` | 多维编译期分发 |
| `run_flash_fwd` | `hopper/flash_fwd_launch_template.h` | kernel 类型组装与启动 |
| `FlashAttnFwdSm90` | `hopper/flash_fwd_kernel_sm90.h` | SM90 kernel 类 |
| `CollectiveMainloopFwdSm90` | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` | TMA 加载 + WGMMA 计算 |
| `CollectiveEpilogueFwd` | `hopper/epilogue_fwd.hpp` | 归一化 + 输出写回 |
| `StaticPersistentTileScheduler` | `hopper/tile_scheduler.hpp` | 非因果持久化调度 |
| `DynamicPersistentTileScheduler` | `hopper/tile_scheduler.hpp` | 因果 LPT 动态调度 |
| `VarlenDynamicPersistentTileScheduler` | `hopper/tile_scheduler.hpp` | 变长序列调度 |
| `num_splits_heuristic` | `hopper/heuristics.h` | Split 数量决策 |
| `should_pack_gqa` | `hopper/heuristics.h` | PackGQA 决策 |
| `prepare_varlen_num_blocks` | `hopper/flash_api.cpp` | Varlen 预处理 PDL kernel |
| `run_mha_fwd_combine` | `hopper/flash_api.cpp` | Split-KV 合并启动 |

# FlashAttention-4 Forward Pass 逐步详解

> **对应代码**：`flash_attn/cute/interface.py`、`flash_attn/cute/flash_fwd_sm100.py`、`flash_attn/cute/softmax.py`  
> **论文**：*FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*（Zadouri et al., 2025）  
> **目标架构**：NVIDIA Blackwell（B200/GB200，SM100+）

---

## 0. 背景：为什么需要 FA4

### 论文 §2.2 + §1：不对称硬件扩展

> *"A critical trend in accelerator evolution is the asymmetric scaling of hardware units. Although Blackwell B200 doubles the tensor core throughput compared to Hopper H100 (2.25 PFLOPS vs. 1 PFLOPS for FP16/BF16), other functional units (shared memory bandwidth, exponential units, and integer/floating point ALUs) scale more slowly or remain unchanged."*

| Blackwell 新特性 | 相对 Hopper 的变化 | FA4 的对应利用 |
|----------------|------------------|--------------|
| BF16 Tensor Core 吞吐 2.25× | **2.25×** ↑ | 更大 tile（128×128），ping-pong pipeline |
| MMA tile 从 64×128 → 128×128 | tile M 翻倍 | `m_block_size=128`，每 CTA 处理两个 tile |
| **Tensor Memory (TMEM) 256KB/SM** | 全新 | S/P/O 累加器全部存 TMEM，不走寄存器 |
| **完全异步 MMA**（写 TMEM 而非寄存器） | 全新 | MMA 与 softmax 真正解耦并行 |
| **2-CTA MMA 模式** | 全新 | M=256 MMA，每 CTA 只加载一半 B |
| MUFU 指数吞吐不变（16 ops/clock/SM） | 不变 | EX2 软件模拟 + 条件性 rescale |
| SMEM 读带宽不变（128 B/clock/SM） | 不变 | sK 与 sV 共用 SMEM，减少 SMEM 用量 |

---

## 1. 入口：`FlashAttnFunc.forward`

**文件**：[flash_attn/cute/interface.py](flash_attn/cute/interface.py#L1337)

```python
class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale=None, causal=False,
                window_size=(None,None), ..., return_lse=False):
        # 1. 处理 block sparse 参数
        block_sparse_tensors = BlockSparseTensorsTorch(...) if ... else None

        # 2. 调用核心 forward 函数
        out, lse = _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, ...)

        # 3. 保存 backward 所需的中间结果
        ctx.save_for_backward(q, k, v, out, lse)
        return out, lse
```

`FlashAttnFunc` 是标准的 `torch.autograd.Function` 子类，负责：
- 将 Python 层参数组织好传入 `_flash_attn_fwd`
- 通过 `ctx.save_for_backward` 保存 `q, k, v, out, lse` 供反向传播使用
- 将 `lse` 标记为 `non_differentiable`（LSE 梯度暂不支持）

---

## 2. Host 端参数处理：`_flash_attn_fwd`

**文件**：[flash_attn/cute/interface.py](flash_attn/cute/interface.py#L116)

这是 host 端最核心的函数，承担六大职责。

### 2.1 输入验证与 Shape 推断

```python
q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
num_head, head_dim = q.shape[-2:]
# 标准 attention: q.shape = (batch, seqlen_q, num_head, head_dim)
# varlen:         q.shape = (total_q, num_head, head_dim)
```

断言检查包括：dtype 必须为 fp16/bf16、所有 tensor 须在 CUDA 上、`num_head % num_head_kv == 0`（GQA 约束）、head_dim 必须满足对齐要求等。

### 2.2 `softmax_scale` 计算

```python
if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(head_dim)
```

对应论文公式：$S = \alpha Q K^\top$，其中 $\alpha = 1/\sqrt{d}$。

### 2.3 SM100 专有：`q_stage` 与 `use_2cta_instrs` 判断

**`q_stage`**（Q tile 的 pipeline 缓冲级数）：

```python
# 仅 SM100+ 使用双 stage
if arch // 10 == 10:
    q_stage = 2 if seqlen_q_packgqa > m_block_size else 1
else:
    q_stage = 1   # Hopper 只需 1 stage
```

> **论文 §3.1.2**：*"We follow a ping-pong schedule similar to FA-3, where two tiles of the output are computed per thread block."*  
> 两个 Q tile（stage 0 和 stage 1）轮流被 MMA warp 和 softmax warp 处理，构成 ping-pong 流水线。

**`use_2cta_instrs`**（是否启用 2-CTA MMA 模式）：

```python
use_2cta_instrs = (
    arch // 10 in [10, 11]        # Blackwell+
    and not causal                # 暂不支持因果 + 2CTA
    and not local
    and not is_split_kv
    and cu_seqlens_q is None      # 非 varlen
    and int(math.ceil(head_dim / 16) * 16) == 128   # hdim=128
    and int(math.ceil(head_dim_v / 16) * 16) == 128
    and seqlen_q_packgqa > 2 * m_block_size          # seqlen 足够长
)
```

> **论文 §2.2**：*"Blackwell supports a 2-CTA tensor core MMA mode in which a CTA pair within the same thread block cluster cooperatively executes a single MMA, allowing the operation to read and write tensor memory from both CTAs."*  
> 2-CTA 模式将 MMA tile M 扩展到 256，每个 CTA 仅加载 V 的一半（B tile 沿 N 分割），SMEM 流量减半。

### 2.4 JIT 编译与缓存（CuTe-DSL）

**论文 §4**：*"FlashAttention-4 reduces compile time by 20-30× compared to FlashAttention-3."*

```python
compile_key = (
    dtype, head_dim, head_dim_v, qhead_per_kvhead,
    causal, score_mod_hash, mask_mod_hash,
    m_block_size, n_block_size, q_stage,
    is_split_kv, pack_gqa, arch,
    use_2cta_instrs,          # ← Blackwell 特有
    ...
)
if compile_key not in _flash_attn_fwd.compile_cache:
    # 构造 FlashAttentionForwardSm100 对象
    fa_fwd = FlashAttentionForwardSm100(
        head_dim, head_dim_v,
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=causal, is_local=local,
        is_split_kv=is_split_kv,
        pack_gqa=pack_gqa,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        q_stage=q_stage,
        is_persistent=...,
        use_2cta_instrs=use_2cta_instrs,   # ← 2-CTA 开关
        ...
    )
    # CuTe-DSL JIT 编译（Python → PTX → SASS）
    _flash_attn_fwd.compile_cache[compile_key] = cute.compile(fa_fwd, ...)
```

CuTe-DSL 是 Python 嵌入式 DSL，编译时通过 JIT 将 Python 代码降低到 PTX，再由 `ptxas` 生成 SASS。**相较 FA3 的 C++ 模板，编译速度提升 20-30×**（论文 Table 4：FA3 编译 55s → FA4 编译 2.5s）。

### 2.5 Split-KV 支持

```python
if is_split_kv:
    out_partial = torch.empty(num_splits, batch, seqlen_q, num_head, head_dim_v,
                              dtype=torch.float32)
    lse_partial = torch.empty(num_splits, ...)
```

SplitKV 将 KV 序列沿 N 维度切分到多个 SM 并行计算（Flash Decoding），最终由 `_flash_attn_fwd_combine` 规约。

### 2.6 调用已编译 kernel

```python
_flash_attn_fwd.compile_cache[compile_key](
    q, k, v, out, lse, softmax_scale, current_stream,
    cu_seqlens_q, cu_seqlens_k, ...
)
# Split-KV 时额外调用 combine
if is_split_kv:
    _flash_attn_fwd_combine(out_partial, lse_partial, out, lse, ...)
```

---

## 3. Kernel 初始化：`FlashAttentionForwardSm100.__init__`

**文件**：[flash_attn/cute/flash_fwd_sm100.py](flash_attn/cute/flash_fwd_sm100.py#L80)

### 3.1 Warp 角色分配

```python
# 每 CTA 共 16 个 warp（512 线程）
self.softmax0_warp_ids  = (0, 1, 2, 3)   # 处理 Q tile stage 0 的 softmax
self.softmax1_warp_ids  = (4, 5, 6, 7)   # 处理 Q tile stage 1 的 softmax
self.correction_warp_ids= (8, 9, 10, 11) # 对 O 做 rescale 并写 SMEM
self.mma_warp_id        = 12             # 驱动 tensor core（QK^T 和 PV）
self.epilogue_warp_ids  = (13,)          # 将 sO 写回 GMEM
self.load_warp_ids      = (14,)          # TMA 从 GMEM 加载 Q/K/V 到 SMEM
self.empty_warp_ids     = (15,)          # 空闲（减少寄存器压力）
```

> **论文 §3.1.2**：*"The natural way to distribute work across these tiles is then to have two warpgroups of 128 threads each, with each thread processing an entire row. This eliminates the need for inter-warp shuffles to reduce the row max, and for multiple statistics registers per thread."*  
> 每个 softmax warp group（128 线程）处理 128 行中的各自行，每线程负责整行 128 个元素，**无需 warp 间 shuffle**。这与 Hopper（FA3）不同——Hopper 的 64×128 tile 需要 4 线程共同处理一行。

### 3.2 Warp 寄存器预算

```python
# head_dim >= 96 时
self.num_regs_softmax   = 192  # softmax warp：需持有整行 128 fp32 + 临时变量
self.num_regs_correction= 80   # correction warp
self.num_regs_other     = 48   # load/mma/epilogue/empty warp
```

> **论文 §3.1.2**：*"For BF16 input data types, we need to hold 128 registers for the input, and potentially 64 registers for the output (plus miscellaneous and temporary registers)."*  
> 高寄存器需求是 Blackwell 128×128 tile 的直接后果。

### 3.3 TMEM 布局（Blackwell 独有）

```python
self.tmem_s_offset  = [0, self.n_block_size]         # S tile 0 / S tile 1
# e.g. [0, 128]（每 tile 128 列 FP32）
self.tmem_o_offset  = [
    tmem_s_offset[-1] + n_block_size + i * head_dim_v_padded
    for i in range(q_stage)
]  # e.g. [256, 384]

self.tmem_p_offset  = [tmem_s_offset[i] + n_block_size // 2
                       for i in range(2)]             # P 偏移 = S 后半段 [64, 192]
self.tmem_s_to_p_offset = n_block_size // 2           # = 64
```

> **论文 §3.1.2**：*"We choose the latter [two tiles of S that overlap with P] because it allows us to start our software pipeline by immediately computing two S tiles. ...one tile of S and two tiles of P, or two tiles of S that overlap with P."*

TMEM 512 列的布局（以 `n_block=128, head_dim_v=128, q_stage=2` 为例）：

```
列偏移  [0 ... 63]   [64 ... 127]  [128 ... 191] [192 ... 255] [256 ... 383] [384 ... 511]
内容    S0 前半       P0（覆盖 S0） S1 前半        P1（覆盖 S1）  O0 累加器      O1 累加器
        (n/2 列)      (n/2 列 BF16→FP32 = 64 列)
```

**设计原因**：TMEM 是 Blackwell 专有内存，MMA 直接写入，无需寄存器中转。S 和 P 原位共享同一 TMEM 区域（softmax 计算后将 P 写回 S 所在的位置），O 占用另外两块区域用于跨 n_block 的累加。

### 3.4 split_P_arrive 优化

```python
self.split_P_arrive = n_block_size // 4 * 3   # = 96（3/4 的列数）
self.split_P_arrive = int(self.split_P_arrive / 32) * 32  # 对齐到 32
```

> **论文 §3.1.2**：*"To reduce register pressure, we stage out storing P: The first three quarters are stored once (and trigger the corresponding MMA operations), and the last quarter is stored separately."*  
> 写入前 3/4 的 P 后立即发信号给 MMA warp，让 PV MMA 提前开始，隐藏 softmax 写 P 的尾部延迟。

---

## 4. Host 端 Kernel 配置：`__call__`

**文件**：[flash_attn/cute/flash_fwd_sm100.py](flash_attn/cute/flash_fwd_sm100.py#L299)

### 4.1 Tensor 轴转置

```python
# Q: (b, s_q, h, d) → (s_q, d, h, b)（便于 TMA 按 (d, s) 排列）
mQ = cute.make_tensor(mQ.iterator,
    cute.select(mQ.layout, mode=[1, 3, 2, 0]))  # [s,d,h,b]
# K: (b, s_k, h_k, d) → (s_k, d, h_k, b)
# V: (b, s_k, h_k, d) → (d, s_k, h_k, b)  ← V 与 K 不同！V 先转置 d 维
```

V 的轴顺序与 K 不同是因为：
- QK^T MMA：Q(M×d) × K^T(d×N)，d 为规约维度 → K 需要 **K-major**（d 在前）
- PV MMA：P(M×N) × V(N×d)，N 为规约维度 → V 需要 **MN-major**（N 在前，即原始 s_k 维在前）

### 4.2 SMEM 布局计算

```python
def _setup_attributes(self):
    smem_size_q = q_stage * m_block * head_dim_padded * q_dtype.width // 8
    smem_size_kv_per_stage = max(smem_K, smem_V) // cta_group_size
    # 224KB SMEM - Q/O 后，剩余空间决定 KV stage 数
    kv_stage = (224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage
    self.kv_stage = kv_stage   # 通常为 3～5
```

sK 和 sV **共用同一块物理 SMEM**（因为在任意时刻，KV pipeline 中同一个 stage 里只有 K 或 V 其中之一是有效的）：

```python
sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
```

这避免了为 K 和 V 分别分配 SMEM，将 SMEM 利用率提高约一倍。

**2-CTA 对 smem 的影响**：
```python
smem_size_kv_per_stage = max(...) // self.cta_group_size  # cta_group_size=2 时除以 2
```
2-CTA 模式下每个 CTA 只加载 B（K 或 V）的一半，SMEM 需求减半。

### 4.3 TMA 描述符构建（Blackwell 特性）

```python
tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)  # GMEM → SMEM 异步批量拷贝
tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
    tma_load_op, mQ, sQ_layout, mma_tiler_qk, tiled_mma_qk, ...)
tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(...)
tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(...)
```

TMA（Tensor Memory Access）是 Hopper/Blackwell 引入的硬件单元，可以将 tensor 从 GMEM 异步搬运到 SMEM，期间 SM 可以执行其他计算。TMA 描述符在 host 端预先计算好，kernel 执行时直接使用。

### 4.4 MMA 配置（SM100 UMMA）

```python
cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
    q_dtype, K_major, K_major, Float32,
    cta_group,
    mma_tiler_qk[:2],   # (2*m_block, n_block) = (256, 128) for 2CTA
)
tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
    v_dtype, K_major, MN_major, Float32,
    cta_group,
    mma_tiler_pv[:2],   # (2*m_block, head_dim_v) = (256, 128) for 2CTA
    p_source=TMEM,       # P 从 TMEM 读（而非 SMEM）← Blackwell 独有
)
```

**Blackwell tcgen05 UMMA 的关键特性**：
1. MMA 结果直接写入 **TMEM**（不走寄存器），彻底解放寄存器压力
2. **完全异步**：MMA 执行期间 SM 可以同时做其他计算（softmax）
3. P 作为 PV MMA 的 A 操作数从 **TMEM** 读取（`p_source=TMEM`），S→P 不需要经过 SMEM

### 4.5 Tile Scheduler 选择

```python
if cu_seqlens_q is not None:
    TileScheduler = SingleTileVarlenScheduler
elif is_causal or is_local:
    TileScheduler = SingleTileLPTScheduler       # LPT 调度
else:
    TileScheduler = StaticPersistentTileScheduler  # 持久化调度
```

> **论文 §3.3**：*"In FlashAttention-4, we use the classical idea of longest-processing-time-first (LPT) scheduling."*  
> 针对因果 masking 的负载不均衡问题，FA4 将 m_block 按**逆序**处理（最长的 KV 范围优先），并在 head 维度做 section 划分以防 L2 cache 抖动。对 MHA 提升 4-8%，对 MQA 提升 7-14%（H200 测试）。

### 4.6 EX2 模拟频率配置

```python
if enable_ex2_emu:
    self.ex2_emu_freq = 16   # 每 16 个元素中有 4 个用 FMA 模拟
    if head_dim == 128 and use_2cta:   self.ex2_emu_freq = 12
    if is_causal and head_dim > 64:    self.ex2_emu_freq = 10
    if pack_gqa and head_dim > 64 and not is_causal:
        self.ex2_emu_freq = 32  # varlen 下更保守（寄存器压力更大）
```

> **论文 §3.1.3**：*"Instead, we apply emulation to only a subset of the entries in each softmax row (10–25%), with the remaining entries computed via hardware MUFU.EX2. The exact fraction is tuned empirically based on the ratio of MMA and exponential throughput for a given tile configuration."*  
> SM103（B300/GB300 的 MUFU 吞吐加倍）不启用 EX2 模拟（`enable_ex2_emu = not is_sm103`）。

---

## 5. GPU Kernel：`kernel()`

**文件**：[flash_attn/cute/flash_fwd_sm100.py](flash_attn/cute/flash_fwd_sm100.py#L745)

### 5.1 TMA 描述符预取

```python
if warp_idx == 0:
    for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
        cpasync.prefetch_descriptor(tma_atom)
```

在 kernel 最开始由 warp 0 将 TMA 描述符预取到 L1/L2 cache，避免后续 TMA 操作因描述符 cache miss 而增加延迟。

### 5.2 TMEM 分配（Blackwell 独有）

```python
# MMA warp 负责分配 TMEM
tmem = cutlass.utils.TmemAllocator(
    storage.tmem_holding_buf,
    barrier_for_retrieve=tmem_alloc_barrier,
    allocator_warp_id=self.mma_warp_id,        # warp 12 执行分配
    is_two_cta=self.use_2cta_instrs,
    two_cta_tmem_dealloc_mbar_ptr=...,
)
# 在 MMA warp 中：
tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))  # 分配全部 512 列
tmem.wait_for_alloc()   # 等待分配完成（其他 warp 在此 barrier 同步）
tmem_ptr = tmem.retrieve_ptr(Float32)  # 其他 warp 获取 TMEM 指针
```

TMEM 必须由单一 warp 分配并通过 barrier 广播指针，其他 warp 通过 `retrieve_ptr` 获取。这是 `tcgen05` 指令集的约束。

### 5.3 多级 Pipeline 建立

```python
# Q pipeline: TMA warp → MMA warp（Q 的 q_stage 级缓冲）
pipeline_q = PipelineTmaUmma.create(num_stages=q_stage, ...)

# KV pipeline: TMA warp → MMA warp（KV 的 kv_stage 级缓冲）
pipeline_kv = PipelineTmaUmma.create(num_stages=kv_stage, ...)

# S→P→O pipeline: MMA warp ↔ softmax/correction warp
# "producer" = MMA warp（S 算完发信号）
# "consumer" = softmax warp（消费 S）+ correction warp（消费 O）
pipeline_s_p_o = PipelineUmmaAsync.create(num_stages=q_stage, ...)

# P last split: softmax warp → MMA warp（P 最后 1/4 写完信号）
pipeline_p_lastsplit = PipelineAsyncUmma.create(num_stages=q_stage, ...)

# O acc: MMA warp → correction warp（O 累加完成信号）
pipeline_o_acc = PipelineUmmaAsync.create(num_stages=q_stage, ...)

# softmax stats: softmax warp → correction warp（row_max/row_sum 传递）
pipeline_sm_stats = PipelineAsync.create(num_stages=q_stage, ...)

# O epi: correction warp → epilogue warp（O 写 SMEM 完成，可写 GMEM）
pipeline_o_epi = PipelineAsync.create(num_stages=q_stage, ...)
```

这 7 个 pipeline 对象将 5 类 warp 之间的所有同步关系完全解耦，是 FA4 能够实现深度流水线重叠的基础。

### 5.4 Warp 分支进入各自角色

```python
# LOAD warp (14)
if warp_idx in load_warp_ids:
    setmaxregister_decrease(48)     # 减少寄存器分配
    self.load(...)

# MMA warp (12)
if warp_idx == mma_warp_id:
    setmaxregister_decrease(48)
    tmem.allocate(...)              # 分配 TMEM（Blackwell 专有）
    self.mma(...)
    tmem.free(tmem_ptr)             # 释放 TMEM

# Softmax warp 0-3 (stage 0) 和 4-7 (stage 1)
if warp_idx <= softmax1_warp_ids[-1]:
    setmaxregister_increase(192)   # 增加寄存器（需要大量寄存器持有 128 列数据）
    tmem.wait_for_alloc()          # 等待 MMA warp 分配 TMEM
    tmem_ptr = tmem.retrieve_ptr() # 获取 TMEM 指针
    self.softmax_loop(stage=0 or 1, ...)

# Correction warp (8-11)
if warp_idx in correction_warp_ids:
    setmaxregister_decrease(80)
    tmem.wait_for_alloc()
    self.correction_loop(...)

# Epilogue warp (13)
if warp_idx in epilogue_warp_ids:
    setmaxregister_decrease(48)
    self.epilogue_s2g(...)
```

`setmaxregister_increase/decrease` 是 SM100 PTX 指令，允许在 kernel 运行时动态调整 warp 的寄存器上限，使不同角色的 warp 使用不同的寄存器预算，避免整个 CTA 因某个 warp 需要多寄存器而导致全员受限。

---

## 6. Load Warp：`load()`

**文件**：[flash_attn/cute/flash_fwd_sm100.py](flash_attn/cute/flash_fwd_sm100.py#L1200)

Load warp（warp 14）通过 TMA 指令将 Q/K/V 从 GMEM 异步搬运到 SMEM。

### 6.1 针对每个 work tile 的加载顺序

```python
# Prologue：预加载第一个 K tile（n_block_max-1，即最后一个 KV tile）
load_K(block=n_block_max-1, ...)   # K0 → sK[kv_stage_0]

# 加载 Q（双 stage）
pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
load_Q_fn(src_idx=0, dst_idx=0, tma_bar_ptr=...)   # Q0 → sQ[0]

pipeline_q.producer_acquire_w_index_phase(1, q_producer_phase)
load_Q_fn(src_idx=1, dst_idx=1, tma_bar_ptr=...)   # Q1 → sQ[1]

# 加载 V0（与 K 的下一个 stage 交叉，共用 SMEM）
load_V(block=n_block_max-1, ...)   # V0 → sV[kv_stage_0]（物理上 = sK[kv_stage_0]）

# Main loop：循环加载剩余 K_i 和 V_i
for i in range(n_block_max - 1 - n_block_min):
    n_block = n_block_max - 2 - i
    load_K(block=n_block, ...)   # K_i
    load_V(block=n_block, ...)   # V_i
```

**逆序遍历**（从 `n_block_max-1` 到 `n_block_min`）：因果 mask 下，高索引的 n_block 有更多有效 token，逆序确保第一个加载的就是最长的块，配合 LPT 调度减少 SM 空等。

**sK/sV 共用 SMEM**：
```python
# K → sK[stage]，下一轮 V 覆盖同一 stage 的 SMEM
sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
```

### 6.2 TMA 批量拷贝

```python
cute.copy(tma_atom_K, tXgX_cur, tXsX_cur,
          tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state))
```

`tma_bar_ptr` 使 TMA 完成后自动递增 mbarrier，通知 MMA warp 数据已就绪。整个 GMEM→SMEM 的搬运过程对 SM 是**完全透明**的（load warp 发出 TMA 指令后即可执行其他工作或等待下一个 pipeline stage）。

---

## 7. MMA Warp：`mma()`

**文件**：[flash_attn/cute/flash_fwd_sm100.py](flash_attn/cute/flash_fwd_sm100.py#L1383)

MMA warp（warp 12）驱动两类 Blackwell tensor core 运算。

### 7.1 Prologue：计算 S0 和 S1

```python
for stage in range(q_stage):   # stage = 0, 1
    # 等待 Q0/Q1 加载完成
    pipeline_q.consumer_wait_w_index_phase(stage, mma_q_consumer_phase)
    # 等待 K0 加载完成（只在 stage=0 时等待）
    if stage == 0:
        pipeline_kv.consumer_wait(mma_kv_consumer_state)

    # tcgen05 UMMA: Q_stage * K0 → S_stage（写 TMEM tmem_s_offset[stage]）
    gemm_Si[stage](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur))

    # 通知 softmax warp：S_stage 已就绪
    pipeline_s_p_o.producer_commit_w_index(stage)
```

`gemm_Si[stage]` 是一个预编译的 PTX 调用（`sm100_utils.gemm_ptx_precomputed_varname`），直接执行 `tcgen05.mma` 指令：
- A（Q）从 SMEM 读
- B（K）从 SMEM 读（SS 操作）
- C（S）写到 **TMEM** `tmem_s_offset[stage]`
- **完全异步**：指令发出后立即返回，无需等待结果写入 TMEM

### 7.2 Main Loop：PV 和 QK 交替执行

```python
O_should_accumulate = False
for i in range(block_loop_count):   # n_block 迭代
    # === PV MMA（P_stage * V_i → O_stage）===
    pipeline_kv.consumer_wait(...)        # 等待 V_i 加载完成
    for stage in range(q_stage):
        # 等待：O 已 rescale（correction warp 完成），且 P 已写入 TMEM
        pipeline_s_p_o.producer_acquire_w_index_phase(stage, ...)

        # P 从 TMEM tmem_p_offset[stage] 读，V 从 SMEM 读（TS 操作）
        # 结果累加到 TMEM tmem_o_offset[stage]
        gemm_Pi[stage](
            tCrB=tOrVi, sB=sV_cur,
            zero_init=not O_should_accumulate,  # 首次不累加
            # split_P_arrive: P 前 3/4 写完即触发此 mbar，提前开始 MMA
            mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage),
            mbar_phase=...,
        )
        # 通知 correction warp：O 累加完成
        pipeline_o_acc.producer_commit_w_index(stage)

    pipeline_kv.consumer_release(...)   # 释放 V_i 对应的 SMEM
    pipeline_kv.consumer_wait(...)      # 等待 K_{i+1} 加载完成

    # === QK MMA（Q_stage * K_{i+1} → S_stage）===
    for stage in range(q_stage):
        gemm_Si[stage](smem_desc_start_b=...)
        pipeline_s_p_o.producer_commit_w_index(stage)

    O_should_accumulate = True
```

> **论文 §3.1.2（Figure 1）**：FA4 的 Ping-Pong pipeline 在每个 n_block 迭代中将 PV MMA（处理 Vi）与 QK MMA（处理 K_{i+1}）重叠，而 softmax warp 同时处理上一个 S，correction warp 同时 rescale 上一个 O。

**`split_P_arrive` 的实现**：
```python
gemm_Pi[stage](
    ...,
    split_arrive=self.split_P_arrive if self.split_P_arrive > 0 else None,
    mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage),
)
```

P 的前 3/4 写完后，softmax warp 触发 `pipeline_p_lastsplit`，MMA warp 无需等待最后 1/4 的 P 即可开始 PV MMA，让 P 写入的尾部与 PV MMA 的头部重叠。

---

## 8. Softmax Warp：`softmax_loop` → `softmax_step`

**文件**：[flash_attn/cute/flash_fwd_sm100.py](flash_attn/cute/flash_fwd_sm100.py#L1680) | [flash_attn/cute/softmax.py](flash_attn/cute/softmax.py)

每个 softmax warp group（warp 0-3 或 warp 4-7）负责处理对应 Q tile stage 的在线 softmax。

### 8.1 初始化

```python
softmax = SoftmaxSm100.create(
    softmax_scale_log2,
    rescale_threshold=8.0 if q_dtype.width == 16 else 0.0,  # BF16/FP16 启用条件 rescale
    softmax_scale=softmax_scale,
)
softmax.reset()   # row_max = -inf, row_sum = 0
```

### 8.2 每个 n_block 的 `softmax_step`

**文件**：[flash_attn/cute/flash_fwd_sm100.py](flash_attn/cute/flash_fwd_sm100.py#L2013)

```python
# Step 1: 等待 S_stage 就绪，从 TMEM 加载到寄存器
pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
tSrS_t2r = make_fragment(...)
cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)  # TMEM → 寄存器（128 个 FP32）
```

从 TMEM 读取 S 行到寄存器，是 softmax 计算的起点。每个线程持有一整行（128 元素）。

**Step 2：条件性 row_max 更新**（论文 §3.1.4 核心创新）

```python
row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)
```

[softmax.py](flash_attn/cute/softmax.py#L198) 中的实现：
```python
def update_row_max(self, acc_S_row, is_first):
    if is_first:
        row_max_new = self._compute_row_max(acc_S_row)
        acc_scale = 0.0   # 首次不需要 rescale
    else:
        row_max_old = self.row_max[0]
        row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
        row_max_safe = row_max_new if row_max_new != -inf else 0.0
        acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2

        # ▼ 条件性 rescale（论文 §3.1.4 公式 6）
        if acc_scale_ >= -self.rescale_threshold:   # threshold = 8.0 * scale_log2
            row_max_new = row_max_old   # 不更新 max
            acc_scale = 1.0             # 跳过 rescale
        else:
            acc_scale = exp2(acc_scale_)

    self.row_max[0] = row_max_new
    return row_max_safe, acc_scale
```

> **论文 §3.1.4 公式（6）**：
> $$O_j = \begin{cases} e^{m_{j-1}-m_j}O_{j-1} + e^{S_j - m_j}V_j & \text{若 } m_j - m_{j-1} > \tau \\ O_{j-1} + e^{S_j - m_{j-1}}V_j & \text{否则} \end{cases}$$  
> 阈值 $\tau = \log_2(256) = 8.0$（代码中 `rescale_threshold=8.0`），对应 rescale 因子超过 256 倍才执行。最终正确性由最后的 $1/\ell_{\text{final}}$ 归一化保证。

**通知 correction warp acc_scale（row_max 更新量）**：
```python
if not is_first:
    sScale[tidx + stage * m_block_size] = acc_scale  # 写 SMEM sScale
sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)  # 发信号
```

correction warp 收到此信号后立即对 O 进行 rescale，无需等待 softmax 的后续步骤完成。

**Step 3：Scale & Subtract row_max**

```python
softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
# 等价于：acc_S[i] = acc_S[i] * scale_log2 - row_max * scale_log2
# 使用 packed FP32x2 FMA 优化（减少指令数）
for i in range(0, N, 2):
    acc[i], acc[i+1] = cute.arch.fma_packed_f32x2(
        (acc[i], acc[i+1]),
        (scale_log2, scale_log2),
        (-row_max * scale_log2, -row_max * scale_log2),
    )
```

**Step 4：EX2 软件模拟 + 类型转换**（论文 §3.1.3 核心创新）

```python
softmax.apply_exp2_convert(
    tSrS_t2r,           # 输入：FP32 缩放后的 S 行
    tSrP_r2t,           # 输出：BF16/FP16 的 P 行（in-place 覆盖 tSrS）
    ex2_emu_freq=self.ex2_emu_freq,        # e.g. 16
    ex2_emu_start_frg=self.ex2_emu_start_frg,  # e.g. 1（跳过第一个 frg）
)
```

[softmax.py](flash_attn/cute/softmax.py#L233) 中的核心循环：
```python
frg_tile = 32   # 每个 frg 处理 32 个元素
for j in range(frg_cnt):   # frg_cnt = 128/32 = 4
    for k in range(0, frg_tile, 2):
        # 判断当前位置是否使用 FMA 软件模拟
        use_emu = (k % ex2_emu_freq < ex2_emu_freq - ex2_emu_res
                   and j >= ex2_emu_start_frg
                   and j < frg_cnt - 1)  # 最后一个 frg 不模拟
        if use_emu:
            # FMA 软件模拟 2^x（Cody-Waite + 多项式，见 utils.ex2_emulation_2）
            acc[k], acc[k+1] = utils.ex2_emulation_2(acc[k], acc[k+1])
        else:
            # 硬件 MUFU.EX2
            acc[k]   = cute.math.exp2(acc[k],   fastmath=True)
            acc[k+1] = cute.math.exp2(acc[k+1], fastmath=True)
    # 写入输出（FP32 → BF16/FP16 类型转换）
    acc_converted[None, j] = acc[None, j].to(output_dtype)
```

> **论文 §3.1.3**：使用 Cody-Waite 范围规约 + 多项式近似实现 $2^x = 2^{\lfloor x \rfloor} \cdot 2^{x-\lfloor x \rfloor}$。  
> - $2^{\lfloor x \rfloor}$ 通过 IEEE 754 指数位操作完成
> - $2^{x_{\text{frac}}}$ 通过 degree-5 多项式近似（Horner 法），FP32 层面最大相对误差 1.44e-7  
> - 转换到 BF16 后，量化误差（3.9e-3）主导，多项式精度与硬件 MUFU 等价  
> - FMA 单元与 MUFU 单元**并行执行**，有效提升指数运算吞吐（论文 Table 2）

**Step 5：将 P 写回 TMEM**

```python
# 分批写入，每批 32 列
for i in range(cute.size(tStP_r2t.shape[2])):
    cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
    # split_P_arrive 位置：前 3/4 写完即通知 MMA warp
    if i + 1 == split_P_arrive_idx:
        cute.arch.fence_view_async_tmem_store()
        pipeline_s_p_o.consumer_release_w_index(stage)   # S slot 释放（P 已写入前 3/4）

# 最后 1/4 写完，通知 pipeline_p_lastsplit
cute.arch.fence_view_async_tmem_store()
pipeline_p_lastsplit.producer_commit_w_index(stage)
```

**Step 6：计算 row_sum（可与 P 写 TMEM 重叠）**

```python
# 必须在写 P 之后计算 row_sum（需要用到转换后的 P 值）
pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
# self.row_sum[0] = row_sum * acc_scale + sum(exp(S_row))
```

注意：`update_row_sum` 使用的是 `tSrS_t2r`（FP32 的 exp 结果），而不是转换后的 BF16 `tSrP_r2t`，保持精度。

**最后：写 softmax 统计量到 SMEM `sScale`**

```python
sScale[tidx + stage * m_block_size] = softmax.row_sum[0]
if mLSE is not None or learnable_sink is not None:
    sScale[tidx + stage * m_block_size + q_stage * m_block_size] = softmax.row_max[0]
sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)
```

`sScale` 是 softmax warp 与 correction warp 共享的 SMEM 缓冲区（大小 `q_stage * m_block_size * 2`），通过 named barrier 同步。

---

## 9. Correction Warp：`correction_loop`

**文件**：[flash_attn/cute/flash_fwd_sm100.py](flash_attn/cute/flash_fwd_sm100.py#L2100)

Correction warp（warp 8-11）承担两个职责：
1. **中间 rescale**：每个 n_block 迭代后，根据 softmax 传来的 acc_scale 对 TMEM 中的 O 做行级缩放
2. **最终 epilogue**：最后一个 n_block 完成后，对 O 做最终归一化并写出到 SMEM（再由 epilogue warp 写 GMEM）

### 9.1 初始信号：无需 rescale

```python
# 第一个 n_block：O 是 0 初始化的，不需要 rescale，直接放行
for stage in range(q_stage):
    pipeline_s_p_o.consumer_release_w_index(stage)  # 告诉 MMA warp：O slot 已就绪
```

### 9.2 Main Loop：中间 rescale

```python
for i in range(total_block_count - 1):
    for stage in range(q_stage):
        # 等待 softmax 传来 acc_scale
        sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
        scale = sScale[tidx + stage * m_block_size]

        # 投票：warp 中任意线程需要 rescale，则全部 rescale（避免 warp 分支）
        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
        if should_rescale:
            self.correction_rescale(thr_mma_pv, tOtO[..., stage], tidx, scale)

        # 通知 MMA warp：O 已 rescale，可以继续累加
        pipeline_s_p_o.consumer_release_w_index(stage)
        pipeline_sm_stats.consumer_release_w_index(q_stage - 1 - stage)
```

> **论文 §3.1.2**：*"Another difference from FA-3 is that since we transfer P via tensor memory rather than register file, we can decouple the rescaling of the output to a separate 'correction' warpgroup and thus take it out of the critical path."*  
> FA3 中 rescale 是 softmax warp 的责任（在关键路径上）。FA4 通过 TMEM 将 O 的 rescale 解耦给独立的 correction warp，使 softmax warp 可以立即开始计算下一个 S，而 correction warp 与 MMA warp 并行执行 rescale。

**`correction_rescale` 的实现**（TMEM 读改写）：
```python
def correction_rescale(self, thr_mma, tOtO, tidx, scale):
    corr_tile_size = 16   # 每次处理 16 列
    for i in range(head_dim_v // corr_tile_size):
        # TMEM → 寄存器
        cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
        # 寄存器中 packed FP32x2 乘法
        for j in range(0, size(tOrO_frg), 2):
            tOrO_frg[j], tOrO_frg[j+1] = cute.arch.mul_packed_f32x2(
                (tOrO_frg[j], tOrO_frg[j+1]), (scale, scale)
            )
        # 寄存器 → TMEM
        cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
    cute.arch.fence_view_async_tmem_store()
```

### 9.3 Epilogue：最终归一化并写 SMEM

```python
for stage in range(q_stage):
    # 等待最后的 softmax stats
    row_sum = sScale[tidx + stage * m_block_size]
    row_max = sScale[tidx + stage * m_block_size + q_stage * m_block_size]

    # 最终缩放因子：1 / row_sum（即论文中的 1/ℓ_final）
    acc_O_nan = (row_sum == 0.0 or row_sum != row_sum)
    scale = cute.arch.rcp_approx(row_sum if not acc_O_nan else 1.0)

    # 等待最后一个 O 累加完成
    pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)

    # correction_epilogue: TMEM → 乘 scale → 类型转换 → SMEM sO
    self.correction_epilogue(thr_mma_pv, tOtO[..., stage], tidx,
                             stage, m_block, seqlen_q, scale, sO[..., stage],
                             mO_cur, gO[..., stage], gmem_tiled_copy_O)

    # 通知 epilogue warp（或 MMA warp）：O 已就绪
    pipeline_s_p_o.consumer_release_w_index(stage)  # 释放 O slot
    pipeline_o_epi.producer_commit_w_index(stage)   # 通知 epilogue warp
```

**`correction_epilogue` 的实现**（TMEM → 寄存器 → 类型转换 → SMEM）：
```python
def correction_epilogue(self, ..., scale, sO, ...):
    corr_tile_size = 8 * 32 // o_dtype.width   # = 128 for BF16
    for i in range(head_dim_v // corr_tile_size):
        # 从 TMEM 加载 FP32 累加器
        cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
        # 乘 final scale
        for j in range(0, size(tOrO_frg), 2):
            tOrO_frg[j], tOrO_frg[j+1] = mul_packed_f32x2(
                (tOrO_frg[j], tOrO_frg[j+1]), (scale, scale)
            )
        # FP32 → BF16/FP16 + 写 SMEM sO
        copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_r2s_i)
    cute.arch.fence_view_async_shared()
```

### 9.4 LSE 写出

```python
if mLSE is not None:
    for stage in range(q_stage):
        LN2 = log(2.0)
        lse = (row_max * scale_log2 + log2(row_sum)) * LN2   # = log(row_sum * exp(row_max))
        # 等价于标准 LSE = log(sum(exp(S - max)) + max
        if tidx < seqlen_q - m_tile_idx * m_block_size:
            gLSE[tidx] = lse  # 写 GMEM
```

---

## 10. Epilogue Warp：`epilogue_s2g`

**文件**：[flash_attn/cute/flash_fwd_sm100.py](flash_attn/cute/flash_fwd_sm100.py#L2631)

Epilogue warp（warp 13）等待 `pipeline_o_epi` 信号，将 SMEM 中的 `sO` 写到 GMEM。

### TMA Store 路径（非 varlen）

```python
if use_tma_O:
    store_O, _, _ = copy_utils.tma_get_copy_fn(tma_atom_O, ...)
    for stage in range(q_stage):
        pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
        store_O(src_idx=stage, dst_idx=stage)             # SMEM → GMEM（异步 TMA）
        cute.arch.cp_async_bulk_commit_group()
    for stage in range(q_stage):
        cute.arch.cp_async_bulk_wait_group(q_stage-1-stage, read=True)
        pipeline_o_epi.consumer_release_w_index(stage)
```

TMA S2G（`CopyBulkTensorTileS2GOp`）将 SMEM 的整块 O tile 异步写回 GMEM，epilogue warp 不需要 staging，效率极高。

### 通用 Copy 路径（varlen）

```python
else:
    for stage in range(q_stage):
        pipeline_o_epi.consumer_wait_w_index_phase(stage, ...)
        self._store_O_to_gmem(sO[..., stage], gO[..., stage], ...)
```

varlen 场景需要处理 seqlen 边界，使用 predicated 的 128-bit 向量拷贝（`CopyUniversalOp`）。

---

## 11. 完整执行流程图

```
Host (interface.py):
  FlashAttnFunc.forward()
    └─→ _flash_attn_fwd()
          ├─ 1. 输入验证 & 参数计算（softmax_scale, q_stage, use_2cta）
          ├─ 2. JIT 编译 → FlashAttentionForwardSm100
          ├─ 3. FlashAttentionForwardSm100.__call__()
          │     ├─ 轴转置（Q/K/V tensor）
          │     ├─ _setup_attributes()（kv_stage, SMEM 布局）
          │     ├─ TMA 描述符构建（Q/K/V/O）
          │     ├─ Tile Scheduler 构建（LPT / Persistent）
          │     └─ kernel.launch(grid, block=512, cluster=cluster_shape)
          └─ 4. [Split-KV] _flash_attn_fwd_combine()

GPU Kernel (SM100, 512 线程/CTA):
  ├─ Warp 15 (empty)    setmaxreg(48), idle
  ├─ Warp 14 (load)     TMA: Q0→sQ[0], Q1→sQ[1], K0,V0,K1,V1,...→sK/sV
  ├─ Warp 12 (MMA)      TMEM alloc
  │   ├─ Prologue:      UMMA: Q0*K0→S0(TMEM), Q1*K0→S1(TMEM)
  │   └─ Loop(n_block): UMMA: P0*Vi→O0(TMEM,acc), Q0*K_{i+1}→S0(TMEM)
  ├─ Warp 0-3 (softmax0) setmaxreg(192)
  │   └─ softmax_step:  S0(TMEM)→reg → cond_rescale → exp2(FMA+MUFU) → P0(TMEM)
  ├─ Warp 4-7 (softmax1) setmaxreg(192)
  │   └─ softmax_step:  S1(TMEM)→reg → cond_rescale → exp2(FMA+MUFU) → P1(TMEM)
  ├─ Warp 8-11 (correction) setmaxreg(80)
  │   ├─ Loop: O(TMEM) *= acc_scale（中间 rescale）
  │   └─ Final: O(TMEM) *= 1/row_sum → BF16 → sO(SMEM)
  └─ Warp 13 (epilogue)
      └─ TMA S2G: sO(SMEM) → mO(GMEM)
```

---

## 12. 七个 Pipeline 的协作关系

```
              pipeline_q          pipeline_kv
Load ────────────────────────────────────────→ MMA
                                                │
                     pipeline_s_p_o(S全满信号)  │
MMA ─────────────────────────────────────────→ Softmax
                                                │
                     pipeline_sm_stats(acc_scale)
Softmax ────────────────────────────────────→ Correction
                                                │
             pipeline_s_p_o(O rescale完成信号) │
Correction ──────────────────────────────────→ MMA
                                                │
             pipeline_p_lastsplit(P 3/4写完)   │
Softmax ────────────────────────────────────→ MMA（提前开始 PV）
                                                │
                     pipeline_o_acc(O累加完成)  │
MMA ─────────────────────────────────────────→ Correction（最终 epilogue）
                                                │
                     pipeline_o_epi(O写SMEM完成)
Correction ──────────────────────────────────→ Epilogue
                                                │
                     TMA S2G                   │
Epilogue ────────────────────────────────────→ GMEM (mO)
```

---

## 13. 核心优化总结

| 优化点 | 论文位置 | Blackwell 硬件支持 | 代码实现 |
|-------|---------|-----------------|--------|
| 完全异步 UMMA + TMEM 存储 | §2.2, §3.1.2 | tcgen05 异步 MMA，256KB TMEM/SM | `tcgen05.mma`，`tmem.allocate` |
| Ping-Pong 双 Q tile pipeline | §3.1.2, Fig.1 | TMEM 存两个 O 累加器 | `q_stage=2`，7 个 pipeline 对象 |
| Correction warp 独立 rescale | §3.1.2 | TMEM 允许跨 warp 读写 | `correction_loop`，`pipeline_s_p_o` |
| 条件性 softmax rescale | §3.1.4, 公式(6) | — | `SoftmaxSm100.update_row_max`，`rescale_threshold=8.0` |
| EX2 软件模拟（FMA + MUFU） | §3.1.3, Table 2 | FMA 与 MUFU 并行执行单元 | `apply_exp2_convert`，`ex2_emulation_2` |
| 2-CTA MMA（M=256） | §2.2 | 2-CTA MMA 模式，cluster | `use_2cta_instrs`，`cta_group=TWO` |
| sK/sV 共用 SMEM | — | SMEM 容量 224KB | `sV = recast_ptr(sK)` |
| split_P_arrive 提前信号 | §3.1.2 | — | `split_P_arrive = 3/4 * n_block` |
| LPT tile 调度 | §3.3 | — | `SingleTileLPTScheduler` |
| CuTe-DSL JIT 编译 | §4, Table 4 | — | `cute.compile`，compile_cache |
| 动态寄存器调整 | §3.1.2 | `setmaxreg` PTX 指令 | `setmaxregister_increase/decrease` |

---

## 参考

- 论文：*FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*（Zadouri, Hoehnerbach, Shah et al., 2025）
- 代码入口：[flash_attn/cute/interface.py#L1337](flash_attn/cute/interface.py#L1337)（`FlashAttnFunc.forward`）
- SM100 Kernel：[flash_attn/cute/flash_fwd_sm100.py](flash_attn/cute/flash_fwd_sm100.py)（`FlashAttentionForwardSm100`）
- Softmax：[flash_attn/cute/softmax.py](flash_attn/cute/softmax.py)（`SoftmaxSm100`）
- CUTLASS 示例：https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha

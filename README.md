---
toc:
  depth_from: 1
  depth_to: 2
  ordered: false
---

# 类脑芯片端到端工具链验证（Hello World）

本项目以最简全连接网络（FCN/MLP）+ MNIST 为验证载体，跑通类脑（神经形态）芯片的 **4 层端到端工具链流水线**，以最低成本验证"复用开源生态"的技术路线，并让工程团队建立对**脉冲（Spike）、权重驻留（Weight Stationary）**与**极低位宽量化**的工程直觉：

```text
PyTorch FP32 基线  →  1.58-bit SNN 量化  →  NIR 标准图  →  硬件/模拟器流式验证
   (Step 1)               (Step 2)           (Step 3)         (Step 4)
```

## 流水线总览

| 步骤 | 目标 | 核心手段 | 产出物 |
| --- | --- | --- | --- |
| **Step 1 操作台** | 建立 FP32 基线，作为精度与能耗的评估参考系 | PyTorch 训练 3 层 MLP，fvcore 静态剖析 | `baseline_fp32_best.pth` |
| **Step 2 降维车间** | 引入脉冲逻辑与 1.58-bit 三值量化 | SpikingJelly + QAT（STE），权重逼近 {-1, 0, 1} | `snn_1.58bit_best.pth` |
| **Step 3 物理映射器** | 软硬解耦，产出标准中间表示 | 权重物理坍缩，构建 NIR 计算图（TVM 挂起） | `snn_mvp_model.nir` |
| **Step 4 流式验证** | 证明"权重驻留"打破内存墙 | 泊松编码 + 流式下发 + 三大商业探针 | 脉冲输入流与期望标签（`.npy`） |

## 目录结构

| 文件 | 所属步骤 | 职责 |
| --- | --- | --- |
| `main.py` | Step 1 | FP32 基线训练入口（含 fvcore Profiling） |
| `model.py` | Step 1 | `BaselineMLP`：3 层 MLP（784 → 256 → 10），模块级自检 |
| `data_loader.py` | Step 1 | MNIST 数据流水线，像素值域严格保持 [0, 1] |
| `utils.py` | Step 1 | 随机种子锁定、设备检测、DataLoader worker 种子 |
| `main_snn.py` | Step 2 | 1.58-bit SNN 训练入口 |
| `snn_models.py` | Step 2 | `SNN_MLP`：时间循环（T 步）脉冲网络，`IFNode` 平替 `ReLU` |
| `ternary_ops.py` | Step 2 | `TernaryQuantize`（QAT 核心算子）与 `TernaryLinear` |
| `check_pth.py` | Step 2 | 校验 `.pth` 权重物理形态（是否已坍缩为三值） |
| `verify_true_accuracy.py` | Step 2 | 用纯物理权重重跑测试集，验证真实准确率 |
| `compiler_tvm_nir.py` | Step 3 | 权重坍缩 + 构建 NIR 图，产出 `.nir` 图纸 |
| `inspect_and_convert_nir.py` | Step 3 | 读取 `.nir`，演示 {-1,0,1} → 兴奋/抑制双阵列拆分 |
| `generate_spikes.py` | Step 4 | 泊松编码，生成硬件输入激励（golden vectors） |
| `install_deps.py` | 工具 | 按本机是否有 NVIDIA GPU 自动安装 torch（CUDA/CPU 分支） |
| `requirements.txt` | 工具 | 依赖清单（静态，按平台分支） |

## 快速开始

### 环境与依赖



```bash
# 方式一：自动检测 GPU，按需安装（推荐）
conda create -n snn_baseline python=3.10 -y
conda activate snn_baseline
python install_deps.py          # 检测到 NVIDIA GPU → CUDA 版 torch；否则 CPU 版

# 方式二：静态清单安装
pip install -r requirements.txt
```

Step 2 / Step 3 的额外依赖（不在此清单内，按需安装）：

```bash
# Step 2：脉冲生态库（建议从基线环境克隆出专属环境）
conda create --name snn_spiking --clone snn_baseline
conda activate snn_spiking
pip install spikingjelly

# Step 3：NIR 标准格式库
pip install nir
```

### 可复现性约定

复现是评估 SNN 精度损失的前提，必须满足：

- 代码入口调用 `set_deterministic_environment(seed=42)`，锁死 Python / NumPy / PyTorch 随机种子及 cudnn 确定性算法。
- `PYTHONHASHSEED` 无法在代码内设置，需由启动命令注入：

```bash
export PYTHONHASHSEED=42
python main.py
```

- 训练入口要求连续运行两次，各 Epoch 指标必须逐位一致（见 Step 1 验收）。

## Step 1 · FP32 基线（PyTorch）

**目标**：训练至收敛，保存基线权重，证明业务逻辑与数据流水线畅通，并采集传统架构的能耗/显存数据作为后续对比基准。

### 设计要点

- **数据流水线**（`data_loader.py`）：直接使用 `torchvision.datasets.MNIST`，仅做 `ToTensor()` 缩放。**严禁添加 `Normalize`**：泊松编码要求输入值域严格落在 [0, 1]（代表发放概率），基线阶段就必须为 Step 4 铺路。
- **网络纪律**（`model.py`）：极简 3 层 MLP（784 → 256 → 10），仅使用 `nn.Linear` + `nn.ReLU`，**严禁引入 BatchNorm / Dropout / 复杂激活**，为 Step 2 的 `ReLU → IFNode` 平替保留纯净计算图。
- **训练配置**：`nn.CrossEntropyLoss` + `Adam(lr=1e-3)`，10 个 Epoch，Batch Size 128。期望测试集准确率达 **97% ~ 98%**。

### 运行与产出

```bash
python main.py
```

产出 `baseline_fp32_best.pth`（最佳权重）。脚本末尾自动执行 fvcore Profiling，输出架构"性能原罪"数据：约 **203,530 参数**、单次推理约 **203,264 次 MACs**、FP32 权重显存占用约 **2385.12 KB**。这组数字是 Step 4 展示类脑架构优势（纯加法、零搬运）的对比靶子。

### 验收

1. **准确率天花板**：10 个 Epoch 内测试准确率突破 97%。若停留在 ~10%（瞎猜），检查 `data_loader.py` 输入；若未达 97%，检查 `model.py` 前向传播。
2. **绝对可复现**：连续运行两次 `python main.py`，每个 Epoch 的 Loss / Acc 必须完全一致。任何微小抖动都说明 `utils.py` 的种子锁定不彻底。
3. **性能原罪数据**：终端成功打印 fvcore Profiling 数据，MACs 数量级在 20 万左右。

## Step 2 · 1.58-bit SNN 量化（SpikingJelly）

**目标**：将 FP32 模型改造为脉冲网络，并通过量化感知训练（QAT）将权重逼近 {-1, 0, 1} 三值状态，实现"乘法降维为加法"。

### 设计要点

- **算子平替**：`nn.ReLU` → SpikingJelly `IFNode`（积分触发神经元，替代梯度 `surrogate.ATan()`）；`nn.Linear` → `TernaryLinear`。
- **时间维度（T）**：SNN 通过在 $T$ 个时间步内积分膜电位来发放脉冲。`SNN_MLP` 在 `forward` 内封装时间循环，输出 T 步平均电压，对外接口与 FP32 模型完全一致，可直接复用 Step 1 的训练循环。MVP 取 **T=4**。
- **工程纪律**：每个 Batch 前向结束后必须调用 `functional.reset_net(self)` 清空膜电位，否则 Batch 间互相污染，模型无法收敛。
- **QAT 与 STE**（`ternary_ops.py`）：前向将权重截断为 `{-1, 0, 1} × scale`；由于量化与脉冲发放不可导，反向传播使用直通估计器（STE）放行梯度。
- **训练配置**：沿用 Step 1 的循环与超参，Epochs 放宽至 15。SNN 存在"冷启动/死神经元"现象（首 Epoch 爬升慢于 FP32），属正常表现。

### 1.58-bit 原理与硬件映射

**为什么叫 1.58-bit**：BitNet b1.58 将 FP16 权重压缩为三种状态 {-1, 0, 1}，因 $\log_2(3) \approx 1.58$ 而得名。权重仅含 ±1 时，$X \times W$ 退化为加法/减法；权重为 0 则直接断路，跳过计算（激活稀疏性）。

**硬件无法物理表示 -1，如何落地（双突触阵列）**：单极性器件只能表示 {0, 1}，因此在编译下发阶段把 1 个三值权重矩阵拆为 2 个纯 {0, 1} 阵列：

| 算法权重 | 兴奋阵列 $W_{pos}$ | 抑制阵列 $W_{neg}$ | 硬件动作 |
| --- | --- | --- | --- |
| 1 | 1 | 0 | Bank 0 加法，升高膜电位 |
| -1 | 0 | 1 | Bank 1 减法，降低膜电位 |
| 0 | 0 | 0 | 物理断路，零功耗 |

数学等效式：$W_{ternary} = W_{pos} - W_{neg}$。**训练阶段必须保留 -1**（抑制能力），若强行抹除负权重，网络失去抑制会"癫痫"式放电，精度断崖下跌；拆分动作留给编译后端，软硬解耦。

### 幽灵权重（Ghost Weights）—— 导出前的关键坑

`TernaryQuantize.forward` 中，前向计算确实使用量化权重，但 PyTorch 底层保存并更新的仍是 FP32 连续权重（反向传播梯度极小，若锁死 {-1,0,1} 则梯度加不上、模型无法学习）。因此：

- **`snn_1.58bit_best.pth` 里保存的是"幽灵权重"（FP32 连续值），不是纯三值。**
- 导出/验证时，必须手动重演坍缩：`scale = w.abs().mean().clamp(min=1e-5)`，`w_q = round(w / scale).clamp(-1, 1) × scale`（见 `compiler_tvm_nir.py` 的 `collapse_to_physical_weight`）。
- **scale 因子的去向**：硬件只烧录三值整数权重。算法层 $W_q \times scale \times X > V_{threshold}$，编译器将等式两边同除 scale，浮点 scale 被吸收进神经元发射阈值（$\frac{V_{threshold}}{scale}$），硬件内部彻底告别小数。

### 运行与产出

```bash
python main_snn.py
```

产出 `snn_1.58bit_best.pth`。**若误将幽灵权重直接交给硬件，流片阶段必然崩溃**——验收必须严格走以下三维度。

### 验收（三维度立体校验）

1. **商业底线（Test 准确率对比）**：Step 2 最终 Epoch 的 Test 准确率对比 Step 1（97.5% ~ 98%）。**只要稳定在 95% ~ 96.5% 即为成功**——以不到 2% 的精度微损，换取功耗与显存数量级的下降。Train 准确率仅作排查 Bug 的辅助指标。
2. **工程健康度（Train 曲线）**：Train 准确率应呈"初期爬升慢、中后期稳步收敛"。若长期停留在 10%~20% 震荡，说明替代梯度失效或量化 scale 计算有误（死神经元）。
3. **物理形态（验明正身）**：运行 `python check_pth.py`，对 `.pth` 执行坍缩后打印权重唯一值，应只出现 **3 个值**（如 `-0.12, 0.0, 0.12`）。若出现上千个不同小数，说明量化被旁路，属"假量化"废品。

**终极兜底**：`python verify_true_accuracy.py` 彻底抛弃量化引擎，用纯物理坍缩权重（{-1,0,1}×scale）塞进标准 `nn.Linear` 重跑测试集——这个数字就是流片后芯片的真实性能。

## Step 3 · NIR 编译映射

**目标**：剥离时间外壳，提取纯粹的空间拓扑，翻译为通用的 **NIR（Neuromorphic Intermediate Representation）** 标准图，作为软硬件交接契约。

### 时间展开天坑

Step 2 训练代码中的 `for t in range(self.T)` 循环若直接交给编译器，会被平铺展开成 T 层完全相同的空间网络——**这是绝对错误的**：类脑芯片底层自带时钟 Tick 机制，SRAM 只需存放**单步权重**，时间循环由硬件时钟完成。因此 Step 3 先定义 `SingleStepSNN`（无时间循环、无 reset），仅保留层级连接，再加载量化权重导出。

### TVM 的现状（战略性挂起）

FCN 结构只有 `Linear` 与脉冲激活，无可融合算子，引入 TVM 收益为零（"高射炮打蚊子"）。且 Windows 官方 wheel（0.25/0.26）不含 relay 前端。因此当前 NIR 图纸由**权重坍缩直接构建**，TVM 的 `jit.trace → relay → SimplifyInference/FoldConstant` 路径已在 `compiler_tvm_nir.py` 中以注释保留；后续推进 Transformer / 复杂 CNN 时再激活。

### 运行与产出

```bash
python compiler_tvm_nir.py
```

产出 `snn_mvp_model.nir`，其拓扑为 `Input → Affine(784→256) → IF → Affine(256→10) → Output`，权重已是物理坍缩后的纯 {-1, 0, 1}（脚本内打印唯一值自检）。

### 硬件交接流程

硬件工程师拿到 `.nir` 后只需做三件事：

1. **解析拓扑**：`nir.read()` 读出 `nir.Affine`（对应交叉阵列/乘加器）与 `nir.IF`（对应神经元电路）。
2. **物理映射**：将三值权重烧录进 SRAM 存储体 / 忆阻器交叉阵列，把发射阈值配置到累加器比较逻辑。
3. **路由连线**：配置片上网络（NoC），保证脉冲按层流动。

硬件团队无需理解 PyTorch 反向传播或替代梯度——他们只看到纯粹的加减法逻辑与连线拓扑。`.nir` 即软件与硬件之间唯一的交接契约。

## Step 4 · 流式验证与硬件探针

**目标**：在周期精确模拟器中按流式方式完成推理，用数据证明"零显存搬运 + 纯加法计算 + 极低延迟"。

### 动作一：泊松编码（输入脉冲流）

芯片输入引脚只接受 0/1 电平，不认识浮点像素。`generate_spikes.py` 将像素值视为发放概率做泊松采样：像素值 0.8 在 $T$ 个时钟周期内每个周期以 80% 概率发一个脉冲。一张静态 28×28 图片由此变为 $T$ 步离散脉冲流：

```bash
python generate_spikes.py
```

产出 `input_spikes_batch0.npy`（形状 `[T, Batch, 784]`，仅含 0/1）与 `expected_output_labels.npy`（期望标签），随 `.nir` 图纸一并交付硬件团队作为黄金测试向量。

### 动作二：权重驻留（Weight Stationary）

模拟器初始化时把 `.nir` 中的三值权重静态烧录进片上 SRAM，此后按 Tick 逐波下发输入脉冲。**整个推理过程严禁读取任何外部存储（DRAM）权重**——权重驻留在计算单元旁，只有脉冲信号在层级间穿梭。

### 三大商业探针

1. **DRAM Access Count（外部显存访问次数）**：总线接口埋点。报告须显示 `DRAM Read/Write = 0 Bytes`。一次 DRAM 访问功耗约为 SRAM 的 100~200 倍，零搬运是"微瓦级功耗"承诺的物理依据（对比：传统 GPU 为 20 万次乘法需从 HBM/GDDR 搬运 2.3 MB 权重）。
2. **SOPs（突触操作计数）**：神经元发脉冲且下游权重非 0 时计数 +1（纯加法/减法，无乘法）。神经元未触发则下游直接跳过（事件驱动稀疏性）。由于 MNIST 黑色背景（像素为 0）与权重稀疏性（大量权重为 0），SOPs 总量将远小于传统架构的 20 万次 MACs——"不发脉冲就不耗电"。
3. **First-Spike Latency（首事件延迟）**：测第一根输入脉冲到输出层首个分类脉冲的时间差。无需等整个 Batch 收齐，特征足够明显时网络在早期 Tick 即开始发放正确类别脉冲——对 DVS 避障等实时场景是核心卖点。

## 编译后端：从 NIR 到 SRAM 烧录文件

算法团队的工作止步于 `.nir`；将三值权重拆分为物理 `{0, 1}` 阵列是编译后端（Compiler/BSP）的职责：

- 硬件通过**物理 SRAM Bank 地址**区分兴奋/抑制：`SRAM_Bank_0` 数据线接累加器正极（执行加法），`SRAM_Bank_1` 接负极（执行减法）。硬件不感知数据逻辑含义，只遵循"Bank 0 加、Bank 1 减"的物理法则。
- 终极交付物不是 NumPy 数组，而是带物理地址的**存储器初始化文件**（`.hex` / `.coe` / `.bin`），例如：

```text
layer1_excitatory_bank0.hex   # w_pos，指示烧录到 Bank 0
layer1_inhibitory_bank1.hex   # w_neg，指示烧录到 Bank 1
```

- 生成方式：`inspect_and_convert_nir.py` 演示了拆分逻辑（`w_pos = (w > 0)`、`w_neg = (w < 0)`），按硬件要求的格式遍历写出即可，半天工作量。
- 需向硬件确认两个参数：**交付文件格式**（.txt / .hex / .coe）与 **SRAM Bank 地址空间起点**。

## 已知问题与注意事项

- **macOS / 沙箱环境 TMPDIR 坑**：`num_workers=2` 的 DataLoader 在部分沙箱环境会因默认 TMPDIR 拦截 torch 共享内存文件创建而报 `RuntimeError: No such file or directory`（实测对照：macOS 上 `fork` 上下文替代亦不可行）。此类机器请用 `TMPDIR=/tmp PYTHONHASHSEED=42 python main.py` 运行；Linux GPU 服务器无此问题，属环境限制而非代码缺陷。
- **fvcore 参数统计重复累加**：`main.py` 中 `sum(params.values())` 会把父层级与子层级参数重复计入（打印值 610,590 = 实际参数 203,530 × 3），不影响 MACs 与显存占用级别的评估。
- **TVM 流程挂起**：见 Step 3 说明，当前 FCN 阶段绕过，NIR 由权重坍缩直接构建。

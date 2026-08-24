""" import torch
import numpy as np

# 加载量化后的权重
state_dict = torch.load("snn_1.58bit_best.pth", map_location="cpu")
weight_tensor = state_dict['fc1.weight']

# 打印权重的唯一值（Unique Values）
unique_vals = torch.unique(weight_tensor).numpy()

print("--- SNN 1.58-bit 权重物理形态开箱 ---")
print(f"权重中包含的不同数值个数: {len(unique_vals)}")
print(f"这些数值分别是: {np.round(unique_vals, decimals=4)}") """


import torch
import numpy as np

# 加载保存的“幽灵权重”
state_dict = torch.load("snn_1.58bit_best.pth", map_location="cpu")
latent_weight = state_dict['fc1.weight']

# ==========================================
# 核心修正：执行物理坍缩 (模拟 TernaryQuantize.forward)
# ==========================================
# 1. 计算当前层的 Scale
scale = latent_weight.abs().mean().clamp(min=1e-5)
# 2. 强制截断到 -1, 0, 1 并乘回 Scale
physical_weight = torch.round(latent_weight / scale).clamp(-1, 1) * scale

# 打印真实的物理权重形态
unique_vals = torch.unique(physical_weight).numpy()

print("--- SNN 1.58-bit 真实物理权重开箱 ---")
print(f"权重中包含的不同数值个数: {len(unique_vals)}")
print(f"这些数值分别是: {np.round(unique_vals, decimals=4)}")
# 此时您应该只会看到类似 [-0.123, 0.0, 0.123] 这 3 个绝美的数字！
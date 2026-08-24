import torch
import torch.nn as nn

class TernaryQuantize(torch.autograd.Function):
    """
    1.58-bit 核心量化算子：
    前向传播时，将权重强制逼近 {-1, 0, 1}。
    反向传播时，使用 STE 机制，允许梯度畅通无阻地流回原始高精度权重。
    """
    @staticmethod
    def forward(ctx, weight):
        # 1. 计算缩放因子 (Scale)：取权重绝对值的平均，防止分母为 0
        scale = weight.abs().mean().clamp(min=1e-5)
        
        # 2. 降维打击：缩放 -> 四舍五入 -> 截断到 [-1, 1]
        weight_q = torch.round(weight / scale).clamp(-1, 1)
        
        # 3. 还原幅度：乘以 scale 保证激活值的方差稳定。
        # 顾问注：在未来通过 TVM 编译到硬件时，这个 scale 会被吸收到下一层神经元的发射阈值中，
        # SRAM 里只会真正烧录 weight_q (即 -1, 0, 1)。
        return weight_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE)：直接放行梯度
        return grad_output

class TernaryLinear(nn.Linear):
    """
    对标传统 nn.Linear 的三值化线性层。
    可直接在网络中平替原来的 nn.Linear。
    """
    def forward(self, input):
        # 每次前向传播时，实时对底层的高精度 FP32 权重进行三值化截断
        quantized_weight = TernaryQuantize.apply(self.weight)
        return nn.functional.linear(input, quantized_weight, self.bias)
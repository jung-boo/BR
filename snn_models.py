import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate

# 假设前面的量化算子保存在同一目录
from ternary_ops import TernaryLinear

class SNN_MLP(nn.Module):
    def __init__(self, T=4):
        """
        T: 仿真时间步长 (Time-steps)。这是 SNN 和 ANN 最根本的区别。
        T 越大，精度越高，但推理延迟越大。为了体现我们的“极低延迟”优势，MVP 阶段建议 T 设置为 4 甚至 2。
        """
        super().__init__()
        self.T = T  
        self.flatten = nn.Flatten()
        
        # Layer 1: 使用 1.58-bit 三值化线性层
        self.fc1 = TernaryLinear(784, 256)
        
        # Layer 1 激活：将原本的 nn.ReLU() 平替为脉冲神经元 IFNode (积分触发器)
        # surrogate.ATan() 就是我们的替代梯度函数，用来解决脉冲不可导的问题
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        
        # Layer 2: 同样使用三值化
        self.fc2 = TernaryLinear(256, 10)

    def forward(self, x):
        x = self.flatten(x)  # Shape: [B, 784]
        
        # 用于累加最后输出层的电压（或脉冲数）
        out_voltage = 0
        
        # 【SNN 核心灵魂：时间循环】
        # 这里我们使用最简单的静态图像直接编码（直接将电流持续输入 T 个时间步）
        for t in range(self.T):
            x_fc1 = self.fc1(x)         # 乘加运算已被量化为极低位宽的加法
            s_t = self.sn1(x_fc1)       # 神经元积分膜电位。只要超过阈值，s_t 就是 1，否则为 0
            v_out = self.fc2(s_t)       # 由于输入 s_t 全是 0 或 1，硬件底层的 MACs 正式降维为纯 ACs (累加)
            out_voltage += v_out
            
        # 【极其致命的工程纪律】
        # 每一个 Batch 的前向传播结束后，必须清空神经元内部残留的“膜电位”。
        # 如果工程师忘记写这行代码，下一个 Batch 的数据会被上一个 Batch 污染，模型绝对无法收敛！
        functional.reset_net(self)
        
        # 输出 T 个时间步的平均电压，可以直接对接原有的 nn.CrossEntropyLoss
        return out_voltage / self.T
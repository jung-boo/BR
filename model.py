import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    """
    最基础的 3 层全连接网络（FP32 基线）。
    战略纪律：严禁在此类中引入 BatchNorm 或 Dropout，为 SNN 转换保持绝对纯净的计算图。
    """
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super(BaselineMLP, self).__init__()
        
        # 将 [Batch, 1, 28, 28] 展平为 [Batch, 784]
        self.flatten = nn.Flatten()
        
        # Layer 1: 线性变换
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # 战略占位符：标准 ReLU 激活。
        # 工程师注：到了 Step 2，只需 `import spikingjelly.activation_based.neuron as neuron`
        # 然后将 `nn.ReLU()` 平替为 `neuron.IFNode(surrogate_function=...)` 即可。
        self.relu1 = nn.ReLU()
        
        # Layer 2: 线性输出层 (Logits)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        前向传播逻辑
        输入 x 维度: [B, C, H, W] -> 例如 [64, 1, 28, 28]
        输出 logits 维度: [B, num_classes] -> 例如 [64, 10]
        """
        x = self.flatten(x)     # Shape: [B, 784]
        x = self.fc1(x)         # Shape: [B, 256]
        x = self.relu1(x)       # Shape: [B, 256]
        logits = self.fc2(x)    # Shape: [B, 10]
        return logits

# ==========================================
# 模块级自检代码 (供工程师快速验证张量血脉)
# ==========================================
if __name__ == "__main__":
    print("[Pipeline] 正在测试 Phase 1.3 网络拓扑初始化...")
    model = BaselineMLP()
    
    # 模拟 Phase 1.2 吐出的一个 Batch 的图像数据
    dummy_batch = torch.randn(64, 1, 28, 28)
    
    # 前向传播测试
    output = model(dummy_batch)
    
    print(f" -> 输入张量维度: {dummy_batch.shape}")
    print(f" -> 输出张量维度: {output.shape} (预期: torch.Size([64, 10]))")
    
    assert output.shape == (64, 10), "严重错误：网络输出维度不符合分类任务预期！"
    print("[Pipeline] Phase 1.3 拓扑结构自测通过。")
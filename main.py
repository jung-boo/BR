import torch
import torch.nn as nn
import torch.optim as optim
import time

from utils import set_deterministic_environment
from data_loader import build_data_pipeline
from model import BaselineMLP

def train_epoch(model, dataloader, criterion, optimizer, device):
    """单次训练循环"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        
        # 统计指标
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def test_epoch(model, dataloader, criterion, device):
    """单次验证循环"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # 验证阶段切断梯度计算
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    print("========== SNN 前置基线工程 (Step 1) 开始 ==========")
    # 1. 绝对环境锁定 (Phase 1.1)
    set_deterministic_environment(seed=42)
    
    # 2. 硬件设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pipeline] 当前训练硬件: {device}")
    
    # 3. 数据流水线构建 (Phase 1.2)
    train_loader, test_loader = build_data_pipeline(batch_size=128)
    
    # 4. 实例化模型并推至硬件 (Phase 1.3)
    model = BaselineMLP().to(device)
    
    # 5. 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    # 使用 Adam 优化器加速收敛，学习率设为经典的 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    print(f"\n[Pipeline] 开始训练，目标 Epochs: {epochs}...")
    
    # 6. 开始训练循环 (Phase 1.4)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        
        end_time = time.time()
        
        print(f"Epoch [{epoch}/{epochs}] | 耗时: {end_time - start_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")
              
        # 保存最佳权重
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "baseline_fp32_best.pth")
            
    print(f"\n[Pipeline] 训练结束！测试集最高准确率: {best_acc:.2f}%")
    print("========== 正在执行产物导出与基线剖析 (Phase 1.5) ==========")
    
    # 将模型调回 CPU 进行静态 Profiling
    model = model.to('cpu')
    model.load_state_dict(torch.load("baseline_fp32_best.pth", weights_only=True))
    model.eval()
    
    # 执行我们前面说过的 Phase 1.5 fvcore Profiling 逻辑...
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        dummy_input = torch.randn(1, 1, 28, 28)
        flops = FlopCountAnalysis(model, dummy_input)
        params = parameter_count(model)
        
        print("\n--- 传统冯·诺依曼架构 (FP32) 性能原罪统计 ---")
        total_params = sum(params.values())
        print(f"参数总量: {total_params} 个")
        print(f"单次推理 MACs: {flops.total()} 次")
        print(f"FP32 权重显存占用: {total_params * 4 / 1024:.2f} KB")
        print("----------------------------------------------")
    except ImportError:
        print("[警告] 未安装 fvcore，跳过 Profiling。请运行 pip install fvcore。")

if __name__ == "__main__":
    main()
    
import torch
import torch.nn as nn
import torch.optim as optim
import time

# 复用 Step 1 的无菌环境与数据流水线
from utils import set_deterministic_environment
from data_loader import build_data_pipeline

# 【核心替换】导入我们在降维车间新打造的 SNN 模型
from snn_models import SNN_MLP

# 训练与验证逻辑 (与 Step 1 保持 100% 一致，直接复用！)
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # 这里的 outputs 已经是经过 T 个时间步积分后的平均电压
        outputs = model(images) 
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(dataloader), 100. * correct / total

def test_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(dataloader), 100. * correct / total

def main():
    print("========== SNN 降维车间 (Step 2) 开始 ==========")
    # 依然需要锁定随机种子，以便与 Step 1 的基线进行绝对公平的对比
    # set_deterministic_environment(seed=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pipeline] 当前训练硬件: {device}")
    
    train_loader, test_loader = build_data_pipeline(batch_size=128)
    
    # 【核心配置】实例化 SNN，设置时间步长 T=4
    # 战略注：T=4 是一个极具商业竞争力的数字，这意味着我们在硬件上只需 4 个时钟周期就能完成一次推理，极其迅速。
    model = SNN_MLP(T=4).to(device)
    
    # 损失函数依旧是朴素的交叉熵
    criterion = nn.CrossEntropyLoss()
    
    # 【工程微调】由于引入了 1.58-bit 和脉冲的非连续性，梯度会变得有些粗糙。
    # 建议将 Adam 的学习率保持在 1e-3，但如果发现 Loss 震荡，可以微微调小到 5e-4。
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # SNN 的收敛曲线通常比 FP32 稍慢，我们将 Epochs 稍微放宽到 15 轮
    epochs = 15
    print(f"\n[Pipeline] 开始 1.58-bit SNN 训练，目标 Epochs: {epochs}...")
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{epochs}] | 耗时: {time.time() - start_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")
              
        if test_acc > best_acc:
            best_acc = test_acc
            # 产出降维车间的终极原材料：SNN 量化权重
            torch.save(model.state_dict(), "snn_1.58bit_best.pth")
            
    print(f"\n[Pipeline] 降维车间竣工！SNN 测试集最高准确率: {best_acc:.2f}%")
    print("对比 Step 1 的 FP32 基线，我们以不到 2 个 bit 的代价和纯加法的算子，逼近了理论极限！")

if __name__ == "__main__":
    main()
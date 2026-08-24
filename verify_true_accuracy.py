import torch
import torch.nn as nn
from data_loader import build_data_pipeline
from snn_models import SNN_MLP
from compiler_tvm_nir import collapse_to_physical_weight


def main():
    print("========== 真相实验室：绝对物理权重精度验证 ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载我们在 Step 2 训练好的“嫌疑”模型
    model = SNN_MLP(T=4)
    state_dict = torch.load("snn_1.58bit_best.pth", map_location="cpu")

    # 2. 暴力制裁：彻底杀死幽灵权重，获取纯正物理状态
    # 利用我们 Step 3 里的打补丁函数，强制转为 {-1, 0, 1} * scale
    true_w1 = collapse_to_physical_weight(state_dict['fc1.weight'])
    true_w2 = collapse_to_physical_weight(state_dict['fc2.weight'])

    # 强制覆写模型内存，现在模型里一丁点 FP32 的幽灵残余都没有了！
    state_dict['fc1.weight'] = torch.tensor(true_w1)
    state_dict['fc2.weight'] = torch.tensor(true_w2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. 跑一遍测试集，看看底裤还在不在
    _, test_loader = build_data_pipeline(batch_size=128)
    criterion = nn.CrossEntropyLoss()

    running_loss, correct, total = 0.0, 0, 0
    print("[Verify] 正在使用纯粹的 1.58-bit 物理权重进行 10000 张测试集推理...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # 这里的正向传播，吃进去的绝对是纯粹的物理坍缩权重
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    final_acc = 100. * correct / total
    print(f"\n[Verify] 暴力剥离后的真实 Test 准确率: {final_acc:.2f}%")
    print("顾问寄语：这个数字，就是您未来流片成功后，芯片能跑出的真实性能！")


if __name__ == "__main__":
    main()

import torch
import numpy as np
import os

# 复用我们绝对安全、决定性锁定的数据流水线与环境锁定
from utils import set_deterministic_environment
from data_loader import build_data_pipeline


def generate_golden_vectors(T=4, batch_size=10, output_dir="./"):
    print("========== 黄金测试向量 (Golden Vectors) 生成器 ==========")
    # 1. 锁定环境，确保每次生成的测试向量比特级一致
    set_deterministic_environment(seed=42)

    # 2. 拉取测试集，我们只需要拿一个 Batch（比如前 10 张图片）来做流式验证
    _, test_loader = build_data_pipeline(batch_size=batch_size)
    images, labels = next(iter(test_loader))
    print(f" -> 原始提取图像维度: {images.shape} (连续浮点像素)")

    # 将图像展平以匹配我们的 FCN 网络拓扑 [10, 1, 28, 28] -> [10, 784]
    images_flat = images.view(batch_size, -1)

    # 3. 【核心：泊松脉冲编码 (Poisson Rate Coding)】
    # 逻辑原理：像素值代表了神经元的发放概率。
    # 我们生成 [0, 1) 的均匀分布随机数，如果 像素值 > 随机数，则在当前 Tick 发放脉冲 (1)，否则为 0。
    spikes_list = []
    for t in range(T):
        # 每一个时钟周期，都要和一批全新的随机阈值比较
        rand_mat = torch.rand_like(images_flat)
        spike_at_t = (images_flat > rand_mat).float()
        spikes_list.append(spike_at_t)

    # 4. 堆叠时间维度：Shape 变为 [Time, Batch, Features] -> [4, 10, 784]
    spikes_tensor = torch.stack(spikes_list)

    # 极其重要：强制转换为 8位整型，剔除所有浮点痕迹，这是硬件唯一认识的语言
    spikes_np = spikes_tensor.numpy().astype(np.int8)
    labels_np = labels.numpy()

    # 5. 落盘验证与交付
    spikes_path = os.path.join(output_dir, "input_spikes_batch0.npy")
    labels_path = os.path.join(output_dir, "expected_output_labels.npy")

    np.save(spikes_path, spikes_np)
    np.save(labels_path, labels_np)

    print("\n--- 交付物生成完毕 ---")
    print(f"1. 脉冲输入流: {spikes_path}")
    print(f"   -> 形状: {spikes_np.shape} [T, Batch, Input_Dim]")
    print(f"   -> 数据校验 (仅应含0和1): {np.unique(spikes_np)}")
    print(f"2. 期望分类标签: {labels_path}")
    print(f"   -> 标签内容: {labels_np}")
    print("\n[商业动作] 请将这两个 .npy 文件连同 .nir 图纸一起打包发给硬件/模拟器团队！")


if __name__ == "__main__":
    generate_golden_vectors(T=4, batch_size=10)

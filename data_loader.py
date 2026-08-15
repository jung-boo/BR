import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def build_data_pipeline(batch_size=64, data_dir="./data"):
    """
    构建标准化的数据摄入流水线。
    战略要点：严格将张量值域控制在 [0.0, 1.0]，为未来的泊松脉冲编码（Poisson Encoding）铺路。
    """
    # 1. 定义数据变换 (Transforms)
    # 警告工程师：绝对不要在这里添加 transforms.Normalize() 将数据变成负数！
    # transforms.ToTensor() 会自动将 PIL 图片转化为 (C, H, W) 张量，并将像素值从 0-255 缩放到 [0.0, 1.0]
    baseline_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 2. 拉取/加载开源数据集 (MNIST)
    # 下载过程仅在第一次运行时触发
    print(f"[Pipeline] 正在拉取或验证 MNIST 数据集于目录: {data_dir} ...")
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=baseline_transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=baseline_transform
    )

    # 3. 实例化 DataLoader
    # 设置 num_workers 提升数据加载效率，pin_memory=True 加速数据向 GPU 的搬运
    # === data_loader.py 修复片段 ===
    # 引入我们刚才在 utils 写的函数
    from utils import seed_worker

    # 1. 创建一个显式的随机数生成器，绑定全局固定种子
    g = torch.Generator()
    g.manual_seed(42)  # 必须与 set_deterministic_environment 中的种子保持一致

    # 2. 实例化 DataLoader 时，严格注入生成器和 worker 种子初始化策略
    # 始终启用 num_workers=2 多进程加载（配合 seed_worker 保持确定性）；
    # pin_memory 仅在 CUDA 可用时开启，加速数据向 GPU 的搬运
    dataloader_kwargs = {'num_workers': 2, 'pin_memory': torch.cuda.is_available()}

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        worker_init_fn=seed_worker,  # <--- 修复注入点 1
        generator=g,                 # <--- 修复注入点 2
        **dataloader_kwargs
    )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False,               # 测试集不打乱，也就不需要 generator
        **dataloader_kwargs
    )


    print(f"[Pipeline] Phase 1.2 数据流水线构建完毕。")
    print(f" -> 训练集批次数量: {len(train_loader)} (Batch Size: {batch_size})")
    print(f" -> 测试集批次数量: {len(test_loader)} (Batch Size: {batch_size})")

    return train_loader, test_loader

# 供主函数调用的入口测试
if __name__ == "__main__":
    train_loader, test_loader = build_data_pipeline(batch_size=64)
    # 抓取一个 Batch 验证维度：应为 [64, 1, 28, 28] 和 [64]
    images, labels = next(iter(train_loader))
    print(f" -> 样本张量维度: {images.shape}, 最大值: {images.max().item()}, 最小值: {images.min().item()}")
import os
import random
import shutil
import subprocess
import numpy as np
import torch


def _has_nvidia_gpu():
    """
    通过 nvidia-smi 检测机器上是否存在 NVIDIA GPU (不依赖 torch 的 CUDA 支持)。
    若 nvidia-smi 不可用或调用失败, 返回 False。
    """
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def get_device():
    """
    自动检测可用训练硬件: CUDA GPU 优先, 否则回退 CPU。
    当机器存在 NVIDIA GPU 但 torch 为 CPU 版 (无 CUDA 支持) 时, 给出明确诊断,
    避免"有 GPU 却静默跑 CPU"的困惑。
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[Pipeline] 检测到 CUDA GPU: {name} (显存 {total_mem:.1f} GB)")
        return device

    if _has_nvidia_gpu():
        print("[Pipeline] 检测到 NVIDIA GPU, 但当前 torch 为 CPU 版本 (无 CUDA 支持)!")
        print("         -> 训练将回退到 CPU。如需使用 GPU, 请安装 CUDA 版 torch:")
        print("            pip install torch --index-url https://download.pytorch.org/whl/cu121")

    device = torch.device("cpu")
    print(f"[Pipeline] 未检测到可用 CUDA GPU, 训练将使用: {device}")
    return device


def seed_worker(worker_id):
    """
    DataLoader 多进程 worker 的随机种子初始化函数。
    确保每个 worker 获取不同的种子，但整体行为依然受主进程的全局种子控制。
    """
    # torch.initial_seed() 获取的是我们在主进程设定的全局种子 (如 42)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_deterministic_environment(seed):
    """
    强制锁定所有的随机种子与底层硬件的非确定性行为。
    这是后续评估 SNN 转换精度损失的绝对基石。
    """
    # 1. 锁定 Python 内置随机库
    random.seed(seed)
    
    # 2. 锁定 Numpy 随机种子
    np.random.seed(seed)
    
    # 3. 锁定 PyTorch CPU 随机种子
    torch.manual_seed(seed)
    
    # 4. 锁定 PyTorch GPU 随机种子（如果使用 CUDA）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 针对多卡情况
        
        # 5. 牺牲一部分 cudnn 加速性能，换取 100% 的可复现性
        # 禁止 cudnn 寻找最优卷积算法（这会导致每次运行结果有微小差异）
        torch.backends.cudnn.benchmark = False
        # 强制使用确定性算法
        torch.backends.cudnn.deterministic = True
        
    # 6. 锁定某些哈希相关的随机性 (针对 Python 3.3+)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"[Pipeline] Phase 1.1: 环境决定性已强制锁定 (Seed = {seed})")
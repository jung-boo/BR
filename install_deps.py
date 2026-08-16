#!/usr/bin/env python3
"""
按本机硬件自动安装依赖 (requirements.txt 的智能执行器):

  · 检测到 NVIDIA GPU (nvidia-smi)  -> 安装 CUDA 版 torch (cu121)
  · 未检测到 GPU                    -> 安装 CPU 版 torch

用法:
  python install_deps.py             # 自动检测硬件并安装
  python install_deps.py --dry-run   # 只打印将要执行的命令, 不实际安装
  python install_deps.py --force cpu # 强制 CPU 分支 (调试/无 GPU 机器)
  python install_deps.py --force cuda# 强制 CUDA 分支 (调试)
"""
import argparse
import pathlib
import shutil
import subprocess
import sys

TORCH_PKGS = {
    "cuda": ["torch==2.2.2+cu121", "torchvision==0.17.2+cu121"],
    "cpu": ["torch==2.2.2", "torchvision==0.17.2"],
}
CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
# CPU 版直接使用 PyPI 默认源 (Windows/macOS 上 PyPI 的 torch 即 CPU 轮子)
CPU_INDEX = None


def has_nvidia_gpu():
    """检测本机是否存在 NVIDIA GPU (不依赖 torch, 通过 nvidia-smi)。"""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def read_common_requirements():
    """
    读取 requirements.txt 中除 torch/torchvision 之外的通用依赖,
    避免脚本内重复维护版本号 (单一事实来源)。
    """
    req_file = pathlib.Path(__file__).resolve().parent / "requirements.txt"
    pkgs = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        # 去掉行内注释 (如 "numpy==1.26.4  # 说明")
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("--"):
            continue
        name = line.split(";")[0].split("==")[0].strip().lower()
        if name in ("torch", "torchvision"):
            continue
        pkgs.append(line)
    return pkgs


def build_commands(branch):
    """构造 pip 安装命令列表: 先装通用依赖, 再装 torch/torchvision。"""
    pip = [sys.executable, "-m", "pip", "install"]

    cmds = [pip + read_common_requirements()]

    torch_cmd = pip + TORCH_PKGS[branch]
    if branch == "cuda":
        torch_cmd += ["--index-url", CUDA_INDEX]
    cmds.append(torch_cmd)
    return cmds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="强制选择安装分支 (默认 auto: 按 nvidia-smi 检测)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印命令, 不实际安装"
    )
    args = parser.parse_args()

    branch = args.force
    if branch == "auto":
        branch = "cuda" if has_nvidia_gpu() else "cpu"

    gpu_msg = "[GPU] 检测到 NVIDIA GPU" if branch == "cuda" else "[NO-GPU] 未检测到 NVIDIA GPU"
    print(f"[install_deps] {gpu_msg} -> 安装 {branch.upper()} 版 torch/torchvision")

    for cmd in build_commands(branch):
        print(f"[install_deps] 执行: {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.check_call(cmd)

    print("[install_deps] 依赖安装完成。")
    if branch == "cuda":
        print("[install_deps] 提示: 安装后可运行 "
              "python -c \"import torch; print(torch.cuda.is_available())\" 确认 GPU 可用。")


if __name__ == "__main__":
    main()

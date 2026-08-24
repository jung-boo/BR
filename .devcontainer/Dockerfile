# 使用纯净的 Ubuntu 22.04 官方镜像
FROM ubuntu:22.04

# ---------------------------------------------------------
# 1. 基础系统环境与 APT 镜像源配置
# ---------------------------------------------------------
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata && \
    apt-get clean

# ---------------------------------------------------------
# 2. 底层 C/C++ 构建、交叉编译与 RTL 仿真工具链
# ---------------------------------------------------------
# 包含：标准 GCC/G++、CMake、针对 MCU 的 ARM 裸机工具链，以及用于 RTL IP 核仿真的 Verilator
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake ninja-build git wget curl unzip tar \
    gcc-arm-none-eabi binutils-arm-none-eabi gdb-multiarch \
    verilator \
    python3 python3-pip python3-dev && \
    apt-get clean

# ---------------------------------------------------------
# 3. Apache TVM 编译器底层依赖
# ---------------------------------------------------------
# 包含：LLVM、Clang (计算图融合与代码生成必备)
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    llvm clang \
    libgtest-dev libtinfo-dev zlib1g-dev && \
    apt-get clean

# ---------------------------------------------------------
# 4. 配置 Python 全局环境与核心深度学习框架
# ---------------------------------------------------------
RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip3 install --upgrade pip 

# ---------------------------------------------------------
# 2. 安装 Miniconda3 并配置环境变量
# ---------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p ${CONDA_DIR} && \
    rm miniconda.sh

# 将 Conda 直接注入系统的全局环境变量，随时可用
ENV PATH=${CONDA_DIR}/bin:${PATH}




# ---------------------------------------------------------
# 3. 写入 Conda 清华源与 pip 阿里源
# ---------------------------------------------------------
RUN echo "channels:" >> ~/.condarc && \
    echo "  - defaults" >> ~/.condarc && \
    echo "show_channel_urls: true" >> ~/.condarc && \
    echo "default_channels:" >> ~/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main" >> ~/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r" >> ~/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2" >> ~/.condarc && \
    echo "custom_channels:" >> ~/.condarc && \
    echo "  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
    echo "  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc


# ---------------------------------------------------------
# 4. 配置基础开发环境 (自带 Python 3.10 及深度学习、计算图核心包)
# ---------------------------------------------------------
# 使用 Conda 安装底层 GPU 框架，用 pip 安装上层算法包

# ---------------------------------------------------------
WORKDIR /workspace
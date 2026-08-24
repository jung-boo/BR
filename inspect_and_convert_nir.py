import nir
import numpy as np


def inspect_and_convert_nir(nir_path="snn_mvp_model.nir"):
    # 1. 读取 NIR 图纸
    graph = nir.read(nir_path)

    # 2. 提取第一层全连接层的权重张量 (包含 -0.72, 0, 0.72)
    w_scaled = graph.nodes['fc1'].weight

    # 3. 统计唯一值，查看当前的真实状态
    unique_vals = np.unique(np.round(w_scaled, decimals=4))
    print(f"NIR 原始权重唯一值: {unique_vals}")

    # 4. 逆向提取 Scale (缩放因子)
    scale = np.max(np.abs(w_scaled))
    print(f"提取出的量化 Scale: {scale:.4f}")

    # 5. 除以 Scale 并四舍五入，还原出极其纯粹的 {-1, 0, 1}
    w_ternary = np.round(w_scaled / scale).astype(np.int8)

    # 6. 【物理映射核心】裂变为兴奋性 (Pos) 和抑制性 (Neg) 两个纯 {0, 1} 阵列
    w_pos = (w_ternary > 0).astype(np.int8)
    w_neg = (w_ternary < 0).astype(np.int8)

    print("\n--- 硬件 SRAM 烧录准备完成 ---")
    print(f"兴奋阵列 (W_pos) 唯一值: {np.unique(w_pos)} | 形状: {w_pos.shape}")
    print(f"抑制阵列 (W_neg) 唯一值: {np.unique(w_neg)} | 形状: {w_neg.shape}")

    # 注意：这里的 scale 怎么处理？
    # 在硬件层，我们只需把下一层神经元 (IFNode) 的发射阈值 (V_threshold)
    # 除以这个 scale，数学上就完全等效了。硬件内部彻底告别小数。


if __name__ == "__main__":
    inspect_and_convert_nir()

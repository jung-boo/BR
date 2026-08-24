import torch
import torch.nn as nn
# TVM 相关导入已注释: Windows 官方 wheel (0.25/0.26) 不含 relay 前端,
# 且当前 NIR 产物由物理权重坍缩直接构建, 不依赖 TVM。
# 若未来在具备完整 TVM (含 relay) 的环境运行, 取消注释即可:
#   import tvm
#   from tvm import relay
import nir
import numpy as np

# 1. 替 TVM "手工剥离"了时间外壳的单步死网络
class SingleStepSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def collapse_to_physical_weight(latent_weight):
    """
    【核心补丁加在这里】：斩断幽灵权重！
    在写入 NIR 图纸前，强制重演前向传播的截断逻辑，将浮点数彻底坍缩为 {-1, 0, 1} * scale
    """
    scale = latent_weight.abs().mean().clamp(min=1e-5)
    physical_w = torch.round(latent_weight / scale).clamp(-1, 1) * scale
    return physical_w.detach().numpy()

def main():
    print("========== SNN 物理映射器 (Step 3) 开始 ==========")
    
    # Phase 3.1: 静态图准备与权重注入
    model = SingleStepSNN()
    state_dict = torch.load("snn_1.58bit_best.pth", map_location="cpu")
    clean_dict = {
        'fc1.weight': state_dict['fc1.weight'], 'fc1.bias': state_dict['fc1.bias'],
        'fc2.weight': state_dict['fc2.weight'], 'fc2.bias': state_dict['fc2.bias']
    }
    model.load_state_dict(clean_dict)
    model.eval()
    
    # Phase 3.2: TVM 图捕获与基础融合 (已注释: Windows 官方 wheel 不含 relay 前端)
    # 若未来在具备完整 TVM (含 relay) 的环境运行, 取消注释即可启用:
    #
    # print("[Compiler] 正在通过 Apache TVM 梳理计算图...")
    # dummy_input = torch.randn(1, 1, 28, 28)
    # scripted_model = torch.jit.trace(model, dummy_input)
    #
    # shape_list = [("input0", dummy_input.shape)]
    # mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    #
    # seq = tvm.transform.Sequential([
    #     relay.transform.SimplifyInference(),
    #     relay.transform.FoldConstant()
    # ])
    # with tvm.transform.PassContext(opt_level=3):
    #     mod = seq(mod)
    # print("[Compiler] TVM Relay IR 优化完成 (算子无冗余)。")

    # Phase 3.3: 提取真实物理权重并构建 NIR 图纸
    print("[Compiler] 正在翻译为 NIR 标准格式并物理坍缩权重...")
    
    # 【核心应用点】：在真正写入 w1 和 w2 时，调用坍缩函数！
    w1 = collapse_to_physical_weight(clean_dict['fc1.weight'])
    b1 = clean_dict['fc1.bias'].detach().numpy()
    
    w2 = collapse_to_physical_weight(clean_dict['fc2.weight'])
    b2 = clean_dict['fc2.bias'].detach().numpy()

    # 防御性自检：打印确保权重只有少量的几个值，如果出来是 [-0.12 0. 0.12] 这种就对了！
    print(f" -> FC1 物理权重唯一值校验: {np.unique(np.round(w1, decimals=4))}")
    
    # 将纯正的三值权重烧入 NIR 图纸
    nodes = {
        "input": nir.Input(input_type=np.array([784])),
        "fc1": nir.Affine(weight=w1, bias=b1),
        "if1": nir.IF(r=np.ones(256), v_threshold=np.ones(256)),
        "fc2": nir.Affine(weight=w2, bias=b2),
        "output": nir.Output(output_type=np.array([10]))
    }
    
    edges = [("input", "fc1"), ("fc1", "if1"), ("if1", "fc2"), ("fc2", "output")]
    
    nir_graph = nir.NIRGraph(nodes=nodes, edges=edges)
    nir.write("snn_mvp_model.nir", nir_graph)
    
    print("\n[Compiler] 映射成功！终极图纸已保存为 snn_mvp_model.nir")

if __name__ == "__main__":
    main()

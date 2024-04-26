import torch
import torch.nn as nn


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def forward(self, x):
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型实例
model = IdentityModel()
model.eval().to(device)

# path = r'test.pt'
# 输出模型结构
# torch.save(model, path)
#
# test = torch.load(path)
# backbone_example = torch.randn([1, 3, 255, 255], device=device)
backbone_example = torch.randint(low=0, high=255, size=[1, 3, 255, 255], device=device, dtype=torch.uint8)
# backbone_example_onnx = 'test1.pt'
# trace_model = torch.jit.trace(test, backbone_example)
# trace_model.save(backbone_example_onnx)

# backbone_example = torch.randn([1, 3, 256, 256], device=device)
backbone_example_onnx = 'example.onnx'
torch.onnx.export(model, backbone_example, backbone_example_onnx, input_names=['input'], output_names=['output'],
                  verbose=True, opset_version=10)
import torch
import torch.nn as nn


class IdentityModel(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=False, d=1):
        super(IdentityModel, self).__init__()

        self.conv = nn.Conv2d(c1, c2, k)

        if c1 == c2:
            # 简单的初始化，但注意这并不会导致严格的恒等映射
            nn.init.constant_(self.conv.weight, 1.0 / c1)
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class SimpleResizeModel(nn.Module):
    def __init__(self):
        super(SimpleResizeModel, self).__init__()
        # 使用 AdaptiveAvgPool2d 来实现形状的变换
        # 注意：这不是学习得到的，只是一个固定的变换
        self.pool = nn.AdaptiveAvgPool2d((224, 224))

    def forward(self, x):
        # 将输入张量通过 pool 层，得到输出张量
        return self.pool(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型实例
size = 32
model = IdentityModel(size, size)
model.eval().to(device)

# path = r'test.pt'
# 输出模型结构
# torch.save(model, path)
#
# test = torch.load(path)
backbone_example = torch.randn(1, size, 277, 960).to(device)
output_tensor = model(backbone_example)
# backbone_example = torch.randint(low=0, high=255, size=[1, 16, 544, 960], device=device, dtype=torch.uint8)
# backbone_example_onnx = 'test1.pt'
# trace_model = torch.jit.trace(test, backbone_example)
# trace_model.save(backbone_example_onnx)

# backbone_example = torch.randn([1, 3, 256, 256], device=device)
backbone_example_onnx = 'resize.onnx'
torch.onnx.export(model, backbone_example, backbone_example_onnx, input_names=['images'], output_names=['output0'],
                  verbose=True, opset_version=11)

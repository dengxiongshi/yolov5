# Profile
import torch
import torch.nn as nn
# from torchsummary import summary
# from torchstat import stat
from torchinfo import summary as su
from ptflops import get_model_complexity_info
import onnx_tool

from models.common import DetectMultiBackend
from models.experimental import attempt_load
from models.yolo import Model
from utils.torch_utils import profile, model_info
import numpy as np

# m1 = nn.ConvTranspose2d(512, 512, 2, 2, 0, 0, 512)
# m2 = nn.ConvTranspose2d(512, 512, 3, 2, 1, 1, 512)
# m3 = nn.ConvTranspose2d(512, 512, 4, 2, 1, 0, 512)
# m4 = nn.ConvTranspose2d(512, 512, 5, 2, 2, 1, 512)
# m5 = nn.ConvTranspose2d(512, 512, 8, 2, 3, 0, 512)
# m6 = nn.ConvTranspose2d(512, 512, 2, 2, 0)
#
# results = profile(input=torch.randn(16, 512, 80, 80), ops=[m1, m2, m3, m4, m5, m6], n=30)
#
# labels_out = np.zeros((3,))
#
# labels = np.array([[          0], [          2]])
# labels_out[labels] = 1


weights = r"/data/yolov5/runs/train/slim/yolov5s-0.6-pruned_202409153/weights/best.pt"
cfg = r'models/yolov5s.yaml'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# model = Model(cfg).to(device)
# cskp = torch.load(model_path, map_location=device)
# model.load_state_dict(cskp['model'], strict=False)
#
# model = model['model'].cuda(device)
# for p in model.parameters():
#     p.requires_grad_(True)

# np, n_g, fs = model_info(model, imgsz=640)
# n_g = n_g / 1e9 * 2

im = torch.rand(1, 3, 640, 640).to(device)
# result = profile(input=im, ops=[model], n=3, device=device)

# model = attempt_load(weights=model_path, device=device)
model = DetectMultiBackend(weights=weights, device=device)
for p in model.parameters():
    p.requires_grad_(True)

if weights.endswith(".pt"):
    # ptflops
    print("==========ptflops==========\n")
    macs, params = get_model_complexity_info(model=model, input_res=(3, 640, 640), as_strings=True, backend='pytorch',
                                             print_per_layer_stat=True, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # torchstat
    # print("==========torchstat==========\n")
    # stat(model=model.to('cpu'), input_size=(3, 640, 640))
    # torchsummary
    # print("==========torchsummary==========\n")
    # summary(model, input_size=(3, 640, 640), batch_size=1)
    # torchinfo
    print("===================torchinfo========================\n")
    su(model=model, input_data=im, device=device, verbose=2, depth=4)

elif weights.endswith(".onnx"):
    onnx_tool.model_profile(weights)
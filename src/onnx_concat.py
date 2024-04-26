import torch
from utils.general import check_requirements
import onnxruntime
import onnx
import onnx.helper as helper
from onnx.helper import TensorProto, GraphProto, AttributeProto
import numpy as np

model_file = r"D:\python_work\yolov5\runs\train\WI_PRW_SSW_SSM_20231127\weights\AnFang_2023_11_29.onnx"
save_file = r"D:\python_work\yolov5\runs\train\WI_PRW_SSW_SSM_20231127\weights\test.onnx"

device = torch.device('cuda:0')

model = onnx.load(model_file)
# model = torch.load(model_file, map_location=device)
node_list = model.graph.node
input_list = model.graph.input
output_list = model.graph.output

# for i in range(len(node_list)):
#     if node_list[i].name == 'Transpose_209':
#         print(i)
#         print(node_list[i])
#         node_rise = node_list[i]
#         if node_rise.output[0] == 'output0':
#             print(i)

# for i in range(len(output_list)):
#     del output_list[0]

# 输入数据
input_data = np.array([1, 3, 12, 20, 10], dtype=np.float32)

# 目标形状
target_shape = (1, 11520, 10)

reshape_node = helper.make_node(
    'Reshape',
    inputs=['output0'],
    outputs=['Reshape363'],
    shape=target_shape
)

# 创建 Graph
# graph_def = onnx.helper.make_graph(
#     [reshape_node],
#     'ReshapeGraph',
#     [onnx.helper.make_tensor_value_info('363', onnx.TensorProto.FLOAT, [5])],
#     [onnx.helper.make_tensor_value_info('Reshape363', onnx.TensorProto.FLOAT, target_shape)],
# )

#
# node_list.insert(node_list[198], reshape_node)

# for output in output_list:
#     d = output.type.tensor_type.shape.dim
#     print(d)

output0 = helper.make_tensor_value_info('Reshape363', TensorProto.FLOAT, [1, 11520, 10])
output_list[0].CopyFrom(output0)
node_list[198].CopyFrom(reshape_node)

# output0 = helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1, 15120, 10])
# model.graph.output.append(output0)

onnx.checker.check_model(model)
onnx.save_model(model, save_file)






# cuda = torch.cuda.is_available() and device.type != 'cpu'
# check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
#
# providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
# session = onnxruntime.InferenceSession(model_file, providers=providers)
# output_names = [x.name for x in session.get_outputs()]
# meta = session.get_modelmeta().custom_metadata_map  # metadata
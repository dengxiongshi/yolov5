import numpy as np
import onnxruntime as ort


def cosine_similariy(x1, x2):
    # 求模
    x1_norm = np.sqrt(np.sum(np.square(x1)))
    x2_norm = np.sqrt(np.sum(np.square(x2)))
    # 内积
    x1_x2 = np.sum(np.multiply(x1, x2))
    # 余弦相似度
    cosin = x1_x2 / (x1_norm * x2_norm)

    print("x1_norm=", x1_norm)
    print("x2_norm=", x2_norm)
    print("x1_x2=", x1_x2)
    print("cosin={}".format(cosin))


def euclidean_distance(x1, x2):
    d0 = np.sqrt(np.sum(np.square(x1 - x2)))
    print("ed={}".format(d0))


def onnx_inference(model_path_, input_data_):
    if len(input_data_.shape) == 3:
        C, H, W = input_data_.shape
        input_data_ = input_data_.reshape(1, C, H, W)
    sess = ort.InferenceSession(model_path_, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    for idx in range(len(sess.get_inputs())):
        print((sess.get_inputs()[idx].name, sess.get_inputs()[idx].shape))
    for idx in range(len(sess.get_outputs())):
        print((sess.get_outputs()[idx].name, sess.get_outputs()[idx].shape))
    output_data_ = sess.run([out_name], {input_name: input_data_})[0]
    return output_data_


if __name__ == '__main__':
    # root_path = '/media/zhaoxinyu/a2a41d26-6d73-4a0d-bc7c-a8c42f8f04e41/NNTOOLS/Noviac/nvtai30/dt/'
    model_path = r"D:\python_work\yolov5\runs\train\yolov5s_xsmall_autoanchpr_conv_half_20240107\weights\AnFang_20240109.onnx"
    input_data_path = r"E:\downloads\compress\datasets\FEEDS\WI_PRW_SSM_0322_v2\quantity_img_bin\img04606.bin"

    input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(1, 3, 384, 640)
    onnx_output = onnx_inference(model_path, input_data)

    sim_outout_path = r"D:\python_work\yolov5\runs\train\yolov5s_xsmall_autoanchpr_conv_half_20240107\weights\out.bin"
    # sim_outout_path = root_path + 'output/' + pattern_name + '/debug/sim_layer/float/Conv_out_Y.bin'
    sim_output = np.fromfile(sim_outout_path, dtype=np.float32).reshape(1, 61200, 8)

    print('cosine: ')
    cosine_similariy(onnx_output, sim_output)

    print('euclidean: ')
    euclidean_distance(onnx_output, sim_output)




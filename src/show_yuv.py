"""
使用BT-601NARROW标准，将yuv420转成bgr
"""
import os.path
import numpy as np
import cv2


def yuv420sp2rgb(ori_data):
    H, W = 1520, 2688
    H_2 = int(H / 2)
    y_data = ori_data[0:1520, :].astype(np.int32)
    uv_data = ori_data[1520:, :].astype(np.int32)
    data = np.zeros([3, H, W]).astype(np.int32)
    r0c0 = 298
    r0c1 = 409
    r0c2 = 0

    r1c0 = 298
    r1c1 = -100
    r1c2 = -208

    r2c0 = 298
    r2c1 = 0
    r2c2 = 516

    input_bias_0 = 16
    input_bias_1 = 128
    input_bias_2 = 128

    data[0, 0:H:2, 0:W:2] = r0c0 * (y_data[0:H:2, 0:W:2] - input_bias_0) + r0c1 * (uv_data[:, 1:W:2] - input_bias_2)
    data[0, 0:H:2, 1:W:2] = r0c0 * (y_data[0:H:2, 1:W:2] - input_bias_0) + r0c1 * (uv_data[:, 1:W:2] - input_bias_2)
    data[0, 1:H:2, 0:W:2] = r0c0 * (y_data[1:H:2, 0:W:2] - input_bias_0) + r0c1 * (uv_data[:, 1:W:2] - input_bias_2)
    data[0, 1:H:2, 1:W:2] = r0c0 * (y_data[1:H:2, 1:W:2] - input_bias_0) + r0c1 * (uv_data[:, 1:W:2] - input_bias_2)

    data[1, 0:H:2, 0:W:2] = r1c0 * (y_data[0:H:2, 0:W:2] - input_bias_0) + r1c1 * (uv_data[:, 0:W:2] - input_bias_1) + r1c2 * (uv_data[:, 1:W:2] - input_bias_2)
    data[1, 0:H:2, 1:W:2] = r1c0 * (y_data[0:H:2, 1:W:2] - input_bias_0) + r1c1 * (uv_data[:, 0:W:2] - input_bias_1) + r1c2 * (uv_data[:, 1:W:2] - input_bias_2)
    data[1, 1:H:2, 0:W:2] = r1c0 * (y_data[1:H:2, 0:W:2] - input_bias_0) + r1c1 * (uv_data[:, 0:W:2] - input_bias_1) + r1c2 * (uv_data[:, 1:W:2] - input_bias_2)
    data[1, 1:H:2, 1:W:2] = r1c0 * (y_data[1:H:2, 1:W:2] - input_bias_0) + r1c1 * (uv_data[:, 0:W:2] - input_bias_1) + r1c2 * (uv_data[:, 1:W:2] - input_bias_2)

    data[2, 0:H:2, 0:W:2] = r2c0 * (y_data[0:H:2, 0:W:2] - input_bias_0) + r2c2 * (uv_data[:, 0:W:2] - input_bias_1)
    data[2, 0:H:2, 1:W:2] = r2c0 * (y_data[0:H:2, 1:W:2] - input_bias_0) + r2c2 * (uv_data[:, 0:W:2] - input_bias_1)
    data[2, 1:H:2, 0:W:2] = r2c0 * (y_data[1:H:2, 0:W:2] - input_bias_0) + r2c2 * (uv_data[:, 0:W:2] - input_bias_1)
    data[2, 1:H:2, 1:W:2] = r2c0 * (y_data[1:H:2, 1:W:2] - input_bias_0) + r2c2 * (uv_data[:, 0:W:2] - input_bias_1)

    out = np.zeros([3, 1520, 2688]).astype(np.float32)
    out = data / 256 / 256
    return out


if __name__ == '__main__':
    path = '/media/zhaoxinyu/a2a41d26-6d73-4a0d-bc7c-a8c42f8f04e41/workspace/net/anfang/data/2024_01_12'
    file_name = '2024_01_12_14_29_50_49_yuv'
    file_t = os.path.join(path, file_name + ".bin")
    yuv_data = np.fromfile(file_t, dtype=np.uint8).reshape(2280, 2688).astype(np.uint8)
    res = yuv420sp2rgb(yuv_data.copy()).astype(np.float32)
    # res = (res - 0.403921569) / 0.225
    pic = cv2.merge(res)

    # file_t = os.path.join(path, "af_bgr01.bin")
    # bgr_data = np.fromfile(file_t, dtype=np.float32).reshape(3, 416, 416)
    # pic = cv2.merge(bgr_data)
    # pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    #
    path = '/media/zhaoxinyu/a2a41d26-6d73-4a0d-bc7c-a8c42f8f04e41/workspace/net/anfang/data/2024_01_12/{}.jpg'.format(file_name)
    cv2.imwrite(path, pic * 255)
    cv2.imshow('pic', pic)
    cv2.waitKey()
    cv2.destroyAllWindows()
"""
使用BT-601NARROW标准，将yuv420转成bgr
"""
import glob
import os.path
import subprocess

import numpy as np
import cv2
from tqdm import tqdm


def yuv420sp2bgr(ori_data, target_size):
    H, W = target_size
    H_2 = int(H / 2)
    y_data = ori_data[0:H, :].astype(np.int32)
    uv_data = ori_data[H:, :].astype(np.int32)
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

    out = np.zeros([3, H, W]).astype(np.float32)
    out = data / 256 / 256
    return out


def bgr2yuv420sp(ori_data):
    C, H, W = ori_data.shape
    YUV_H = H * 3 // 2
    dst = np.zeros((YUV_H, W))

    r_data = ori_data[0, :, :]
    g_data = ori_data[1, :, :]
    b_data = ori_data[2, :, :]

    y_data = ((66 * r_data + 129 * g_data + 25 * b_data + 128) >> 8) + 16
    u_data = ((-38 * r_data - 74 * g_data + 112 * b_data + 128) >> 8) + 128
    v_data = ((112 * r_data - 94 * g_data - 18 * b_data + 128) >> 8) + 128

    dst[0:H, :] = y_data.clip(0, 255)
    dst[H:, 0:W:2] = u_data[0:H:2, 0:W:2].clip(0, 255)
    dst[H:, 1:W:2] = v_data[0:H:2, 0:W:2].clip(0, 255)

    return dst.astype(np.uint8)


def ffmpeg_yuv2jpg(yuv_path, save_path, target_size):
    h, w = target_size
    ffmpeg_command = [
        'ffmpeg',
        '-s', f'{w}x{h}',
        '-i', yuv_path,
        '-frames:v', f'{1}',
        save_path
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f'YUV转换成功，图片保存在 {save_path}')
    except subprocess.CalledProcessError as e:
        print(f'YUV转换失败: {e}')



img_dir = r"C:\Users\dengxs\Desktop\dengxs\20241012\vi"

save_dir = img_dir

h, w = 1080, 1920
target_size = (h, w)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_list = glob.glob(img_dir + "/*.yuv")

pbar = tqdm(img_list, desc=f'Converting {img_dir}')

for p in pbar:

    img_name = os.path.basename(p)
    name = os.path.splitext(img_name)[0]

    save_name = os.path.join(save_dir, name + ".jpg")
    # yuv = np.fromfile(p, dtype=np.uint8)
    # data = yuv.reshape((int(h + h / 2), w))
    #
    # bgr = cv2.cvtColor(data, cv2.COLOR_YUV420SP2BGR)

    # yuv_image = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    # bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV420SP2BGR)

    # res = yuv420sp2bgr(data, target_size)
    # pic = cv2.merge(res) * 255
    # pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

    # cv2.imwrite(save_name, bgr)

    ffmpeg_yuv2jpg(p, save_name, target_size)




from tqdm import tqdm
import glob
import os
from PIL import Image
import numpy as np


def resize_and_save(input_path, output_path, target_size):
    # 打开图像
    img = Image.open(input_path)

    # 调整大小
    img_resized = img.resize(target_size, Image.ANTIALIAS)

    # 将图像转换为 NumPy 数组
    img_array = np.array(img_resized, dtype=np.float32) / 255.0  # 归一化到 [0, 1] 范围

    # 将数组保存为二进制文件
    img_array.tofile(output_path)

img_dir = r"E:\downloads\compress\datasets\FEEDS\WI_PRW_SSM_0322_v2\quantity_img"

save_dir = r"E:\downloads\compress\datasets\FEEDS\WI_PRW_SSM_0322_v2\quantity_img_bin"

target_image_size = (384, 640)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_list = glob.glob(img_dir + "/*.jpg")

pbar = tqdm(img_list, desc=f'Converting {img_dir}')

for p in pbar:

    img_name = os.path.basename(p)
    name = os.path.splitext(img_name)[0]

    save_name = os.path.join(save_dir, name + ".bin")
    img = Image.open(p)

    # 调整大小
    img_resized = img.resize(target_image_size, Image.ANTIALIAS)

    # 将图像转换为 NumPy 数组
    img_array = np.array(img_resized, dtype=np.float32) / 255.0  # 归一化到 [0, 1] 范围

    # 将数组保存为二进制文件
    img_array.tofile(save_name)


    # resize_and_save(p, save_name, target_image_size)
    # im = Image.open(p)
    # im_resize = im.resize((w, h))

    # with open(p, 'rb') as file:
    #     image_data = file.read()

    # with open(save_name, 'wb') as file:
    #     file.write(im_resize)
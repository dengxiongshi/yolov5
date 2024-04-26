import cv2
import numpy as np
import glob
from tqdm import tqdm

image_folder = r'D:\python_work\sim_detection\datasets\car-11\out\Head_2\nanotrack_om'
video_name = r'D:\python_work\sim_detection\datasets\car-11\out\\nanotrack_om.mp4'

# 查找图片文件并进行排序
images = sorted(glob.glob(image_folder + '/*.jpg'))

# 读取第一张图片以获取宽度和高度信息
img = cv2.imread(images[0])
height, width, _ = img.shape

# 创建视频编码器
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

pbar = tqdm(images, desc=f'Converting: {image_folder}')

# 逐帧将图片写入视频
for image in pbar:
    img = cv2.imread(image)
    video.write(img)

# 释放资源
video.release()
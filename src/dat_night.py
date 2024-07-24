import cv2
import numpy as np
from sklearn.cluster import KMeans


def is_daytime(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # 计算全局平均亮度
    global_mean_brightness = np.mean(hsv[:, :, 2])

    # 局部亮度分析：检测图像中的高亮区域
    bright_areas = cv2.inRange(hsv, (0, 0, 200), (255, 255, 255))
    bright_ratio = np.sum(bright_areas) / (bright_areas.shape[0] * bright_areas.shape[1])

    # 边缘检测
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])

    # 检测天空区域
    sky_mask = cv2.inRange(hsv, (90, 0, 0), (140, 255, 255))
    sky_ratio = np.sum(sky_mask) / (sky_mask.shape[0] * sky_mask.shape[1])

    # 阈值
    global_brightness_threshold = 50
    bright_area_threshold = 0.01
    edge_density_threshold = 0.02
    sky_threshold = 0.1

    if (global_mean_brightness > global_brightness_threshold or bright_ratio > bright_area_threshold) and (
            edge_density > edge_density_threshold or sky_ratio > sky_threshold):
        return "Daytime"
    else:
        return "Nighttime"


# 测试代码
image_path = r"F:\BaiduNetdiskDownload\Data\train\weather_00042.jpg"
result = is_daytime(image_path)
print(result)

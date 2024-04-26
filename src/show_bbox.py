import cv2
import glob
import matplotlib.pyplot as plt
import os
import numpy as np

import xml.etree.ElementTree as ET
from tqdm import tqdm


# 由原标签得出检测矩形框左上角和右下角的坐标分别为：xmin,ymin,xmax,ymax
def Xmin_Xmax_Ymin_Ymax(img_path, txt_path):
    """
    :param img_path: 图片文件的路径
    :param txt_path: 标签文件的路径
    :return:
    """
    img = cv2.imread(img_path)
    # 获取图片的高宽
    h, w, _ = img.shape
    # 读取TXT文件 中的中心坐标和框大小
    with open(txt_path, "r") as fp:
        # 以空格划分
        contline = fp.readline().split(' ')
    if len(contline) > 1:
        # 计算框的左上角坐标和右下角坐标,使用strip将首尾空格去掉
        xmin = float((contline[1]).strip()) - float(contline[3].strip()) / 2
        xmax = float(contline[1].strip()) + float(contline[3].strip()) / 2

        ymin = float(contline[2].strip()) - float(contline[4].strip()) / 2
        ymax = float(contline[2].strip()) + float(contline[4].strip()) / 2

        # 将坐标（0-1之间的值）还原回在图片中实际的坐标位置
        xmin, xmax = w * xmin, w * xmax
        ymin, ymax = h * ymin, h * ymax

        return (contline[0], xmin, ymin, xmax, ymax)
    else:
        return (0, 0, 0, 2, 2)


# 坐标转换
def xywh2xyxy(x, w1, h1, img):
    label, x, y, w, h = x
    # print("原图宽高:\nw1={}\nh1={}".format(w1, h1))
    # 边界框反归一化
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1
    # print("反归一化后输出：\n第一个:{}\t第二个:{}\t第三个:{}\t第四个:{}\t\n\n".format(x_t, y_t, w_t, h_t))
    # 计算坐标
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2

    # print('标签:{}'.format(labels[int(label)]))
    # print("左上x坐标:{}".format(top_left_x))
    # print("左上y坐标:{}".format(top_left_y))
    # print("右下x坐标:{}".format(bottom_right_x))
    # print("右下y坐标:{}".format(bottom_right_y))
    color = (0, 255, 0)
    # 绘制矩形框
    cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), color, 2)
    """
    # (可选)给不同目标绘制不同的颜色框
    if int(label) == 0:
       cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), (0, 255, 0), 2)
    elif int(label) == 1:
       cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), (255, 0, 0), 2)
    """
    return img


def draw_boxes(image_path, label_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 读取标签文件
    with open(label_path, 'r') as source:
        lines = [line.strip() for line in source.readlines() if line.strip()]

    for line in lines:
        # 解析YOLO标签信息
        parts = line.strip().split(' ')
        class_id, x_center, y_center, width, height = map(float, parts)

        # 计算矩形框的坐标
        h, w, _ = image.shape
        x = int((x_center - width / 2) * w)
        y = int((y_center - height / 2) * h)
        box_width = int(width * w)
        box_height = int(height * h)

        # 画矩形框
        color = (0, 255, 0)  # Green color
        thickness = 2
        cv2.rectangle(image, (x, y), (x + box_width, y + box_height), color, thickness)

    # 保存绘制后的图像
    cv2.imwrite(output_path, image)


# 根据label坐标画出目标框
def plot_tangle(image_path, image_label, save_dir):
    raw_data_path_list = glob.glob(image_path + '/*.jpg')

    # save_dir = image_path.replace('images', 'images_label')
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    pbar = tqdm(raw_data_path_list, desc=f'Converting {image_path}')  # 进度条
    # with alive_bar(len(raw_data_path_list)) as bar:
    for f in pbar:
        img_path = f

        base_name = os.path.basename(img_path)
        # print(os.path.splitext(base_name)[0])
        # temp = img_path.replace('images', 'labels')
        temp = os.path.join(image_label, base_name[0:-4] + ".txt")
        # output = img_path.replace('images', 'images_label')

        output = os.path.join(save_dir, base_name)
        draw_boxes(img_path, temp, output)

            # bar()


def txt_bbox(image_path, label_path, output_path):
    images_list = glob.glob(image_path + '/*.jpg')

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    pbar = tqdm(images_list, desc=f'Converting {image_path}')  # 进度条

    for p in pbar:
        img_path = p

        base_name = os.path.basename(img_path)
        # print(os.path.splitext(base_name)[0])
        # temp = img_path.replace('images', 'labels')
        label_txt = os.path.join(label_path, base_name[0:-4] + ".txt")
        output = os.path.join(output_path, base_name)

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        with open(label_txt, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
        # 绘制每一个目标
        for x in lb:
            # 反归一化并得到左上和右下坐标，画出矩形框
            img = xywh2xyxy(x, w, h, img)

        cv2.imwrite(output, img)



def visualize_annotation(image_path, annotation_path, save_img):
    # image_path_list = glob.glob(image_path + '/*.jpg')
    annotation_path_list = glob.glob(annotation_path + '/*.xml')

    # save_dir = image_path.replace('images', 'images_label')
    if os.path.exists(save_img) == False:
        os.makedirs(save_img)

    pbar = tqdm(annotation_path_list, disable=f'Converting {annotation_path}')
    for p in pbar:
        annotation = p
        annotation_name = os.path.basename(annotation)

        img = os.path.join(image_path, annotation_name.replace('xml', 'jpg'))
        # 读取图像
        image = cv2.imread(img)
        # output_image = img.replace('images', 'images_label')
        output_image = os.path.join(save_img, annotation_name.replace('xml', 'jpg'))
        # 读取 XML 标注文件
        tree = ET.parse(annotation)
        root = tree.getroot()

        # 遍历 XML 文件中的所有对象
        for obj in root.findall('object'):
            # 获取对象的名称
            name = obj.find('name').text

            # 获取对象的边界框坐标
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))

            # 在图像上绘制矩形框
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(output_image, image)
        # 显示图像
        # cv2.imshow('Annotated Image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = r"E:\downloads\compress\datasets\FEEDS\WI_PRW_SSM_0322_v2\new\images"
    image_label = r"E:\downloads\compress\datasets\FEEDS\WI_PRW_SSM_0322_v2\new\labels"
    annotation_path = r"E:\downloads\compress\datasets\yolo_wider\train\Annotations"
    save_dir = r"E:\downloads\compress\datasets\FEEDS\WI_PRW_SSM_0322_v2\new\labels_images"
    patter = "txt"

    if patter == "xml":
        visualize_annotation(image_path, annotation_path, save_dir)
    elif patter == "txt":
        # plot_tangle(image_path, image_label, save_dir)
        txt_bbox(image_path, image_label, save_dir)
    pass

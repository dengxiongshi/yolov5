from tqdm import tqdm
import glob
import os
import cv2
import numpy as np


def IOU(box1, box2):
    """
        计算两个边界框的IOU值
        :param box1: 边界框1的坐标 (x1, y1, x2, y2)
        :param box2: 边界框2的坐标 (x1, y1, x2, y2)
        :return: IOU值
        """
    # 计算交集面积和并集面积
    intersection_area = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    # union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (
    #             box2[3] - box2[1]) - intersection_area
    union_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # 计算IOU值
    iou = intersection_area / union_area
    return iou


def find_largest_isolated_box(image, yolo_labels):

    # 获取图像的宽度和高度
    # image_height, image_width, _ = image.shape
    #
    # # 解析初始标签
    # class_id, xmin_initial, ymin_initial, xmax_initial, ymax_initial = xywh2xyxy(image, yolo_labels[0])
    #
    # max_area_box = (xmin_initial, ymin_initial, xmax_initial, ymax_initial)

    # 初始化孤立框列表
    isolated_boxes = []
    row_count = len(yolo_labels)
    if row_count == 1:
        isolated_boxes.append(' '.join(yolo_labels))
    else:
        # 遍历所有标签框
        for i in range(row_count):
            label = yolo_labels[i].split()

            class_initial, xmin_initial, ymin_initial, xmax_initial, ymax_initial = xywh2xyxy(image, yolo_labels[i])
            box_initial = (xmin_initial, ymin_initial, xmax_initial, ymax_initial)
            # 检查是否与其他框重叠
            is_isolated = True
            for j in range(row_count):
                if i != j:
                    # box = yolo_labels[j]
                    class_current, xmin_current, ymin_current, xmax_current, ymax_current = xywh2xyxy(image, yolo_labels[j])
                    box_current = (xmin_current, ymin_current, xmax_current, ymax_current)

                    iou = IOU(box_initial, box_current)
                    if iou > 0.3:
                        is_isolated = False
                        break

            if is_isolated:
                isolated_boxes.append(' '.join(label))

    labels_copy = []
    for line in isolated_boxes:
        parts = line.split()
        id, w, h = int(parts[0]), float(parts[3]), float(parts[4])
        if (w * h * 100 > 5.5) and np.any((id != 0) & (id != 1)):
            labels_copy.append(' '.join(parts))

    return labels_copy


def xywh2xyxy(image, boxs):
    image_height, image_width, _ = image.shape
    box = boxs.split()

    clasid = int(box[0])
    x_center = int(float(box[1]) * image_width)
    y_center = int(float(box[2]) * image_height)
    width = int(float(box[3]) * image_width)
    height = int(float(box[4]) * image_height)

    xmin = max(0, x_center - width // 2)
    ymin = max(0, y_center - height // 2)
    xmax = min(image_width, x_center + width // 2)
    ymax = min(image_height, y_center + height // 2)

    return clasid, xmin, ymin, xmax, ymax


def xywh2xyxy_image(x, w1, h1, img):
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


def convert(bb, box):
    dw = 1. / (bb[2] - bb[0])
    dh = 1. / (bb[3] - bb[1])

    x = (box[1] + box[3]) / 2 - bb[0] - 1
    y = (box[2] + box[4]) / 2 - bb[1] - 1
    w = box[3] - box[1]
    h = box[4] - box[2]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    yolo = (box[0], x, y, w, h)

    return yolo


def crop_around_box(image, box, margin):
    # 逆归一化
    image_height, image_width, _ = image.shape

    xmin = max(0, box[1] - margin)
    ymin = max(0, box[2] - margin)
    xmax = min(image_width, box[3] + margin)
    ymax = min(image_height, box[4] + margin)

    bb = (xmin, ymin, xmax, ymax)
    yolo = convert(bb, box)

    return image[ymin:ymax, xmin:xmax], yolo


def convert_and_save_label(cropped_box, output_path):
    # h, w = img.shape[:2]

    b = cropped_box[1:]

    # bb = convert((w, h), b)

    with open(output_path, 'w') as f:
        f.write(str(int(cropped_box[0])) + " " + " ".join([str(a) for a in b]) + '\n')


def visualize(image, label):
    img = image.copy()
    h, w = img.shape[:2]
    with open(label, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
    # 绘制每一个目标
    for x in lb:
        # 反归一化并得到左上和右下坐标，画出矩形框
        img = xywh2xyxy_image(x, w, h, img)

    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', img)
    cv2.waitKey()


images_dir = r"F:\datasets\VisDrone2019\VisDrone2019-DET-val\images"
labels_dir = r"F:\datasets\VisDrone2019\VisDrone2019-DET-val\labels"

save_image = r'F:\datasets\VisDrone2019\VisDrone2019-DET-val\cut_images'
save_lable = r"F:\datasets\VisDrone2019\VisDrone2019-DET-val\cut_labels"

os.makedirs(save_image, exist_ok=True)
os.makedirs(save_lable, exist_ok=True)
# images_list = glob.glob(images_dir + '/*.jpg')
labels_list = glob.glob(labels_dir + '/*.txt')

pbar = tqdm(labels_list, desc=f'Converting {labels_dir}')

isolated_threshold = 0.11
crop_margin = 5

for p in pbar:
    basename = os.path.basename(p)
    name = os.path.splitext(basename)[0]

    if name == '0000076_02667_d_0000010':
        s = 1

    image_name = os.path.join(images_dir, name + '.jpg')

    image = cv2.imread(image_name)

    # original_label = np.loadtxt(p)
    with open(p, 'r') as source:
        original_label = [line.strip() for line in source.readlines() if line.strip()]

    # 找到最大的孤立框
    largest_boxs = find_largest_isolated_box(image, original_label)

    i = 0
    for line in largest_boxs:
        # 在孤立框周围裁剪图像
        id, x1, y1, x2, y2 = xywh2xyxy(image, line)
        box = (id, x1, y1, x2, y2)

        cropped_image, yolo = crop_around_box(image, box, crop_margin)

        label_name = os.path.join(save_lable, name + f'_{i}.txt')
        image_name = os.path.join(save_image, name + f'_{i}.jpg')
        i = i + 1
        # 转换并保存标签
        convert_and_save_label(yolo, label_name)

        # 可视化结果
        # visualize(cropped_image, label_name)

        # 保存裁剪后的图像
        cv2.imwrite(image_name, cropped_image)

# images_file = r"F:\datasets\VisDrone2019\VisDrone2019-DET-train\images\9999966_00000_d_0000135.jpg"
# labels_file = r"F:\datasets\VisDrone2019\VisDrone2019-DET-train\labels\9999966_00000_d_0000135.txt"
#
# save_dir = r"F:\datasets\VisDrone2019\VisDrone2019-DET-train"



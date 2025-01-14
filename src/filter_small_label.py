import math
import os
import glob

import cv2
from tqdm import tqdm


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

    return xmin, ymin, xmax, ymax


def filter_small_label(source_file, destination_file, mode='w'):
    with open(source_file, 'r') as source:
        data = [line.strip() for line in source.readlines() if line.strip()]

    labels = []
    for line in data:
        parts = line.split()
        if len(parts) > 0:
            width, height = float(parts[3]), float(parts[4])
            sign = math.sqrt(width * height)
            if sign >= 0.03:
                labels.append(' '.join(parts))

    with open(destination_file, mode) as destination:
        destination.write('\n'.join(labels))


def del_label(source_file, destination_file):
    with open(source_file, 'r') as source:
        data = [line.strip() for line in source.readlines() if line.strip()]

    labels = []
    for line in data:
        parts = line.split()
        if len(parts) > 0:
            parts = parts[0:5]

            labels.append(' '.join(parts))

    with open(destination_file, 'a') as destination:
        destination.write('\n'.join(labels))


def get_files(directory):
    files = [os.path.join(directory, file) for file in os.listdir(directory) if
             os.path.isfile(os.path.join(directory, file))]
    return files


if __name__ == "__main__":
    # src_image = r"E:\downloads\compress\datasets\VisDrone2019\train_data\images\val"
    src_label = r"E:\downloads\compress\datasets\stanford_cars\test\train_data\labels\val"
    save_label = r"E:\downloads\compress\datasets\stanford_cars\test\train_data\labels_no_small\val"

    os.makedirs(save_label, exist_ok=True)

    src_label_list = glob.glob(src_label + '/*.txt')
    # save_label_list = glob.glob(save_label + '/*.txt')
    pbar = tqdm(src_label_list, desc=f'Converting {src_label}')

    for p in pbar:
        src_file = p
        src_name = os.path.basename(src_file)

        # image_name = os.path.join(src_image, src_name)

        save_file = os.path.join(save_label, src_name)
        filter_small_label(src_file, save_file)

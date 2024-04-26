#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: xiaoshi
# FILE_NAME: xml_to_label
# TIME:  11:54
import glob
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import cv2
from concurrent import futures
from os import getcwd
from alive_progress import alive_bar

import numpy as np
import yaml


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(images, annotation, labels, classes, xml_id):
    in_file = open(os.path.join(annotation, xml_id + ".xml"), encoding='UTF-8')
    out_file = open(os.path.join(labels, xml_id + ".txt"), 'w')
    img_file = os.path.join(images, xml_id + ".jpg")
    img = cv2.imread(img_file)
    h, w = img.shape[:2]

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult) == 1:
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def process_file(file):
    # 这里的pbar是线程专用的进度条
    name = os.path.basename(file)
    xml = os.path.splitext(name)[0]
    convert_annotation(image_dir, annotations_dir, labels_dir, classes, xml)
    # pbar.update()  # 更新进度条


if __name__ == "__main__":
    # classes = ["face",
    #            "person",
    #            "car",
    #            "bus",
    #            "aeroplane",
    #            "bicycle",
    #            "bird",
    #            "boat",
    #            "bottle",
    #            "cat",
    #            "chair",
    #            "cow",
    #            "diningtable",
    #            "dog",
    #            "horse",
    #            "motorbike",
    #            "pottedplant",
    #            "sheep",
    #            "sofa",
    #            "train",
    #            "tvmonitor"
    #            ]

    classes = ["face",
               "person",
               "car",
               "bus",
               "truck"
               ]
    # classes = ["fire", "smoke"]

    image_dir = r"E:\downloads\compress\datasets\video\2024_01_18\images"
    annotations_dir = r"E:\downloads\compress\datasets\video\2024_01_18\Annotations"
    labels_dir = r"E:\downloads\compress\datasets\video\2024_01_18\labels"

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # xmlfiles = os.listdir(annotations_dir)

    files = glob.glob(image_dir + '/*.jpg')

    total_files = len(files)

    # 创建一个总的进度条
    with tqdm(total=total_files, desc=f'Converting {image_dir}', ncols=100) as pbar:
        # 创建一个线程池
        with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # 提交任务到线程池
            future = [executor.submit(process_file, file) for file in files]

            # 等待所有任务完成并更新进度条
            for future in futures.as_completed(future):
                # 处理完一个文件后更新进度条
                pbar.update()



                # num_files = len(files)
    # cpu_nums = os.cpu_count()
    #
    # with futures.ThreadPoolExecutor(max_workers=cpu_nums) as executor:
    #     future = []
    #     for file in files:
    #         pbar = tqdm(total=1, desc=f'Converting {image_dir}', position=files.index(file))
    #         exe = executor.submit(process_file, file, pbar)
    #         future.append(exe)
    #
    #     futures.wait(future)

    # pbar = tqdm(files, desc=f'Converting {image_dir}')  # 进度条
    #
    # for file in pbar:
    #     # if xml_id == 15:
    #     name = os.path.basename(file)
    #     xml = os.path.splitext(name)[0]
    #     # print(file)
    #     convert_annotation(image_dir, annotations_dir, labels_dir, classes, xml)

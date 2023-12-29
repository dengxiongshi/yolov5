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
        if cls not in classes or int(difficult) == 1:
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

    image_dir = r"E:\downloads\compress\datasets\yolo_wider\val\new\images"
    annotations_dir = r"E:\downloads\compress\datasets\yolo_wider\val\new\Annotations"
    labels_dir = r"E:\downloads\compress\datasets\yolo_wider\val\new\labels"

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # xmlfiles = os.listdir(annotations_dir)

    files = glob.glob(annotations_dir + '/*.xml')
    pbar = tqdm(files, desc=f'Converting {annotations_dir}')  # 进度条

    for file in pbar:
        # if xml_id == 15:
        name = os.path.basename(file)
        xml = os.path.splitext(name)[0]
        # print(file)
        convert_annotation(image_dir, annotations_dir, labels_dir, classes, xml)

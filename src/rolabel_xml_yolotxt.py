#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:MXY
# roLabelImg标注的xml结果转换为yolo的标签
import glob
import shutil
import xml.etree.ElementTree as ET
import os
import math
from os import getcwd
from tqdm import tqdm



def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] * dw
    y = box[1] * dh
    w = box[2] * dw
    h = box[3] * dh
    return x, y, w, h


def convert_annotation(annotation, labels, classes, xml_id):
    in_file = open(os.path.join(annotation, xml_id + ".xml"), encoding='UTF-8')
    out_file = open(os.path.join(labels, xml_id + ".txt"), 'w')
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
        xmlbox = obj.find('robndbox')
        cx = float(xmlbox.find('cx').text)
        cy = float(xmlbox.find('cy').text)
        w = float(xmlbox.find('w').text)
        h = float(xmlbox.find('h').text)
        theta = float(xmlbox.find('angle').text)
        # b1, b2, b3, b4 = b
        # if w < h:
        #     b = (cx, cy, h, w)
        #     theta = int(((theta * 180 / math.pi) + 90) % 180)
        # else:  # 如果有b3≈b4, 那么OBB是正方形, 两个theta值都适用, 但是不利于训练
        #     theta = int(theta * 180 / math.pi)

        # 计算旋转框的四个角
        angle_rad = math.radians(theta)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        x1 = cx - w / 2 * cos_a - h / 2 * sin_a
        y1 = cy - w / 2 * sin_a + h / 2 * cos_a

        x2 = cx + w / 2 * cos_a - h / 2 * sin_a
        y2 = cy + w / 2 * sin_a + h / 2 * cos_a

        x3 = cx + w / 2 * cos_a + h / 2 * sin_a
        y3 = cy + w / 2 * sin_a - h / 2 * cos_a

        x4 = cx - w / 2 * cos_a + h / 2 * sin_a
        y4 = cy - w / 2 * sin_a - h / 2 * cos_a

        xmin = min(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        xmax = max(x1, x2, x3, x4)
        ymax = max(y1, y2, y3, y4)

        if xmax > width:
            xmax = width

        if ymax > height:
            ymax = height

        b = (xmin, xmax, ymin, ymax)


        bb = convert((width, height), b)
        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + " " + str(theta) + '\n')
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == "__main__":
    # ------------------------------ 修改内容 ----------------------------------
    classes = ['face', 'person', 'car', 'bus', 'truck']

    annotations_dir = r"E:\downloads\compress\datasets\UAV_ROD_Data\train\annotations"
    labels_dir = r"E:\downloads\compress\datasets\UAV_ROD_Data\train\Annotations_labels"

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # xmlfiles = os.listdir(annotations_dir)
    xmlfiles = glob.glob(annotations_dir + '/*.xml')

    pbar = tqdm(xmlfiles, desc=f'Converting {annotations_dir}')

    for file in pbar:
        # if xml_id == 15:
        filename = os.path.basename(file)
        xml = os.path.splitext(filename)[0]
        # print(file)
        convert_annotation(annotations_dir, labels_dir, classes, xml)


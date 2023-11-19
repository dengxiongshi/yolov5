#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: xiaoshi
# FILE_NAME: xml_to_label
# TIME:  11:54

import xml.etree.ElementTree as ET
import os
from os import getcwd

import numpy as np
import yaml


wd = getcwd()
print(wd)

data_path = r"D:\python_code\yolo5_caffe_hisi3559\02.yolov5\board\20221202"

# path = wd + data_path
path = data_path

my_yaml = path + "/board.yaml"

# with open(my_yaml, errors='ignore') as f:
#     classes = yaml.safe_load(f)['names']

classes = ['unload', 'load']   # 改成自己的类别
# abs_path = os.getcwd()
# print(abs_path)

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

def convert_annotation(image_id):
    in_file = open(path + '/Annotations/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open(path + '/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
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


if not os.path.exists(path + '/labels/'):
    os.makedirs(path + '/labels/')

# xml文件存放目录
annotations_dir = path + '/Annotations'
xmlfiles = os.listdir(annotations_dir)

for image_id in xmlfiles:
    image = os.path.splitext(image_id)[0]
    convert_annotation(image)


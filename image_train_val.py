# !/usr/bin/ python3
# -*- coding: UTF-8 -*-
# @author: xiaoshi
# @file: split_train_val.py
# @Time: 2022/7/26 9:38
# coding:utf-8

import os
import random
import argparse

# parser = argparse.ArgumentParser()
# #xml文件的地址，根据自己的数据进行修改 xml一般存放在Annotations下
# parser.add_argument('--xml_path', default='board20221110/Annotations', type=str, help='input xml label path')
# #数据集的划分，地址选择自己数据下的ImageSets/Main
# parser.add_argument('--txt_path', default='board20221110/ImageSets/Main', type=str, help='output txt label path')
# opt = parser.parse_args()

datapath = r'D:\python_code\yolov5\datasets\board20221117'
image_path = datapath + '/images'
ImageAllName = os.listdir(image_path)                                            # 获取path路径下所有文件名称

trainval_percent = 1.0
train_percent = 0.8
val_percent = 0.2
# test_percent = 0.1

# xmlfilepath = opt.xml_path
# txtsavepath = opt.txt_path
# total_xml = os.listdir(xmlfilepath)
# if not os.path.exists(txtsavepath):
#     os.makedirs(txtsavepath)

num = len(ImageAllName)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
va = int(tv * val_percent)
# te = tv - tr - va

trainval = random.sample(list_index, tv)
all_result = trainval.copy()
train = random.sample(trainval, tr)
val = random.sample(trainval, va)
# for i in range(tr):
#     if train[i] in all_result:
#         all_result.remove(train[i])

# val = random.sample(all_result, va)
# for i in range(va):
#     if val[i] in all_result:
#         all_result.remove(val[i])
#
# test = all_result

file_trainval = open(datapath + '/trainval.txt', 'w')

file_test = open(datapath + '/test.txt', 'w')
file_train = open(datapath + '/train.txt', 'w')
file_val = open(datapath + '/val.txt', 'w')

for i in list_index:
    name = image_path + '/' + ImageAllName[i] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()

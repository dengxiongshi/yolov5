#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: xiaoshi
# FILE_NAME: background_label
# TIME:  21:39
import os

mydir = r"D:\python_code\yolo5_caffe_hisi3559\02.yolov5\board\20221202\nounload"
savedir = r"D:\python_code\yolo5_caffe_hisi3559\02.yolov5\board\20221202\nounload_label"
if not os.path.exists(savedir):
    os.makedirs(savedir)

dirname = os.listdir(mydir)

for id, old_name in enumerate(dirname):
    filename = os.path.splitext(old_name)[0] + '.txt'
    r = os.path.join(savedir, filename)
    file = open(r, 'w')
    file.write(" ")
    file.flush()
    file.close()
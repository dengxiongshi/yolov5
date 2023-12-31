import os
from random import sample

import numpy as np
from PIL import Image, ImageDraw

from random_data import get_random_data, get_random_data_with_MixUp
from utils import convert_annotation, get_classes

#-----------------------------------------------------------------------------------#
#   Origin_VOCdevkit_path   原始标签所在的路径
#   Out_VOCdevkit_path      输出标签所在的路径
#                                   处理后的标签为灰度图，如果设置的值太小会看不见具体情况。
#-----------------------------------------------------------------------------------#
Origin_VOCdevkit_path   = r"D:\python_code\yolo5_caffe_hisi3559\02.yolov5\board\20221202"
Out_VOCdevkit_path      = r"D:\python_code\yolo5_caffe_hisi3559\02.yolov5\board\20221202"
#-----------------------------------------------------------------------------------#
#   Out_Num                 利用mixup生成多少组图片
#   input_shape             生成的图片大小
#-----------------------------------------------------------------------------------#
# Out_Num                 = 5
input_shape             = [640, 640]

#-----------------------------------------------------------------------------------#
#   下面定义了xml里面的组成模块，无需改动。
#-----------------------------------------------------------------------------------#
headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""

objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
    
tailstr = '''\
</annotation>
'''
if __name__ == "__main__":
    Origin_JPEGImages_path  = os.path.join(Origin_VOCdevkit_path, "unload")
    Origin_Annotations_path = os.path.join(Origin_VOCdevkit_path, "unload_annotations")
    
    Out_JPEGImages_path  = os.path.join(Out_VOCdevkit_path, "unload_new")
    Out_Annotations_path = os.path.join(Out_VOCdevkit_path, "unload_annotations_new")
    
    if not os.path.exists(Out_JPEGImages_path):
        os.makedirs(Out_JPEGImages_path)
    if not os.path.exists(Out_Annotations_path):
        os.makedirs(Out_Annotations_path)
    #---------------------------#
    #   遍历标签并赋值
    #---------------------------#
    xml_names = os.listdir(Origin_Annotations_path)
    nums = len(xml_names)

    def write_xml(anno_path, jpg_pth, head, input_shape, boxes, unique_labels, tail):
        f = open(anno_path, "w")
        f.write(head%(jpg_pth, input_shape[0], input_shape[1], 3))
        for i, box in enumerate(boxes):
            f.write(objstr%(str(unique_labels[int(box[4])]), box[0], box[1], box[2], box[3]))
        f.write(tail)
    
    #------------------------------#
    #   循环生成xml和jpg
    #------------------------------#
    for index in range(nums):
        name = xml_names[index].split(".")[0]
        #------------------------------#
        #   获取一个图像与标签
        #------------------------------#
        sample_xmls     = sample(xml_names, 1)
        unique_labels   = get_classes(sample_xmls, Origin_Annotations_path)
        
        jpg_name  = os.path.join(Origin_JPEGImages_path, os.path.splitext(sample_xmls[0])[0] + '.jpg')
        xml_name  = os.path.join(Origin_Annotations_path, sample_xmls[0])
            
        line = convert_annotation(jpg_name, xml_name, unique_labels)
        
        #------------------------------#
        #   各自数据增强
        #------------------------------#
        image_data, box_data  = get_random_data(line, input_shape) 
        
        img = Image.fromarray(image_data.astype(np.uint8))

        imgname = 'unload2_' + name + '.jpg'
        xmlname = 'unload2_' + name + '.xml'
        img.save(os.path.join(Out_JPEGImages_path, imgname))
        write_xml(os.path.join(Out_Annotations_path, xmlname), os.path.join(Out_JPEGImages_path, imgname), \
                    headstr, input_shape, box_data, unique_labels, tailstr)

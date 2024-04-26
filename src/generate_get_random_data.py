import glob
import os
from random import sample
from tqdm import tqdm

import numpy as np
from PIL import Image, ImageDraw

from random_data import get_random_data, get_random_data_with_MixUp
from utils_ import convert_annotation, get_classes

# -----------------------------------------------------------------------------------#
#   Origin_VOCdevkit_path   原始标签所在的路径
#   Out_VOCdevkit_path      输出标签所在的路径
#                                   处理后的标签为灰度图，如果设置的值太小会看不见具体情况。
# -----------------------------------------------------------------------------------#
Origin_VOCdevkit_path = r"E:\downloads\compress\datasets\stanford_cars\test"
Out_VOCdevkit_path = r"E:\downloads\compress\datasets\stanford_cars\test"
# -----------------------------------------------------------------------------------#
#   Out_Num                 利用mixup生成多少组图片
#   input_shape             生成的图片大小
# -----------------------------------------------------------------------------------#
# Out_Num                 = 5
input_shape = [640, 640]

# -----------------------------------------------------------------------------------#
#   下面定义了xml里面的组成模块，无需改动。
# -----------------------------------------------------------------------------------#
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
    Origin_JPEGImages_path = os.path.join(Origin_VOCdevkit_path, "images")
    Origin_Annotations_path = os.path.join(Origin_VOCdevkit_path, "Annotations")

    Out_JPEGImages_path = os.path.join(Out_VOCdevkit_path, "images_new")
    Out_Annotations_path = os.path.join(Out_VOCdevkit_path, "Annotations_new")

    if not os.path.exists(Out_JPEGImages_path):
        os.makedirs(Out_JPEGImages_path)
    if not os.path.exists(Out_Annotations_path):
        os.makedirs(Out_Annotations_path)
    # ---------------------------#
    #   遍历标签并赋值
    # ---------------------------#
    # xml_names = os.listdir(Origin_Annotations_path)
    # nums = len(xml_names)

    Annotation_list = glob.glob(Origin_Annotations_path + '/*.xml')

    pbar = tqdm(Annotation_list, desc=f'Converting {Origin_Annotations_path}')


    def write_xml(anno_path, jpg_pth, head, input_shape, boxes, unique_labels, tail):
        f = open(anno_path, "w")
        f.write(head % (jpg_pth, input_shape[0], input_shape[1], 3))
        for i, box in enumerate(boxes):
            f.write(objstr % (str(unique_labels[int(box[4])]), box[0], box[1], box[2], box[3]))
        f.write(tail)


    # ------------------------------#
    #   循环生成xml和jpg
    # ------------------------------#
    for p in pbar:
        basename = os.path.basename(p)
        name = os.path.splitext(basename)[0]
        # ------------------------------#
        #   获取一个图像与标签
        # ------------------------------#
        unique_labels = get_classes(basename, Origin_Annotations_path)

        jpg_name = os.path.join(Origin_JPEGImages_path, name + '.jpg')
        xml_name = p

        line = convert_annotation(jpg_name, xml_name, unique_labels)

        # ------------------------------#
        #   各自数据增强
        # ------------------------------#
        image_data, box_data = get_random_data(line, input_shape)

        img = Image.fromarray(image_data.astype(np.uint8))

        imgname = 'new3_' + name + '.jpg'
        xmlname = 'new3_' + name + '.xml'
        img.save(os.path.join(Out_JPEGImages_path, imgname))
        write_xml(os.path.join(Out_Annotations_path, xmlname), os.path.join(Out_JPEGImages_path, imgname), \
                  headstr, input_shape, box_data, unique_labels, tailstr)
# 文件名称   ：roxml_to_dota.py
# 功能描述   ：把rolabelimg标注的xml文件转换成dota能识别的xml文件，
#             再转换成dota格式的txt文件
#            把旋转框 cx,cy,w,h,angle，转换成四点坐标x1,y1,x2,y2,x3,y3,x4,y4
import os
import xml.etree.ElementTree as ET
import math
import numpy as np
import glob
from tqdm import tqdm


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


# 转换成四点坐标
def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc
    yoff = yp - yc
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return int(xc + pResx), int(yc + pResy)


def bndbox_func(lis):  # 传入有一个元素的列表，其中每个元素是一个列表形式的坐标

    # 将坐标的每一个元素映射为float
    for i in range(0, len(lis)):
        lis[i] = list(map(float, lis[i]))

    # 将lis转为array，格式为[[198. 301.]，[151. 335.]，[164. 359.]，[212. 325.]]
    coordinate = np.array(lis)

    # 求中心点坐标（array相当于矩阵，可以直接进行类似矩阵的运算）
    center = coordinate[0]
    for _ in range(1, 4):
        center = center + coordinate[_]
    center = center / 4

    # 复制一份坐标，避免原坐标被破坏
    coordinate_temp = coordinate.copy()

    # 找出x轴小于中心坐标点的点 left_coordinate,两个点就是左上和左下，一个点就是左下
    left_coordinate = []  # 用于存储x轴小于中心坐标点的点
    delete_index = []
    for _ in range(4):  # 将 x轴小于中心坐标点的点存储进left_coordinate
        if (coordinate[_][0] < center[0]):
            left_coordinate.append(coordinate[_])  # list(array1,array2)
            delete_index.append(_)  # list(index1,index2)

    # 若上面找出有两个点，其余点即为右上和右下
    right_coordinate = np.delete(coordinate_temp, delete_index, axis=0)  # array
    # 若上面找出有一个点
    coordinate_temp = np.delete(coordinate_temp, delete_index, axis=0)  # array

    # 将left_coordinate做备份
    left_coordinate_temp = left_coordinate.copy()

    # 此时对角线和x轴斜交
    if (len(left_coordinate_temp) == 2):
        # 比较左边两个点的y轴，大的为左上
        if (left_coordinate[0][1] < left_coordinate[1][1]):
            left_bottom = left_coordinate[0]  # array
            left_top = left_coordinate[1]  # array
        elif (left_coordinate[0][1] > left_coordinate[1][1]):
            left_bottom = left_coordinate[1]  # array
            left_top = left_coordinate[0]  # array

        # 比较右边两个点的y轴，大的为右上
        if (right_coordinate[0][1] < right_coordinate[1][1]):
            right_bottom = right_coordinate[0]  # array
            right_top = right_coordinate[1]  # array
        elif (right_coordinate[0][1] > right_coordinate[1][1]):
            right_bottom = right_coordinate[1]  # array
            right_top = right_coordinate[0]  # array

    # 此时对角线和x轴垂直
    elif (len(left_coordinate_temp) == 1):
        left_bottom = left_coordinate[0]  # 左下
        delete_index = []

        for _ in range(3):
            if (coordinate_temp[_][0] == center[0] and coordinate_temp[_][1] > center[1]):
                left_top = coordinate_temp[_]  # 左上
                delete_index.append(_)
            if (coordinate_temp[_][0] == center[0] and coordinate_temp[_][1] < center[1]):
                right_bottom = coordinate_temp[_]  # 右下
                delete_index.append(_)

        coordinate_temp = np.delete(coordinate_temp, delete_index, axis=0)
        right_top = coordinate_temp[0]  # 右上

    return left_top[0], left_top[1], right_bottom[0], right_bottom[1]


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

        obj_bnd = obj.find('robndbox')
        # obj_bnd.tag = 'bndbox'  # 修改节点名
        obj_cx = obj_bnd.find('cx')
        obj_cy = obj_bnd.find('cy')
        obj_w = obj_bnd.find('w')
        obj_h = obj_bnd.find('h')
        obj_angle = obj_bnd.find('angle')
        cx = float(obj_cx.text)
        cy = float(obj_cy.text)
        w = float(obj_w.text)
        h = float(obj_h.text)
        angle = float(obj_angle.text)
        # obj_bnd.remove(obj_cx)  # 删除节点
        # obj_bnd.remove(obj_cy)
        # obj_bnd.remove(obj_w)
        # obj_bnd.remove(obj_h)
        # obj_bnd.remove(obj_angle)

        x0, y0 = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
        x1, y1 = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
        x2, y2 = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
        x3, y3 = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)

        coordinate = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]

        x_min, y_min, x_max, y_max = bndbox_func(coordinate)  # 转化后的矩形坐标

        # 标注越界修正
        if x_max > width:
            x_max = width
        if y_max > height:
            y_max = height

        b = (x_min, x_max, y_min, y_max)

        bb = convert((width, height), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')



if __name__ == "__main__":

    classes = ['face', 'person', 'car', 'bus', 'truck']

    annotations_dir = r"E:\downloads\compress\datasets\UAV_ROD_Data\train\annotations"
    labels_dir = r"E:\downloads\compress\datasets\UAV_ROD_Data\train\Annotations_labels"

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # xmlfiles = os.listdir(annotations_dir)
    xmlfiles = glob.glob(annotations_dir + '/*.xml')

    pbar = tqdm(xmlfiles, disable=f'Converting {annotations_dir}')

    for file in pbar:
        # if xml_id == 15:
        filename = os.path.basename(file)
        xml = os.path.splitext(filename)[0]
        # print(file)
        convert_annotation(annotations_dir, labels_dir, classes, xml)

    # filelist = os.listdir(roxml_path)
    # for file in filelist:
    #     edit_xml(os.path.join(roxml_path, file), os.path.join(dotaxml_path, file))
    #
    # # -----**** 第二步：把旋转框xml文件转换成txt格式 ****-----
    # totxt(dotaxml_path, out_path)

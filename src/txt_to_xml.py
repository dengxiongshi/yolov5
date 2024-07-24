#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: xiaoshi
# FILE_NAME: txt_to_xml
# TIME:  11:09
from xml.dom.minidom import Document
import os
import cv2
import glob
from tqdm import tqdm


# def makexml(txtPath, xmlPath, picPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
def makexml(picPath, txtPath, xmlPath, dic):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
    """此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
    在自己的标注图片文件夹下建三个子文件夹，分别命名为picture、txt、xml
    """
    if os.path.exists(xmlPath) == False:
        os.makedirs(xmlPath)

    # files = os.listdir(txtPath)
    files = glob.glob(txtPath + '/*.txt')
    pbar = tqdm(files, desc=f'Converting {txtPath}')  # 进度条
    # i = 0
    for p in pbar:
        name = os.path.basename(p)
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        # txtFile = open(txtPath + name)
        with open(os.path.join(txtPath, name)) as txtFile:
            txtList = [line.strip() for line in txtFile.readlines() if line.strip()]
        # txtList = txtFile.readlines()
        # img = cv2.imread(picPath + name[0:-4] + ".jpg")
        image_path = os.path.join(picPath, name[0:-4] + '.jpg')
        # i = i + 1
        # if i == 1078:
        #     s = 0
        img = cv2.imread(image_path)
        if img is None:
            print(image_path)
        Pheight, Pwidth, Pdepth = img.shape

        folder = xmlBuilder.createElement("folder")  # folder标签
        foldercontent = xmlBuilder.createTextNode("driving_annotation_dataset")
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)  # folder标签结束

        filename = xmlBuilder.createElement("filename")  # filename标签
        filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)  # filename标签结束

        imagepath = xmlBuilder.createElement("path")
        imagepathcontent = xmlBuilder.createTextNode(image_path)
        imagepath.appendChild(imagepathcontent)
        annotation.appendChild(imagepath)

        size = xmlBuilder.createElement("size")  # size标签
        width = xmlBuilder.createElement("width")  # size子标签width
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)  # size子标签width结束

        height = xmlBuilder.createElement("height")  # size子标签height
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)  # size子标签height结束

        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)  # size子标签depth结束

        annotation.appendChild(size)  # size标签结束

        # if name == "img01280.txt":
        #     s = 0

        for j in txtList:
            oneline = j.strip().split(" ")
            object = xmlBuilder.createElement("object")  # object 标签
            picname = xmlBuilder.createElement("name")  # name标签
            namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
            picname.appendChild(namecontent)
            object.appendChild(picname)  # name标签结束

            pose = xmlBuilder.createElement("pose")  # pose标签
            posecontent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(posecontent)
            object.appendChild(pose)  # pose标签结束

            truncated = xmlBuilder.createElement("truncated")  # truncated标签
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)  # truncated标签结束

            difficult = xmlBuilder.createElement("difficult")  # difficult标签
            difficultcontent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultcontent)
            object.appendChild(difficult)  # difficult标签结束

            bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
            xmin = xmlBuilder.createElement("xmin")  # xmin标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)  # xmin标签结束

            ymin = xmlBuilder.createElement("ymin")  # ymin标签
            mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)  # ymin标签结束

            xmax = xmlBuilder.createElement("xmax")  # xmax标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)  # xmax标签结束

            ymax = xmlBuilder.createElement("ymax")  # ymax标签
            mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)  # ymax标签结束

            object.appendChild(bndbox)  # bndbox标签结束

            annotation.appendChild(object)  # object标签结束

        f = open(os.path.join(xmlPath, name[0:-4] + ".xml"), 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()


if __name__ == "__main__":
    picPath = r"E:\downloads\compress\datasets\fire_smoke\fire-8\images"  # 图片所在文件夹路径，后面的/一定要带上
    txtPath = r"E:\downloads\compress\datasets\fire_smoke\fire-8\labels"  # txt所在文件夹路径，后面的/一定要带上
    xmlPath = r"E:\downloads\compress\datasets\fire_smoke\fire-8\Annotations"  # xml文件保存路径，后面的/一定要带上

    # dic = {
    #     "0": "single",
    #     "1": "double"
    # }
    dic = {
        "0": "fire",
        "1": "smoke"
    }
    # dic = {
    #     "0": "car",
    #     "1": "person",
    #     "2": "face",
    #     "3": "bus",
    #     "4": "truck"
    # }
    # dic = {
    #     "0": "face",
    #     "1": "person",
    #     "2": "car",
    #     "3": "bus",
    #     "4": "aeroplane",
    #     "5": "bicycle",
    #     "6": "bird",
    #     "7": "boat",
    #     "8": "bottle",
    #     "9": "cat",
    #     "10": "chair",
    #     "11": "cow",
    #     "12": "diningtable",
    #     "13": "dog",
    #     "14": "horse",
    #     "15": "motorbike",
    #     "16": "pottedplant",
    #     "17": "sheep",
    #     "18": "sofa",
    #     "19": "train",
    #     "20": "tvmonitor"
    # }

    makexml(picPath, txtPath, xmlPath, dic)

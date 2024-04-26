import os
import pandas as pd
from xml.dom.minidom import Document
import json


# coco格式的json转voc格式的xml
def generate_label_file(json_path):
    with open(json_path, 'r') as load_f:
        f = json.load(load_f)
    df_anno = pd.DataFrame(f['annotations'])
    imgs = f['images']
    cata = {}
    df_cate = f['categories']
    for item in df_cate:
        cata[item['id']] = item['name']
    for im in imgs:
        flag = 0
        filename = im['file_name']
        height = im['height']
        img_id = im['id']
        width = im['width']

        doc = Document()
        annotation = doc.createElement('annotation')
        doc.appendChild(annotation)
        filenamedoc = doc.createElement("filename")
        annotation.appendChild(filenamedoc)
        filename_txt = doc.createTextNode(filename)
        filenamedoc.appendChild(filename_txt)
        size = doc.createElement("size")
        annotation.appendChild(size)

        widthdoc = doc.createElement("width")
        size.appendChild(widthdoc)
        width_txt = doc.createTextNode(str(width))
        widthdoc.appendChild(width_txt)
        heightdoc = doc.createElement("height")
        size.appendChild(heightdoc)
        height_txt = doc.createTextNode(str(height))
        heightdoc.appendChild(height_txt)

        annos = df_anno[df_anno["image_id"].isin([img_id])]
        for index, row in annos.iterrows():
            bbox = row["bbox"]
            category_id = row["category_id"]
            cate_name = cata[category_id]
            if classes:
                if cate_name not in classes:
                    print(cate_name + ",don`t in classes")
                    continue
            flag = 1
            object = doc.createElement('object')
            annotation.appendChild(object)

            name = doc.createElement('name')
            object.appendChild(name)
            name_txt = doc.createTextNode(cate_name)
            name.appendChild(name_txt)

            pose = doc.createElement('pose')
            object.appendChild(pose)
            pose_txt = doc.createTextNode('Unspecified')
            pose.appendChild(pose_txt)

            truncated = doc.createElement('truncated')
            object.appendChild(truncated)
            truncated_txt = doc.createTextNode('0')
            truncated.appendChild(truncated_txt)

            difficult = doc.createElement('difficult')
            object.appendChild(difficult)
            difficult_txt = doc.createTextNode('0')
            difficult.appendChild(difficult_txt)

            bndbox = doc.createElement('bndbox')
            object.appendChild(bndbox)

            xmin = doc.createElement('xmin')
            bndbox.appendChild(xmin)
            xmin_txt = doc.createTextNode(str(int(bbox[0])))
            xmin.appendChild(xmin_txt)

            ymin = doc.createElement('ymin')
            bndbox.appendChild(ymin)
            ymin_txt = doc.createTextNode(str(int(bbox[1])))
            ymin.appendChild(ymin_txt)

            xmax = doc.createElement('xmax')
            bndbox.appendChild(xmax)
            xmax_txt = doc.createTextNode(str(int(bbox[0] + bbox[2])))
            xmax.appendChild(xmax_txt)

            ymax = doc.createElement('ymax')
            bndbox.appendChild(ymax)
            ymax_txt = doc.createTextNode(str(int(bbox[1] + bbox[3])))
            ymax.appendChild(ymax_txt)

            if '/' in filename:
                filename = filename.split('/')[-1]
        if flag == 1:
            xml = os.path.join(xml_path, filename.replace('.jpg', '.xml'))
            f = open(xml, "w")
            f.write(doc.toprettyxml(indent="  "))
            f.close()


if __name__ == '__main__':
    # json文件路径、xml保存路径
    json_path = r"C:\Users\soli\Desktop\smoke_add.json"
    xml_path = r"C:\Users\soli\Desktop\xm"

    # 指定转换类别
    classes = ['face', 'person', 'car']  # classes为空时，转换所有类别

    # 执行转换
    generate_label_file()

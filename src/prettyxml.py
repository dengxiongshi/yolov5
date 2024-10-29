import glob
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET

from tqdm import tqdm


def format_xml(input_file, output_file):
    # 读取 XML 文件
    with open(input_file, 'r') as file:
        xml_content = file.read()

    # 解析 XML
    dom = xml.dom.minidom.parseString(xml_content)

    # 格式化 XML
    pretty_xml = dom.toprettyxml()

    # 将格式化后的 XML 保存到文件
    with open(output_file, 'w') as output_file:
        output_file.write(pretty_xml)


def edit_xml(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    path = root.find('path')
    basename = os.path.basename(path.text)

    path_value = os.path.join(image_path, basename)
    path.text = path_value

    tree.write(output_file)


if __name__ == "__main__":

    xml_path = r"F:\BaiduNetdiskDownload\BoadData\seaships\annotations"
    image_path = r"F:\BaiduNetdiskDownload\BoadData\seaships\images"

    save_dir = xml_path.replace('annotations', 'Annotations_label')

    if os.path.exists(save_dir)  == False:
        os.makedirs(save_dir)

    raw_data_path_list = glob.glob(xml_path + '/*.xml')

    pbar = tqdm(raw_data_path_list, desc=f'Converting {xml_path}')  # 进度条

    for file in pbar:
        input_file = file
        output_file = input_file.replace('annotations', 'Annotations_label')

        edit_xml(input_file, output_file)


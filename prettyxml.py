import glob
import os
import xml.dom.minidom

from alive_progress import alive_bar


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


if __name__ == "__main__":

    xml_path = r"E:\downloads\compress\FEEDS\WI_PRW_SSM_0322_v2\Annotations"

    save_dir = xml_path.replace('Annotations', 'Annotations_label')
    if os.path.exists(save_dir)  == False:
        os.makedirs(save_dir)

    raw_data_path_list = glob.glob(xml_path + '/*.xml')

    with alive_bar(len(raw_data_path_list)) as bar:
        for i in range(len(raw_data_path_list)):
            input_file = raw_data_path_list[i]
            output_file = input_file.replace('Annotations', 'Annotations_label')

            format_xml(input_file, output_file)

            bar()

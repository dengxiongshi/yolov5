import numpy as np
import csv

file_path = r"C:\Users\dengxs\Desktop\gyro_raw_data.csv"

data = []
# 打开CSV文件
with open(file_path, 'r') as file:
    # 创建CSV读取器
    reader = csv.reader(file)
    # 遍历每一行
    for row in reader:
        # 打印每一行数据
        data.append(row)


title = data[0]
parts = title[0].split(';')
head = []

for tt in parts:
    head.append(str(tt))

datasets = []
datasets.append(head)

for i in range(len(data) - 1):
    tt = []
    da = data[i + 1][0].split(';')
    for part in da:
        if part:
            tt.append(int(part))

    datasets.append(tt)


save_path = r"C:\Users\dengxs\Desktop\save_gyro.csv"

# 打开一个CSV文件进行写入
with open(save_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入标题行
    writer.writerow(datasets[0])
    # 写入数据行
    for i in range(len(datasets) - 1):
        da = datasets[i + 1]
        # for j in range(len(da)):
        writer.writerow(da)
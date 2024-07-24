from shutil import copyfile

from sklearn.model_selection import train_test_split
import os

# 设置数据集路径
dataset_folder = "/determined/alluxio/dengxiongshi/datasets/weather/traindata/trainData_20240611"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)


images_path = "/determined/alluxio/dengxiongshi/datasets/weather/thunder"
class_names = ["thunder"]


# 获取所有图像文件的路径
image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

# 划分训练集和测试集
train_files, test = train_test_split(image_files, test_size=0.25, random_state=42)
valid_files, test_files = train_test_split(test, test_size=0.5, random_state=42)

# 创建训练集和测试集的文件夹
image_train = dataset_folder + '/train/' + class_names[0]
image_val = dataset_folder + '/val/' + class_names[0]
image_test = dataset_folder + '/test/' + class_names[0]


os.makedirs(image_train, exist_ok=True)
os.makedirs(image_val, exist_ok=True)
os.makedirs(image_test, exist_ok=True)


# 将图像文件和标签文件移动到对应的文件夹
for file in train_files:
    img_path = os.path.join(images_path, file)
    copyfile(img_path, os.path.join(image_train, file))

for file in valid_files:
    img_path = os.path.join(images_path, file)
    copyfile(img_path, os.path.join(image_val, file))

for file in test_files:
    img_path = os.path.join(images_path, file)
    copyfile(img_path, os.path.join(image_test, file))


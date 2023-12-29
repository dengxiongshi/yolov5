from shutil import copyfile

from sklearn.model_selection import train_test_split
import os

# 设置数据集路径
dataset_folder = r'E:\downloads\compress\datasets\FEEDS\WI_PRW_SSM_0322_v2\datasets'

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)


images_path = r"E:\downloads\compress\datasets\FEEDS\WI_PRW_SSM_0322_v2\JPEGImages"
labels_path = r"E:\downloads\compress\datasets\FEEDS\WI_PRW_SSM_0322_v2\labels_final"

# 获取所有图像文件的路径
image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

# 划分训练集和测试集
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

# 创建训练集和测试集的文件夹
image_train = dataset_folder + '/images/train'
image_val = dataset_folder + '/images/val'
label_train = dataset_folder + '/labels/train'
label_val = dataset_folder + '/labels/val'

os.makedirs(image_train, exist_ok=True)
os.makedirs(image_val, exist_ok=True)
os.makedirs(label_train, exist_ok=True)
os.makedirs(label_val, exist_ok=True)

# 将图像文件和标签文件移动到对应的文件夹
for file in train_files:
    img_path = os.path.join(images_path, file)
    label_path = os.path.join(labels_path, file.replace('.jpg', '.txt'))
    copyfile(img_path, os.path.join(image_train, file))
    copyfile(label_path, os.path.join(label_train, file.replace('.jpg', '.txt')))

for file in test_files:
    img_path = os.path.join(images_path, file)
    label_path = os.path.join(labels_path, file.replace('.jpg', '.txt'))
    copyfile(img_path, os.path.join(image_val, file))
    copyfile(label_path, os.path.join(label_val, file.replace('.jpg', '.txt')))

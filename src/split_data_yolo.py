from shutil import copyfile

from sklearn.model_selection import train_test_split
import os

# 设置数据集路径
from tqdm import tqdm

dataset_folder = r"\\10.10.10.8\determined\alluxio\dengxiongshi\datasets\weather\weather_mlabel\trainData_20240717"

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)


images_path = r"E:\downloads\compress\datasets\天气\weather_classification\thunder"
labels_path = r"E:\downloads\compress\datasets\天气\weather_classification\thunder_label"

# 获取所有图像文件的路径
image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

# 划分训练集和测试集
train_files, test = train_test_split(image_files, test_size=0.25, random_state=42)
valid_files, test_files = train_test_split(test, test_size=0.5, random_state=42)

# 创建训练集和测试集的文件夹
image_train = dataset_folder + '/images/train'
image_val = dataset_folder + '/images/val'
image_test = dataset_folder + '/images/test'
label_train = dataset_folder + '/labels/train'
label_val = dataset_folder + '/labels/val'
label_test = dataset_folder + '/labels/test'

os.makedirs(image_train, exist_ok=True)
os.makedirs(image_val, exist_ok=True)
os.makedirs(image_test, exist_ok=True)
os.makedirs(label_train, exist_ok=True)
os.makedirs(label_val, exist_ok=True)
os.makedirs(label_test, exist_ok=True)

pbar_train = tqdm(train_files, desc=f'{image_train}')
pbar_val = tqdm(valid_files, desc=f'{image_val}')
pbar_test = tqdm(test_files, desc=f'{image_test}')
# 将图像文件和标签文件移动到对应的文件夹
for file in pbar_train:
    img_path = os.path.join(images_path, file)
    label_path = os.path.join(labels_path, file.replace('.jpg', '.txt'))
    copyfile(img_path, os.path.join(image_train, file))
    copyfile(label_path, os.path.join(label_train, file.replace('.jpg', '.txt')))

for file in pbar_val:
    img_path = os.path.join(images_path, file)
    label_path = os.path.join(labels_path, file.replace('.jpg', '.txt'))
    copyfile(img_path, os.path.join(image_val, file))
    copyfile(label_path, os.path.join(label_val, file.replace('.jpg', '.txt')))

for file in pbar_test:
    img_path = os.path.join(images_path, file)
    label_path = os.path.join(labels_path, file.replace('.jpg', '.txt'))
    copyfile(img_path, os.path.join(image_test, file))
    copyfile(label_path, os.path.join(label_test, file.replace('.jpg', '.txt')))

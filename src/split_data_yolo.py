import glob
from shutil import copyfile

from sklearn.model_selection import train_test_split
import os

# 设置数据集路径
from tqdm import tqdm

dataset_folder = r"F:\BaiduNetdiskDownload\BoadData\train_data"

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)


images_path = r"F:\BaiduNetdiskDownload\BoadData\Ship Identification\images"
labels_path = r"F:\BaiduNetdiskDownload\BoadData\Ship Identification\labels"

IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
files = sorted(glob.glob(os.path.join(images_path, "*.*")))
image_files = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
# 获取所有图像文件的路径
# image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

# 划分训练集和测试集
train_files, test = train_test_split(image_files, test_size=0.25, random_state=42)
# valid_files, test_files = train_test_split(test, test_size=0.5, random_state=42)

# 创建训练集和测试集的文件夹
image_train = dataset_folder + '/images/train'
label_train = dataset_folder + '/labels/train'

image_val = dataset_folder + '/images/val'
label_val = dataset_folder + '/labels/val'

# image_test = dataset_folder + '/images/test'
# label_test = dataset_folder + '/labels/test'

os.makedirs(image_train, exist_ok=True)
os.makedirs(label_train, exist_ok=True)

os.makedirs(image_val, exist_ok=True)
os.makedirs(label_val, exist_ok=True)

# os.makedirs(image_test, exist_ok=True)
# os.makedirs(label_test, exist_ok=True)

pbar_train = tqdm(train_files, desc=f'{image_train}')
pbar_val = tqdm(test, desc=f'{image_val}')
# pbar_test = tqdm(test_files, desc=f'{image_test}')
# 将图像文件和标签文件移动到对应的文件夹
for file in pbar_train:
    img_path = file
    basename = os.path.basename(file)
    filename = os.path.splitext(basename)[0]
    label_path = os.path.join(labels_path, filename + '.txt')

    copyfile(img_path, os.path.join(image_train, basename))
    copyfile(label_path, os.path.join(label_train, filename + '.txt'))

for file in pbar_val:
    img_path = file
    basename = os.path.basename(file)
    filename = os.path.splitext(basename)[0]
    label_path = os.path.join(labels_path, filename + '.txt')

    copyfile(img_path, os.path.join(image_val, basename))
    copyfile(label_path, os.path.join(label_val, filename + '.txt'))

# for file in pbar_test:
#     img_path = os.path.join(images_path, file)
#     label_path = os.path.join(labels_path, file.replace('.jpg', '.txt'))
#     copyfile(img_path, os.path.join(image_test, file))
#     copyfile(label_path, os.path.join(label_test, file.replace('.jpg', '.txt')))

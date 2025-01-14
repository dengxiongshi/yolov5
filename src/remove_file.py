import glob
import os
import shutil

from tqdm import tqdm

src_path = r"E:\downloads\compress\datasets\quantization_images\images"
dst_path = r"\\10.10.10.8\determined\alluxio\public\dengxiongshi\datasets\boat_person_car\train_data_20241114\labels\val"
save_path = r"E:\downloads\compress\datasets\quantization_images\labels"

os.chdir(src_path)
files = os.listdir()
files.sort()

pbar = tqdm(files, desc=f'Converting {src_path}')  # 进度条

for p in pbar:
    basename = os.path.basename(p)
    name = os.path.splitext(basename)[0]

    dst_file = os.path.join(dst_path, name + ".txt")
    save_file = os.path.join(save_path, name + ".txt")

    if os.path.exists(dst_file):
        shutil.copyfile(dst_file, save_file)



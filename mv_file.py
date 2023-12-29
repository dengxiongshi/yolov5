from shutil import copyfile
import os
import glob
from tqdm import tqdm

image_dir = r"E:\downloads\compress\datasets\video\test\test"

Annotation_dir = r"E:\downloads\compress\datasets\video\test\new"

save_dir = r"E:\downloads\compress\datasets\video\test\Annotations"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

image_list = glob.glob(image_dir + '/*.jpg')

pbar = tqdm(image_list, desc=f'Converting {image_dir}')

for p in pbar:

    basename = os.path.basename(p)
    name = os.path.splitext(basename)[0]

    annotation_name = os.path.join(Annotation_dir, name + '.xml')

    copyfile(annotation_name, os.path.join(save_dir, name + '.xml'))


from shutil import copyfile
import os
import glob
from tqdm import tqdm

image_dir = r"E:\downloads\compress\datasets\fire_smoke\fire_smoke_datasets-master\images"

Annotation_dir = r"E:\downloads\compress\datasets\fire_smoke\fire_smoke_datasets-master\Annotations"

save_dir = r"E:\downloads\compress\datasets\fire_smoke\fire_smoke_datasets-master\Annotations_new"

os.makedirs(save_dir, exist_ok=True)

image_list = glob.glob(image_dir + '/*.jpg')

pbar = tqdm(image_list, desc=f'Converting {image_dir}')

for p in pbar:

    basename = os.path.basename(p)
    name = os.path.splitext(basename)[0]

    annotation_name = os.path.join(Annotation_dir, name + '.xml')
    annotation_new = os.path.join(save_dir, name + '.xml')

    copyfile(annotation_name, annotation_new)

    # copyfile(p, os.path.join(save_dir, name + '.jpg'))


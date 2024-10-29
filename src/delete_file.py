import os
import glob
from tqdm import tqdm


def delete_file(filename):
    os.remove(filename)


image_dir = r"F:\BaiduNetdiskDownload\BoadData\seaships\labels"

Annotation_dir = r"F:\BaiduNetdiskDownload\BoadData\seaships\images"

# save_dir = r"E:\downloads\compress\datasets\VOC2007\Annotations"

# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

Annotation_list = glob.glob(image_dir + '/*.*')

pbar = tqdm(Annotation_list, desc=f'Converting {Annotation_dir}')

for p in pbar:
    # delete_file(p)

    basename = os.path.basename(p)
    name = os.path.splitext(basename)[0]

    image_name = os.path.join(Annotation_dir, name + '.jpg')

    if not os.path.exists(image_name):
        delete_file(p)


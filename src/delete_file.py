import os
import glob
from tqdm import tqdm


def delete_file(filename):
    os.remove(filename)


image_dir = r"D:\train_data\labels\train"

Annotation_dir = r"E:\downloads\compress\datasets\fire_smoke\fire_smoke_datasets-master\images"

# save_dir = r"E:\downloads\compress\datasets\VOC2007\Annotations"

# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

Annotation_list = glob.glob(image_dir + '/*.jpg')

pbar = tqdm(Annotation_list, desc=f'Converting {Annotation_dir}')

for p in pbar:
    delete_file(p)

    # basename = os.path.basename(p)
    # name = os.path.splitext(basename)[0]
    #
    # image_name = os.path.join(image_dir, name + '.txt')
    #
    # if not os.path.exists(image_name):
    #     delete_file(p)


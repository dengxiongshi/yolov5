import cv2
import os
import glob
from tqdm import tqdm


def list_subdirectories(directory):
    return [p for p in os.listdir(directory) if os.path.isdir(os.path.join(directory, p))]


def get_subdir_paths(directory):
    subdir_paths = [os.path.join(dp, d) for dp, dn, fn in os.walk(directory) for d in dn]
    return subdir_paths


image_dir = r'F:\BaiduNetdiskDownload\weather\rainy_image_dataset\training'

# dirs = list_subdirectories(image_dir)
dirs = get_subdir_paths(image_dir)

for dir in dirs:
    images = glob.glob(dir + '/*')
    pbar = tqdm(images, desc=f'{dir}')
    for image in pbar:
        if os.path.splitext(image)[1] != '.jpg':
            print(image)
        else:
            data = cv2.imread(image)
            if data is None:
                print(image)


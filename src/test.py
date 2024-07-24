import cv2
import os
import glob
from tqdm import tqdm


def list_subdirectories(directory):
    return [p for p in os.listdir(directory) if os.path.isdir(os.path.join(directory, p))]


def get_subdir_paths(directory):
    subdir_paths = [os.path.join(dp, d) for dp, dn, fn in os.walk(directory) for d in dn]
    return subdir_paths


def is_txt_file_empty(file_path):
    with open(file_path, 'r') as file:
        file_content = file.readlines()
        if file_content:
            return False
        else:
            return True


image_dir = r'E:\downloads\compress\datasets\天气\weather_classification\rainy'

save_dir = r'E:\downloads\compress\datasets\天气\weather_classification\rainy_label'

label_dir = r'\\10.10.10.8\determined\alluxio\dengxiongshi\datasets\weather\weather_mlabel\trainData_20240717\labels\train'

# os.makedirs(save_dir, exist_ok=True)

images = glob.glob(label_dir + '/*')
pbar = tqdm(images, desc=f'{image_dir}')
for image in pbar:
    # basename = os.path.basename(image)
    # save_file = os.path.join(save_dir, basename.replace(basename[-4:], '.txt'))
    #
    # data = ['3']
    # with open(save_file, 'a') as destination:
    #     destination.write('\n'.join(data))

    # if os.path.exists(save_file):
    #     continue
    # else:
    #     print(save_file)
    if is_txt_file_empty(image):
        print(os.path.basename(image))
    else:
        continue



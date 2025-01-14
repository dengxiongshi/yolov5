import glob
import os.path

import cv2
import numpy as np
import tqdm


def letterbox_image(image, target_size):
    src_height, src_width = image.shape[:2]
    target_width, target_height = target_size
    scale = min(target_width / src_width, target_height / src_height)
    new_width = int(src_width * scale)
    new_height = int(src_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_height, target_width, 3), 128, dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    return canvas


def convert(image):
    img = 1 - image

    return img


def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 1.0)


image_path = r"C:\Users\dengxs\Desktop\dengxs\20241104\ir_denoise"
save_path = r"C:\Users\dengxs\Desktop\dengxs\20241104\ir_resize_3"

os.makedirs(save_path, exist_ok=True)

images = glob.glob(image_path + '/*.jpg')

pbar = tqdm.tqdm(images, desc=f'Converting {image_path}')

image_size = (1920, 1536)

for p in pbar:
    basename = os.path.basename(p)
    save_name = os.path.join(save_path, basename)

    src_image = cv2.imread(p)
    # src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    # image = convert(src_image)
    # image = apply_gaussian_filter(image)
    
    image = letterbox_image(src_image, image_size)
    # src_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(save_name, image)

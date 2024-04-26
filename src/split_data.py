from shutil import copyfile
import os
import glob

parent_dir = r'F:\datasets\COCO'

image_path = r'F:\datasets\COCO\JPEGImages'
Annotation = r"F:\datasets\COCO\new\Annotations"

images_list = glob.glob(image_path + '/*.jpg')

image_len = len(images_list)

i = 0

while i <= image_len:

    images = images_list[i:i + 1000]

    save_dir = os.path.join(parent_dir, f'{i}')

    save_image = os.path.join(save_dir, 'images')
    save_annotation = os.path.join(save_dir, 'Annotations')

    os.makedirs(save_image, exist_ok=True)
    os.makedirs(save_annotation, exist_ok=True)

    for file in images:
        basename = os.path.basename(file)
        name = os.path.splitext(basename)[0]
        annotation_file = os.path.join(Annotation, name + '.xml')

        copyfile(file, os.path.join(save_image, basename))
        copyfile(annotation_file, os.path.join(save_annotation, name + '.xml'))

    i = i + 1000

if i > image_len:
    images = images_list[i - 1000:image_len]

    save_dir = os.path.join(parent_dir, f'{i}')

    save_image = os.path.join(save_dir, 'images')
    save_annotation = os.path.join(save_dir, 'Annotations')

    os.makedirs(save_image, exist_ok=True)
    os.makedirs(save_annotation, exist_ok=True)

    for file in images:
        basename = os.path.basename(file)
        name = os.path.splitext(basename)[0]
        annotation_file = os.path.join(Annotation, name + '.xml')

        copyfile(file, os.path.join(save_image, basename))
        copyfile(annotation_file, os.path.join(save_annotation, name + '.xml'))

import os
import glob
from tqdm import tqdm

if __name__ == "__main__":
    src_label = r"C:\Users\dengxs\Desktop\images"
    save_label = r"F:\datasets\COCO\new\labels"

    image_lists = glob.glob(src_label + '/*.jpg')

    pbar = tqdm(image_lists, desc=f'Converting {src_label}')

    for p in pbar:
        basename = os.path.basename(p)
        name = os.path.splitext(basename)[0]

        txtname = os.path.join(src_label, name + '.txt')

        if not os.path.exists(txtname):
            with open(txtname, 'a') as destination:
                destination.write('\n'.join(' '))

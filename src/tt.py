import os
import glob
from tqdm import tqdm


def change_label(source_file, destination_file):
    with open(source_file, 'r') as source:
        data = [line.strip() for line in source.readlines() if line.strip()]

    labels = []
    for line in data:
        parts = line.split()
        if len(parts) > 0:
            number = int(parts[0])
            if number == 3 or number == 4:
                parts[0] = '2'

            labels.append(' '.join(parts))

    with open(destination_file, 'a') as destination:
        destination.write('\n'.join(labels))


if __name__ == "__main__":
    src_label = "/data/yolov5/datasets/trainData_20240118/labels/val"
    save_label = "/data/yolov5/datasets/trainData_20240105/labels/train"

    # if os.path.exists(save_label) == False:
    #     os.makedirs(save_label)

    src_label_list = glob.glob(src_label + '/*.txt')
    # save_label_list = glob.glob(save_label + '/*.txt')
    pbar = tqdm(src_label_list, desc=f'Converting {src_label}')

    for p in pbar:
        with open(p, 'r') as source:
            data = [line.strip() for line in source.readlines() if line.strip()]

        for line in data:
            parts = line.split()
            if len(parts) > 0:
                number = int(parts[0])
                if number in [0, 1, 2]:
                    continue
                else:

                    print(p)

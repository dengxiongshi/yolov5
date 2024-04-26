import os
import glob
from tqdm import tqdm


def change_label(source_file, destination_file):
    with open(source_file, 'r') as source:
        data = [line.strip() for line in source.readlines() if line.strip()]

    modified_lines = []
    for line in data:
        parts = line.split()
        if len(parts) > 0:
            number = int(parts[0])
            if number in [3, 4, 10]:
                # if float(parts[3]) > 0.010 and float(parts[4]) > 0.0190:
                modified_lines.append(' '.join(parts))

    labels = []
    for line in modified_lines:
        parts = line.split()
        if len(parts) > 0:
            number = int(parts[0])
            if number == 4:
                parts[0] = '2'
            elif number == 10:
                parts[0] = '1'
            # van变成truck
            # elif number == 4:
            #     parts[0] = '2'
            # elif number == 5:
            #     parts[0] = '4'
            # elif number == 8:
            #     parts[0] = '3'

            labels.append(' '.join(parts))

    with open(destination_file, 'a') as destination:
        destination.write('\n'.join(labels))


if __name__ == "__main__":
    src_label = r"F:\datasets\ExDark_coco_yolo\labels"
    save_label = r"F:\datasets\ExDark_coco_yolo\labels_final"

    if os.path.exists(save_label) == False:
        os.makedirs(save_label)

    src_label_list = glob.glob(src_label + '/*.txt')
    # save_label_list = glob.glob(save_label + '/*.txt')
    pbar = tqdm(src_label_list, desc=f'Converting {src_label}')

    for p in pbar:
        src_file = p
        src_name = os.path.basename(src_file)

        save_file = os.path.join(save_label, src_name)
        change_label(src_file, save_file)

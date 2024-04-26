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


def del_label(source_file, destination_file):
    with open(source_file, 'r') as source:
        data = [line.strip() for line in source.readlines() if line.strip()]

    labels = []
    for line in data:
        parts = line.split()
        if len(parts) > 0:
            parts = parts[0:5]

            labels.append(' '.join(parts))

    with open(destination_file, 'a') as destination:
        destination.write('\n'.join(labels))


if __name__ == "__main__":
    src_label = r"E:\downloads\compress\datasets\licensePlate_detect\detect_plate_datasets\train_data\CRPD_TRAIN"
    save_label = r"E:\downloads\compress\datasets\licensePlate_detect\detect_plate_datasets\train_data\CRPD_TRAIN\labels"

    if os.path.exists(save_label) == False:
        os.makedirs(save_label)

    src_label_list = glob.glob(src_label + '/*.txt')
    # save_label_list = glob.glob(save_label + '/*.txt')
    pbar = tqdm(src_label_list, desc=f'Converting {src_label}')

    for p in pbar:
        src_file = p
        src_name = os.path.basename(src_file)

        save_file = os.path.join(save_label, src_name)
        del_label(src_file, save_file)

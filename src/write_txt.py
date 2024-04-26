import os
import glob
from tqdm import tqdm

def append_file(source_file, destination_file):
    with open(source_file, 'r') as source:
        data = [line.strip() for line in source.readlines() if line.strip()]

    with open(destination_file, 'a') as destination:
        destination.write('\n'.join(data))


if __name__ == "__main__":
    src_label = r"F:\datasets\COCO\other_labels_final"
    save_label = r"F:\datasets\COCO\new\labels"

    if os.path.exists(save_label) == False:
        os.makedirs(save_label)

    src_label_list = glob.glob(src_label + '/*.txt')
    # save_label_list = glob.glob(save_label + '/*.txt')
    pbar = tqdm(src_label_list, desc=f'Converting {src_label}')

    for p in pbar:
        src_file = p
        src_name = os.path.basename(src_file)

        save_file = os.path.join(save_label, src_name)
        append_file(src_file, save_file)

        # for j in range(len(save_label_list)):
        #     save_file = save_label_list[j]
        #     save_name = os.path.basename(save_file)
        #
        #     if src_name == save_name:
        #         append_file(src_file, save_file)
        #         break

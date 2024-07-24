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
    src_label = r"E:\downloads\compress\datasets\天气\weather_classification\thunder"
    save_label = r"E:\downloads\compress\datasets\天气\weather_classification\thunder_label"

    class_name = []

    if os.path.exists(save_label) == False:
        os.makedirs(save_label)


    with open(src_label, 'r') as source:
        data = [line.strip() for line in source.readlines() if line.strip()]

    pbar = tqdm(data, desc=f'Converting {src_label}')

    for line in pbar:
        line = line.split()
        name = line[0]
        save_file = os.path.join(save_label, name.replace(name[-4:], ".txt"))

        label = [int(x) for x in line[1:]]

        indices = [str(index) for index, value in enumerate(label) if value == 1]

        with open(save_file, 'a') as destination:
            destination.write('\n'.join(indices))

    # src_label_list = glob.glob(src_label + '/*.txt')
    # save_label_list = glob.glob(save_label + '/*.txt')
    # pbar = tqdm(src_label_list, desc=f'Converting {src_label}')

    # for p in pbar:
    #     with open(p, 'r') as source:
    #         data = [line.strip() for line in source.readlines() if line.strip()]
    #
    #     for line in data:
    #         parts = line.split()
    #         if len(parts) > 0:
    #             number = int(parts[0])
    #             if number in [0, 1, 2]:
    #                 continue
    #             else:
    #
    #                 print(p)

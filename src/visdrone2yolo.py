from utils.general import download, os, Path
import glob


def visdrone2yolo(dir):
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    if os.path.exists(os.path.join(dir, 'labels')) == False:
        os.makedirs(os.path.join(dir, 'labels'))
    # (os.path.join(dir, 'labels')).mkdir(parents=True, exist_ok=True)  # make labels directory
    txt_list = glob.glob(os.path.join(dir, 'annotations') + '/*.txt')
    pbar = tqdm(txt_list, desc=f'Converting {dir}')  # 进度条
    for f in pbar:
        name = os.path.basename(f)
        imagename = name.replace('txt', 'jpg')
        image_path = os.path.join(dir, 'images', imagename)
        img_size = Image.open(image_path).size
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(os.sep + 'annotations' + os.sep, os.sep + 'labels' + os.sep), 'w') as fl:
                    fl.writelines(lines)  # write label.txt


# Download
dir = r"E:\downloads\compress\datasets\VisDrone2019_DET"  # dataset root dir

# Convert
for d in 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev':
    visdrone2yolo(os.path.join(dir, d))  # convert VisDrone annotations to YOLO labels

import os

import cv2
import numpy as np
from tqdm import tqdm

raw_path = '/determined/alluxio/public/iDehaze_Image/RGB/train_os04a10_till0509_haze.txt'

save_path = '/determined/alluxio/dengxiongshi/datasets/weather/fog'
os.makedirs(save_path, exist_ok=True)

with open(raw_path, 'r') as source:
    data = [line.strip() for line in source.readlines() if line.strip()]

pbar = tqdm(data, desc=f'Converting raw to bgr {raw_path}')

H, W = 1520, 2688
pattern = 'rggb'

for p in pbar:
    basename = os.path.basename(p)
    name = basename.replace('bin', 'jpg')
    save_name = os.path.join(save_path, name)
    rawdata = np.fromfile(p, dtype=np.uint16)
    rawdata = rawdata.reshape(H, W)
    img = self_isp(rawdata / 65472, pattern)
    cv2.imwrite(save_name, img)











